from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("train")

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B")
ENV_SERVER_URL = f"http://127.0.0.1:{os.getenv('ENV_SERVER_PORT', '8000')}"
VLLM_URL = f"http://127.0.0.1:{os.getenv('VLLM_PORT', '8080')}"
MAX_TURNS = 20
MAX_COMPLETION_LEN = 4096
HISTORY_WINDOW = 8


@dataclass
class EpisodeTranscript:
    bug_type: str
    difficulty: float
    steps: int
    total_reward: float
    resolution_score: float
    resolved: bool
    turns: list[dict] = field(default_factory=list)


def _wait_for_service(url: str, name: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code < 500:
                logger.info("service %s reachable at %s", name, url)
                return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"{name} not reachable at {url} after {timeout}s")


def _load_system_prompt() -> str:
    return (ROOT / "prompts" / "system.txt").read_text(encoding="utf-8")


def _build_chat_prompt(
    tokenizer,
    system_prompt: str,
    history: list[dict],
    observation: str,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-HISTORY_WINDOW:]:
        messages.append({"role": "user", "content": turn["observation"]})
        messages.append({"role": "assistant", "content": turn["action"]})
    messages.append({"role": "user", "content": observation})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_episode(tokenizer, generate_fn, system_prompt: str) -> tuple[list[dict], EpisodeTranscript]:
    reset = httpx.post(f"{ENV_SERVER_URL}/reset", timeout=120.0).json()
    observation: str = reset["observation"]
    info: dict = reset["info"]
    transcript = EpisodeTranscript(
        bug_type=info["bug_type"],
        difficulty=info["difficulty"],
        steps=0,
        total_reward=0.0,
        resolution_score=0.0,
        resolved=False,
    )
    history: list[dict] = []
    turn_segments: list[dict] = []

    for turn in range(MAX_TURNS):
        prompt_text = _build_chat_prompt(tokenizer, system_prompt, history, observation)
        generation = generate_fn(prompt_text)
        action_text = generation["text"]
        prompt_ids: list[int] = generation["prompt_ids"]
        completion_ids: list[int] = generation["completion_ids"]

        step = httpx.post(
            f"{ENV_SERVER_URL}/step",
            json={"action": action_text},
            timeout=120.0,
        ).json()
        reward = float(step["reward"])
        done = bool(step["done"])
        transcript.steps += 1
        transcript.total_reward += reward
        turn_segments.append(
            {
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "reward": reward,
            }
        )
        transcript.turns.append(
            {
                "turn": turn,
                "observation": observation,
                "action": action_text,
                "reward": reward,
                "info": step.get("info", {}),
            }
        )
        history.append({"observation": observation, "action": action_text})
        observation = step["observation"]
        if done:
            info_out = step.get("info", {})
            transcript.resolution_score = float(info_out.get("resolution_score", 0.0))
            transcript.resolved = bool(info_out.get("resolved", False))
            break

    return turn_segments, transcript


def _assign_sparse_rewards(segments: list[dict]) -> tuple[list[int], list[int], list[float]]:
    if not segments:
        return [], [], []
    first_prompt = segments[0]["prompt_ids"]
    all_completion: list[int] = []
    per_token_rewards: list[float] = []
    for seg in segments:
        comp = list(seg["completion_ids"])
        all_completion.extend(comp)
        if not comp:
            continue
        per_token_rewards.extend([0.0] * (len(comp) - 1))
        per_token_rewards.append(float(seg["reward"]))
    return list(first_prompt), all_completion, per_token_rewards


def _append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _vllm_generate(prompt_text: str, tokenizer, max_new_tokens: int = 256) -> dict:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "max_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "stop": ["\n\n"],
    }
    resp = httpx.post(f"{VLLM_URL}/v1/completions", json=payload, timeout=120.0)
    resp.raise_for_status()
    body = resp.json()
    text = body["choices"][0]["text"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return {"text": text, "prompt_ids": prompt_ids, "completion_ids": completion_ids}


def build_grpo_trainer(tokenizer):
    cfg = GRPOConfig(
        output_dir=str(ROOT / "checkpoints"),
        num_generations=8,
        loss_type="dapo",
        epsilon=0.2,
        epsilon_high=0.28,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=1,
        learning_rate=2e-6,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        beta=0.01,
        max_completion_length=MAX_COMPLETION_LEN,
        logging_steps=1,
        save_steps=50,
        use_vllm=True,
        vllm_mode="colocate",
        report_to="none",
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    def rollout_func(prompts: list[str], args, processing_class, **kwargs) -> dict:
        system_prompt = _load_system_prompt()
        batch_prompt_ids: list[list[int]] = []
        batch_completion_ids: list[list[int]] = []
        batch_rewards: list[list[float]] = []
        for _ in prompts:
            segments, transcript = run_episode(
                tokenizer,
                lambda p: _vllm_generate(p, tokenizer),
                system_prompt,
            )
            p_ids, c_ids, rewards = _assign_sparse_rewards(segments)
            batch_prompt_ids.append(p_ids)
            batch_completion_ids.append(c_ids)
            batch_rewards.append(rewards)
            _append_csv(
                LOGS_DIR / "rewards.csv",
                {
                    "ts": datetime.utcnow().isoformat(),
                    "bug_type": transcript.bug_type,
                    "difficulty": f"{transcript.difficulty:.3f}",
                    "steps": transcript.steps,
                    "total_reward": f"{transcript.total_reward:.3f}",
                    "resolution_score": f"{transcript.resolution_score:.3f}",
                    "resolved": int(transcript.resolved),
                },
            )
            _append_jsonl(LOGS_DIR / "transcripts.jsonl", transcript.__dict__)
        return {
            "prompt_ids": batch_prompt_ids,
            "completion_ids": batch_completion_ids,
            "rewards": batch_rewards,
        }

    dummy_dataset = Dataset.from_list([{"prompt": "episode"}] * 1024)

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        args=cfg,
        train_dataset=dummy_dataset,
        processing_class=tokenizer,
        peft_config=lora,
        rollout_func=rollout_func,
    )
    return trainer


def main() -> None:
    _wait_for_service(f"{ENV_SERVER_URL}/health", "env-server")
    _wait_for_service(f"{VLLM_URL}/v1/models", "vllm")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = build_grpo_trainer(tokenizer)
    logger.info("starting GRPO training loop")
    trainer.train()


if __name__ == "__main__":
    main()
