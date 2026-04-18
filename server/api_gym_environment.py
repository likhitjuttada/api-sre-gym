from __future__ import annotations

import importlib
import json
import logging
import os
import random
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gym.registry import all_bug_types, get_app, tier_for
from gym.runner import GymRunner
from server.action_parser import ParsedAction, canonical_hash_key, parse as parse_action
from server.adversarial_designer import AdversarialDesigner
from server.curriculum import Curriculum
from server.judge import LLMJudge

load_dotenv()
logger = logging.getLogger(__name__)


MAX_STEPS_BY_TIER = {1: 15, 2: 18, 3: 22, 4: 25}
MALFORMED_PENALTY = -0.3
REPEAT_PENALTY = -0.5
TIMEOUT_PENALTY = -2.0
RESOLUTION_THRESHOLD = 3.0


@dataclass
class EpisodeState:
    bug_type: str
    difficulty: float
    ground_truth: dict[str, Any]
    base_url: str
    steps_taken: int = 0
    max_steps: int = 15
    rewards: list[float] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    request_counts: dict[str, int] = field(default_factory=dict)
    resolved: bool = False
    done: bool = False


class StepRequest(BaseModel):
    action: str


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: dict[str, Any]


class ResetResponse(BaseModel):
    observation: str
    info: dict[str, Any]


def _designer_probability(difficulty: float) -> float:
    if difficulty < 0.4:
        return 0.3
    if difficulty < 0.7:
        return 0.6
    return 0.9


def _format_response(action: ParsedAction, http_resp: httpx.Response | None, steps_remaining: int) -> str:
    if http_resp is None:
        return (
            "[ERROR] request could not be executed\n"
            f"[REMAINING_STEPS] {steps_remaining}"
        )
    headers_of_interest = ("content-type", "authorization", "www-authenticate",
                           "x-ratelimit-remaining", "x-ratelimit-limit", "location", "set-cookie")
    hdr_lines = []
    for k, v in http_resp.headers.items():
        if k.lower() in headers_of_interest:
            hdr_lines.append(f"{k}: {v}")
    body_text = http_resp.text
    if len(body_text) > 500:
        body_text = body_text[:500] + "... [truncated]"
    return (
        f"[STATUS] {http_resp.status_code} {http_resp.reason_phrase}\n"
        f"[HEADERS] {'; '.join(hdr_lines) or '(none)'}\n"
        f"[BODY]\n{body_text}\n"
        f"[REMAINING_STEPS] {steps_remaining}"
    )


class Environment:
    def __init__(self) -> None:
        self.runner = GymRunner()
        self.curriculum = Curriculum()
        self.designer = AdversarialDesigner()
        self.judge = LLMJudge()
        self.ep: EpisodeState | None = None
        self.enable_adversarial = os.getenv("ENABLE_ADVERSARIAL", "1") != "0"

    def _pick_module(self, bug_type: str, difficulty: float) -> str:
        if self.enable_adversarial and random.random() < _designer_probability(difficulty):
            weak = self.curriculum.weak_spots()
            generated = self.designer.design(bug_type, difficulty, weak)
            if generated:
                return generated
        return get_app(bug_type)

    def _finalize_if_abandoned(self) -> None:
        if self.ep and not self.ep.done:
            total = sum(self.ep.rewards)
            self.curriculum.record_episode(self.ep.bug_type, total, resolved=False)
            self.curriculum.save()

    def reset(self) -> ResetResponse:
        self._finalize_if_abandoned()
        self.runner.shutdown()

        bug_type, difficulty = self.curriculum.get_next_episode()
        module_path = self._pick_module(bug_type, difficulty)
        try:
            mod = importlib.import_module(module_path)
            ground_truth = dict(getattr(mod, "GROUND_TRUTH"))
        except Exception as exc:
            logger.warning("ground truth import failed for %s: %s; falling back", module_path, exc)
            module_path = get_app(bug_type)
            mod = importlib.import_module(module_path)
            ground_truth = dict(getattr(mod, "GROUND_TRUTH"))

        base_url = self.runner.start(module_path)
        max_steps = MAX_STEPS_BY_TIER.get(tier_for(bug_type), 20)
        self.ep = EpisodeState(
            bug_type=bug_type,
            difficulty=difficulty,
            ground_truth=ground_truth,
            base_url=base_url,
            max_steps=max_steps,
        )
        observation = (
            f"[BASE_URL] {base_url}\n"
            "[TASK] Explore this API to find any anomaly, bug, or unexpected behavior. "
            "Issue one HTTP request per turn. When identified, reply with `FOUND: <short description>`.\n"
            f"[REMAINING_STEPS] {max_steps}"
        )
        return ResetResponse(
            observation=observation,
            info={"bug_type": bug_type, "difficulty": difficulty, "max_steps": max_steps},
        )

    def step(self, action_text: str) -> StepResponse:
        if self.ep is None or self.ep.done:
            raise HTTPException(status_code=400, detail="no active episode; call /reset first")

        ep = self.ep
        ep.steps_taken += 1
        steps_remaining = ep.max_steps - ep.steps_taken

        parsed = parse_action(action_text)

        if parsed.kind == "invalid":
            reward = MALFORMED_PENALTY
            ep.rewards.append(reward)
            ep.history.append(f"TURN {ep.steps_taken}: [INVALID] {parsed.error}")
            observation = f"[ERROR] {parsed.error}\n[REMAINING_STEPS] {steps_remaining}"
            done = self._check_timeout(ep)
            return StepResponse(
                observation=observation,
                reward=reward + (TIMEOUT_PENALTY if done else 0.0),
                done=done,
                info={"parsed": asdict(parsed)},
            )

        if parsed.kind == "found":
            res = self.judge.verify_resolution(ep.ground_truth, parsed.raw_claim or "", ep.history)
            resolved = res.score > RESOLUTION_THRESHOLD
            ep.resolved = resolved
            ep.done = True
            efficiency = max(0.0, 1.0 - ep.steps_taken / ep.max_steps)
            bonus = res.score + (efficiency if resolved else 0.0)
            ep.rewards.append(bonus)
            ep.history.append(f"TURN {ep.steps_taken}: FOUND {parsed.raw_claim}")
            self.curriculum.record_episode(ep.bug_type, sum(ep.rewards), resolved=resolved)
            self.curriculum.save()
            observation = f"[RESOLUTION] score={res.score:.2f} feedback={res.feedback}"
            return StepResponse(
                observation=observation,
                reward=bonus,
                done=True,
                info={"resolved": resolved, "resolution_score": res.score, "feedback": res.feedback},
            )

        key = canonical_hash_key(parsed)
        count = ep.request_counts.get(key, 0) + 1
        ep.request_counts[key] = count

        if count >= 3:
            reward = REPEAT_PENALTY
            ep.rewards.append(reward)
            ep.history.append(f"TURN {ep.steps_taken}: [REPEAT x{count}] {key}")
            observation = f"[REPEAT] This action was already attempted {count - 1} times.\n[REMAINING_STEPS] {steps_remaining}"
            done = self._check_timeout(ep)
            return StepResponse(
                observation=observation,
                reward=reward + (TIMEOUT_PENALTY if done else 0.0),
                done=done,
                info={"repeat_count": count},
            )

        http_resp: httpx.Response | None = None
        try:
            http_resp = httpx.request(
                parsed.method or "GET",
                f"{ep.base_url}{parsed.path or '/'}",
                json=parsed.body if parsed.body else None,
                headers=parsed.headers or None,
                timeout=5.0,
            )
        except httpx.HTTPError as exc:
            observation = f"[ERROR] http_error: {exc}\n[REMAINING_STEPS] {steps_remaining}"
            ep.history.append(f"TURN {ep.steps_taken}: {key} -> http_error: {exc}")
            reward = MALFORMED_PENALTY
            ep.rewards.append(reward)
            done = self._check_timeout(ep)
            return StepResponse(
                observation=observation,
                reward=reward + (TIMEOUT_PENALTY if done else 0.0),
                done=done,
                info={"http_error": str(exc)},
            )

        observation = _format_response(parsed, http_resp, steps_remaining)
        ep.history.append(
            f"TURN {ep.steps_taken}: {key} -> {http_resp.status_code} {http_resp.text[:120]}"
        )
        score = self.judge.evaluate_step(
            bug_category=ep.bug_type.split("_")[0],
            difficulty=ep.difficulty,
            action_text=f"{parsed.method} {parsed.path} {json.dumps(parsed.body) if parsed.body else ''}",
            response_summary=observation,
            recent_history=ep.history,
        )
        ep.rewards.append(score.score)
        done = self._check_timeout(ep)
        return StepResponse(
            observation=observation,
            reward=score.score + (TIMEOUT_PENALTY if done else 0.0),
            done=done,
            info={"judge_feedback": score.feedback},
        )

    def _check_timeout(self, ep: EpisodeState) -> bool:
        if ep.steps_taken >= ep.max_steps and not ep.done:
            ep.done = True
            self.curriculum.record_episode(ep.bug_type, sum(ep.rewards), resolved=False)
            self.curriculum.save()
            return True
        return False


_env: Environment | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    _env = Environment()
    try:
        yield
    finally:
        if _env is not None:
            _env.runner.shutdown()


app = FastAPI(title="api-testing-gym", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/reset", response_model=ResetResponse)
def reset() -> ResetResponse:
    assert _env is not None
    return _env.reset()


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    assert _env is not None
    return _env.step(req.action)
