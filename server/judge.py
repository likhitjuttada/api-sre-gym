from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _persona(difficulty: float) -> str:
    if difficulty < 0.34:
        return "junior"
    if difficulty < 0.67:
        return "senior"
    return "principal"


@dataclass
class StepScore:
    score: float
    feedback: str


@dataclass
class ResolutionScore:
    score: float
    feedback: str


class LLMJudge:
    def __init__(
        self,
        step_model: str | None = None,
        resolution_model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client: Any | None = None
        self.step_model = step_model or os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001")
        self.resolution_model = resolution_model or os.getenv("RESOLUTION_MODEL", "claude-sonnet-4-6")
        self._step_prompt = _load_prompt("judge_step.txt")
        self._resolution_prompt = _load_prompt("judge_resolution.txt")

    @property
    def client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def evaluate_step(
        self,
        bug_category: str,
        difficulty: float,
        action_text: str,
        response_summary: str,
        recent_history: list[str],
    ) -> StepScore:
        history_text = "\n".join(recent_history[-8:]) if recent_history else "(none)"
        user = (
            f"Persona: {_persona(difficulty)} (difficulty={difficulty:.2f})\n"
            f"Bug category: {bug_category}\n\n"
            f"Recent history (up to 8 turns):\n{history_text}\n\n"
            f"Current action:\n{action_text}\n\n"
            f"Response summary:\n{response_summary}\n\n"
            'Return JSON: {"score": float in [-1, 1], "feedback": "one sentence"}'
        )
        try:
            resp = self.client.messages.create(
                model=self.step_model,
                max_tokens=256,
                temperature=0.3,
                system=self._step_prompt,
                messages=[{"role": "user", "content": user}],
            )
            text = resp.content[0].text if resp.content else ""
        except Exception as exc:
            logger.warning("judge step api error: %s", exc)
            return StepScore(score=0.0, feedback=f"judge_error: {exc}")

        data = _extract_json(text)
        if not data:
            return StepScore(score=0.0, feedback=f"judge_parse_error: {text[:120]}")
        score = float(data.get("score", 0.0))
        score = max(-1.0, min(1.0, score))
        feedback = str(data.get("feedback", ""))[:240]
        return StepScore(score=score, feedback=feedback)

    def verify_resolution(
        self,
        ground_truth: dict[str, Any],
        agent_claim: str,
        full_history: list[str],
    ) -> ResolutionScore:
        history_text = "\n".join(full_history[-40:]) if full_history else "(none)"
        user = (
            f"Ground truth:\n{json.dumps(ground_truth, indent=2)}\n\n"
            f"Agent claim:\n{agent_claim}\n\n"
            f"Full action history (last 40 lines):\n{history_text}\n\n"
            'Return JSON: {"score": float in [0, 5], "feedback": "one sentence"}'
        )
        try:
            resp = self.client.messages.create(
                model=self.resolution_model,
                max_tokens=256,
                temperature=0.1,
                system=self._resolution_prompt,
                messages=[{"role": "user", "content": user}],
            )
            text = resp.content[0].text if resp.content else ""
        except Exception as exc:
            logger.warning("judge resolution api error: %s", exc)
            return ResolutionScore(score=0.0, feedback=f"judge_error: {exc}")

        data = _extract_json(text)
        if not data:
            return ResolutionScore(score=0.0, feedback=f"judge_parse_error: {text[:120]}")
        score = float(data.get("score", 0.0))
        score = max(0.0, min(5.0, score))
        feedback = str(data.get("feedback", ""))[:240]
        return ResolutionScore(score=score, feedback=feedback)
