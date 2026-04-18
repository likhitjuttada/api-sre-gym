from __future__ import annotations

import ast
import importlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import httpx

from gym.runner import GymRunner

logger = logging.getLogger(__name__)

_GENERATED_DIR = Path(__file__).resolve().parent.parent / "gym" / "generated"
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_FEW_SHOT_APPS = [
    Path(__file__).resolve().parent.parent / "gym" / "apps" / "wrong_status.py",
    Path(__file__).resolve().parent.parent / "gym" / "apps" / "data_leak.py",
]

_CODE_FENCE_RE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


class AdversarialDesigner:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client: Any | None = None
        self.model = model or os.getenv("DESIGNER_MODEL", "claude-sonnet-4-6")
        self._prompt = (_PROMPTS_DIR / "designer.txt").read_text(encoding="utf-8")
        _GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def _build_user_message(
        self,
        bug_type: str,
        difficulty: float,
        weak_spots: list[str],
    ) -> str:
        examples = "\n\n".join(
            f"### {p.name}\n```python\n{p.read_text(encoding='utf-8')}\n```"
            for p in _FEW_SHOT_APPS
        )
        weak_text = ", ".join(weak_spots) if weak_spots else "(none)"
        return (
            f"Target bug type: {bug_type}\n"
            f"Difficulty: {difficulty:.2f}\n"
            f"Agent weak spots to exploit: {weak_text}\n\n"
            f"Reference examples:\n{examples}\n\n"
            "Generate a new buggy FastAPI module following the same contract. "
            "It must be importable, include a GROUND_TRUTH dict at module level, "
            "use gym.apps._base.make_app and run_app, and contain the target bug."
        )

    def design(
        self,
        bug_type: str,
        difficulty: float,
        weak_spots: list[str],
    ) -> str | None:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.7,
                system=self._prompt,
                messages=[{"role": "user", "content": self._build_user_message(bug_type, difficulty, weak_spots)}],
            )
            text = resp.content[0].text if resp.content else ""
        except Exception as exc:
            logger.warning("designer api error: %s", exc)
            return None

        code = _extract_code(text)
        if not code or "GROUND_TRUTH" not in code or "make_app" not in code:
            logger.warning("designer output missing required constructs")
            return None

        try:
            ast.parse(code)
        except SyntaxError as exc:
            logger.warning("designer output has syntax error: %s", exc)
            return None

        module_name = f"_gen_{uuid.uuid4().hex[:10]}"
        file_path = _GENERATED_DIR / f"{module_name}.py"
        file_path.write_text(code, encoding="utf-8")

        module_path = f"gym.generated.{module_name}"
        if not self._validate_runtime(module_path, file_path):
            try:
                file_path.unlink()
            except OSError:
                pass
            return None
        return module_path

    def _validate_runtime(self, module_path: str, file_path: Path) -> bool:
        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:
            logger.warning("designer module import failed: %s", exc)
            return False
        if not hasattr(mod, "GROUND_TRUTH") or not hasattr(mod, "app"):
            logger.warning("designer module missing GROUND_TRUTH or app")
            return False

        runner = GymRunner(startup_timeout=8.0)
        try:
            base_url = runner.start(module_path)
        except RuntimeError as exc:
            logger.warning("designer runtime probe failed: %s", exc)
            return False

        try:
            h = httpx.get(f"{base_url}/healthz", timeout=2.0)
            if h.status_code != 200:
                return False
            o = httpx.get(f"{base_url}/openapi.json", timeout=2.0)
            if o.status_code != 200:
                return False
        except httpx.HTTPError as exc:
            logger.warning("designer probe http error: %s", exc)
            return False
        finally:
            runner.shutdown()
        return True
