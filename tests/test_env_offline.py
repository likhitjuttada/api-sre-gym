import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake")
os.environ["ENABLE_ADVERSARIAL"] = "0"


class FakeStepScore:
    def __init__(self, score: float, feedback: str = "ok") -> None:
        self.score = score
        self.feedback = feedback


class FakeResolutionScore:
    def __init__(self, score: float, feedback: str = "ok") -> None:
        self.score = score
        self.feedback = feedback


def main() -> None:
    from server import api_gym_environment as env_mod
    from server.judge import LLMJudge

    def fake_step(self, *a, **kw):
        return FakeStepScore(0.5, "fake ok")

    def fake_resolve(self, *a, **kw):
        return FakeResolutionScore(4.5, "fake solved")

    with patch.object(LLMJudge, "evaluate_step", new=fake_step), \
         patch.object(LLMJudge, "verify_resolution", new=fake_resolve):
        environment = env_mod.Environment()
        reset = environment.reset()
        print("RESET", reset.info)
        assert "BASE_URL" in reset.observation
        bug = reset.info["bug_type"]
        print("bug:", bug)

        r1 = environment.step("GET /healthz")
        print("STEP1", r1.reward, r1.done)
        assert abs(r1.reward - 0.5) < 1e-6
        assert r1.done is False

        r2 = environment.step("not a real action")
        print("STEP2", r2.reward, r2.done)
        assert r2.reward < 0

        r3 = environment.step("FOUND: I found the bug in the api")
        print("STEP3", r3.reward, r3.done, r3.info)
        assert r3.done is True
        assert r3.reward > 3.0

        environment.runner.shutdown()
        print("PASS")


if __name__ == "__main__":
    main()
