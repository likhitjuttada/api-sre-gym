# API Testing RL Gym

Train a small open-source LLM (Qwen3-1.7B) to perform exploratory API testing via GRPO. A frontier model (Claude) acts as adversarial scenario designer and judge, evolving buggy mock APIs to target the agent's weaknesses. Architecture modeled after [kube-sre-gym](https://github.com/sid-rp/kube-sre-gym).

## How it works

1. The **environment server** spins up a buggy FastAPI app per episode (wrong status codes, missing auth, data leaks, hidden endpoints, schema mismatches, rate-limit/auth bypasses, race conditions).
2. The **agent model** issues one HTTP request per turn against the app, reads the response, and eventually reports `FOUND: <description>`.
3. Per-step **Claude Haiku** scores the information value of each request (-1 to +1). Final **Claude Sonnet** verifies whether the agent correctly identified the bug (0 to 5).
4. A **curriculum** tracks per-bug mastery and escalates difficulty; once the agent masters tier N, tier N+1 unlocks.
5. An **adversarial designer** (Claude Sonnet) generates new buggy apps targeting weak spots, validated by AST parse -> import -> runtime probe.
6. **GRPO** (TRL 0.29) with LoRA-16 fine-tunes the agent using sparse per-token rewards (reward lands on last token of each action segment).

## Requirements

- Linux or WSL (vLLM does not support native Windows)
- H100 or A100 GPU (80GB recommended)
- Python 3.10+
- Anthropic API key

## Setup

```bash
# clone & enter
cd rlvr

# virtualenv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install
pip install -r requirements.txt

# env
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY and (optionally) HF_TOKEN
```

## Run

Three services in three terminals (in this order).

### 1. vLLM inference server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --port 8080 \
  --gpu-memory-utilization 0.4
```

### 2. Environment server

```bash
uvicorn server.api_gym_environment:app --port 8000
```

### 3. Training

```bash
python train.py
```

`train.py` waits for both services to be healthy before launching GRPO. Checkpoints go to `checkpoints/`, episode transcripts to `logs/transcripts.jsonl`, reward CSV to `logs/rewards.csv`.

## Codebase structure

```
rlvr/
├── train.py                          # GRPO entry point
├── requirements.txt
├── .env.example
├── prompts/
│   ├── system.txt                    # agent persona
│   ├── judge_step.txt                # per-step scorer (blind to ground truth)
│   ├── judge_resolution.txt          # final-claim verifier
│   └── designer.txt                  # adversarial app generator
├── server/
│   ├── api_gym_environment.py        # FastAPI reset/step endpoints, episode state
│   ├── judge.py                      # LLMJudge: Haiku (step) + Sonnet (resolution)
│   ├── adversarial_designer.py       # Claude Sonnet generates new buggy apps
│   ├── action_parser.py              # Parse model output -> ParsedAction
│   └── curriculum.py                 # Per-bug mastery + tier escalation
├── gym/
│   ├── runner.py                     # Subprocess launcher for mock apps
│   ├── registry.py                   # bug_type -> module path; tier map
│   ├── apps/                         # 8 hand-written buggy apps
│   │   ├── _base.py                  # GROUND_TRUTH contract + make_app helper
│   │   ├── wrong_status.py           # tier 1
│   │   ├── missing_auth.py           # tier 1
│   │   ├── data_leak.py              # tier 2
│   │   ├── hidden_endpoint.py        # tier 2
│   │   ├── schema_mismatch.py        # tier 2
│   │   ├── rate_limit_bypass.py      # tier 3
│   │   ├── auth_bypass.py            # tier 3
│   │   └── race_condition.py         # tier 4
│   └── generated/                    # Claude-designed apps land here
└── tests/
    ├── test_parser.py                # action parser unit tests
    ├── test_runner.py                # all 8 apps start + /healthz + /openapi.json
    └── test_env_offline.py           # full reset/step cycle with mocked judge
```

## Ground truth contract

Every buggy app in `gym/apps/` exports a module-level `GROUND_TRUTH: GroundTruth` dict:

```python
GROUND_TRUTH: GroundTruth = {
    "bug_type": "missing_auth",
    "location": "GET /admin/stats",
    "description": "...",
    "detection_criteria": ["missing auth", "/admin/stats", ...],
}
```

The environment imports this via `importlib` on reset. The per-step judge never sees it (prevents answer leakage in feedback). The resolution verifier sees the full dict.

## Reward shaping

| Signal | Value | Source |
|---|---|---|
| Malformed action | -0.3 | server-side, skips judge |
| Repeated request (3rd time) | -0.5 | server-side, skips judge |
| Per-step information gain | [-1, +1] | Claude Haiku |
| Resolution bonus | [0, 5] | Claude Sonnet |
| Efficiency bonus (on resolve) | [0, 1] | `1 - steps/max_steps` |
| Timeout penalty | -2.0 | on exceeding `max_steps` |

Rewards are assigned to the **last completion token** of each action segment; all other positions are zero-filled.

## Tests

No real API keys or GPUs needed for these:

```bash
python tests/test_parser.py          # 8/8 parser cases
python tests/test_runner.py          # all 8 apps boot + healthz + openapi
python tests/test_env_offline.py     # reset/step/found cycle with mocked judge
```

## Configuration

Environment variables (see `.env.example`):

| Var | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for judge + designer |
| `HF_TOKEN` | — | Optional, for gated models |
| `ENV_SERVER_PORT` | 8000 | FastAPI env server port |
| `VLLM_PORT` | 8080 | vLLM OpenAI API port |
| `MODEL_NAME` | `Qwen/Qwen3-1.7B` | HF model id for agent |
| `JUDGE_MODEL` | `claude-haiku-4-5-20251001` | Per-step scorer |
| `DESIGNER_MODEL` | `claude-sonnet-4-6` | Adversarial designer |
| `RESOLUTION_MODEL` | `claude-sonnet-4-6` | Resolution verifier |
| `ENABLE_ADVERSARIAL` | `1` | Set `0` to disable Claude-designed apps |

## Estimated cost

- ~$300 Anthropic spend for 10K training episodes (Haiku per-step + Sonnet resolution).
- Designer calls scale with difficulty; expect another $50-100 for heavy adversarial training.

## Known limitations

- vLLM is Linux-only in practice; develop on WSL or a cloud Linux box.
- The env server is single-tenant; GRPO's 8 rollouts serialize through it.
- TRL 0.29 GRPO rollout_func signature may have shifted between minor releases; verify against the installed version before launching a long run.
