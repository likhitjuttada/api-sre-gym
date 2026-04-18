from __future__ import annotations

BUG_REGISTRY: dict[str, str] = {
    "wrong_status": "gym.apps.wrong_status",
    "missing_auth": "gym.apps.missing_auth",
    "data_leak": "gym.apps.data_leak",
    "hidden_endpoint": "gym.apps.hidden_endpoint",
    "schema_mismatch": "gym.apps.schema_mismatch",
    "rate_limit_bypass": "gym.apps.rate_limit_bypass",
    "auth_bypass": "gym.apps.auth_bypass",
    "race_condition": "gym.apps.race_condition",
}


TIERS: dict[int, list[str]] = {
    1: ["wrong_status", "missing_auth"],
    2: ["data_leak", "hidden_endpoint", "schema_mismatch"],
    3: ["rate_limit_bypass", "auth_bypass"],
    4: ["race_condition", "chained_vuln"],
}


def all_bug_types() -> list[str]:
    return list(BUG_REGISTRY.keys())


def get_app(bug_type: str, generated_module: str | None = None) -> str:
    if generated_module:
        return generated_module
    if bug_type not in BUG_REGISTRY:
        raise KeyError(f"unknown bug_type: {bug_type}")
    return BUG_REGISTRY[bug_type]


def tier_for(bug_type: str) -> int:
    for tier, bugs in TIERS.items():
        if bug_type in bugs:
            return tier
    return 4
