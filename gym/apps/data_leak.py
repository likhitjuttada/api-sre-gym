from fastapi import FastAPI

from gym.apps._base import GroundTruth, make_app, run_app

GROUND_TRUTH: GroundTruth = {
    "bug_type": "data_leak",
    "location": "GET /users/{id}",
    "description": "GET /users/{id} leaks sensitive fields (password_hash, ssn, api_key) in the response",
    "detection_criteria": [
        "data leak",
        "password_hash",
        "ssn",
        "api_key",
        "sensitive",
        "PII",
    ],
}

app: FastAPI = make_app(GROUND_TRUTH)

_USERS: dict[int, dict] = {
    1: {
        "id": 1,
        "name": "alice",
        "email": "alice@example.com",
        "password_hash": "$2b$12$abcdefghijklmnopqrstuv",
        "ssn": "123-45-6789",
        "api_key": "sk_live_aaaabbbbccccdddd",
    },
    2: {
        "id": 2,
        "name": "bob",
        "email": "bob@example.com",
        "password_hash": "$2b$12$zyxwvutsrqponmlkjihgfe",
        "ssn": "987-65-4321",
        "api_key": "sk_live_9999888877776666",
    },
}


@app.get("/users")
def list_users() -> list[dict]:
    return [{"id": u["id"], "name": u["name"]} for u in _USERS.values()]


@app.get("/users/{user_id}")
def get_user(user_id: int) -> dict:
    return _USERS.get(user_id, {"error": "not found"})


if __name__ == "__main__":
    run_app(app)
