from fastapi import FastAPI
from pydantic import BaseModel

from gym.apps._base import GroundTruth, make_app, run_app

GROUND_TRUTH: GroundTruth = {
    "bug_type": "wrong_status",
    "location": "POST /users",
    "description": "POST /users creates a user but returns 200 OK instead of 201 Created",
    "detection_criteria": [
        "wrong status",
        "200",
        "201",
        "POST /users",
        "should return 201",
    ],
}

app: FastAPI = make_app(GROUND_TRUTH)

_USERS: dict[int, dict] = {1: {"id": 1, "name": "alice"}}
_NEXT_ID = [2]


class UserCreate(BaseModel):
    name: str


@app.get("/users")
def list_users() -> list[dict]:
    return list(_USERS.values())


@app.get("/users/{user_id}")
def get_user(user_id: int) -> dict:
    return _USERS.get(user_id, {"error": "not found"})


@app.post("/users")
def create_user(user: UserCreate) -> dict:
    uid = _NEXT_ID[0]
    _NEXT_ID[0] += 1
    rec = {"id": uid, "name": user.name}
    _USERS[uid] = rec
    return rec


if __name__ == "__main__":
    run_app(app)
