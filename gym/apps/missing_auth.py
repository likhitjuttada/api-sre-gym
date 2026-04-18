from fastapi import FastAPI, Header, HTTPException

from gym.apps._base import GroundTruth, make_app, run_app

GROUND_TRUTH: GroundTruth = {
    "bug_type": "missing_auth",
    "location": "GET /admin/stats",
    "description": "GET /admin/stats exposes internal metrics without requiring authentication, while other /admin/* routes correctly require it",
    "detection_criteria": [
        "missing auth",
        "/admin/stats",
        "no authentication",
        "unauthenticated",
        "no authorization",
    ],
}

app: FastAPI = make_app(GROUND_TRUTH)

_VALID_TOKEN = "secret-admin-token"


def _require_auth(authorization: str | None) -> None:
    if not authorization or authorization != f"Bearer {_VALID_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/admin/users")
def admin_users(authorization: str | None = Header(default=None)) -> dict:
    _require_auth(authorization)
    return {"users": ["alice", "bob"]}


@app.get("/admin/config")
def admin_config(authorization: str | None = Header(default=None)) -> dict:
    _require_auth(authorization)
    return {"mode": "prod", "debug": False}


@app.get("/admin/stats")
def admin_stats() -> dict:
    return {
        "requests_total": 12450,
        "errors_total": 37,
        "internal_db_host": "pg-prod-01.internal",
        "build_sha": "a1b2c3d4",
    }


@app.get("/")
def root() -> dict:
    return {"name": "admin-api"}


if __name__ == "__main__":
    run_app(app)
