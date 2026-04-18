from fastapi import FastAPI

from gym.apps._base import GroundTruth, make_app, run_app

GROUND_TRUTH: GroundTruth = {
    "bug_type": "hidden_endpoint",
    "location": "GET /.env",
    "description": "An undocumented /.env endpoint returns environment variables including secrets; it is excluded from the OpenAPI schema",
    "detection_criteria": [
        "hidden endpoint",
        "/.env",
        "undocumented",
        "environment variables",
        "leaked secrets",
    ],
}

app: FastAPI = make_app(GROUND_TRUTH)


@app.get("/items")
def list_items() -> list[dict]:
    return [{"id": 1, "name": "widget"}, {"id": 2, "name": "gadget"}]


@app.get("/items/{item_id}")
def get_item(item_id: int) -> dict:
    return {"id": item_id, "name": "widget"}


@app.get("/.env", include_in_schema=False)
def dotenv() -> dict:
    return {
        "DATABASE_URL": "postgres://prod:hunter2@db.internal:5432/app",
        "STRIPE_SECRET_KEY": "sk_live_xxxxxxxxxxxxxxxxxxxx",
        "JWT_SIGNING_KEY": "supersecret-do-not-share",
    }


if __name__ == "__main__":
    run_app(app)
