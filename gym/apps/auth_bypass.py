from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from gym.apps._base import GroundTruth, make_app, run_app

GROUND_TRUTH: GroundTruth = {
    "bug_type": "auth_bypass",
    "location": "GET /secret (trailing slash variant)",
    "description": "Middleware enforces auth on exact path /secret but /secret/ (trailing slash) is routed to the same handler without middleware matching",
    "detection_criteria": [
        "auth bypass",
        "trailing slash",
        "/secret/",
        "bypass",
        "middleware",
    ],
}

app: FastAPI = make_app(GROUND_TRUTH)

_TOKEN = "Bearer valid-token"


@app.middleware("http")
async def auth_guard(request: Request, call_next):
    if request.url.path == "/secret":
        if request.headers.get("authorization") != _TOKEN:
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
    return await call_next(request)


@app.get("/secret")
def secret_main() -> dict:
    return {"secret": "the cake is a lie"}


@app.get("/secret/")
def secret_slash() -> dict:
    return {"secret": "the cake is a lie"}


@app.get("/public")
def public() -> dict:
    return {"msg": "hello"}


if __name__ == "__main__":
    run_app(app)
