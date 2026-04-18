from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from gym.apps._base import GroundTruth, make_app, run_app

GROUND_TRUTH: GroundTruth = {
    "bug_type": "rate_limit_bypass",
    "location": "GET /api/search",
    "description": "Rate limit on /api/search is keyed by exact path; adding a query parameter like ?_=1 bypasses the limit entirely",
    "detection_criteria": [
        "rate limit bypass",
        "query parameter",
        "?_=",
        "bypass",
        "/api/search",
    ],
}

app: FastAPI = make_app(GROUND_TRUTH)

_LIMIT = 3
_counts: dict[str, int] = {}


@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    if request.url.path == "/api/search" and not request.url.query:
        key = request.url.path
        _counts[key] = _counts.get(key, 0) + 1
        if _counts[key] > _LIMIT:
            return JSONResponse(
                status_code=429,
                content={"error": "rate limit exceeded"},
                headers={"X-RateLimit-Remaining": "0"},
            )
    return await call_next(request)


@app.get("/api/search")
def search(q: str = "") -> dict:
    return {"results": [{"id": 1, "title": f"match for {q}"}]}


@app.get("/api/health")
def apihealth() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    run_app(app)
