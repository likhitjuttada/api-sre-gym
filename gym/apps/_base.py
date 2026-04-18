from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, TypedDict

from fastapi import FastAPI


class GroundTruth(TypedDict):
    bug_type: str
    location: str
    description: str
    detection_criteria: list[str]


def make_app(ground_truth: GroundTruth) -> FastAPI:
    app = FastAPI(title=f"buggy-{ground_truth['bug_type']}", version="0.1.0")

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, Any]:
        return {"ok": True, "bug_type": ground_truth["bug_type"]}

    return app


def run_app(app: FastAPI) -> None:
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
