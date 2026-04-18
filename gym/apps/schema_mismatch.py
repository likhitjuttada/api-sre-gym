from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from gym.apps._base import GroundTruth, make_app, run_app

GROUND_TRUTH: GroundTruth = {
    "bug_type": "schema_mismatch",
    "location": "GET /products",
    "description": "OpenAPI schema declares GET /products returns {items: [...], total: int} but implementation returns a bare array",
    "detection_criteria": [
        "schema mismatch",
        "declared",
        "array",
        "wrapped object",
        "openapi",
        "documented",
    ],
}

app: FastAPI = make_app(GROUND_TRUTH)


class ProductListResponse(BaseModel):
    items: list[dict]
    total: int


@app.get("/products", response_model=ProductListResponse)
def list_products():
    return JSONResponse(
        content=[
            {"id": 1, "name": "book"},
            {"id": 2, "name": "pen"},
        ]
    )


@app.get("/products/{pid}")
def get_product(pid: int) -> dict:
    return {"id": pid, "name": "book"}


if __name__ == "__main__":
    run_app(app)
