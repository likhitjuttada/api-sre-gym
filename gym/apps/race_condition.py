import asyncio
import time

from fastapi import FastAPI
from pydantic import BaseModel

from gym.apps._base import GroundTruth, make_app, run_app

GROUND_TRUTH: GroundTruth = {
    "bug_type": "race_condition",
    "location": "POST /transfer",
    "description": "POST /transfer reads balance, sleeps, then writes -- concurrent transfers see stale balance and allow overdraft",
    "detection_criteria": [
        "race condition",
        "concurrent",
        "double spend",
        "overdraft",
        "TOCTOU",
        "time of check",
    ],
}

app: FastAPI = make_app(GROUND_TRUTH)

_balance = {"alice": 100, "bob": 0}


class Transfer(BaseModel):
    src: str
    dst: str
    amount: int


@app.get("/balances")
def balances() -> dict:
    return dict(_balance)


@app.post("/transfer")
async def transfer(t: Transfer) -> dict:
    src_bal = _balance.get(t.src, 0)
    if src_bal < t.amount:
        return {"ok": False, "reason": "insufficient_funds", "balance": src_bal}
    await asyncio.sleep(0.05)
    _balance[t.src] = src_bal - t.amount
    _balance[t.dst] = _balance.get(t.dst, 0) + t.amount
    return {"ok": True, "balance_src": _balance[t.src], "balance_dst": _balance[t.dst]}


if __name__ == "__main__":
    run_app(app)
