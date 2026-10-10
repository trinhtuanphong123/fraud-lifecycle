"""FastAPI service: POST /score with one transaction, get probability, decision and reasons.

    uv run uvicorn fraud.serving.app:app          # bundle from artifacts/bundle or $FRAUD_BUNDLE
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from fraud.serving.bundle import BUNDLE_DIR, load_bundle
from fraud.serving.history import HistoryStore
from fraud.serving.scorer import Scorer


class Transaction(BaseModel):
    """A raw IEEE-CIS transaction. Four fields are required; any other column may be sent, missing ones are NaN."""

    model_config = ConfigDict(extra="allow")
    TransactionID: int
    TransactionDT: int = Field(ge=0, description="seconds since the dataset's reference time")
    TransactionAmt: float = Field(gt=0)
    card1: int


class Reason(BaseModel):
    feature: str
    value: float | int | str | None
    contribution: float = Field(description="TreeSHAP contribution to the log-odds of fraud")
    description: str


class ExpectedCost(BaseModel):
    approve: float
    decline: float
    review: float


class ScoreResponse(BaseModel):
    transaction_id: int
    fraud_probability: float
    raw_score: float
    decision: str
    decline_threshold: float
    expected_cost: ExpectedCost
    review_recommended: bool
    explained: bool = Field(description="whether reasons were computed (by default only for declines)")
    reasons: list[Reason]
    model_version: str


def create_app(bundle_dir: Path | None = None) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        directory = Path(bundle_dir or os.environ.get("FRAUD_BUNDLE", BUNDLE_DIR))
        state["scorer"] = Scorer(load_bundle(directory), HistoryStore.load(directory))
        yield
        state.clear()

    app = FastAPI(title="Fraud scoring", version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        scorer = state.get("scorer")
        return {"status": "ok" if scorer else "loading", "model_version": scorer.bundle.version if scorer else None,
                "history_keys": len(scorer.store.state) if scorer else 0}

    @app.post("/score", response_model=ScoreResponse)
    def score(tx: Transaction, record: bool = True, explain: Literal["auto", "always", "never"] = "auto") -> dict:
        """Score one transaction; with record=true it is then added to the card's history.

        Reasons (TreeSHAP) are computed for declines by default (explain=auto).
        explain=always forces them, at ~0.65 s per call; use it when a reviewer
        opens a review case, off the payment path.
        """
        return state["scorer"].score(tx.model_dump(), record=record, explain=explain)

    return app


app = create_app()
