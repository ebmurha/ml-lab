"""FastAPI service for calibrated credit-default risk predictions."""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from .config import DEFAULT_ARTIFACT_PATH
from .inference import InferenceService

LOGGER = logging.getLogger("credit_default_risk.predictions")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

Money = Annotated[int, Field(ge=-10_000_000, le=10_000_000)]
PaymentStatus = Annotated[int, Field(ge=-2, le=9)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sex: Literal[1, 2]
    education: Annotated[int, Field(ge=1, le=4)]
    marriage: Annotated[int, Field(ge=1, le=3)]
    limit_bal: Annotated[int, Field(ge=1, le=10_000_000)]
    age: Annotated[int, Field(ge=18, le=120)]
    pay_0: PaymentStatus
    pay_2: PaymentStatus
    pay_3: PaymentStatus
    pay_4: PaymentStatus
    pay_5: PaymentStatus
    pay_6: PaymentStatus
    bill_amt1: Money
    bill_amt2: Money
    bill_amt3: Money
    bill_amt4: Money
    bill_amt5: Money
    bill_amt6: Money
    pay_amt1: Money
    pay_amt2: Money
    pay_amt3: Money
    pay_amt4: Money
    pay_amt5: Money
    pay_amt6: Money


class PredictionResponse(BaseModel):
    default_probability: float
    risk_band: Literal["low", "medium", "high"]
    model_version: str


def create_app(service: InferenceService | None = None) -> FastAPI:
    state: dict[str, InferenceService | None] = {"service": service}

    def get_service() -> InferenceService:
        if state["service"] is None:
            artifact_path = os.getenv("MODEL_PATH", str(DEFAULT_ARTIFACT_PATH))
            try:
                state["service"] = InferenceService.load(artifact_path)
            except FileNotFoundError as error:
                raise HTTPException(status_code=503, detail="Model artifact unavailable") from error
        return state["service"]

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        get_service()
        yield

    app = FastAPI(
        title="Credit Default Risk API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> dict[str, Any]:
        started = time.perf_counter()
        result = get_service().predict(payload.model_dump())
        latency_ms = (time.perf_counter() - started) * 1_000
        LOGGER.info(
            json.dumps(
                {
                    "event": "prediction",
                    "latency_ms": round(latency_ms, 3),
                    "model_version": result["model_version"],
                    "drift": result.pop("drift"),
                    "request_id": request.headers.get("x-request-id"),
                }
            )
        )
        return result

    return app


app = create_app()


def run() -> None:
    uvicorn.run("credit_default_risk.api:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
