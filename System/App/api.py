"""Prediction API for the Somalia Food Price Index model.

Wraps the same `inference` module the dashboard uses, so both always serve
the same model.

    uvicorn api:app --app-dir System/App --port 8000
"""

from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import DATA_PATH, predict_hybrid_residual, recursive_forecast_hybrid

# Drivers a client may override; anything omitted falls back to the market's
# most recent observed value.
DRIVERS = [
    "market_price_maize",
    "market_price_rice",
    "market_price_sorghum",
    "market_price_oil",
    "population",
    "exchange_rate_typical",
    "food_price_critical",
    "cpi_communication",
    "cpi_housing_utilities",
]

_df = pd.read_csv(
    DATA_PATH,
    usecols=["Date", "adm1_name", "mkt_name", "food_price_index", *DRIVERS],
    parse_dates=["Date"],
).rename(columns={"adm1_name": "region", "mkt_name": "district"})

# One sorted history per market, built once at startup.
_HISTORY = {
    key: g.sort_values("Date").reset_index(drop=True)
    for key, g in _df.groupby(["region", "district"])
}
_MARKETS: dict[str, list[str]] = {}
for _region, _district in sorted(_HISTORY):
    _MARKETS.setdefault(_region, []).append(_district)


class Scenario(BaseModel):
    region: str = Field(examples=["Banadir"])
    district: str = Field(description="Market name", examples=["Bakaara"])
    market_price_maize: Optional[float] = Field(None, gt=0)
    market_price_rice: Optional[float] = Field(None, gt=0)
    market_price_sorghum: Optional[float] = Field(None, gt=0)
    market_price_oil: Optional[float] = Field(None, gt=0)
    population: Optional[float] = Field(None, gt=0)
    exchange_rate_typical: Optional[float] = Field(None, ge=0)
    food_price_critical: Optional[float] = Field(None, ge=0)
    cpi_communication: Optional[float] = Field(None, gt=0)
    cpi_housing_utilities: Optional[float] = Field(None, gt=0)


class PredictRequest(Scenario):
    food_price_index_rolling_mean_3: Optional[float] = Field(
        None, gt=0, description="Defaults to the mean of the market's last 3 observed months."
    )


class ForecastRequest(Scenario):
    months: int = Field(3, ge=1, le=24, description="Months ahead of the last observation.")


class Prediction(BaseModel):
    region: str
    district: str
    food_price_index: float
    inputs: dict


class ForecastPoint(BaseModel):
    date: str
    food_price_index: float


class Forecast(BaseModel):
    region: str
    district: str
    last_observed: str
    inputs: dict
    forecast: list[ForecastPoint]


def _resolve(req: Scenario) -> tuple[pd.DataFrame, dict]:
    """Return the market's history and the full driver set for a request."""
    history = _HISTORY.get((req.region, req.district))
    if history is None:
        raise HTTPException(404, f"Unknown market '{req.district}' in region '{req.region}'. See GET /markets.")
    latest = history.iloc[-1]
    inputs = {"region": req.region, "district": req.district}
    for name in DRIVERS:
        value = getattr(req, name)
        inputs[name] = float(latest[name] if value is None else value)
    return history, inputs


app = FastAPI(
    title="Somalia Food Price Index API",
    description="Predict and forecast the market-level Food Price Index (FPI).",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "markets": len(_HISTORY)}


@app.get("/markets")
def markets():
    """Regions and the markets available in each."""
    return _MARKETS


@app.post("/predict", response_model=Prediction)
def predict(req: PredictRequest):
    """FPI for one market-month given its drivers."""
    history, inputs = _resolve(req)
    rolling = req.food_price_index_rolling_mean_3
    if rolling is None:
        rolling = float(history["food_price_index"].tail(3).mean())
    inputs["food_price_index_rolling_mean_3"] = rolling
    fpi = float(predict_hybrid_residual(pd.DataFrame([inputs])))
    return {"region": req.region, "district": req.district, "food_price_index": fpi, "inputs": inputs}


@app.post("/forecast", response_model=Forecast)
def forecast(req: ForecastRequest):
    """Recursive month-by-month FPI forecast after the last observation."""
    history, inputs = _resolve(req)
    last = history["Date"].iloc[-1]
    out = recursive_forecast_hybrid(
        history[["Date", "food_price_index"]],
        inputs,
        last + pd.DateOffset(months=1),
        last + pd.DateOffset(months=req.months),
    )
    return {
        "region": req.region,
        "district": req.district,
        "last_observed": last.strftime("%Y-%m"),
        "inputs": inputs,
        "forecast": [
            {"date": d.strftime("%Y-%m"), "food_price_index": float(v)}
            for d, v in zip(out["Date"], out["food_price_index"])
        ],
    }
