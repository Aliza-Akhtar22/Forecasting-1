# forecast_router.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session
from database import SessionLocal
from forecast_service import run_forecast 

router = APIRouter()

class ForecastRequest(BaseModel):
    model_type: str  # "prophet" or "random_forest"
    table_name: str
    period: int
    ds_column: str
    y_column: str
    regressors: List[str]
    growth_rates: List[float]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/forecast")
def forecast_router(req: ForecastRequest, db: Session = Depends(get_db)):
    try:
        result_df = run_forecast(
            db=db,
            model_type=req.model_type,
            table_name=req.table_name.lower(),
            ds_col=req.ds_column,
            y_col=req.y_column,
            regressor_cols=req.regressors,
            growth_rates=req.growth_rates,
            period=req.period
        )
        result_df["ds"] = result_df["ds"].astype(str)
        return {"forecast": result_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
