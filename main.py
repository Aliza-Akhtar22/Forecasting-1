from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from sqlalchemy import inspect
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from forecast import dynamic_forecast

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    table_name: str
    period: int
    ds_column: str
    y_column: str
    regressors: List[str]
    growth_rates: List[float]



@app.get("/tables")
def get_tables():
    inspector = inspect(engine)
    return {"tables": inspector.get_table_names()}

@app.get("/columns")
def get_columns(table_name: str):
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name.lower())
        return {"columns": [col["name"] for col in columns]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/forecast")
def forecast_from_db(req: ForecastRequest):
    db: Session = SessionLocal()
    try:
        result_df = dynamic_forecast(
            db=db,
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
    finally:
        db.close()
