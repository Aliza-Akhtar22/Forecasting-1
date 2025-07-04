from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect
from database import engine
from forecast_router import router as forecast_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API router
app.include_router(forecast_router, prefix="/api")

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
