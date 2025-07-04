from forecast import dynamic_forecast
from rf_model import forecast_with_random_forest
from sqlalchemy.orm import Session
import pandas as pd
from sqlalchemy import text


def run_forecast(
    db: Session,
    model_type: str,
    table_name: str,
    ds_col: str,
    y_col: str,
    regressor_cols: list,
    growth_rates: list,
    period: int
):
    # Load data from DB
    query = f"SELECT {ds_col}, {y_col}, {', '.join(regressor_cols)} FROM {table_name}"
    df = pd.read_sql(text(query), db.bind)

    if model_type == "prophet":
        result = dynamic_forecast(
            db=db,
            table_name=table_name,
            ds_col=ds_col,
            y_col=y_col,
            regressor_cols=regressor_cols,
            growth_rates=growth_rates,
            period=period
        )
        return result

    elif model_type == "random_forest":
        result = forecast_with_random_forest(
            df=df,
            ds_col=ds_col,
            y_col=y_col,
            regressor_cols=regressor_cols,
            growth_rates=growth_rates,
            period=period
        )
        return result

    else:
        raise ValueError("Invalid model_type. Choose 'prophet' or 'random_forest'.")
