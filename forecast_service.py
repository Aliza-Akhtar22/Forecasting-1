from forecast import dynamic_forecast
from rf_model import forecast_with_random_forest
from xgb_model import forecast_with_xgboost
from sqlalchemy.orm import Session
import pandas as pd
from sqlalchemy import text
from sklearn.metrics import mean_squared_error
import numpy as np


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

    elif model_type == "xgboost":
        result = forecast_with_xgboost(
            df=df,
            ds_col=ds_col,
            y_col=y_col,
            regressor_cols=regressor_cols,
            growth_rates=growth_rates,
            period=period
        )
        return result

    else:
        raise ValueError("Invalid model_type. Choose 'prophet', 'random_forest', or 'xgboost'.")


def evaluate_models(
    db: Session,
    table_name: str,
    ds_col: str,
    y_col: str,
    regressor_cols: list,
    growth_rates: list,
    period: int
):
    query = f"SELECT {ds_col}, {y_col}, {', '.join(regressor_cols)} FROM {table_name}"
    df = pd.read_sql(text(query), db.bind)

    df = df.dropna(subset=[ds_col, y_col] + regressor_cols)
    df[ds_col] = pd.to_datetime(df[ds_col], dayfirst=True)
    df = df.sort_values(ds_col)

    
    train_df = df.iloc[:-period]
    test_df = df.iloc[-period:]

    results = {}

    # Prophet
    try:
        temp_table = "__temp_forecast_eval"
        train_df.to_sql(temp_table, db.bind, if_exists="replace", index=False)
        prophet_forecast = dynamic_forecast(
            db=db,
            table_name=temp_table,
            ds_col=ds_col,
            y_col=y_col,
            regressor_cols=regressor_cols,
            growth_rates=growth_rates,
            period=period
        )
        rmse_prophet = np.sqrt(mean_squared_error(test_df[y_col], prophet_forecast["yhat"]))
        results["prophet"] = rmse_prophet
    except Exception as e:
        results["prophet"] = f"Error: {str(e)}"

    # Random Forest
    try:
        rf_forecast = forecast_with_random_forest(
            df=train_df,
            ds_col=ds_col,
            y_col=y_col,
            regressor_cols=regressor_cols,
            growth_rates=growth_rates,
            period=period
        )
        rmse_rf = np.sqrt(mean_squared_error(test_df[y_col], rf_forecast["yhat"]))
        results["random_forest"] = rmse_rf
    except Exception as e:
        results["random_forest"] = f"Error: {str(e)}"

    # XGBoost
    try:
        xgb_forecast = forecast_with_xgboost(
            df=train_df,
            ds_col=ds_col,
            y_col=y_col,
            regressor_cols=regressor_cols,
            growth_rates=growth_rates,
            period=period
        )
        rmse_xgb = np.sqrt(mean_squared_error(test_df[y_col], xgb_forecast["yhat"]))
        results["xgboost"] = rmse_xgb
    except Exception as e:
        results["xgboost"] = f"Error: {str(e)}"

    # Determining the best model
    numeric_errors = {k: v for k, v in results.items() if isinstance(v, (int, float))}
    best_model = min(numeric_errors, key=numeric_errors.get) if numeric_errors else "None"

    return {
        "rmse_scores": results,
        "recommended_model": best_model
    }
