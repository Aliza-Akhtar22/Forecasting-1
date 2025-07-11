from forecast import dynamic_forecast
from rf_model import forecast_with_random_forest
from xgb_model import forecast_with_xgboost
from sqlalchemy.orm import Session
import pandas as pd
from sqlalchemy import text
from sklearn.metrics import mean_absolute_error
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    """Handle zero values in y_true gracefully."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if not np.any(non_zero):
        return np.inf  
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


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
        return dynamic_forecast(
            db=db,
            table_name=table_name,
            ds_col=ds_col,
            y_col=y_col,
            regressor_cols=regressor_cols,
            growth_rates=growth_rates,
            period=period
        )

    elif model_type == "random_forest":
        return forecast_with_random_forest(
            df=df,
            ds_col=ds_col,
            y_col=y_col,
            regressor_cols=regressor_cols,
            growth_rates=growth_rates,
            period=period
        )

    elif model_type == "xgboost":
        return forecast_with_xgboost(
            df=df,
            ds_col=ds_col,
            y_col=y_col,
            regressor_cols=regressor_cols,
            growth_rates=growth_rates,
            period=period
        )

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

    def compute_metrics(y_true, y_pred):
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred)
        }

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
        metrics = compute_metrics(test_df[y_col], prophet_forecast["yhat"])
        results["prophet"] = metrics
    except Exception as e:
        results["prophet"] = {"error": str(e)}

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
        metrics = compute_metrics(test_df[y_col], rf_forecast["yhat"])
        results["random_forest"] = metrics
    except Exception as e:
        results["random_forest"] = {"error": str(e)}

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
        metrics = compute_metrics(test_df[y_col], xgb_forecast["yhat"])
        results["xgboost"] = metrics
    except Exception as e:
        results["xgboost"] = {"error": str(e)}

    # Determining best model based on MAE
    valid_models = {k: v for k, v in results.items() if isinstance(v, dict) and "mae" in v}
    best_model = min(valid_models, key=lambda k: valid_models[k]["mae"]) if valid_models else "None"

    return {
        "evaluation_metrics": results,
        "recommended_model": best_model
    }
