# xgb_model.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

def prepare_xgb_features(df: pd.DataFrame, ds_col: str, y_col: str, regressor_cols: list):
    df = df.dropna(subset=[ds_col, y_col] + regressor_cols)
    df[ds_col] = pd.to_datetime(df[ds_col], dayfirst=True)
    df = df.sort_values(ds_col)

    df['month'] = df[ds_col].dt.month
    df['year'] = df[ds_col].dt.year
    df['quarter'] = df[ds_col].dt.quarter

    for lag in [1, 2, 3]:
        df[f"lag_{lag}"] = df[y_col].shift(lag)

    df = df.dropna()
    X = df[regressor_cols + ['month', 'year', 'quarter'] + [f"lag_{i}" for i in [1, 2, 3]]]
    y = df[y_col]
    return X, y, df

def forecast_with_xgboost(df: pd.DataFrame, ds_col: str, y_col: str, regressor_cols: list, growth_rates: list, period: int):
    X, y, processed_df = prepare_xgb_features(df, ds_col, y_col, regressor_cols)

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    last_row = processed_df.iloc[-1]
    forecast_rows = []
    growth_factors = [1] * len(regressor_cols)

    for i in range(1, period + 1):
        future_date = last_row[ds_col] + pd.DateOffset(months=i)
        for j in range(len(growth_factors)):
            growth_factors[j] *= (1 + growth_rates[j] / 100)

        future_regressors = [last_row[regressor_cols[j]] * growth_factors[j] for j in range(len(regressor_cols))]
        features = future_regressors + [future_date.month, future_date.year, future_date.quarter] + [last_row[f"lag_{k}"] for k in [1, 2, 3]]

        prediction = model.predict([features])[0]

        forecast_rows.append({
            "ds": future_date,
            "y": int(prediction),
            "yhat": int(prediction),
            **{reg: val for reg, val in zip(regressor_cols, future_regressors)}
        })

        last_row = last_row.copy()
        last_row[y_col] = prediction
        for k in [3, 2, 1]:
            last_row[f"lag_{k}"] = last_row[f"lag_{k - 1}"] if k != 1 else prediction
        for idx, reg in enumerate(regressor_cols):
            last_row[reg] = future_regressors[idx]
        last_row[ds_col] = future_date

    return pd.DataFrame(forecast_rows)
