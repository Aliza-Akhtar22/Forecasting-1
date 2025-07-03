import pandas as pd
from prophet import Prophet
from sqlalchemy import text

def dynamic_forecast(db, table_name, ds_col, y_col, regressor_cols, growth_rates, period):
    
    selected_columns = [ds_col, y_col] + regressor_cols
    query = f"SELECT {', '.join(selected_columns)} FROM {table_name}"
    df = pd.read_sql(text(query), db.bind)

    
    df = df.dropna(subset=[ds_col, y_col] + regressor_cols)
    df[ds_col] = pd.to_datetime(df[ds_col], dayfirst=True)
    df.rename(columns={ds_col: "ds", y_col: "y"}, inplace=True)
    df = df[["ds", "y"] + regressor_cols]

    
    model = Prophet()
    for reg in regressor_cols:
        model.add_regressor(reg)
    model.fit(df)

    
    last_date = df["ds"].iloc[-1]
    initial_values = df.iloc[-1][regressor_cols]
    growth_factors = [1] * len(regressor_cols)

    future_data = []
    for i in range(1, period + 1):
        new_date = last_date + pd.DateOffset(months=i)
        reg_values = []
        for j in range(len(regressor_cols)):
            growth_factors[j] *= (1 + growth_rates[j] / 100)
            reg_values.append(initial_values[j] * growth_factors[j])
        row = {"ds": new_date, **dict(zip(regressor_cols, reg_values))}
        future_data.append(row)

    future_df = pd.DataFrame(future_data)

    
    forecast = model.predict(future_df)

    
    future_df["yhat"] = forecast["yhat"].astype(int)
    future_df["yhat_lower"] = forecast["yhat_lower"].astype(int)
    future_df["yhat_upper"] = forecast["yhat_upper"].astype(int)
    future_df["y"] = future_df["yhat"]  # Optionally including raw prediction

    result = future_df[["ds", "y", "yhat", "yhat_lower", "yhat_upper"] + regressor_cols]
    return result
