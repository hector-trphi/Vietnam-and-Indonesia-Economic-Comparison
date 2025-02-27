import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Dữ liệu GDP từ 2010-2024 (giả lập)
data = {
    "Year": np.arange(2010, 2025),
    "Vietnam_GDP": [115, 135, 155, 175, 200, 220, 245, 275, 310, 345, 370, 400, 450, 490, 520],
    "Indonesia_GDP": [755, 800, 850, 910, 980, 1050, 1125, 1200, 1300, 1405, 1500, 1605, 1700, 1800, 1920]
}
df = pd.DataFrame(data)

# Dự đoán bằng ARIMA
def arima_forecast(series):
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=6)
    return forecast

vn_gdp_pred_arima = arima_forecast(df["Vietnam_GDP"])
id_gdp_pred_arima = arima_forecast(df["Indonesia_GDP"])

# Dự đoán bằng Hồi quy tuyến tính (Linear Regression)
X = df[["Year"]]
y_vn = df["Vietnam_GDP"]
y_id = df["Indonesia_GDP"]

vn_model = LinearRegression()
vn_model.fit(X, y_vn)
id_model = LinearRegression()
id_model.fit(X, y_id)

future_years = np.arange(2025, 2031).reshape(-1, 1)
vn_gdp_pred_lr = vn_model.predict(future_years)
id_gdp_pred_lr = id_model.predict(future_years)

# Tính sai số MAE
arima_mae_vn = mean_absolute_error(df["Vietnam_GDP"].iloc[-6:], vn_gdp_pred_arima)
arima_mae_id = mean_absolute_error(df["Indonesia_GDP"].iloc[-6:], id_gdp_pred_arima)
lr_mae_vn = mean_absolute_error(df["Vietnam_GDP"].iloc[-6:], vn_gdp_pred_lr)
lr_mae_id = mean_absolute_error(df["Indonesia_GDP"].iloc[-6:], id_gdp_pred_lr)

print(f"Vietnam ARIMA MAE: {arima_mae_vn:.2f}, Linear Regression MAE: {lr_mae_vn:.2f}")
print(f"Indonesia ARIMA MAE: {arima_mae_id:.2f}, Linear Regression MAE: {lr_mae_id:.2f}")

# Biểu đồ so sánh dự đoán GDP
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["Vietnam_GDP"], 'bo-', label='Vietnam GDP')
plt.plot(df["Year"], df["Indonesia_GDP"], 'ro-', label='Indonesia GDP')
plt.plot(future_years, vn_gdp_pred_arima, 'b--', label='Vietnam GDP (ARIMA)')
plt.plot(future_years, id_gdp_pred_arima, 'r--', label='Indonesia GDP (ARIMA)')
plt.plot(future_years, vn_gdp_pred_lr, 'g-.', label='Vietnam GDP (Linear Regression)')
plt.plot(future_years, id_gdp_pred_lr, 'y-.', label='Indonesia GDP (Linear Regression)')
plt.xlabel("Year")
plt.ylabel("GDP (Billion USD)")
plt.title("Vietnam vs Indonesia GDP Prediction")
plt.legend()
plt.grid()
plt.show()

