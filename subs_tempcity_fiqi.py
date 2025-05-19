import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('datasets/city_temperature_asia.csv')
df = df[df['AvgTemperature'] > -50]
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
city = 'Jakarta'
df_city = df[df['City'] == city].sort_values('Date').reset_index(drop=True)
df_city['AvgTemperature_C'] = (df_city['AvgTemperature'] - 32) * 5/9

# Feature engineering
df_city['Month'] = df_city['Date'].dt.month
df_city['DayOfWeek'] = df_city['Date'].dt.dayofweek
df_city['RollingMean_7'] = df_city['AvgTemperature'].rolling(window=7).mean()

# Baseline
df_city['Naive_Pred'] = df_city['AvgTemperature_C'].shift(1)
df_eval = df_city.dropna(subset=['Naive_Pred'])
mae_naive = mean_absolute_error(df_eval['AvgTemperature_C'], df_eval['Naive_Pred'])
rmse_naive = np.sqrt(mean_squared_error(df_eval['AvgTemperature_C'], df_eval['Naive_Pred']))

# Modeling
features = ['Month', 'DayOfWeek', 'RollingMean_7']
df_ml = df_city.dropna(subset=features + ['AvgTemperature_C'])
X = df_ml[features]
y = df_ml['AvgTemperature_C']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae_ml = mean_absolute_error(y_test, y_pred)
rmse_ml = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Baseline Naive - MAE: {mae_naive:.2f}, RMSE: {rmse_naive:.2f}")
print(f"XGBoostRegressor - MAE: {mae_ml:.2f}, RMSE: {rmse_ml:.2f}")