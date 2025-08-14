import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Bước 1: Đọc dữ liệu
file_path = r"openaq_location_measurments_2024_t1.csv"
df = pd.read_csv(file_path)
df['datetime'] = pd.to_datetime(df['datetimeLocal'])

# Bước 2: Pivot bảng
df_clean = df[['datetime', 'parameter', 'value']]
df_pivot = df_clean.pivot_table(index='datetime', columns='parameter', values='value', aggfunc='mean')
df_pivot = df_pivot.sort_index()
df_pivot = df_pivot.ffill()

# Bước 3: Chọn dữ liệu đầu vào
features = ['pm25', 'pm10', 'co', 'no2', 'o3', 'so2'] #co, no2, o3, pm10, pm25, so2
df_features = df_pivot[features].dropna()

# Scale dữ liệu về [0, 1] cho LSTM
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_features)

# Bước 4: Tạo tập dữ liệu cho LSTM
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size][0])  # pm25
    return np.array(X), np.array(y)

window_size = 24  # 24 giờ gần nhất
X, y = create_sequences(data_scaled, window_size)

# Bước 5: Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Bước 6: Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Bước 7: Huấn luyện
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Bước 8: Dự đoán
y_pred = model.predict(X_test)

# Inverse scale kết quả dự đoán (chỉ pm25)
pm25_index = df_features.columns.get_loc('pm25')
y_test_rescaled = y_test * scaler.data_range_[pm25_index] + scaler.data_min_[pm25_index]
y_pred_rescaled = y_pred.flatten() * scaler.data_range_[pm25_index] + scaler.data_min_[pm25_index]

# Bước 9: Vẽ biểu đồ
plt.figure(figsize=(10,5))
plt.plot(y_test_rescaled[:100], label='Thực tế PM2.5', color='blue')
plt.plot(y_pred_rescaled[:100], label='Dự đoán PM2.5', color='red')
plt.title('Dự đoán PM2.5 bằng LSTM')
plt.xlabel('Thời gian')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


from sklearn.metrics import mean_absolute_error, mean_squared_error

# MAE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
# RMSE
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
# MAPE (%)
mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100

# In kết quả
print(f"MAE: {mae:.2f} µg/m³")
print(f"RMSE: {rmse:.2f} µg/m³")
print(f"MAPE: {mape:.2f}%")

# Nếu muốn độ chính xác phần trăm (Accuracy)
print(100 - mape)