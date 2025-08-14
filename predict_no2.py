import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, feature_index, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size][feature_index])
    return np.array(X), np.array(y)

def predict(data_scaled, scaler, df_features):
    feature_name = 'no2'
    window_size = 24

    # Xác định vị trí cột chỉ số cần dự báo
    feature_index = df_features.columns.get_loc(feature_name)

    # Tạo chuỗi dữ liệu cho LSTM
    X, y = create_sequences(data_scaled, feature_index, window_size)

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Xây dựng mô hình
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Huấn luyện
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=0)

    # Dự đoán
    y_pred = model.predict(X_test)

    # Scale ngược lại
    y_test_rescaled = y_test * scaler.data_range_[feature_index] + scaler.data_min_[feature_index]
    y_pred_rescaled = y_pred.flatten() * scaler.data_range_[feature_index] + scaler.data_min_[feature_index]

    # Ép giá trị dự báo về >= 0
    y_pred_rescaled = np.clip(y_pred_rescaled, 0, None)
    
    return y_test_rescaled, y_pred_rescaled
