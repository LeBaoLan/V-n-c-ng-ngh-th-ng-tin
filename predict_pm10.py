import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(data, feature_index, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size][feature_index])
    return np.array(X), np.array(y)

def predict(data_scaled, scaler, df_features):
    feature_name = 'pm10'
    window_size = 24

    # Index cột PM10
    feature_index = df_features.columns.get_loc(feature_name)

    # Chuỗi dữ liệu
    X, y = create_sequences(data_scaled, feature_index, window_size)

    # Chia train/test theo thời gian
    split_idx = int(len(X)*0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Mô hình
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(64))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Huấn luyện
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, callbacks=[early_stop], verbose=0)

    # Dự đoán
    y_pred = model.predict(X_test)

    # Scale ngược
    y_test_rescaled = y_test * scaler.data_range_[feature_index] + scaler.data_min_[feature_index]
    y_pred_rescaled = y_pred.flatten() * scaler.data_range_[feature_index] + scaler.data_min_[feature_index]

    # Ép giá trị dự báo về >= 0
    y_pred_rescaled = np.clip(y_pred_rescaled, 0, None)
    
    return y_test_rescaled, y_pred_rescaled
