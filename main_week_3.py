import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Đọc dữ liệu
file_path = r"openaq_location_measurments_2024_t1.csv"
df = pd.read_csv(file_path)
df['datetime'] = pd.to_datetime(df['datetimeLocal'])

# Pivot bảng
df_clean = df[['datetime', 'parameter', 'value']]
df_pivot = df_clean.pivot_table(index='datetime', columns='parameter', values='value', aggfunc='mean')
df_pivot = df_pivot.sort_index().ffill()

# Chọn các chỉ số cần
features = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2']
df_features = df_pivot[features].dropna()

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_features)

# Gọi các mô hình dự báo
from predict_pm25 import predict as predict_pm25
from predict_pm10 import predict as predict_pm10
from predict_co import predict as predict_co
from predict_no2 import predict as predict_no2
from predict_o3 import predict as predict_o3
from predict_so2 import predict as predict_so2

predictors = {
    'CO': predict_co,
    'NO2': predict_no2,
    'O3': predict_o3,
    'PM10': predict_pm10,
    'PM2.5': predict_pm25,
    'SO2': predict_so2
}

# === 1. Dự báo duy nhất và lưu lại ===
predictions_dict = {}

for name, func in predictors.items():
    y_true, y_pred = func(data_scaled, scaler, df_features)
    predictions_dict[name] = (y_true, y_pred)

# === 2. Tính AQI tổng hợp từ các chỉ số đã dự báo ===
def compute_and_plot_aqi(y_true_array, y_pred_array):
    def compute_individual_aqi(conc, breakpoints):
        aqi_values = []
        for c in conc:
            matched = False
            for (Clow, Chigh, Ilow, Ihigh) in breakpoints:
                if Clow <= c <= Chigh:
                    aqi = ((Ihigh - Ilow) / (Chigh - Clow)) * (c - Clow) + Ilow
                    aqi_values.append(aqi)
                    matched = True
                    break
            if not matched:
                aqi_values.append(np.nan)
        return np.array(aqi_values)

    aqi_bp = {
        'PM2.5': [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
                  (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)],
        'PM10':  [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
                  (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)],
        'CO':    [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
                  (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500)],
        'NO2':   [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
                  (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500)],
        'O3':    [(0.0, 0.054, 0, 50), (0.055, 0.070, 51, 100), (0.071, 0.085, 101, 150),
                  (0.086, 0.105, 151, 200), (0.106, 0.200, 201, 300)],
        'SO2':   [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
                  (186, 304, 151, 200), (305, 604, 201, 300), (605, 1004, 301, 500)]
    }

    names = ['CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2']
    true_aqi_components = []
    pred_aqi_components = []

    for i, name in enumerate(names):
        bp = aqi_bp[name]
        true_component = compute_individual_aqi(y_true_array[:, i], bp)
        pred_component = compute_individual_aqi(y_pred_array[:, i], bp)
        true_aqi_components.append(true_component)
        pred_aqi_components.append(pred_component)

    true_aqi = np.nanmax(np.stack(true_aqi_components), axis=0)
    pred_aqi = np.nanmax(np.stack(pred_aqi_components), axis=0)

    return true_aqi, pred_aqi

# === 3. Tính mảng AQI và vẽ ===
all_y_true = [predictions_dict[name][0] for name in predictors.keys()]
all_y_pred = [predictions_dict[name][1] for name in predictors.keys()]

y_aqi_true_array = np.array(all_y_true).T
y_aqi_pred_array = np.array(all_y_pred).T

y_aqi_true_graph, y_aqi_pred_graph = compute_and_plot_aqi(y_aqi_true_array, y_aqi_pred_array)

# === 4. Vẽ từng chỉ số ===
plt.figure(figsize=(15, 10))

for i, name in enumerate(predictors.keys()):
    y_true_graph, y_pred_graph = predictions_dict[name]

    plt.subplot(4, 2, i + 1)
    plt.plot(y_true_graph[:100], label='Thực tế', color='blue')
    plt.plot(y_pred_graph[:100], label='Dự báo', color='red')
    plt.title(f'Dự báo vs Thực tế: {name}')
    plt.xlabel('Thời gian (giờ)')
    plt.ylabel(name)
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True)

plt.subplot(4, 2, 7)
plt.plot(y_aqi_true_graph[:100], label='Thực tế', color='blue')
plt.plot(y_aqi_pred_graph[:100], label='Dự báo', color='red')
plt.title(f'AQI Dự báo vs Thực tế')
plt.xlabel('Thời gian (giờ)')
plt.ylabel('AQI')
plt.legend(loc='upper right', fontsize=8)
plt.grid(True)

# Tạo bảng AQI ở subplot cuối cùng
plt.subplot(4, 2, 8)
plt.axis('off')

cell_text = [
    ['Tốt', '0 to 50', 'Không ảnh hưởng đến sức khỏe'],
    ['Trung Bình', '51 to 100', 'Ở mức chấp nhận được.'],
    ['Kém', '101 to 150', 'Ảnh hưởng sức khỏe. Nhóm nhạy cảm nên hạn chế thời gian ra ngoài'],
    ['Xấu', '151 to 200', 'Nhóm nhạy cảm tránh ra ngoài. Những người khác hạn chế ra ngoài'],
    ['Rất xấu', '201 to 300', 'Cảnh báo sức khỏe khẩn cấp. Ảnh hưởng đến tất cả cư dân'],
    ['Nguy hại', '301 to 500', 'Báo động: Có thể ảnh hưởng nghiêm trọng đến sức khỏe mọi người']
]

col_widths = [0.16, 0.16, 0.56]

table = plt.table(
    cellText=cell_text,
    colLabels=['Chất lượng không khí', 'Khoảng giá trị AQI', 'Mức độ cảnh báo y tế'],
    cellLoc='center',
    loc='center',
    colWidths=col_widths
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.8)
plt.text(0.5, 1.18, 'Bảng tra chất lượng không khí', ha='center', va='bottom', fontsize=13, fontweight='bold', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

# Tính độ chính xác cho AQI
mask_aqi = y_aqi_true_graph != 0
aqi_mape = np.mean(np.abs((y_aqi_true_graph[mask_aqi] - y_aqi_pred_graph[mask_aqi]) / y_aqi_true_graph[mask_aqi])) * 100
aqi_accuracy = 100 - aqi_mape
print(f"\nĐộ chính xác dự báo AQI: {aqi_accuracy:.2f}%")