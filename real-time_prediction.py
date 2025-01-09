import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Bước 1: Load lại dữ liệu từ file CSV
file_path = 'BTCUSDT_processed_f32.csv'  # Đường dẫn file dữ liệu
data = pd.read_csv(file_path)

# Bước 2: Chuẩn hóa dữ liệu
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)  # Chuẩn hóa dữ liệu

# Bước 3: Lấy sliding window cuối cùng (400 dòng cuối cùng)
window_size = 400
last_window = standardized_data[-window_size:]  # Lấy 400 dòng cuối cùng
last_window = np.expand_dims(last_window, axis=0)  # Thêm chiều để phù hợp với input của mô hình

# Bước 4: Tải mô hình đã huấn luyện
model = load_model('financial_cnn_lstm_model.keras')
print("Mô hình đã được tải lại.")

# Bước 5: Dự đoán giá trị close tiếp theo
predicted_close = model.predict(last_window)[0][0]  # Lấy giá trị dự đoán từ mô hình

# Bước 6: Chuyển giá trị về thang giá trị gốc
predicted_close_original = scaler.inverse_transform(
    [[0] * (data.shape[1] - 1) + [predicted_close]]
)[0, -1]

# In kết quả
print(f"Giá trị close tiếp theo được dự đoán - giá real-time: {predicted_close_original}")
