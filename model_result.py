import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Bước 1: Tải lại dữ liệu
file_path = 'BTCUSDT_processed_f32.csv'  # Đường dẫn file dữ liệu
data = pd.read_csv(file_path)

# Bước 2: Chuẩn hóa dữ liệu
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)

# Bước 3: Tạo sliding window
def create_sliding_window(data, window_size, target_col_index):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :])  # Lấy cửa sổ có kích thước window_size làm input
        y.append(data[i + window_size, target_col_index])  # Giá trị close của dòng tiếp theo làm output
    return np.array(X), np.array(y)

# Định nghĩa tham số
window_size = 400
target_col_index = data.columns.get_loc("close")  # Vị trí cột "close"

# Tạo dữ liệu đầu vào (X) và đầu ra (y)
X, y = create_sliding_window(standardized_data, window_size, target_col_index)

# Chia dữ liệu thành train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tải lại mô hình
model = load_model('financial_cnn_lstm_model.keras')
print("Mô hình Neural Network đã được tải lại.")

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Chuyển giá trị dự đoán về thang giá trị gốc nếu đã chuẩn hóa
y_pred_original = scaler.inverse_transform([[0] * (data.shape[1] - 1) + [y] for y in y_pred.flatten()])[:, -1]
y_test_original = scaler.inverse_transform([[0] * (data.shape[1] - 1) + [y] for y in y_test.flatten()])[:, -1]

# Vẽ biểu đồ so sánh
plt.figure(figsize=(14, 7))
plt.plot(y_test_original[:100], label='Giá trị thực tế', color='blue', marker='o', markersize=4)
plt.plot(y_pred_original[:100], label='Giá trị dự đoán', color='red', linestyle='dashed', marker='x', markersize=4)
plt.title("So sánh giá trị thực tế và dự đoán (100 điểm đầu tiên)")
plt.xlabel("Index")
plt.ylabel("Giá trị Close")
plt.legend()
plt.grid()
plt.show()

####################################
