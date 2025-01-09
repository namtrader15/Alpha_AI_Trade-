# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
# Load dữ liệu
file_path = 'BTCUSDT_processed_f32.csv'  # Thay bằng đường dẫn file của anh
data = pd.read_csv(file_path)

# Chuẩn hóa dữ liệu bằng StandardScaler
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)

# Hàm tạo sliding window
def create_sliding_window(data, window_size, target_col_index):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :])  # Lấy cửa sổ có kích thước window_size làm input
        y.append(data[i + window_size, target_col_index])  # Giá trị close của dòng tiếp theo làm output
    return np.array(X), np.array(y)

# Kích thước sliding window và cột mục tiêu
window_size = 400
target_col_index = data.columns.get_loc("close")  # Lấy vị trí cột "close"

# Tạo dữ liệu đầu vào (X) và đầu ra (y)
X, y = create_sliding_window(standardized_data, window_size, target_col_index)

# Chia tập dữ liệu thành train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kiểm tra kích thước dữ liệu
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

######################################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout

# Kiểm tra kích thước đầu vào
input_shape = X_train.shape[1:]  # (400, 19)

# Xây dựng mô hình CNN + LSTM
model = Sequential([
    # Lớp Input
    Input(shape=input_shape),

    # Lớp CNN
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    Dropout(0.2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),

    # Lớp LSTM
    LSTM(64, activation='relu', return_sequences=False),

    # Dense Layer
    Dense(64, activation='relu'),
    Dense(1)  # Dự đoán giá trị tiếp theo của cột close
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Hiển thị thông tin mô hình
model.summary()

# Huấn luyện mô hình vs early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])

# Đánh giá mô hình
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")


# Lưu mô hình sau khi huấn luyện
#model.save('financial_cnn_lstm_model.h5')
model.save('financial_cnn_lstm_model.keras')
print("Mô hình đã được lưu thành công.")


# Tải lại mô hình để kiểm tra
loaded_model = load_model('financial_cnn_lstm_model.keras')
#loaded_model = load_model('financial_cnn_lstm_model.h5')
print("Mô hình đã được tải lại thành công.")

#########################################
#Step3: Vẽ biểu đồ để trực quan hóa kết quả.

import matplotlib.pyplot as plt

# Dự đoán giá trị trên tập kiểm tra
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

# Tính toán độ sai lệch
error = y_test_original - y_pred_original
print(f"Độ sai lệch trung bình (Mean Error): {error.mean():.4f}")
print(f"Độ sai lệch tuyệt đối trung bình (Mean Absolute Error): {abs(error).mean():.4f}")

#######################################
#Step 4: MCTS
import pickle
import numpy as np
from collections import defaultdict

# Khởi tạo nút MCTS
class MCTSNode:
    def __init__(self, state, action=None, parent=None):
        self.state = state  # Sliding window hiện tại
        self.action = action  # Hành động (Buy, Sell, Hold)
        self.parent = parent  # Nút cha
        self.children = []  # Danh sách các nút con
        self.visits = 0  # Số lần thăm nút
        self.value = 0  # Giá trị reward

    def add_child(self, child_state, action):
        child_node = MCTSNode(state=child_state, action=action, parent=self)
        self.children.append(child_node)
        return child_node

# Hàm tính reward
def calculate_reward(current_close, predicted_close, action):
    if action == "Buy":
        return max(0, predicted_close - current_close)  # Lợi nhuận khi giá tăng
    elif action == "Sell":
        return max(0, current_close - predicted_close)  # Lợi nhuận khi giá giảm
    elif action == "Hold":
        return -0.01  # Phạt nhỏ cho việc không hành động
    return 0

# Hàm thực hiện MCTS
def run_mcts(root, model, iterations=100):
    for _ in range(iterations):
        # 1. Thăm dò
        node = root
        while node.children:
            node = select_best_child(node)

        # 2. Mở rộng (Expand)
        actions = ["Buy", "Sell", "Hold"]
        current_close = node.state[-1, -1]  # Giá trị close hiện tại
        predicted_close = model.predict(node.state[np.newaxis, ...])[0][0]  # Giá trị dự đoán

        for action in actions:
            reward = calculate_reward(current_close, predicted_close, action)
            child_state = node.state  # Giữ nguyên state cho ví dụ đơn giản
            child_node = node.add_child(child_state, action)
            child_node.value = reward

        # 3. Backpropagation
        backpropagate(node)

# Hàm chọn nút con tốt nhất
def select_best_child(node):
    best_score = -np.inf
    best_child = None
    for child in node.children:
        score = child.value / (child.visits + 1)  # UCT Score đơn giản
        if score > best_score:
            best_score = score
            best_child = child
    return best_child

# Hàm backpropagation
def backpropagate(node):
    while node:
        node.visits += 1
        if node.parent:
            node.value += node.parent.value  # Cộng dồn giá trị từ cha
        node = node.parent

# Hàm lưu cây MCTS
def save_mcts_tree(root, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(root, file)
    print(f"Cây MCTS đã được lưu tại: {file_path}")

# Hàm tải lại cây MCTS
def load_mcts_tree(file_path):
    with open(file_path, 'rb') as file:
        root = pickle.load(file)
    print(f"Cây MCTS đã được tải lại từ: {file_path}")
    return root

# Thử nghiệm MCTS với một sliding window
root_state = X_test[0]  # Lấy một sliding window từ tập kiểm tra
root = MCTSNode(state=root_state)

# Chạy MCTS
run_mcts(root, model, iterations=100)

# Lưu cây MCTS
save_mcts_tree(root, 'mcts_tree.pkl')

# Tải lại cây MCTS
loaded_root = load_mcts_tree('mcts_tree.pkl')

# Hiển thị kết quả
for child in loaded_root.children:
    print(f"Action: {child.action}, Reward: {child.value:.4f}, Visits: {child.visits}")

