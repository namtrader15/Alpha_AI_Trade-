# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Input, Conv1D, Flatten
from tensorflow.keras.models import Model, Sequential

# Load dữ liệu
file_path = 'BTCUSDT_processed_f32.csv'  # Đường dẫn file của anh
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

# Hàm Positional Encoding
def positional_encoding(window_size, d_model):
    positions = tf.range(window_size, dtype=tf.float32)[:, tf.newaxis]
    div_terms = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / tf.cast(d_model, tf.float32)))
    angle_rads = positions * div_terms

    # Gộp sin và cos
    angle_rads = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)

    # Đảm bảo kích thước khớp với d_model
    if d_model % 2 == 1:  # Nếu số chiều lẻ, thêm một chiều 0
        angle_rads = tf.pad(angle_rads, [[0, 0], [0, 1]])

    return tf.cast(angle_rads[tf.newaxis, ...], dtype=tf.float32)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(d_model),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)  # Self-Attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Kiến trúc CNN + Transformer
def create_cnn_transformer_model(input_shape, num_heads=16, ff_dim=512, num_transformer_blocks=8, num_cnn_layers=4):
    inputs = Input(shape=input_shape)

    # Tầng CNN nhiều lớp
    x = inputs
    for i in range(num_cnn_layers):
        x = Conv1D(filters=128 // (2 ** i), kernel_size=3, activation='relu', padding='same')(x)
        x = Dropout(0.2)(x)

    # Lấy số chiều đầu ra từ tầng CNN
    cnn_output_dim = x.shape[-1]

    # Thêm Positional Encoding
    pos_enc = positional_encoding(input_shape[0], cnn_output_dim)
    x = x + pos_enc  # Thêm positional encoding vào chuỗi CNN đã trích xuất

    # Thêm các Transformer Blocks
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(d_model=cnn_output_dim, num_heads=num_heads, ff_dim=ff_dim)(x, training=True)

    # Fully Connected Layer
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1)(x)  # Dự đoán giá trị "close"
    model = Model(inputs=inputs, outputs=x)
    return model


# Khởi tạo mô hình CNN + Transformer
input_shape = X_train.shape[1:]  # (window_size, num_features)
cnn_transformer_model = create_cnn_transformer_model(input_shape)

# Biên dịch mô hình
cnn_transformer_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Hiển thị thông tin mô hình
cnn_transformer_model.summary()

# Huấn luyện mô hình với EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = cnn_transformer_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])

# Lưu mô hình đã huấn luyện
model_path = 'cnn_transformer_model.keras'
cnn_transformer_model.save(model_path)
print(f"Mô hình đã được lưu thành công tại {model_path}")

# Đánh giá mô hình sau khi huấn luyện
test_loss, test_mae = cnn_transformer_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")


# Dự đoán giá trị trên tập kiểm tra
y_pred = cnn_transformer_model.predict(X_test)

# Chuyển giá trị dự đoán về thang giá trị gốc nếu đã chuẩn hóa
y_pred_original = scaler.inverse_transform([[0] * (data.shape[1] - 1) + [y] for y in y_pred.flatten()])[:, -1]
y_test_original = scaler.inverse_transform([[0] * (data.shape[1] - 1) + [y] for y in y_test.flatten()])[:, -1]

# Vẽ biểu đồ so sánh
import matplotlib.pyplot as plt
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
