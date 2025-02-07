import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, MultiHeadAttention, Dense, Dropout, LayerNormalization

import tensorflow as tf
from tensorflow.keras.layers import Layer, MultiHeadAttention, Dense, Dropout, LayerNormalization

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        # Khởi tạo các lớp này trong phương thức build()
        self.att = None
        self.ffn = None
        self.layernorm1 = None
        self.layernorm2 = None
        self.dropout1 = None
        self.dropout2 = None

    def build(self, input_shape):
        # Khởi tạo các layer trong phương thức build
        self.att = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model)
        self.ffn = tf.keras.Sequential([
            Dense(self.ff_dim, activation="relu"),
            Dense(self.d_model),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.rate)
        self.dropout2 = Dropout(self.rate)

        # Gọi super().build() để hoàn tất quá trình build
        super(TransformerBlock, self).build(input_shape)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)  # Self-Attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



# 1. Tải mô hình đã huấn luyện trước đó
model_path = 'cnn_transformer_model.keras'  # Đường dẫn đến mô hình đã huấn luyện
model = tf.keras.models.load_model(model_path, custom_objects={'TransformerBlock': TransformerBlock})

# 2. Đọc và chuẩn hóa dữ liệu mới từ file 1month_BTCUSDT_processed_f32.csv
new_data = pd.read_csv('1month_BTCUSDT_processed_f32.csv')
scaler = StandardScaler()

# Chuẩn hóa dữ liệu mới (lưu ý rằng chúng ta đã sử dụng scaler trước đó, đảm bảo rằng chúng ta sử dụng cùng một scaler đã huấn luyện)
new_data_standardized = scaler.fit_transform(new_data)

# 3. Hàm tạo dữ liệu đầu vào (X) và đầu ra (y) từ dữ liệu mới (sử dụng sliding window)
def create_sliding_window(data, window_size, target_col_index):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :])  # Lấy cửa sổ có kích thước window_size làm input
        y.append(data[i + window_size, target_col_index])  # Giá trị "close" của dòng tiếp theo làm output
    return np.array(X), np.array(y)

# Kích thước cửa sổ trượt và cột mục tiêu
window_size = 400
target_col_index = new_data.columns.get_loc("close")  # Giả sử cột "close" là cột mục tiêu

# Tạo dữ liệu đầu vào (X) và đầu ra (y) từ dữ liệu mới
X_new, y_new = create_sliding_window(new_data_standardized, window_size, target_col_index)

# 4. Fine-tune mô hình với dữ liệu mới
# Cài đặt optimizer và learning rate nhỏ để tránh làm thay đổi quá nhiều trọng số đã học
optimizer = Adam(learning_rate=0.0001)  # Learning rate thấp để fine-tune

# Biên dịch lại mô hình với optimizer và loss function
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Đặt EarlyStopping để tránh overfitting trong quá trình fine-tuning
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Huấn luyện lại mô hình với dữ liệu mới
history = model.fit(X_new, y_new, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# 5. Lưu mô hình đã fine-tune
model.save('cnn_transformer_model_finetuned.keras')
print("Mô hình đã được fine-tune và lưu lại.")

# Đánh giá mô hình trên tập kiểm tra
test_loss, test_mae = model.evaluate(X_new, y_new)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
