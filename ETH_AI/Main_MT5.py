import time
import pandas as pd
import MetaTrader5 as mt5
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from Entry_Super import get_final_trend
from atr_check import atr_stop_loss_finder
from Real_time import TransformerBlock, process_data

# Thông tin tài khoản MT5
MT5_ACCOUNT = 7510016
MT5_PASSWORD = '7lTa+zUw'
MT5_SERVER = 'VantageInternational-Demo'

# Define constants
from binance.client import Client
BINANCE_API_KEY = "your_api_key"
BINANCE_API_SECRET = "your_api_secret"
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
if client is None:
    raise ValueError("Client không được khởi tạo. Kiểm tra API Key và Secret.")
SYMBOL = 'ETHUSD'
MARKET_INTERVAL = '1h'
WINDOW_SIZE = 400
ATR_LENGTH = 14
ATR_MULTIPLIER = 1.5
RUN_INTERVAL = 3700  # >1 hour in seconds
RETRY_INTERVAL = 10  # Retry every 10 seconds if an error occurs
MODEL_PATH = 'cnn_transformer_model.keras'
DATA_FILE = 'ETHUSDT_processed_f32.csv'

# Load AI model
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'TransformerBlock': TransformerBlock})

# Hàm kết nối với MT5
def connect_mt5():
    while True:
        try:
            if not mt5.initialize():
                print("Lỗi khi khởi động MT5:", mt5.last_error())
                time.sleep(RETRY_INTERVAL)
                continue

            authorized = mt5.login(MT5_ACCOUNT, password=MT5_PASSWORD, server=MT5_SERVER)
            if not authorized:
                error_code, error_message = mt5.last_error()
                print(f"Lỗi kết nối đến MT5: Mã lỗi {error_code} - {error_message}")
                mt5.shutdown()
                time.sleep(RETRY_INTERVAL)
                continue

            print("Kết nối thành công đến MT5 với tài khoản:", MT5_ACCOUNT)
            return True
        except Exception as e:
            print(f"Lỗi khi kết nối MT5: {e}")
            time.sleep(RETRY_INTERVAL)

# Hàm lấy giá mark từ MT5
def get_mark_price():
    while True:
        try:
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick:
                return round(tick.ask, 2)  # Giá mua (ask)
            else:
                print(f"Không thể lấy giá hiện tại cho {SYMBOL}. Thử lại sau {RETRY_INTERVAL}s.")
                time.sleep(RETRY_INTERVAL)
        except Exception as e:
            print(f"Lỗi khi lấy giá Mark: {e}")
            time.sleep(RETRY_INTERVAL)

# Hàm đóng tất cả các lệnh
def close_all_orders():
    while True:
        try:
            open_positions = mt5.positions_get(symbol=SYMBOL)
            if open_positions is None or len(open_positions) == 0:
                print("Không có lệnh mở nào để đóng.")
                return

            for position in open_positions:
                action = mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": action,
                    "position": position.ticket,
                    "price": get_mark_price(),
                    "deviation": 20,
                    "magic": position.magic,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(close_request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Đóng lệnh thành công: {position.ticket}")
                else:
                    print(f"Lỗi khi đóng lệnh: {position.ticket}, Mã lỗi: {result.retcode if result else 'N/A'}")
            return
        except Exception as e:
            print(f"Lỗi khi đóng lệnh: {e}. Thử lại sau {RETRY_INTERVAL}s.")
            time.sleep(RETRY_INTERVAL)

# Hàm đặt lệnh trên MT5
def place_order_mt5(order_type, predicted_price, risk_amount=50):
    while True:
        try:
            # Lấy giá mark hiện tại
            mark_price = get_mark_price()
            if mark_price is None:
                continue

            # Lấy ATR stop loss từ dữ liệu Binance hoặc nội bộ
            atr_short_stop, atr_long_stop = atr_stop_loss_finder(client, "ETHUSDT", ATR_LENGTH, ATR_MULTIPLIER)
            atr_short_stop = round(atr_short_stop, 2)
            atr_long_stop = round(atr_long_stop, 2)

            # Xác định Stop Loss và Take Profit
            if order_type == "buy":
                stop_loss_price = atr_long_stop
                take_profit_price = predicted_price  # TP là giá dự đoán của AI
            else:
                stop_loss_price = atr_short_stop
                take_profit_price = predicted_price  # TP là giá dự đoán của AI

            # Tính Risk và Reward
            risk = abs(mark_price - stop_loss_price)
            reward = abs(take_profit_price - mark_price)
            rr_ratio = reward / risk if risk > 0 else 0

            # In thông tin tỷ lệ R:R
            print(f"DEBUG: Order Type: {order_type}, Mark Price: {mark_price}, SL: {stop_loss_price}, TP: {take_profit_price}, Risk: {risk:.2f}, Reward: {reward:.2f}, R:R = {rr_ratio:.2f}")

            # Chỉ đặt lệnh nếu R:R > 1
            if rr_ratio <= 1.2:
                print("Lệnh bị từ chối vì tỷ lệ R:R <= 1.2")
                print("Tìm kiếm cơ hội khác sau 3700s...")
                time.sleep(RUN_INTERVAL)
                continue

            # Kiểm tra giá trị SL và log thông tin
            if order_type == "buy" and stop_loss_price >= mark_price:
                print("Lỗi: Giá Stop Loss (SL) phải nhỏ hơn Mark Price cho lệnh Buy.")
                return
            if order_type == "sell" and stop_loss_price <= mark_price:
                print("Lỗi: Giá Stop Loss (SL) phải lớn hơn Mark Price cho lệnh Sell.")
                return

            # Tính khối lượng giao dịch dựa trên mức rủi ro
            distance = abs(mark_price - stop_loss_price)
            contract_size = mt5.symbol_info(SYMBOL).trade_contract_size
            volume = round(risk_amount / (distance * contract_size), 2)

            # Thiết lập lệnh Market
            order = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL,
                "deviation": 20,
                "sl": float(f"{stop_loss_price:.2f}"),
                "tp": float(f"{take_profit_price:.2f}"),
                "magic": 234000,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Gửi lệnh và kiểm tra kết quả
            result = mt5.order_send(order)
            if result is None:
                error_code, error_message = mt5.last_error()
                print("Gửi lệnh thất bại: Không nhận được kết quả từ MT5.")
                print(f"Thông tin lệnh: {order}, Lỗi: {error_code} - {error_message}")
                time.sleep(10)
                continue

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Lệnh {order_type} thành công: {volume} lots ở giá {mark_price}, SL={stop_loss_price}, TP={take_profit_price}")
                time.sleep(500)
                continue
            else:
                print(f"Gửi lệnh thất bại. Mã lỗi: {result.retcode}, Thông tin lỗi: {result}")
                time.sleep(10)
                continue
            return
        except Exception as e:
            print(f"Lỗi khi đặt lệnh: {e}. Thử lại sau {RETRY_INTERVAL}s.")
            time.sleep(RETRY_INTERVAL)

def trading_bot():
    while True:
        try:
            # Bước 1: Kết nối với MT5 nếu chưa kết nối
            if not connect_mt5():
                print("Không thể kết nối MT5. Thử lại sau 10 giây...")
                time.sleep(10)
                continue

            # Bước 2: Lấy dự đoán giá từ AI
            last_window, scaler = process_data(DATA_FILE, WINDOW_SIZE)
            prediction = model.predict(last_window)
            predicted_price = round(scaler.inverse_transform([[0] * (last_window.shape[2] - 1) + [prediction.flatten()[0]]])[:, -1][0], 2)

            # Bước 3: Xác định xu hướng thị trường
            market_trend = get_final_trend(client)
            mark_price = get_mark_price()

            # Hiển thị thông tin debug
            print(f"Predicted Price: {predicted_price:.2f}, Mark Price: {mark_price:.2f}, Market Trend: {market_trend}")

            # Bước 4: Kiểm tra và đóng các lệnh ngược xu hướng
            open_positions = mt5.positions_get(symbol=SYMBOL)
            if open_positions:
                for position in open_positions:
                    if (market_trend == "Xu hướng tăng" and position.type == mt5.ORDER_TYPE_SELL) or \
                       (market_trend == "Xu hướng giảm" and position.type == mt5.ORDER_TYPE_BUY):
                        print(f"Đóng lệnh ngược xu hướng: {position.ticket}")
                        close_all_orders()
                        time.sleep(5)
                        continue
                    else:
                        time.sleep(3700)
                        continue
                    
            # Bổ sung: không đặt lệnh nếu đã có lệnh tồn tại cùng xu hướng
            if open_positions:  # Nếu đã có lệnh mở, không đặt lệnh mới
                print("Đã có lệnh mở. Không thực hiện lệnh mới.")
                time.sleep(3700)  # Đợi 1 giờ (có thể điều chỉnh lại thời gian nếu cần)
                continue

            # Bước 5: Quyết định giao dịch
            if predicted_price > mark_price and market_trend == "Xu hướng tăng":
                print("Condition met for Buy.")
                place_order_mt5("buy", predicted_price)
            elif predicted_price < mark_price and market_trend == "Xu hướng giảm":
                print("Condition met for Sell.")
                place_order_mt5("sell", predicted_price)
            else:
                print("Conditions not met for trading. Retrying...")
                time.sleep(3700)
                continue
        except Exception as e:
            print(f"Error occurred: {e}. Thử lại sau 10 giây...")

        # Đợi trước khi lặp lại
        time.sleep(10)


if __name__ == "__main__":
    trading_bot()
