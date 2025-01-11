import os
import pandas as pd
from binance.client import Client
import logging
from datetime import datetime, timedelta, timezone
import numpy as np

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BinanceDataCollector:
    def __init__(self, api_key, api_secret, symbol):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol

    def fetch_ohlcv(self, symbol, interval='1h', months_back=12):
        """Thu thập dữ liệu OHLCV trong khoảng thời gian nhất định"""
        try:
            # Tính toán thời gian bắt đầu và kết thúc
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=30 * months_back)

            klines = self.client.get_historical_klines(
                symbol, interval, start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S'))
            if not klines:
                logging.warning("Không có dữ liệu OHLCV thu thập được.")
                return pd.DataFrame()

            ohlcv_data = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

            # Định dạng dữ liệu
            ohlcv_data['timestamp'] = pd.to_datetime(ohlcv_data['timestamp'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                               'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            ohlcv_data[numeric_columns] = ohlcv_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

            logging.info("Dữ liệu OHLCV thu thập thành công với %d dòng.", len(ohlcv_data))
            return ohlcv_data
        except Exception as e:
            logging.error("Error fetching OHLCV data: %s", e)
            return pd.DataFrame()

    def calculate_indicators(self, ohlcv_data):
        """Tính toán các chỉ báo kỹ thuật"""
        try:
            # Price Change 24h
            ohlcv_data['price_change_24h'] = ohlcv_data['close'].pct_change(periods=24) * 100

            # Buy/Sell Volume and Delta
            ohlcv_data['Buy_Volume'] = pd.to_numeric(ohlcv_data['taker_buy_base_asset_volume'], errors='coerce')
            ohlcv_data['Sell_Volume'] = ohlcv_data['volume'] - ohlcv_data['Buy_Volume']
            ohlcv_data['Delta_Volume'] = ohlcv_data['Buy_Volume'] - ohlcv_data['Sell_Volume']

            # RSI
            delta = ohlcv_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            ohlcv_data['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            ema_short = ohlcv_data['close'].ewm(span=12, adjust=False).mean()
            ema_long = ohlcv_data['close'].ewm(span=26, adjust=False).mean()
            ohlcv_data['MACD'] = ema_short - ema_long
            ohlcv_data['Signal_Line'] = ohlcv_data['MACD'].ewm(span=9, adjust=False).mean()

            # VWAP
            ohlcv_data['VWAP'] = (ohlcv_data['quote_asset_volume'] / ohlcv_data['volume']).cumsum()

            # Parabolic SAR (Giả lập đơn giản)
            ohlcv_data['Parabolic_SAR'] = None
            af = 0.02
            trend = 1
            ep = ohlcv_data['high'][0]
            sar = ohlcv_data['low'][0]
            for i in range(1, len(ohlcv_data)):
                if trend == 1:
                    sar = sar + af * (ep - sar)
                    if ohlcv_data['low'][i] < sar:
                        trend = -1
                        sar = ep
                        ep = ohlcv_data['low'][i]
                        af = 0.02
                    else:
                        ep = max(ep, ohlcv_data['high'][i])
                        af = min(af + 0.02, 0.2)
                else:
                    sar = sar - af * (sar - ep)
                    if ohlcv_data['high'][i] > sar:
                        trend = 1
                        sar = ep
                        ep = ohlcv_data['high'][i]
                        af = 0.02
                    else:
                        ep = min(ep, ohlcv_data['low'][i])
                        af = min(af + 0.02, 0.2)
                ohlcv_data.loc[i, 'Parabolic_SAR'] = sar

            logging.info("Tính toán các chỉ báo kỹ thuật thành công với %d dòng.", len(ohlcv_data))
            return ohlcv_data
        except Exception as e:
            logging.error("Error calculating indicators: %s", e)
            return ohlcv_data

    def get_long_short_ratio(self, symbol, interval='1h'):
        """Thu thập dữ liệu Long/Short Ratio bằng requests"""
        import requests
        try:
            url = "https://fapi.binance.com/futures/data/topLongShortAccountRatio"
            params = {
                'symbol': symbol,
                'period': interval
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                ratio_data = pd.DataFrame(data)
                ratio_data['timestamp'] = pd.to_datetime(ratio_data['timestamp'], unit='ms')
                ratio_data[['longAccount', 'shortAccount', 'longShortRatio']] = ratio_data[
                    ['longAccount', 'shortAccount', 'longShortRatio']].apply(pd.to_numeric, errors='coerce')

                logging.info("Thu thập dữ liệu Long/Short Ratio thành công.")
                return ratio_data
            else:
                logging.error("Lỗi lấy dữ liệu Long/Short Ratio: %s", response.text)
                return pd.DataFrame()
        except Exception as e:
            logging.error("Error fetching Long/Short Ratio: %s", e)
            return pd.DataFrame()

    def save_to_csv(self, data, filename):
        """Lưu dữ liệu vào CSV"""
        try:
            if data.empty:
                logging.warning("Không có dữ liệu để lưu.")
                return
            data.to_csv(filename, index=False)
            logging.info("Dữ liệu đã được lưu vào: %s", filename)
        except Exception as e:
            logging.error("Error saving data to CSV: %s", e)


def main():
    # Đọc API key và secret từ biến môi trường hoặc file cấu hình
    api_key = os.getenv('BINANCE_API_KEY', 'your_actual_api_key')
    api_secret = os.getenv('BINANCE_API_SECRET', 'your_actual_api_secret')
    symbol = 'BTCUSDT'

    # Khởi tạo collector
    collector = BinanceDataCollector(api_key, api_secret, symbol)

    # Thu thập dữ liệu OHLCV
    logging.info("Bắt đầu thu thập dữ liệu OHLCV...")
    ohlcv_data = collector.fetch_ohlcv(symbol, interval='1h', months_back=12)

    # Tính toán các chỉ báo kỹ thuật
    if not ohlcv_data.empty:
        ohlcv_data = collector.calculate_indicators(ohlcv_data)

    # Thu thập dữ liệu Long/Short Ratio
    long_short_ratio = collector.get_long_short_ratio(symbol=symbol)
    if not long_short_ratio.empty:
        ohlcv_data = pd.merge(ohlcv_data, long_short_ratio, on='timestamp', how='left')

    # Lưu dữ liệu vào file CSV
    if not ohlcv_data.empty:
        collector.save_to_csv(ohlcv_data, f"{symbol}_ohlcv_with_indicators.csv")
################################################
        # Render to Final_Dataset
    # Đọc dữ liệu từ file
    file_path = "BTCUSDT_ohlcv_with_indicators.csv"
    data = pd.read_csv(file_path)

    # Loại bỏ các cột không cần thiết
    columns_to_drop = ['timestamp', 'close_time', 'symbol', 'longAccount', 'shortAccount', 'longShortRatio']
    data = data.drop(columns=columns_to_drop)

    # Điền giá trị NaN bằng giá trị trung bình của từng cột
    data = data.fillna(data.mean())

    # Chuyển đổi dữ liệu về kiểu float32 và làm tròn đến 2 chữ số thập phân
    data = data.astype(np.float32).round(2)

    # Lưu dữ liệu đã xử lý vào file mới
    processed_file_path = "BTCUSDT_processed_f32.csv"
    data.to_csv(processed_file_path, index=False)

    print("Dữ liệu đã được xử lý và lưu tại:", processed_file_path)
      
if __name__ == "__main__":
    main()
