from binance.client import Client
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Hàm tính xác suất kết hợp của hai mô hình không hoàn toàn độc lập
def combined_probability(p1, p2):
    combined_prob = p1 + p2 - (p1 * p2)
    return combined_prob

# Hàm tính Parabolic SAR
def calculate_parabolic_sar(data, acceleration=0.02, maximum=0.2):
    high = data['high']
    low = data['low']
    close = data['close']

    sar = [close.iloc[0]]
    ep = high.iloc[0]
    af = acceleration
    trend = 1

    for i in range(1, len(close)):
        if trend == 1:
            sar.append(sar[i-1] + af * (ep - sar[i-1]))
            if low.iloc[i] < sar[i]:
                trend = -1
                sar[i] = ep
                af = acceleration
                ep = low.iloc[i]
        else:
            sar.append(sar[i-1] + af * (ep - sar[i-1]))
            if high.iloc[i] > sar[i]:
                trend = 1
                sar[i] = ep
                af = acceleration
                ep = high.iloc[i]

        if trend == 1 and high.iloc[i] > ep:
            ep = high.iloc[i]
            af = min(af + acceleration, maximum)
        elif trend == -1 and low.iloc[i] < ep:
            ep = low.iloc[i]
            af = min(af + acceleration, maximum)

    data['parabolic_sar'] = sar
    return data

# Lấy dữ liệu thời gian thực từ Binance
def get_realtime_klines(client, symbol, interval, lookback, end_time=None):
    if end_time:
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            endTime=int(end_time.timestamp() * 1000),
            limit=lookback
        )
    else:
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=lookback
        )
    data = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
    ])
    data[['open', 'high', 'low', 'close']] = data[['open', 'high', 'low', 'close']].astype(float)
    data['volume'] = data['volume'].astype(float)

    # Tính giá trị Heikin-Ashi
    ha_open = (data['open'].shift(1) + data['close'].shift(1)) / 2
    ha_open.iloc[0] = (data['open'].iloc[0] + data['close'].iloc[0]) / 2
    ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    ha_high = pd.concat([data['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([data['low'], ha_open, ha_close], axis=1).min(axis=1)

    data['open'] = ha_open
    data['high'] = ha_high
    data['low'] = ha_low
    data['close'] = ha_close

    # Tính EMA
    data['ema1'] = data['close'].ewm(span=5, adjust=False).mean()
    data['ema2'] = data['close'].ewm(span=11, adjust=False).mean()
    data['ema3'] = data['close'].ewm(span=15, adjust=False).mean()
    data['ema8'] = data['close'].ewm(span=34, adjust=False).mean()

    # Sinh tín hiệu EMA Ribbon
    data['Longema'] = (data['ema2'] > data['ema8']).astype(int)
    data['Redcross'] = (data['ema1'] < data['ema2']).astype(int)
    data['Bluetriangle'] = (data['ema2'] > data['ema3']).astype(int)

    return data

# Tính RSI
def calculate_rsi(data, window):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Tính MACD
def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data['close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Phân tích xu hướng
def analyze_trend(client, symbol, interval, lookback, end_time=None):
    data = get_realtime_klines(client, symbol, interval, lookback, end_time)
    rsi = calculate_rsi(data, 14)
    macd, signal_line = calculate_macd(data)
    data = calculate_parabolic_sar(data)

    data['rsi'] = rsi
    data['macd'] = macd
    data['signal_line'] = signal_line
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

    features = data[['rsi', 'macd', 'signal_line', 'Longema', 'Redcross', 'Bluetriangle', 'parabolic_sar']].dropna()
    target = data['target'].dropna()

    min_length = min(len(features), len(target))
    features = features.iloc[:min_length]
    target = target.iloc[:min_length]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, refit=True, verbose=0)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    latest_features = features_scaled[-1].reshape(1, -1)
    prediction_prob = grid.predict_proba(latest_features)[0]

    # Loại bỏ các giá trị nằm trong khoảng 0.45–0.55
    if 0.45 <= prediction_prob[0] <= 0.55 or 0.45 <= prediction_prob[1] <= 0.55:
        prediction_prob = None
        prediction = None
    else:
        prediction = grid.predict(latest_features)[0]

    return prediction, accuracy * 100, f1 * 100


# Hàm xác định xu hướng cuối cùng
# Hàm xác định xu hướng cuối cùng
def get_final_trend(client):
    # Phân tích xu hướng cho hai khung thời gian
    trend_h1, accuracy_h1, f1_h1 = analyze_trend(client, "ETHUSDT", "1h", 1500)
    trend_h4, accuracy_h4, f1_h4 = analyze_trend(client, "ETHUSDT", "4h", 1500)

    # Kiểm tra nếu một trong hai giá trị là None
    if trend_h1 is None or trend_h4 is None:
        return "Xu hướng không rõ ràng"

    # Tính xác suất kết hợp
    combined_acc = combined_probability(accuracy_h1 / 100, accuracy_h4 / 100)

    # Kiểm tra các điều kiện để quyết định kết quả
    if (trend_h1 == 1 and trend_h4 == 1 and combined_acc >= 0.89) or \
       (trend_h1 == 1 and accuracy_h1 > 72 and f1_h1 > 72) or \
       (trend_h4 == 1 and accuracy_h4 > 69 and f1_h4 > 70):
        return "Xu hướng tăng"
        
    elif (trend_h1 == 0 and trend_h4 == 0 and combined_acc >= 0.89) or \
         (trend_h1 == 0 and accuracy_h1 > 72 and f1_h1 > 72) or \
         (trend_h4 == 0 and accuracy_h4 > 69 and f1_h4 > 70):
        return "Xu hướng giảm"
        
    # Nếu một trong các khung thời gian có xu hướng không rõ ràng
    elif trend_h1 == -1 or trend_h4 == -1:
        return "Xu hướng không rõ ràng"
    
    else:
        return "Xu hướng không rõ ràng"
