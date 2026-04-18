import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime
import yfinance as yf

# Add the project root to sys.path to import model package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Kronos, KronosTokenizer, KronosPredictor

LOOKBACK = 400
PRED_LEN = 6

# Define Assets
CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "LINKUSDT", "SOLUSDT", "ORDIUSDT", "INJUSDT", "NEARUSDT"]
YF_US_STOCKS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "CRCL"]
YF_A_SHARES = ["000651.SZ", "600036.SS", "002142.SZ", "002594.SZ"]
YF_COMMODITIES = {"黄金": "GC=F", "白银": "SI=F"}

def fetch_gateio_klines(symbol, interval, limit):
    """Fetch candlestick data from Gate.io public API."""
    gate_symbol = symbol.replace("USDT", "_USDT")
    url = "https://api.gateio.ws/api/v4/spot/candlesticks"
    params = {
        "currency_pair": gate_symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    # Gate.io kline columns: [timestamp, volume, close, high, low, open, amount, window_closed]
    columns = ['timestamps', 'volume', 'close', 'high', 'low', 'open', 'amount', 'window_closed']
    df = pd.DataFrame(data, columns=columns)
    
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = df[col].astype(float)
        
    df['timestamps'] = pd.to_datetime(df['timestamps'].astype(float), unit='s')
    return df

def fetch_yfinance_klines(symbol, interval):
    """Fetch candlestick data from Yahoo Finance."""
    # We need LOOKBACK candles, 2 years of daily data is enough.
    df = yf.download(symbol, interval=interval, period="2y", progress=False)
    
    if df.empty:
        raise ValueError(f"No data fetched for {symbol}")
        
    df = df.reset_index()
    
    # Flatten multi-index if it exists (yfinance behavior depending on version)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [tuple_col[0] for tuple_col in df.columns]

    # Standardize column names
    rename_mapping = {}
    for col in df.columns:
        if 'Date' in col: rename_mapping[col] = 'timestamps'
        elif 'Open' == col: rename_mapping[col] = 'open'
        elif 'High' == col: rename_mapping[col] = 'high'
        elif 'Low' == col: rename_mapping[col] = 'low'
        elif 'Close' == col: rename_mapping[col] = 'close'
        elif 'Volume' == col: rename_mapping[col] = 'volume'
        
    df.rename(columns=rename_mapping, inplace=True)
    
    df['amount'] = df['close'] * df['volume']
    
    df = df.tail(LOOKBACK).copy()
    df.reset_index(drop=True, inplace=True)
    return df


def send_feishu_message(webhook_url, title, results):
    """Send a structured message to Feishu webhook."""
    if not webhook_url:
        print("Warning: FEISHU_WEBHOOK environment variable is not set.")
        return
        
    text = "\n\n".join(results)
    payload = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": title,
                    "content": [[{"tag": "text", "text": text}]]
                }
            }
        }
    }
    
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(webhook_url, json=payload, headers=headers)
        response.raise_for_status()
        print("Successfully sent message to Feishu.")
    except Exception as e:
        print(f"Failed to send Feishu message: {e}")


def predict_for_symbol(symbol, display_name, predictor, df, icon):
    x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_timestamp = df['timestamps'].copy()

    time_diff = x_timestamp.iloc[-1] - x_timestamp.iloc[-2]
    y_timestamp = pd.Series([x_timestamp.iloc[-1] + time_diff * i for i in range(1, PRED_LEN + 1)])

    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1, verbose=False
    )

    current_close = x_df['close'].iloc[-1]
    predicted_close = pred_df['close'].iloc[-1]
    percent_change = ((predicted_close - current_close) / current_close) * 100

    if percent_change > 0: trend_icon = "📈 涨 (Up)"
    elif percent_change < 0: trend_icon = "📉 跌 (Down)"
    else: trend_icon = "➡️ 平 (Flat)"

    return (
        f"{icon} 【{display_name}】: {trend_icon}\n"
        f"• 现价: {current_close:.4f}\n"
        f"• 预测: {predicted_close:.4f}\n"
        f"• 预计变化幅: {percent_change:+.2f}%"
    )


def main():
    webhook_url = os.environ.get("FEISHU_WEBHOOK")
    
    print("Loading Kronos Tokenizer and Model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    
    results = []
    
    # 1. Crypto (4h)
    results.append("====== 🪙 加密货币 (4h) ======")
    for symbol in CRYPTO_SYMBOLS:
        print(f"Processing Crypto: {symbol}...")
        try:
            df = fetch_gateio_klines(symbol, "4h", LOOKBACK)
            res = predict_for_symbol(symbol, symbol.replace("USDT", ""), predictor, df, "🪙")
            results.append(res)
        except Exception as e:
            print(f"Error {symbol}: {e}")
            results.append(f"🪙 【{symbol.replace('USDT', '')}】: ⚠️ 预测失败")
            
    # 2. US Stocks (1d)
    results.append("====== 🇺🇸 美股 (1d) ======")
    for symbol in YF_US_STOCKS:
        print(f"Processing US Stock: {symbol}...")
        try:
            df = fetch_yfinance_klines(symbol, "1d")
            res = predict_for_symbol(symbol, symbol, predictor, df, "🗽")
            results.append(res)
        except Exception as e:
            print(f"Error {symbol}: {e}")
            results.append(f"🗽 【{symbol}】: ⚠️ 预测失败")
            
    # 3. A-Shares (1d)
    results.append("====== 🇨🇳 A股 (1d) ======")
    for symbol in YF_A_SHARES:
        print(f"Processing A-Share: {symbol}...")
        try:
            df = fetch_yfinance_klines(symbol, "1d")
            display_name = symbol.split('.')[0]
            res = predict_for_symbol(symbol, display_name, predictor, df, "🇨🇳")
            results.append(res)
        except Exception as e:
            print(f"Error {symbol}: {e}")
            results.append(f"🇨🇳 【{symbol.split('.')[0]}】: ⚠️ 预测失败")

    # 4. Commodities (1d)
    results.append("====== 🏆 大宗商品 (1d) ======")
    for name, symbol in YF_COMMODITIES.items():
        print(f"Processing Commodity: {name} ({symbol})...")
        try:
            df = fetch_yfinance_klines(symbol, "1d")
            res = predict_for_symbol(symbol, name, predictor, df, "🏆")
            results.append(res)
        except Exception as e:
            print(f"Error {name}: {e}")
            results.append(f"🏆 【{name}】: ⚠️ 预测失败")
            
    title = f"🤖 Kronos 智能预测报告 (多资产趋势)"
    send_feishu_message(webhook_url, title, results)

if __name__ == "__main__":
    main()
