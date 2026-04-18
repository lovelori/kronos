import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime

# Add the project root to sys.path to import model package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Kronos, KronosTokenizer, KronosPredictor

SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "LINKUSDT", "SOLUSDT"]
INTERVAL = "4h"
LOOKBACK = 400
PRED_LEN = 6  # Predict 6 periods (24 hours)

def fetch_gateio_klines(symbol, interval, limit):
    """Fetch candlestick data from Gate.io public API."""
    # Gate.io uses format like BTC_USDT
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
    
    # Gate.io kline columns: [timestamp, volume, close, high, low, open, amount]
    columns = ['timestamps', 'volume', 'close', 'high', 'low', 'open', 'amount']
    df = pd.DataFrame(data, columns=columns)
    
    # Ensure numerical types for Kronos
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = df[col].astype(float)
        
    df['timestamps'] = pd.to_datetime(df['timestamps'].astype(float), unit='s')
    return df

def send_feishu_message(webhook_url, title, results):
    """Send a structured message to Feishu webhook."""
    if not webhook_url:
        print("Warning: FEISHU_WEBHOOK environment variable is not set. Skipping notification.")
        return
        
    text = "\n\n".join(results)
    payload = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": title,
                    "content": [
                        [{"tag": "text", "text": text}]
                    ]
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

def main():
    webhook_url = os.environ.get("FEISHU_WEBHOOK")
    
    print("Loading Kronos Tokenizer and Model...")
    # Load the best open-source model: Kronos-base
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    
    print("========================================")
    
    results = []
    
    for symbol in SYMBOLS:
        print(f"Fetching and predicting for {symbol}...")
        try:
            df = fetch_gateio_klines(symbol, INTERVAL, LOOKBACK)
            
            x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
            x_timestamp = df['timestamps'].copy()
            
            # Formulate future predictions timestamps
            # Using the delta between the last two points to extrapolate
            time_diff = x_timestamp.iloc[-1] - x_timestamp.iloc[-2]
            y_timestamp = pd.Series([x_timestamp.iloc[-1] + time_diff * i for i in range(1, PRED_LEN + 1)])
            
            # Predict
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=PRED_LEN,
                T=1.0,  # default temperature
                top_p=0.9,
                sample_count=1,
                verbose=False
            )
            
            current_close = x_df['close'].iloc[-1]
            predicted_close = pred_df['close'].iloc[-1]
            
            percent_change = ((predicted_close - current_close) / current_close) * 100
            
            if percent_change > 0:
                trend_icon = "📈 涨 (Up)"
            elif percent_change < 0:
                trend_icon = "📉 跌 (Down)"
            else:
                trend_icon = "➡️ 平 (Flat)"
                
            coin_name = symbol.replace("USDT", "")
            result_str = (
                f"🪙 【{coin_name}】: {trend_icon}\n"
                f"• 现价(Current): {current_close:.4f}\n"
                f"• 24h后预测(Predicted): {predicted_close:.4f}\n"
                f"• 预计变化幅(Change): {percent_change:+.2f}%"
            )
            results.append(result_str)
            print(f"{symbol} processed successfully.")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            results.append(f"🪙 【{symbol.replace('USDT', '')}】: ⚠️ 预测失败\n• 错误信息: {e}")
            
    print("========================================")
    
    # Notify through Feishu
    title = f"🤖 Kronos 智能预测报告 (24小时趋势)"
    send_feishu_message(webhook_url, title, results)

if __name__ == "__main__":
    main()
