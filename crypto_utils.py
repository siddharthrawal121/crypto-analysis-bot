import json
import pandas as pd
import pandas_ta as ta
import requests
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import logging
from datetime import datetime
from config import TELEGRAM, COINS, INTERVAL, SECONDARY_INTERVAL, BINANCE_LIMIT, SUPPORT_RESISTANCE, debug_print

def get_crypto_data(symbol, interval=INTERVAL):
    try:
        debug_print(f"Fetching {symbol} ({interval}) data...", level="info")
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={BINANCE_LIMIT}"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_quote_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        debug_print(f"Error fetching {symbol} data: {str(e)}", level="error")
        return None

def calculate_advanced_indicators(df):
    try:
        if df is None or df.empty:
            return None
            
        debug_print("Calculating technical indicators...", level="info")
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.bbands(length=20, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.obv(append=True)
        df.ta.vwap(append=True)
        return df.dropna()
    except Exception as e:
        debug_print(f"Indicator error: {str(e)}", level="error")
        return None

def calculate_support_resistance(df):
    if df is None or df.empty:
        return None
        
    try:
        cfg = SUPPORT_RESISTANCE
        lookback = cfg['LOOKBACK_PERIODS']
        current_price = df['close'].iloc[-1]
        
        # Calculate Fibonacci levels
        recent_high = df['high'].rolling(lookback).max().iloc[-1]
        recent_low = df['low'].rolling(lookback).min().iloc[-1]
        fib_levels = {
            '0.236': recent_high - (recent_high - recent_low) * 0.236,
            '0.382': recent_high - (recent_high - recent_low) * 0.382,
            '0.5': recent_high - (recent_high - recent_low) * 0.5,
            '0.618': recent_high - (recent_high - recent_low) * 0.618
        }
        
        # Combine methods
        levels = []
        levels.extend(df['low'].rolling(lookback).min().iloc[-3:].tolist())
        levels.extend(df['high'].rolling(lookback).max().iloc[-3:].tolist())
        levels.extend(fib_levels.values())
        
        # Filter relevant levels
        relevant_levels = [lvl for lvl in levels 
                         if abs(lvl - current_price)/current_price <= cfg['RELEVANCE_THRESHOLD']]
        
        # Cluster levels
        clustered = []
        for level in sorted(relevant_levels):
            if not clustered:
                clustered.append(level)
            else:
                if abs(level - clustered[-1])/clustered[-1] > cfg['CLUSTER_TOLERANCE']:
                    clustered.append(level)
        
        # Separate support/resistance
        support = [lvl for lvl in clustered if lvl < current_price][:cfg['MAX_LEVELS']]
        resistance = [lvl for lvl in clustered if lvl > current_price][:cfg['MAX_LEVELS']]
        
        return {
            'current_price': current_price,
            'support': sorted(support, reverse=True),
            'resistance': sorted(resistance),
            'fib_levels': fib_levels
        }
    except Exception as e:
        debug_print(f"Support/Resistance error: {str(e)}", level="error")
        return None

def generate_chart(symbol, df, levels):
    try:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3,1,1]})
        
        # Price Plot
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], color='#00ff9d', linewidth=1.5, label='Price')
        
        # Plot support/resistance
        if levels:
            current_price = levels['current_price']
            for sup in levels['support']:
                ax1.axhline(sup, color='#00ff9d', linestyle='--', alpha=0.7, label='Support' if sup == levels['support'][0] else "")
            for res in levels['resistance']:
                ax1.axhline(res, color='#ff006a', linestyle='--', alpha=0.7, label='Resistance' if res == levels['resistance'][0] else "")
            ax1.axhline(current_price, color='#ffffff', linestyle='-', linewidth=1, alpha=0.5, label='Current Price')
            
        # Bollinger Bands
        if 'BBL_20_2.0' in df.columns:
            ax1.fill_between(df.index, df['BBL_20_2.0'], df['BBU_20_2.0'], color='#2b2b2b', alpha=0.3)
            ax1.plot(df.index, df['BBM_20_2.0'], color='#787878', linewidth=1, label='Bollinger Mid')
            
        ax1.set_title(f"{symbol} Price Analysis")
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.1)
        
        # RSI
        axes[1].plot(df.index, df['RSI_14'], color='#ffcc00', linewidth=1)
        axes[1].axhline(70, color='#ff006a', linestyle='--', alpha=0.7)
        axes[1].axhline(30, color='#00ff9d', linestyle='--', alpha=0.7)
        axes[1].fill_between(df.index, 30, 70, color='#2b2b2b', alpha=0.3)
        axes[1].set_title("RSI (14)")
        axes[1].grid(alpha=0.1)
        
        # MACD
        axes[2].plot(df.index, df['MACD_12_26_9'], color='#00ff9d', linewidth=1, label='MACD')
        axes[2].plot(df.index, df['MACDs_12_26_9'], color='#ff006a', linewidth=1, label='Signal')
        axes[2].bar(df.index, df['MACDh_12_26_9'], 
                   color=np.where(df['MACDh_12_26_9'] > 0, '#00ff9d', '#ff006a'), 
                   alpha=0.5, width=0.01)
        axes[2].set_title("MACD")
        axes[2].grid(alpha=0.1)
        axes[2].legend()
        
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0e1116')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        debug_print(f"Chart error: {str(e)}", level="error")
        return None

def generate_insights(symbol, df, levels):
    try:
        message = []
        current_price = df['close'].iloc[-1]
        message.append(f"🔥 *{symbol} Technical Analysis* 🔥\n")
        message.append(f"📈 Current Price: ${current_price:.2f}\n")
        
        if levels:
            message.append("*Support/Resistance Levels:*")
            message.append("```")
            message.append("Support Levels:")
            for sup in levels['support']:
                message.append(f"🟢 ${sup:.2f} ({(sup-current_price)/current_price*100:.1f}%)")
            message.append("\nResistance Levels:")
            for res in levels['resistance']:
                message.append(f"🔴 ${res:.2f} ({(res-current_price)/current_price*100:.1f}%)")
            message.append("```\n")
            
        # RSI Analysis
        if 'RSI_14' in df.columns:
            rsi = df['RSI_14'].iloc[-1]
            status = "OVERSOLD 🚨" if rsi < 30 else "OVERBOUGHT ⚠️" if rsi > 70 else "Neutral"
            message.append(f"📊 *RSI (14):* {rsi:.1f} - {status}")
            
        # MACD Analysis
        if 'MACD_12_26_9' in df.columns:
            macd = df['MACD_12_26_9'].iloc[-1]
            signal = df['MACDs_12_26_9'].iloc[-1]
            trend = "Bullish 🟢" if macd > signal else "Bearish 🔴"
            message.append(f"📉 *MACD:* {macd:.4f} | Signal: {signal:.4f} - {trend}")
            
        # Volume Analysis
        if 'OBV' in df.columns:
            obv_trend = "Upward" if df['OBV'].iloc[-1] > df['OBV'].iloc[-2] else "Downward"
            message.append(f"💹 *Volume Trend:* {obv_trend}")
            
        message.append("\n⚠️ *Disclaimer:* This is not financial advice. Always do your own research.")
        
        return "\n".join(message)
    except Exception as e:
        debug_print(f"Insight error: {str(e)}", level="error")
        return f"⚠️ Error generating insights for {symbol}"

def send_telegram_message(text, image=None):
    try:
        if image:
            url = f"https://api.telegram.org/bot{TELEGRAM['token']}/sendPhoto"
            files = {'photo': image}
            data = {'chat_id': TELEGRAM['chat_id'], 'caption': text[:900], 'parse_mode': 'Markdown'}
            response = requests.post(url, files=files, data=data)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM['token']}/sendMessage"
            data = {'chat_id': TELEGRAM['chat_id'], 'text': text, 'parse_mode': 'Markdown'}
            response = requests.post(url, json=data)
            
        response.raise_for_status()
        return True
    except Exception as e:
        debug_print(f"Telegram error: {str(e)}", level="error")
        return False