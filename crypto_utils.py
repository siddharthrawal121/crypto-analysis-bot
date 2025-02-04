import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime, timedelta
import json
import os
import sys

from config import (
    TELEGRAM, SUPPORT_RESISTANCE, TRADING_PARAMS, ML_PARAMS,
    BINANCE_LIMIT, SENTIMENT_ANALYSIS, debug_print
)

# For stacking ensemble and grid search tuning:
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

PREDICTION_HISTORY_FILE = "prediction_history.json"

def get_crypto_data(symbol, interval):
    base_url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': BINANCE_LIMIT
    }
    try:
        debug_print(f"Fetching data for {symbol}...", "info")
        response = requests.get(base_url, params=params, timeout=15)
        if not response.ok:
            debug_print(f"API Error {response.status_code} for {symbol}", "error")
            return None
        data = response.json()
        if len(data) < TRADING_PARAMS['MIN_DATA_POINTS']:
            debug_print(f"Insufficient data points for {symbol}", "warning")
            return None
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_quote_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.set_index('timestamp')[numeric_cols].dropna()
        return df.iloc[-BINANCE_LIMIT:]
    except Exception as e:
        debug_print(f"Data fetch failed for {symbol}: {str(e)}", "error")
        return None

def add_extra_indicators(df):
    """
    Adds extra technical indicators:
      - Fibonacci retracement levels from the last 100 periods.
      - Ichimoku Cloud using pandas_ta with add_suffix to avoid duplicate column names.
      - Pivot Points using pandas_ta.
    Returns updated df and a dict of Fibonacci levels.
    """
    try:
        # Fibonacci retracement levels
        recent = df.tail(100)
        high = recent['high'].max()
        low = recent['low'].min()
        diff = high - low
        fib_levels = {
            "Fib_23.6": high - 0.236 * diff,
            "Fib_38.2": high - 0.382 * diff,
            "Fib_50": high - 0.5 * diff,
            "Fib_61.8": high - 0.618 * diff
        }
        # Ichimoku Cloud with add_suffix to avoid duplicate column names
        ichimoku_df = df.ta.ichimoku()
        ichimoku_df = ichimoku_df.add_suffix('_ichimoku')
        df = df.join(ichimoku_df)
        # Pivot Points (using a 14-period)
        pivots = ta.pivots(df['high'], df['low'], df['close'], length=14)
        df = df.join(pivots)
        return df, fib_levels
    except Exception as e:
        debug_print(f"Extra indicator calculation failed: {str(e)}", "error")
        return df, {}

def calculate_advanced_indicators(df):
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        debug_print("Missing base columns for indicators", "error")
        return None
    try:
        df = df.copy()
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        macd_df = ta.macd(df['close'])
        df['MACD'] = macd_df['MACD_12_26_9']
        df['MACD_signal'] = macd_df['MACDs_12_26_9']
        df['EMA_50'] = ta.ema(df['close'], length=50)
        df['EMA_200'] = ta.ema(df['close'], length=200)
        bbands = ta.bbands(df['close'], length=20)
        df['BBU'] = bbands['BBU_20_2.0']
        df['BBL'] = bbands['BBL_20_2.0']
        df['BB_width'] = df['BBU'] - df['BBL']
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        # Add extra indicators
        df, fib_levels = add_extra_indicators(df)
        df = df.dropna()
        df.attrs['fib_levels'] = fib_levels
        return df
    except Exception as e:
        debug_print(f"Indicator calculation failed: {str(e)}", "error")
        return None

def calculate_support_resistance(df):
    try:
        if df.empty:
            return None
        current_price = df['close'].iloc[-1]
        lookback = min(SUPPORT_RESISTANCE['LOOKBACK_PERIODS'], len(df))
        swing_highs = df['high'].rolling(lookback).max().dropna().tolist()
        swing_lows = df['low'].rolling(lookback).min().dropna().tolist()
        levels = swing_lows + swing_highs
        levels = [lvl for lvl in levels if abs(lvl - current_price) / current_price <= SUPPORT_RESISTANCE['RELEVANCE_THRESHOLD']]
        if len(levels) > 1:
            from sklearn.cluster import KMeans
            n_clusters = min(SUPPORT_RESISTANCE['MAX_LEVELS'], len(levels))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(np.array(levels).reshape(-1, 1))
            clustered = [round(np.mean([lvl for i, lvl in enumerate(levels) if clusters[i]==c]), 2) for c in set(clusters)]
        else:
            clustered = levels
        support = [l for l in clustered if l < current_price]
        resistance = [l for l in clustered if l > current_price]
        return {
            'current_price': round(current_price, 2),
            'support': sorted(support, reverse=True)[:SUPPORT_RESISTANCE['MAX_LEVELS']],
            'resistance': sorted(resistance)[:SUPPORT_RESISTANCE['MAX_LEVELS']]
        }
    except Exception as e:
        debug_print(f"S/R calculation failed: {str(e)}", "error")
        return None

def generate_chart(symbol, df, sr_levels, predicted_price=None):
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14,7))
        ax.plot(df.index, df['close'], label="Price", color="#00FF9D")
        ax.plot(df.index, df['EMA_50'], label="EMA 50", color="#FF006A")
        ax.plot(df.index, df['EMA_200'], label="EMA 200", color="#00FFFF")
        # Plot Ichimoku Cloud if available
        if 'ISA_9_ichimoku' in df.columns and 'ISB_26_ichimoku' in df.columns:
            ax.fill_between(df.index, df['ISA_9_ichimoku'], df['ISB_26_ichimoku'], color='purple', alpha=0.2, label='Ichimoku Cloud')
        if sr_levels:
            for level in sr_levels.get('support', []):
                ax.axhline(level, color="green", linestyle="--", alpha=0.7)
            for level in sr_levels.get('resistance', []):
                ax.axhline(level, color="red", linestyle="--", alpha=0.7)
            ax.axhline(sr_levels['current_price'], color="white", linewidth=1, alpha=0.5)
        if predicted_price is not None:
            ax.axhline(predicted_price, color="orange", linestyle=":", linewidth=2)
            ax.text(df.index[-1], predicted_price, f" Predicted: ${predicted_price:.2f}", color="orange", fontsize=10, verticalalignment="bottom")
        ax.set_title(f"{symbol} Price Action", fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(alpha=0.1)
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0E1116')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        debug_print(f"Chart generation failed: {str(e)}", "error")
        return None

def generate_insights(symbol, df, sr_levels, fib_levels=None):
    try:
        current_price = df['close'].iloc[-1]
        intraday_high = df['high'].max()
        intraday_low = df['low'].min()
        atr = df['ATR_14'].iloc[-1] if 'ATR_14' in df.columns else None
        rsi = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else None
        macd_bullish = df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]
        ema_trend = df['EMA_50'].iloc[-1] > df['EMA_200'].iloc[-1]
        trend = "Upward" if ema_trend else "Downward"
        
        # Next market close (assume daily close at 00:00 UTC)
        now_utc = datetime.utcnow()
        tomorrow_utc = now_utc + timedelta(days=1)
        next_close = datetime(tomorrow_utc.year, tomorrow_utc.month, tomorrow_utc.day, 0, 0, 0)
        time_to_close = next_close - now_utc
        hours_to_close = time_to_close.total_seconds() / 3600

        message = (
            f"👋 *Market Update for {symbol}*\n\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"💰 *Current Price:* ${current_price:.2f}\n"
            f"📈 *Intraday High:* ${intraday_high:.2f} | *Intraday Low:* ${intraday_low:.2f}\n"
            f"⏳ *Next Daily Close (UTC):* {next_close.strftime('%Y-%m-%d %H:%M')} (in {hours_to_close:.1f} hrs)\n\n"
            "🔍 *Technical Indicators:*\n"
            f"- RSI (14): {rsi:.1f} " +
                ("(Oversold <30)" if rsi < 30 else ("(Overbought >70)" if rsi > 70 else "(Neutral)")) + "\n"
            f"- MACD: {'Bullish' if macd_bullish else 'Bearish'}\n"
            f"- EMA 50 vs EMA 200: {'Bullish' if ema_trend else 'Bearish'} (Trend: *{trend}*)\n"
            f"- ATR (14): " + (f"${atr:.2f}" if atr else "N/A") + "\n"
            f"- Bollinger Band Width: ${df['BB_width'].iloc[-1]:.2f}\n\n"
            "🔑 *Support & Resistance Levels:*\n"
            f"   - *Current Price:* ${sr_levels['current_price']:.2f}\n"
            "   - *Supports:* " + (", ".join([f"${s:.2f}" for s in sr_levels.get('support', [])]) or "-") + "\n"
            "   - *Resistances:* " + (", ".join([f"${r:.2f}" for r in sr_levels.get('resistance', [])]) or "-") + "\n\n"
            "📉 *Risk Management Settings:*\n"
            f"- Risk Reward Ratio: {TRADING_PARAMS['RISK_REWARD_RATIO']}\n"
            f"- Stop Loss ATR Multiplier: {TRADING_PARAMS['STOP_LOSS_ATR_MULTIPLIER']}\n\n"
        )
        if fib_levels is None:
            fib_levels = df.attrs.get('fib_levels', None)
        if fib_levels:
            message += "*Fibonacci Retracement Levels (Last 100 Periods):*\n"
            for key, level in fib_levels.items():
                message += f"- {key}: ${level:.2f}\n"
            message += "\n"
        message += (
            "🤖 *Ensemble Price Prediction:*\n"
            "   - Check your Telegram for the latest prediction update!\n\n"
            "💬 *Sentiment Analysis:* (in progress...)\n\n"
            "⚠️ *Disclaimer:* This is for educational purposes only. Trade responsibly!\n\n"
            "🚀 *Stay informed, stay ahead, and trade smart!*"
        )
        return message
    except Exception as e:
        debug_print(f"Insight generation failed: {str(e)}", "error")
        return f"⚠️ Error generating insights for {symbol}"

def tuned_stacking_prediction(df, symbol):
    """
    Use an ensemble stacking approach with grid search tuning and extra features
    to predict the future price. Returns predicted price, RMSE, previous accuracy (if any), and trend.
    """
    try:
        if df.empty or len(df) < 50:
            debug_print("Not enough data for prediction", "error")
            return None, None, None, None

        df = df.copy()
        # Basic feature engineering
        df['Return'] = df['close'].pct_change()
        df['MA_50'] = df['close'].rolling(window=50).mean()
        df['MA_200'] = df['close'].rolling(window=200).mean()
        df['RSI'] = ta.rsi(df['close'], length=14)
        macd_df = ta.macd(df['close'])
        df['MACD'] = macd_df['MACD_12_26_9']
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        # Extra technical indicators:
        df['OBV'] = ta.obv(df['close'], df['volume'])
        adx_df = ta.adx(df['high'], df['low'], df['close'])
        df['ADX'] = adx_df['ADX_14']
        # Add extra indicators (Fibonacci, Ichimoku, Pivot Points)
        df, fib_levels = add_extra_indicators(df)
        df = df.dropna()

        features = ['Return', 'MA_50', 'MA_200', 'RSI', 'MACD', 'ATR', 'OBV', 'ADX']
        target = 'close'

        if len(df) < ML_PARAMS['PREDICTION_HORIZON'] + 1:
            debug_print("Not enough data points after feature calculation", "error")
            return None, None, None, None

        # Determine trend from MA_50 vs MA_200
        ema_trend = df['MA_50'].iloc[-1] > df['MA_200'].iloc[-1]
        trend = "Upward" if ema_trend else "Downward"

        X = df[features][:-ML_PARAMS['PREDICTION_HORIZON']]
        y = df[target].shift(-ML_PARAMS['PREDICTION_HORIZON'])[:-ML_PARAMS['PREDICTION_HORIZON']]
        if X.empty or y.empty:
            debug_print("Empty features or target for prediction", "error")
            return None, None, None, None

        split_index = int(len(X) * ML_PARAMS['TRAIN_TEST_SPLIT'])
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        rf = RandomForestRegressor(random_state=42)
        xgb = XGBRegressor(random_state=42, objective='reg:squarederror')

        if ML_PARAMS['TUNE_MODEL']:
            param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
            grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='neg_mean_squared_error')
            grid_rf.fit(X_train, y_train)
            rf = grid_rf.best_estimator_

            param_grid_xgb = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7]}
            grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='neg_mean_squared_error')
            grid_xgb.fit(X_train, y_train)
            xgb = grid_xgb.best_estimator_

        estimators = [
            ('rf', rf),
            ('xgb', xgb)
        ]
        stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=RandomForestRegressor(n_estimators=50, random_state=42)
        )
        stacking_model.fit(X_train, y_train)
        y_pred = stacking_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        future_features = df[features].iloc[-1].to_frame().T
        future_price = stacking_model.predict(future_features)[0]

        history = {}
        if os.path.exists(PREDICTION_HISTORY_FILE):
            try:
                with open(PREDICTION_HISTORY_FILE, "r") as f:
                    history = json.load(f)
            except Exception:
                debug_print("Failed to decode prediction history. Resetting.", "warning")
                history = {}
        if symbol in history and history[symbol].get("predicted_price") is not None and history[symbol].get("actual_price") is not None:
            last_predicted = history[symbol]["predicted_price"]
            last_actual = history[symbol]["actual_price"]
            accuracy = 100 - (abs(last_predicted - last_actual) / last_actual * 100)
        else:
            accuracy = None

        history[symbol] = {
            "predicted_price": future_price,
            "actual_price": None,
            "timestamp": datetime.now().isoformat()
        }
        with open(PREDICTION_HISTORY_FILE, "w") as f:
            json.dump(history, f)

        return future_price, rmse, accuracy, trend
    except Exception as e:
        debug_print(f"Ensemble price prediction failed: {str(e)}", "error")
        return None, None, None, None

def analyze_sentiment(text):
    from textblob import TextBlob
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
        return sentiment, polarity
    except Exception as e:
        debug_print(f"Sentiment analysis failed: {str(e)}", "error")
        return None, None

def send_telegram_message(text, image=None):
    try:
        if not TELEGRAM.get('token') or not TELEGRAM.get('chat_id'):
            debug_print("Telegram credentials missing", "error")
            return False
        if image:
            url = f"https://api.telegram.org/bot{TELEGRAM['token']}/sendPhoto"
            files = {'photo': ('chart.png', image.getvalue(), 'image/png')}
            data = {
                'chat_id': TELEGRAM['chat_id'],
                'caption': text[:1024],
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, files=files, data=data)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM['token']}/sendMessage"
            data = {
                'chat_id': TELEGRAM['chat_id'],
                'text': text[:4096],
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, json=data)
        if response.status_code != 200:
            debug_print(f"Telegram API error: {response.text}", "error")
            return False
        return True
    except Exception as e:
        debug_print(f"Telegram communication failed: {str(e)}", "error")
        return False
