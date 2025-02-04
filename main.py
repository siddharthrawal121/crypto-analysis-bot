import time
import sys
import json
import os
from config import COINS, INTERVAL, TRADING_PARAMS, debug_print
from crypto_utils import (
    get_crypto_data,
    calculate_advanced_indicators,
    calculate_support_resistance,
    generate_chart,
    generate_insights,
    tuned_stacking_prediction,
    analyze_sentiment,
    send_telegram_message
)

class CryptoBot:
    def __init__(self):
        self.running = True
        self.analysis_count = 0
        self.coins = COINS
        self.interval = INTERVAL
        self.interval_seconds = self._get_interval_seconds()

    def _get_interval_seconds(self):
        interval_map = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        return interval_map.get(self.interval, 3600)

    def safe_analyze(self, coin):
        for attempt in range(TRADING_PARAMS['MAX_RETRIES']):
            try:
                debug_print(f"Analyzing {coin} (attempt {attempt+1})...", "info")
                df = get_crypto_data(coin, self.interval)
                if df is None or len(df) < TRADING_PARAMS['MIN_DATA_POINTS']:
                    debug_print(f"Insufficient raw data for {coin}", "warning")
                    return False
                df = calculate_advanced_indicators(df)
                if df is None or df.empty:
                    debug_print(f"Indicator calculation failed for {coin}", "warning")
                    return False
                sr_levels = calculate_support_resistance(df)
                future_price, rmse, accuracy, trend = tuned_stacking_prediction(df, coin)
                chart = generate_chart(coin, df, sr_levels, predicted_price=future_price)
                fib_levels = df.attrs.get('fib_levels', None)
                analysis = generate_insights(coin, df, sr_levels, fib_levels=fib_levels)

                from datetime import datetime, timedelta
                now_utc = datetime.utcnow()
                tomorrow_utc = now_utc + timedelta(days=1)
                next_close = datetime(tomorrow_utc.year, tomorrow_utc.month, tomorrow_utc.day, 0, 0, 0)
                next_close_str = next_close.strftime('%Y-%m-%d %H:%M UTC')

                price_message = (
                    f"🔮 *{coin} Future Price Prediction*\n\n"
                    f"👉 *Predicted Next Close:* **${future_price:.2f}**\n"
                    f"📊 *Model RMSE:* **{rmse:.2f}**\n"
                    f"📈 *Trend:* **{trend}**\n" +
                    (f"🎯 *Previous Prediction Accuracy:* **{accuracy:.2f}%**\n" if accuracy is not None else "") +
                    f"⏰ *Next Market Close (UTC):* **{next_close_str}**\n\n" +
                    "*Technical Indicators Snapshot:*\n"
                    f"- EMA 50 vs EMA 200: {'Bullish' if trend=='Upward' else 'Bearish'}\n"
                    f"- RSI (14) reflects market momentum\n"
                    f"- Extra indicators (MACD, ATR, OBV, ADX) refine predictions\n\n"
                    "🚀 *Stay ahead in the market and trade smart!*"
                )
                send_telegram_message(price_message)

                if os.path.exists("prediction_history.json"):
                    with open("prediction_history.json", "r") as f:
                        history = json.load(f)
                    if coin in history:
                        history[coin]["actual_price"] = df['close'].iloc[-1]
                    with open("prediction_history.json", "w") as f:
                        json.dump(history, f)

                news_sentiment, news_polarity = analyze_sentiment(f"Latest news about {coin}")
                social_sentiment, social_polarity = analyze_sentiment(f"Latest tweets about {coin}")
                sentiment_message = (
                    f"💬 *Sentiment Analysis for {coin}*\n\n"
                    f"📰 *News Sentiment:* **{news_sentiment}** (Polarity: **{news_polarity:.2f}**)\n"
                    f"🐦 *Social Sentiment:* **{social_sentiment}** (Polarity: **{social_polarity:.2f}**)"
                )
                send_telegram_message(sentiment_message)

                if chart and analysis:
                    send_telegram_message(f"📈 *{coin} Detailed Market Update & Chart*", chart)
                    time.sleep(1)
                    send_telegram_message(analysis)
                return True
            except Exception as e:
                debug_print(f"Analysis failed for {coin}: {str(e)}", "error")
                time.sleep(5)
        return False

    def main_loop(self):
        debug_print("🚀 Starting the Ultimate Crypto Analysis Bot!", "info")
        try:
            while self.running:
                self.analysis_count += 1
                start_time = time.time()
                debug_print(f"\n=== Analysis Cycle #{self.analysis_count} ===", "info")
                results = {}
                for coin in self.coins:
                    results[coin] = self.safe_analyze(coin)
                    time.sleep(2)
                success_rate = sum(results.values()) / len(results)
                debug_print(f"Cycle completed with {success_rate:.0%} success rate", "info")
                elapsed = time.time() - start_time
                sleep_time = max(self.interval_seconds - (elapsed % self.interval_seconds), 60)
                debug_print(f"Next analysis in {sleep_time/60:.1f} minutes", "info")
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            self.shutdown()
        except Exception as e:
            debug_print(f"Critical error: {str(e)}", "critical")
            self.shutdown()

    def shutdown(self):
        self.running = False
        debug_print("Bot shutdown initiated. Goodbye!", "info")
        sys.exit(0)

if __name__ == "__main__":
    bot = CryptoBot()
    bot.main_loop()
