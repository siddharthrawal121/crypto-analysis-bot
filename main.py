import time
from crypto_utils import (
    get_crypto_data,
    calculate_advanced_indicators,
    calculate_support_resistance,
    generate_chart,
    generate_insights,
    send_telegram_message
)
from config import COINS, INTERVAL, SECONDARY_INTERVAL, debug_print

def analyze_coin(coin):
    try:
        debug_print(f"Analyzing {coin}...", level="info")
        
        # Get data for both timeframes
        df_primary = get_crypto_data(coin, INTERVAL)
        df_secondary = get_crypto_data(coin, SECONDARY_INTERVAL)
        
        if df_primary is None or df_secondary is None:
            return False
            
        # Calculate indicators
        df_primary = calculate_advanced_indicators(df_primary)
        df_secondary = calculate_advanced_indicators(df_secondary)
        
        # Calculate support/resistance
        levels = calculate_support_resistance(df_primary)
        
        # Generate outputs
        chart = generate_chart(coin, df_primary, levels)
        analysis = generate_insights(coin, df_primary, levels)
        
        # Send to Telegram
        if chart and analysis:
            send_telegram_message(f"*{coin} Market Update*", chart)
            time.sleep(1)
            send_telegram_message(analysis)
            return True
            
        return False
    except Exception as e:
        debug_print(f"Analysis failed for {coin}: {str(e)}", level="error")
        return False

def main_loop():
    debug_print("Starting analysis bot...", level="info")
    while True:
        try:
            debug_print("\n" + "="*40 + " NEW CYCLE " + "="*40, level="info")
            start_time = time.time()
            
            for coin in COINS:
                analyze_coin(coin)
                time.sleep(1)
                
            cycle_time = time.time() - start_time
            debug_print(f"Cycle completed in {cycle_time:.1f} seconds", level="info")
            time.sleep(max(3600 - cycle_time, 300))  # 1 hour cycle
        except KeyboardInterrupt:
            debug_print("Bot stopped by user", level="info")
            break
        except Exception as e:
            debug_print(f"Main loop error: {str(e)}", level="error")
            time.sleep(600)

if __name__ == "__main__":
    main_loop()