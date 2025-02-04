import os
from dotenv import load_dotenv
load_dotenv()

TELEGRAM = {
    "token": os.getenv("TELEGRAM_TOKEN"),
    "chat_id": os.getenv("TELEGRAM_CHAT_ID")
}

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
# Run every 30 minutes
INTERVAL = '30m'
BINANCE_LIMIT = 1000

TRADING_PARAMS = {
    'RISK_REWARD_RATIO': 2.0,
    'STOP_LOSS_ATR_MULTIPLIER': 1.5,
    'MIN_DATA_POINTS': 200,
    'MAX_RETRIES': 3
}

SUPPORT_RESISTANCE = {
    'LOOKBACK_PERIODS': 200,
    'RELEVANCE_THRESHOLD': 0.1,
    'CLUSTER_TOLERANCE': 0.005,
    'MAX_LEVELS': 5,
    'SWING_WINDOW': 5
}

# Machine Learning parameters – using a stacking ensemble with grid search tuning.
ML_PARAMS = {
    'PREDICTION_HORIZON': 1,
    'TRAIN_TEST_SPLIT': 0.8,
    'MODEL_TYPE': 'StackingEnsemble',
    'TUNE_MODEL': True
}

SENTIMENT_ANALYSIS = {
    'ENABLE_NEWS_SENTIMENT': True,
    'ENABLE_SOCIAL_MEDIA_SENTIMENT': True
}

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def debug_print(message, level="info"):
    level = level.lower()
    log_levels = {
        'debug': logger.debug,
        'info': logger.info,
        'warning': logger.warning,
        'error': logger.error,
        'critical': logger.critical
    }
    log_func = log_levels.get(level, logger.info)
    log_func(message)
