import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Telegram Configuration
TELEGRAM = {
    "token": os.getenv("TELEGRAM_TOKEN"),
    "chat_id": os.getenv("TELEGRAM_CHAT_ID")
}

# Trading Parameters
COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
INTERVAL = '1h'
SECONDARY_INTERVAL = '4h'
BINANCE_LIMIT = 200

# Support/Resistance Configuration
SUPPORT_RESISTANCE = {
    'LOOKBACK_PERIODS': 100,
    'RELEVANCE_THRESHOLD': 0.15,
    'CLUSTER_TOLERANCE': 0.005,
    'MAX_LEVELS': 5
}

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_bot.log"),
        logging.StreamHandler()
    ]
)

def debug_print(message, level="info"):
    logger = logging.getLogger()
    level = level.lower()
    if level == "error":
        logger.error(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "debug":
        logger.debug(message)
    else:
        logger.info(message)