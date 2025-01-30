# config.py
import os
from datetime import timedelta

class Settings:
    # === API Configuration ===
    EXCHANGE_CONFIGS = {
        "binance": {
            "api_key": os.getenv("BINANCE_API_KEY", "YOUR_BINANCE_API_KEY"),
            "api_secret": os.getenv("BINANCE_API_SECRET", "YOUR_BINANCE_SECRET_KEY"),
            "testnet": True  # Set to False for live trading
        },
        "bybit": {
            "api_key": os.getenv("BYBIT_API_KEY", "YOUR_BYBIT_API_KEY"),
            "api_secret": os.getenv("BYBIT_API_SECRET", "YOUR_BYBIT_SECRET_KEY"),
            "testnet": True
        },
        "kraken": {
            "api_key": os.getenv("KRAKEN_API_KEY", "YOUR_KRAKEN_API_KEY"),
            "private_key": os.getenv("KRAKEN_PRIVATE_KEY", "YOUR_KRAKEN_PRIVATE_KEY")
        }
    }

    # === Trading Parameters ===
    TRADING_PAIR = "BTCUSDT"           # Base trading pair (BTCUSDT, ETHUSDT, etc.)
    ACTIVE_EXCHANGES = ["binance", "bybit"]  # Exchanges to monitor
    EXCHANGE_FEES = {                  # Maker/taker fees (adjust for your account tier)
        "binance": 0.001,              # 0.1%
        "bybit": 0.0006,               # 0.06%
        "kraken": 0.0026               # 0.26%
    }
    
    # === Risk Management ===
    ARBITRAGE_THRESHOLD = 0.0015       # Minimum profit % to consider (0.15%)
    MIN_PROFIT_PCT = 0.001             # Absolute minimum profit to execute (0.1%)
    MAX_TRADE_AMOUNT = 0.01            # Max BTC to trade per execution (0.01 BTC)
    TRADE_AMOUNT = 0.001               # Default trade size (0.001 BTC)
    ORDER_TIMEOUT = 30                 # Seconds to wait for order fulfillment

    # === Execution Parameters ===
    CHECK_INTERVAL = 5                 # Price check interval (seconds)
    ERROR_RETRY_DELAY = 60             # Seconds to wait after critical errors
    PRICE_FRESHNESS = 2                # Maximum acceptable price age (seconds)

    # === Machine Learning ===
    ML_MODEL_PATH = "models/arbitrage_model.pkl"
    ML_RETRAIN_INTERVAL = timedelta(hours=24)  # Retrain model daily
    DATA_LOG_PATH = "data/arbitrage_data.csv"
    FEATURE_WINDOW = 60                # Minutes of historical data for ML features

    # === Logging & Monitoring ===
    LOG_LEVEL = "INFO"                 # DEBUG, INFO, WARNING, ERROR
    LOG_FILE_PATH = "logs/bot.log"
    PERFORMANCE_METRICS = True         # Enable trading performance tracking

    # === Safety Features ===
    DRY_RUN = True                     # Test mode (no real trades executed)
    MAX_DAILY_LOSS = -0.05             # -5% daily loss limit (auto-shutdown)
    POSITION_LIMITS = {
        "binance": 0.1,                # Max BTC exposure on Binance
        "bybit": 0.1                   # Max BTC exposure on Bybit
    }

# Singleton configuration instance
config = Settings()