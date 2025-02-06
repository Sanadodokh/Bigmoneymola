# config.py
import os
from typing import Dict, TypedDict
from decimal import Decimal

class ExchangeConfig(TypedDict):
    api_key: str
    api_secret: str
    fee: Decimal
    rate_limit: int
    base_currency: str
    min_order_size: Decimal

class AppConfig:
    def __init__(self):
        self.EXCHANGES: Dict[str, ExchangeConfig] = {
            'binance': {
                'api_key': os.getenv('BINANCE_API_KEY', ''),
                'api_secret': os.getenv('BINANCE_API_SECRET', ''),
                'fee': Decimal('0.001'),
                'rate_limit': 10,
                'base_currency': 'USDT',
                'min_order_size': Decimal('0.001')
            },
            'kucoin': {
                'api_key': os.getenv('KUCOIN_API_KEY', ''),
                'api_secret': os.getenv('KUCOIN_API_SECRET', ''),
                'fee': Decimal('0.0018'),
                'rate_limit': 8,
                'base_currency': 'USDT',
                'min_order_size': Decimal('0.01')
            }
        }
        
        self.MIN_PROFITABILITY = Decimal('0.002')
        self.MAX_TRADE_SIZE = Decimal('0.1')
        self.COOLDOWN_PERIOD = 5
        self.SIMULATION_MODE = os.getenv('SIMULATION_MODE', 'true').lower() == 'true'
        self.INITIAL_BALANCE = Decimal('10000')
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

config = AppConfig()