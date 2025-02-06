# exchange_connector.py
import aiohttp
import asyncio
import time
from typing import Dict, Optional
from decimal import Decimal
from config import config
from exceptions import ExchangeAPIError, RateLimitError

class SecureExchangeConnector:
    def __init__(self, exchange_name: str):
        if exchange_name not in config.EXCHANGES:
            raise ConfigurationError(f"Invalid exchange: {exchange_name}")
            
        self.name = exchange_name
        self.conf = config.EXCHANGES[exchange_name]
        self.session = aiohttp.ClientSession()
        self.last_request = 0.0
        self.rate_limit_sem = asyncio.Semaphore(self.conf['rate_limit'])

    async def fetch_order_book(self, symbol: str) -> Dict:
        """Fetch order book with rate limiting and error handling"""
        url = self._get_orderbook_url(symbol)
        
        try:
            async with self.rate_limit_sem:
                await self._enforce_rate_limit()
                
                async with self.session.get(url) as response:
                    if response.status == 429:
                        reset_time = float(response.headers.get('x-ratelimit-reset', 60))
                        raise RateLimitError(self.name, reset_time)
                        
                    if not response.ok:
                        raise ExchangeAPIError(self.name, url)
                        
                    return self._parse_orderbook(await response.json())
                    
        except aiohttp.ClientError as e:
            raise ExchangeAPIError(self.name, url) from e

    async def _enforce_rate_limit(self):
        """Enforce exchange-specific rate limits"""
        elapsed = time.time() - self.last_request
        min_interval = 1 / self.conf['rate_limit']
        
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
            
        self.last_request = time.time()

    def _get_orderbook_url(self, symbol: str) -> str:
        """Get exchange-specific order book URL"""
        endpoints = {
            'binance': f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=100",
            'kucoin': f"https://api.kucoin.com/api/v1/market/orderbook/level2_100?symbol={symbol}"
        }
        return endpoints[self.name]

    def _parse_orderbook(self, data: Dict) -> Dict:
        """Parse exchange-specific order book format"""
        parsers = {
            'binance': lambda d: {
                'bids': [[Decimal(p), Decimal(q)] for p