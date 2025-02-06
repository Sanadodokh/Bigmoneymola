# arbitrage_engine.py
import asyncio
from decimal import Decimal, getcontext
from typing import Dict, Optional  # Add this import
import logging
from dataclasses import dataclass

# Configure precision
getcontext().prec = 8

@dataclass
class ArbitrageOpportunity:
    buy_exchange: str
    sell_exchange: str
    symbol: str
    buy_price: Decimal
    sell_price: Decimal
    volume: Decimal
    fees: Dict[str, Decimal]

class ArbitrageEngine:
    def __init__(self, config: Dict):
        self.min_profit = Decimal(config['MIN_PROFIT'])
        self.max_trade_size = Decimal(config['MAX_TRADE_SIZE'])
        self.fee_structures = config['FEES']
        self.logger = logging.getLogger('arbitrage.engine')

    async def analyze_orderbooks(self, orderbooks: Dict) -> Optional[ArbitrageOpportunity]:
        """Core arbitrage detection logic"""
        best_opportunity = None
        max_profit = Decimal('0')

        for symbol in orderbooks['symbols']:
            for buy_ex in orderbooks['exchanges']:
                for sell_ex in orderbooks['exchanges']:
                    if buy_ex == sell_ex:
                        continue

                    try:
                        buy_price = orderbooks[buy_ex][symbol]['asks'][0][0]
                        sell_price = orderbooks[sell_ex][symbol]['bids'][0][0]
                        
                        # Calculate effective prices with fees
                        effective_buy = buy_price * (1 + self.fee_structures[buy_ex])
                        effective_sell = sell_price * (1 - self.fee_structures[sell_ex])
                        
                        spread = (effective_sell - effective_buy) / effective_buy

                        # Check if this is the best opportunity
                        if spread > max_profit and spread >= self.min_profit:
                            max_profit = spread
                            best_opportunity = ArbitrageOpportunity(
                                buy_exchange=buy_ex,
                                sell_exchange=sell_ex,
                                symbol=symbol,
                                buy_price=buy_price,
                                sell_price=sell_price,
                                volume=self.max_trade_size,
                                fees=self.fee_structures
                            )

                    except KeyError as e:
                        self.logger.warning(f"Missing data for {symbol} on {buy_ex} or {sell_ex}: {str(e)}")
                    except Exception as e:
                        self.logger.error(f"Unexpected error analyzing {symbol}: {str(e)}")

        return best_opportunity