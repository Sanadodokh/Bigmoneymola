# bot.py
import asyncio
import logging
from decimal import Decimal
from typing import Dict, List
from exchange_connector import SecureExchangeConnector
from arbitrage_engine import ArbitrageEngine, ArbitrageOpportunity
from risk_manager import RiskManager
from market_simulator import MarketSimulator
from config import config

logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('arbitrage.log'),
        logging.StreamHandler()
    ]
)

class ArbitrageBot:
    def __init__(self, exchange_name: str):
        self.logger = logging.getLogger('ArbitrageBot')
        self.exchange_name = exchange_name
        self.connector = SecureExchangeConnector(exchange_name)
        self.engine = ArbitrageEngine(config)
        self.risk_manager = RiskManager(config)
        self.simulator = MarketSimulator(config.INITIAL_BALANCE)
        self.active_symbols: List[str] = []

    async def run(self):
        """Main execution loop"""
        self.logger.info("Initializing arbitrage bot")
        
        try:
            self.active_symbols = await self._get_valid_symbols()
            await self._main_loop()
        except Exception as e:
            self.logger.error(f"Fatal error: {str(e)}", exc_info=True)
        finally:
            await self._shutdown()

    async def _main_loop(self):
        """Core trading loop"""
        while True:
            try:
                order_books = await self._fetch_market_data()
                opportunity = await self.engine.analyze_orderbooks(order_books)
                
                if opportunity:
                    await self._process_opportunity(opportunity)
                    
                await asyncio.sleep(config.COOLDOWN_PERIOD)
                    
            except Exception as e:
                self.logger.error(f"Cycle error: {str(e)}", exc_info=True)
                await asyncio.sleep(10)  # Error cooldown

    async def _fetch_market_data(self) -> Dict:
        """Fetch order books for all symbols"""
        tasks = [
            self.connector.fetch_order_book(symbol)
            for symbol in self.active_symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            symbol: result for symbol, result in zip(self.active_symbols, results)
            if not isinstance(result, Exception)
        }

    async def _process_opportunity(self, opportunity: ArbitrageOpportunity):
        """Handle detected arbitrage opportunity"""
        try:
            if not self.risk_manager.approve_trade(opportunity):
                self.logger.debug("Risk manager blocked trade")
                return

            if config.SIMULATION_MODE:
                profit = await self.simulator.execute_order(opportunity)
                self.logger.info(f"Simulated trade: {profit:.4f}% profit")
            else:
                profit = await self._execute_real_trade(opportunity)
                self.logger.info(f"Live trade executed: {profit:.4f}% profit")

            self.engine.update_model(opportunity, profit)
            
        except Exception as e:
            self.logger.error(f"Trade processing failed: {str(e)}", exc_info=True)

    async def _execute_real_trade(self, opportunity: ArbitrageOpportunity) -> Decimal: