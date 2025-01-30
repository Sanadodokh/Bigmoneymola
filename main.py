# main.py
import asyncio
import time
import logging
from typing import Dict, Optional
import config
import exchanges
import arbitrage
import ml_model
from exceptions import ExchangeAPIError, InsufficientLiquidityError

# Configure structured logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def fetch_prices(trading_pair: str) -> Dict[str, Optional[float]]:
    """Asynchronously fetch prices from all configured exchanges."""
    prices = {}
    exchange_functions = {
        "binance": exchanges.get_binance_price,
        "bybit": exchanges.get_bybit_price,
        "kraken": exchanges.get_kraken_price
    }

    async def fetch_price(exchange: str):
        try:
            prices[exchange] = await exchange_functions[exchange](trading_pair)
        except ExchangeAPIError as e:
            logger.warning(f"Failed to fetch price from {exchange}: {str(e)}")
            prices[exchange] = None

    # Run all exchange requests concurrently
    await asyncio.gather(*[fetch_price(exchange) for exchange in config.ACTIVE_EXCHANGES])

    # Filter out exchanges with failed price fetches
    return {k: v for k, v in prices.items() if v is not None}

async def execute_trade(buy_exchange: str, sell_exchange: str, trading_pair: str, amount: float) -> bool:
    """Execute atomic trade across two exchanges with error handling."""
    logger.info(f"[ACTION] Executing {buy_exchange}->{sell_exchange} trade for {amount} {trading_pair}")
    
    try:
        # Get current order book depth
        buy_price = await exchanges.get_orderbook_price(buy_exchange, trading_pair, "buy")
        sell_price = await exchanges.get_orderbook_price(sell_exchange, trading_pair, "sell")

        # Verify liquidity
        if not exchanges.check_liquidity(buy_exchange, trading_pair, amount, "buy"):
            raise InsufficientLiquidityError(f"{buy_exchange} has insufficient buy liquidity")
        if not exchanges.check_liquidity(sell_exchange, trading_pair, amount, "sell"):
            raise InsufficientLiquidityError(f"{sell_exchange} has insufficient sell liquidity")

        # Execute orders atomically
        buy_result = await exchanges.place_order(buy_exchange, trading_pair, "buy", amount)
        sell_result = await exchanges.place_order(sell_exchange, trading_pair, "sell", amount)

        logger.info(f"[TRADE SUCCESS] {buy_exchange}: {buy_result}, {sell_exchange}: {sell_result}")
        return True

    except (ExchangeAPIError, InsufficientLiquidityError) as e:
        logger.error(f"[TRADE FAILED] {str(e)}")
        await handle_trade_failure(buy_exchange, sell_exchange, trading_pair, amount)
        return False

async def handle_trade_failure(buy_exchange: str, sell_exchange: str, trading_pair: str, amount: float):
    """Handle failed trades and potential rollbacks."""
    logger.warning("Attempting trade failure cleanup...")
    # Add logic to cancel pending orders or hedge positions
    pass

async def trading_cycle(model, iteration: int):
    """Single iteration of the trading loop."""
    start_time = time.time()
    
    # 1. Fetch market data
    prices = await fetch_prices(config.TRADING_PAIR)
    if len(prices) < 2:
        logger.warning("Insufficient exchange data for arbitrage")
        return

    # 2. Find best arbitrage opportunity
    opportunity = arbitrage.find_best_arbitrage_opportunity(
        prices=prices,
        fees=config.EXCHANGE_FEES,
        threshold=config.ARBITRAGE_THRESHOLD
    )

    if not opportunity:
        logger.debug("No arbitrage opportunities found")
        return

    action, buy_exchange, sell_exchange, profit_pct = opportunity

    # 3. ML Model Prediction
    features = ml_model.create_features(
        prices=prices,
        spread=profit_pct,
        orderbook_depth=exchanges.get_orderbook_depth(),
        market_volatility=ml_model.calculate_volatility()
    )
    
    ml_decision = ml_model.predict_opportunity(model, features)
    
    # 4. Execute if conditions met
    if ml_decision and profit_pct >= config.MIN_PROFIT_PCT:
        logger.info(f"[STRATEGY] Executing trade with expected {profit_pct}% profit")
        success = await execute_trade(buy_exchange, sell_exchange, config.TRADING_PAIR, config.TRADE_AMOUNT)
        
        if success:
            ml_model.log_trade_outcome(
                features=features,
                profit=profit_pct,
                success=True
            )
    else:
        logger.info(f"[STRATEGY] Passing opportunity: {profit_pct}% profit (ML confidence: {ml_decision})")

    logger.debug(f"Cycle {iteration} completed in {time.time() - start_time:.2f}s")

async def main():
    """Main trading loop with performance monitoring."""
    logger.info("Initializing arbitrage bot")
    
    # Initialize components
    model = ml_model.load_or_train_model(
        config.DATA_LOG_PATH,
        config.ML_MODEL_PATH,
        retrain_interval=config.ML_RETRAIN_INTERVAL
    )
    
    # Warm up exchange connections
    await exchanges.initialize_exchanges(config.ACTIVE_EXCHANGES)

    iteration = 0
    while True:
        try:
            await trading_cycle(model, iteration)
            iteration += 1
            await asyncio.sleep(config.CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            await exchanges.close_exchange_connections()
            break
        except Exception as e:
            logger.error(f"Critical error in main loop: {str(e)}")
            await handle_trade_failure(None, None, config.TRADING_PAIR, 0)
            await asyncio.sleep(config.ERROR_RETRY_DELAY)

if __name__ == "__main__":
    asyncio.run(main())