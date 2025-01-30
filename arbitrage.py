# arbitrage.py
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("arbitrage.log"), logging.StreamHandler()]
)

def calculate_effective_price(price: float, fee_rate: float) -> float:
    """
    Calculate the effective price after accounting for fees.
    
    Args:
        price: The original price (must be > 0)
        fee_rate: The trading fee rate (e.g., 0.001 for 0.1%, must be >= 0)
    
    Returns:
        Effective price including fees
    
    Raises:
        ValueError: If inputs are invalid
    """
    if price <= 0:
        raise ValueError(f"Invalid price: {price}. Must be positive.")
    if fee_rate < 0:
        raise ValueError(f"Invalid fee rate: {fee_rate}. Cannot be negative.")
    
    return price * (1 + fee_rate)

def find_best_arbitrage_opportunity(
    prices: dict,
    fees: dict,
    threshold: float = 0.002
) -> Optional[Tuple[str, str, str, float]]:
    """
    Find the most profitable arbitrage opportunity across multiple exchanges.
    
    Args:
        prices: Dictionary of current prices {exchange: price}
        fees: Dictionary of fee rates {exchange: fee_rate}
        threshold: Minimum profitable difference (default: 0.2%)
    
    Returns:
        Tuple: (action, buy_exchange, sell_exchange, profit_percentage)
        or None if no opportunity exists
    """
    max_profit = 0.0
    best_opportunity = None
    exchanges = list(prices.keys())
    
    # Compare all unique exchange pairs
    for i, buy_exchange in enumerate(exchanges):
        for sell_exchange in exchanges[i+1:]:
            try:
                # Calculate effective prices
                buy_price = calculate_effective_price(prices[buy_exchange], fees[buy_exchange])
                sell_price = calculate_effective_price(prices[sell_exchange], fees[sell_exchange])
                
                # Calculate potential profit
                if sell_price > buy_price:
                    profit = (sell_price - buy_price) / buy_price
                    if profit > max_profit and profit >= threshold:
                        max_profit = profit
                        best_opportunity = (
                            f"BUY_{buy_exchange.upper()}_SELL_{sell_exchange.upper()}",
                            buy_exchange,
                            sell_exchange,
                            round(profit * 100, 4)  # Percentage with 4 decimal places
                        )
                
            except ValueError as e:
                logging.error(f"Skipping {buy_exchange}-{sell_exchange} pair: {str(e)}")
                continue
    
    if best_opportunity:
        logging.info(f"Arbitrage opportunity found: {best_opportunity[0]} with {best_opportunity[3]}% profit")
        return best_opportunity
    
    logging.debug("No arbitrage opportunities meeting threshold")
    return None

# Example usage:
if __name__ == "__main__":
    # Sample data
    exchange_prices = {
        "binance": 28750.50,
        "bybit": 28785.30,
        "kraken": 28720.10,
        "okx": 28795.75
    }
    
    exchange_fees = {
        "binance": 0.001,
        "bybit": 0.0005,
        "kraken": 0.002,
        "okx": 0.00075
    }
    
    opportunity = find_best_arbitrage_opportunity(exchange_prices, exchange_fees, 0.0015)
    print(opportunity)