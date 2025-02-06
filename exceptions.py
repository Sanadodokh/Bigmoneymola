# exceptions.py
class ArbitrageError(Exception):
    """Base class for all arbitrage-related exceptions"""
    code: int = 500

class ExchangeAPIError(ArbitrageError):
    """Raised when exchange API communication fails"""
    code = 503
    def __init__(self, exchange: str, endpoint: str):
        super().__init__(f"API Error on {exchange} ({endpoint})")

class RateLimitError(ExchangeAPIError):
    """Raised when exchange rate limits are hit"""
    code = 429
    def __init__(self, exchange: str, reset_time: float):
        super().__init__(exchange, "Rate Limited")
        self.reset_time = reset_time

class InsufficientLiquidityError(ArbitrageError):
    """Raised when order book lacks required liquidity"""
    code = 406
    def __init__(self, symbol: str, required: float, available: float):
        super().__init__(f"Insufficient liquidity for {symbol}: Need {required}, Available {available}")

class OrderExecutionError(ArbitrageError):
    """Raised when trade execution fails"""
    code = 400
    def __init__(self, order_id: str, reason: str):
        super().__init__(f"Order {order_id} failed: {reason}")

class ModelPredictionError(ArbitrageError):
    """Raised when ML model prediction fails"""
    code = 500
    def __init__(self, model_name: str):
        super().__init__(f"Prediction failed for {model_name}")

class ConfigurationError(ArbitrageError):
    """Raised for invalid configuration"""
    code = 500