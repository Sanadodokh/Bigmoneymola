# exceptions.py
class ExchangeAPIError(Exception): pass
class InsufficientLiquidityError(Exception): pass
class OrderExecutionError(Exception): pass
class ExchangeAPIError(Exception):
    """Base exception for exchange API errors"""
    
class InsufficientLiquidityError(Exception):
    """Raised when order book lacks required liquidity"""
    
class OrderExecutionError(Exception):
    """Raised when trade execution fails"""
    
class ModelPredictionError(Exception):
    """Raised when ML model prediction fails"""
