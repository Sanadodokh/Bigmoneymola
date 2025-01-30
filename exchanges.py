# exchanges.py

import requests
import time
import hmac
import hashlib
import json
import config

#########################################
# BINANCE
#########################################

def get_binance_price(symbol):
    """
    Fetch the latest price from Binance's public API.
    symbol: e.g. "BTCUSDT", "ETHUSDT"
    """
    base_url = "https://api.binance.com"
    endpoint = f"/api/v3/ticker/price?symbol={symbol}"

    try:
        response = requests.get(base_url + endpoint, timeout=5)
        response.raise_for_status()
        data = response.json()
        return float(data["price"])
    except Exception as e:
        print(f"[Binance] Error fetching price for {symbol}: {e}")
        return None

def place_binance_order(symbol, side, quantity):
    """
    Example function to place an order on Binance (simplified).
    Real usage requires security (signing the request).
    This is for demo purposes only.
    """
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/order"
    timestamp = int(time.time() * 1000)

    params = {
        "symbol": symbol,
        "side": side.upper(),           # "BUY" or "SELL"
        "type": "MARKET",
        "quantity": quantity,
        "timestamp": timestamp,
    }

    # Create signature
    query_string = "&".join([f"{k}={v}" for k,v in params.items()])
    signature = hmac.new(config.BINANCE_API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()

    headers = {
        "X-MBX-APIKEY": config.BINANCE_API_KEY
    }

    params["signature"] = signature

    try:
        response = requests.post(base_url + endpoint, params=params, headers=headers)
        response.raise_for_status()
        order_data = response.json()
        return order_data
    except Exception as e:
        print(f"[Binance] Error placing order: {e}")
        return None

#########################################
# BYBIT
#########################################

def get_BYBIT_price(symbol):
    """
    Fetch the latest price from BYBIT public API.
    BYBIT uses different naming, e.g. "BTC-USD".
    We'll do a quick conversion from "BTCUSDT" to "BTC-USD".
    """
    base_currency = symbol[:3]  # e.g. "BTC"
    quote_currency = symbol[3:] # e.g. "USDT" -> "USD"
    if quote_currency == "USDT":
        quote_currency = "USD"

    pair = f"{base_currency}-{quote_currency}"  # e.g. "BTC-USD"

    url = f"https://api.BYBIT.com/v2/prices/{pair}/spot"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        # data["data"]["amount"] might be the price
        return float(data["data"]["amount"])
    except Exception as e:
        print(f"[BYBIT] Error fetching price for {symbol}: {e}")
        return None

def place_BYBIT_order(symbol, side, quantity):
    """
    Example function for placing a market order on BYBIT.
    This is heavily simplified. Real usage requires OAuth or API key authentication,
    plus additional parameters like 'funds' or 'size'.
    """
    # BYBIT Pro (Advanced Trade) API docs differ from the standard "BYBIT" docs.
    # We'll skip the full authenticated call complexity for brevity.
    print(f"[BYBIT] Placing {side} order for {quantity} {symbol} (demo only).")
    # In reality, you must sign requests similarly to Binance with your secret key.
    return {"status": "success", "message": "Simulated BYBIT order."}
