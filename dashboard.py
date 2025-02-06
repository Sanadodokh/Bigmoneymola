# dashboard.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from config import config

app = FastAPI(title="Arbitrage Dashboard")
security = APIKeyHeader(name="X-API-Key")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthCheck(BaseModel):
    status: str
    exchange: str
    symbols: int
    uptime: float

class TradeMetrics(BaseModel):
    total_trades: int
    successful_trades: int
    total_profit: