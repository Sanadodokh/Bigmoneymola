# New file: fastapi_server.py
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
async def predict(features: dict):
    model = joblib.load("xgboost_arbitrage.pkl")
    return {"profit": model.predict([features])[0]}
