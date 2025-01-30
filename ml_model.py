# ml_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report
import joblib
import logging
from datetime import datetime
from typing import Dict, Any
import config

logging.basicConfig(level=config.config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class ArbitrageModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])
        self.feature_columns = [
            'spread_pct', 
            'volatility_5m',
            'orderbook_imbalance',
            'trade_amount',
            'hour_of_day',
            'liquidity_ratio'
        ]

    def create_features(self, prices: Dict[str, float], orderbook_depth: Dict[str, Any]) -> Dict[str, float]:
        """Create ML features from market data"""
        try:
            # Calculate spread metrics
            sorted_prices = sorted(prices.values())
            spread_pct = (sorted_prices[-1] - sorted_prices[0]) / sorted_prices[0]
            
            # Calculate volatility (standard deviation of prices)
            volatility_5m = np.std(list(prices.values()))
            
            # Order book metrics
            bids = sum([d['bids'] for d in orderbook_depth.values()])
            asks = sum([d['asks'] for d in orderbook_depth.values()])
            orderbook_imbalance = (bids - asks) / (bids + asks)
            
            # Temporal feature
            hour_of_day = datetime.now().hour
            
            # Liquidity metrics
            liquidity_ratios = [d['liquidity'] for d in orderbook_depth.values()]
            liquidity_ratio = min(liquidity_ratios) / max(liquidity_ratios)
            
            return {
                'spread_pct': spread_pct,
                'volatility_5m': volatility_5m,
                'orderbook_imbalance': orderbook_imbalance,
                'trade_amount': config.config.TRADE_AMOUNT,
                'hour_of_day': hour_of_day,
                'liquidity_ratio': liquidity_ratio
            }
            
        except Exception as e:
            logger.error(f"Feature creation failed: {str(e)}")
            raise ModelPredictionError("Feature engineering error")

    def load_or_train_model(self, data_path: str, model_path: str) -> Pipeline:
        """Manage model lifecycle with hyperparameter tuning"""
        try:
            if os.path.exists(model_path):
                logger.info("Loading existing model")
                return joblib.load(model_path)
                
            logger.info("Training new model")
            data = pd.read_csv(data_path, parse_dates=['timestamp'])
            
            # Feature engineering
            data['hour_of_day'] = data['timestamp'].dt.hour
            data = data.dropna()
            
            # Train-test split with time series
            split_idx = int(len(data) * 0.8)
            X_train = data[self.feature_columns].iloc[:split_idx]
            y_train = data['label'].iloc[:split_idx]
            X_test = data[self.feature_columns].iloc[split_idx:]
            y_test = data['label'].iloc[split_idx:]
            
            # Hyperparameter tuning
            param_dist = {
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            }
            
            self.pipeline = RandomizedSearchCV(
                self.pipeline,
                param_dist,
                n_iter=5,
                cv=TimeSeriesSplit(n_splits=3).split(X_train),
                scoring='f1'
            )
            
            self.pipeline.fit(X_train, y_train)
            logger.info(f"Best params: {self.pipeline.best_params_}")
            
            # Evaluate
            y_pred = self.pipeline.predict(X_test)
            logger.info(classification_report(y_test, y_pred))
            
            joblib.dump(self.pipeline, model_path)
            return self.pipeline
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise ModelPredictionError("Model training error")

    def predict_opportunity(self, features: Dict[str, float]) -> float:
        """Make prediction with confidence score"""
        try:
            feature_array = np.array([[features[col] for col in self.feature_columns]])
            return self.pipeline.predict_proba(feature_array)[0][1]  # Probability of "ACT"
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise ModelPredictionError("Prediction error")

def log_trade_outcome(features: Dict[str, float], profit: float, success: bool):
    """Log trade outcomes for model retraining"""
    try:
        new_row = {
            **features,
            'label': success and profit > 0,
            'timestamp': datetime.now(),
            'profit': profit
        }
        
        pd.DataFrame([new_row]).to_csv(
            config.config.DATA_LOG_PATH,
            mode='a',
            header=not os.path.exists(config.config.DATA_LOG_PATH),
            index=False
        )
    except Exception as e:
        logger.error(f"Failed to log trade outcome: {str(e)}")