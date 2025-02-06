import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import joblib
import logging
from datetime import datetime
from typing import Dict, Any
import os
from config import config
from exceptions import ModelPredictionError

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class ArbitrageModel:
    def __init__(self):
        self.pipeline = self._create_pipeline()
        self.feature_columns = [
            'spread_pct', 'volatility_5m', 'orderbook_imbalance',
            'trade_amount', 'hour_of_day', 'liquidity_ratio',
            'spread_zscore', 'volume_ratio', 'market_trend'
        ]
        self.model_version = datetime.now().strftime("%Y%m%d%H%M")

    def _create_pipeline(self):
        return Pipeline([
            ('scaler', RobustScaler()),
            ('sampler', SMOTE(sampling_strategy='minority')),
            ('model', HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            ))
        ])

    def create_features(self, prices: Dict[str, float], orderbook: Dict) -> Dict[str, float]:
        """Create enhanced market features with error handling"""
        try:
            # Price-based features
            price_values = list(prices.values())
            spread_pct = (max(price_values) - min(price_values)) / min(price_values)
            volatility_5m = np.std(price_values)
            
            # Order book features
            bid_volumes = sum([level['bid_volume'] for level in orderbook.values()])
            ask_volumes = sum([level['ask_volume'] for level in orderbook.values()])
            orderbook_imbalance = (bid_volumes - ask_volumes) / (bid_volumes + ask_volumes + 1e-8)
            
            # Temporal and market context
            hour = datetime.now().hour
            market_trend = np.polyfit(range(len(price_values)), price_values, 1)[0]
            
            return {
                'spread_pct': spread_pct,
                'volatility_5m': volatility_5m,
                'orderbook_imbalance': orderbook_imbalance,
                'trade_amount': config.TRADE_AMOUNT,
                'hour_of_day': hour,
                'liquidity_ratio': orderbook['liquidity_ratio'],
                'spread_zscore': (spread_pct - np.mean(price_values)) / (np.std(price_values) + 1e-8),
                'volume_ratio': bid_volumes / (ask_volumes + 1e-8),
                'market_trend': market_trend
            }
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}", exc_info=True)
            raise ModelPredictionError("Feature creation error")

    def load_or_train_model(self, data_path: str, model_dir: str) -> Pipeline:
        """Enhanced model training with version control"""
        try:
            os.makedirs(model_dir, exist_ok=True)
            latest_model = self._find_latest_model(model_dir)
            
            if latest_model:
                logger.info(f"Loading model: {latest_model}")
                self.pipeline = joblib.load(latest_model)
                return self.pipeline
                
            logger.info("Training new model...")
            data = self._prepare_data(data_path)
            X, y = data[self.feature_columns], data['label']
            
            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            self.pipeline = RandomizedSearchCV(
                self.pipeline,
                self._get_hyperparams(),
                n_iter=20,
                cv=tscv,
                scoring='f1',
                n_jobs=-1
            )
            
            self.pipeline.fit(X, y)
            self._evaluate_model(X, y)
            self._save_model(model_dir)
            return self.pipeline

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            raise ModelPredictionError("Model training error")

    def _prepare_data(self, data_path):
        """Data preprocessing pipeline"""
        data = pd.read_csv(data_path, parse_dates=['timestamp'])
        data = data.dropna().sort_values('timestamp')
        
        # Feature engineering
        data['hour_of_day'] = data['timestamp'].dt.hour
        data['market_trend'] = data.groupby('timestamp')['spread_pct'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        return data

    def _get_hyperparams(self):
        return {
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5, 7],
            'model__max_iter': [200, 300],
            'model__l2_regularization': [0, 0.1, 0.5]
        }

    def _evaluate_model(self, X, y):
        """Comprehensive model evaluation"""
        cv_scores = cross_val_score(
            self.pipeline.best_estimator_,
            X,
            y,
            cv=TimeSeriesSplit(3),
            scoring='f1'
        )
        logger.info(f"CV F1 Scores: {cv_scores}")
        logger.info(f"Best Model Params: {self.pipeline.best_params_}")

    def _save_model(self, model_dir):
        """Save model with versioning"""
        model_path = os.path.join(model_dir, f"arbitrage_model_{self.model_version}.pkl")
        joblib.dump(self.pipeline.best_estimator_, model_path)
        logger.info(f"Model saved: {model_path}")

    def _find_latest_model(self, model_dir):
        """Find most recent model version"""
        models = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        return max([os.path.join(model_dir, f) for f in models], key=os.path.getctime) if models else None

    def predict_opportunity(self, features: Dict) -> float:
        """Make prediction with confidence and error handling"""
        try:
            feature_df = pd.DataFrame([features])[self.feature_columns]
            return self.pipeline.predict_proba(feature_df)[0][1]
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise ModelPredictionError("Prediction error")

def log_trade_outcome(features: Dict, profit: float, success: bool):
    """Robust data logging with error handling"""
    try:
        log_entry = features.copy()
        log_entry.update({
            'timestamp': datetime.now().isoformat(),
            'profit': profit,
            'success': success,
            'label': success and profit > config.MIN_PROFIT_THRESHOLD
        })
        
        log_df = pd.DataFrame([log_entry])
        
        # Append to CSV with locking
        with open(config.DATA_LOG_PATH, 'a') as f:
            lock(f, LOCK_EX)
            header = not os.path.exists(config.DATA_LOG_PATH) or os.stat(config.DATA_LOG_PATH).st_size == 0
            log_df.to_csv(f, header=header, index=False)
            unlock(f)
            
    except Exception as e:
        logger.error(f"Failed to log trade: {str(e)}", exc_info=True)