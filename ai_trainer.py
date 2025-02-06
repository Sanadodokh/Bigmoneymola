# ai_trainer.py
import numpy as np
from ml_model import RiskPredictor
from data.arbitrage_data import load_training_data

class AITrainer:
    def __init__(self):
        self.model = RiskPredictor()
        self.data = load_training_data()
        
    def preprocess_data(self):
        """Clean and normalize training data"""
        # Remove outliers
        self.data = self.data[(self.data['profit'] > -0.1) & (self.data['profit'] < 0.5)]
        
        # Normalize features
        self.features = (self.data[['spread', 'liquidity', 'volatility']] 
                         - self.data.mean()) / self.data.std()
        
    def train_model(self, epochs=100):
        """Train risk prediction model"""
        X = self.features.values
        y = self.data['success'].values
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model.fit(X, y, epochs=epochs, validation_split=0.2)
        
    def save_model(self, path='models/arbitrage_model.pkl'):
        self.model.save(path)