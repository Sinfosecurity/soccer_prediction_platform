from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
from typing import Tuple, Dict, List, Optional
from datetime import datetime
from app.core.logging_config import logger
from app.core.config import get_settings
from .preprocessing import FeaturePreprocessor
from .feature_engineering import FeatureEngineer

settings = get_settings()

class SoccerPredictionModel:
    def __init__(self):
        self.model = None
        self.preprocessor = FeaturePreprocessor()
        
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        historical_matches: Optional[List[Dict]] = None
    ) -> Dict:
        """Train the model and return performance metrics"""
        try:
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Initialize and train base model
            base_model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
            
            # Calibrate probabilities
            self.model = CalibratedClassifierCV(
                base_model,
                cv=5,
                method='sigmoid'
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Calculate metrics
            train_score = self.model.score(X_train, y_train)
            val_score = self.model.score(X_val, y_val)
            
            metrics = {
                'accuracy': float(val_score),
                'train_accuracy': float(train_score),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Model trained successfully. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict(
        self,
        odds_data: Dict,
        historical_matches: Optional[List[Dict]] = None
    ) -> Tuple[int, float]:
        """Make prediction for a match"""
        try:
            if not self.model:
                raise ValueError("Model not trained or loaded")
                
            # Preprocess features
            X = self.preprocessor.preprocess_match_data(
                odds_data,
                historical_matches
            )
            
            # Get prediction and probability
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            confidence = float(probabilities.max())
            
            return int(prediction), confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def save_model(self, path: str):
        """Save model to disk"""
        try:
            joblib.dump({
                'model': self.model,
                'preprocessor': self.preprocessor
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str):
        """Load model from disk"""
        try:
            loaded = joblib.load(path)
            self.model = loaded['model']
            self.preprocessor = loaded['preprocessor']
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 