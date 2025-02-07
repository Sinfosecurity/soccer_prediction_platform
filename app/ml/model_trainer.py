from typing import Dict, List, Optional
import os
from datetime import datetime
from app.core.logging_config import logger
from app.core.config import get_settings
from .model import SoccerPredictionModel
from .model_registry import ModelRegistry

settings = get_settings()

class ModelTrainer:
    def __init__(self):
        self.model = SoccerPredictionModel()
        self.registry = ModelRegistry()
        
    async def train_and_evaluate(
        self,
        training_data: Dict,
        validation_data: Optional[Dict] = None,
        historical_matches: Optional[List[Dict]] = None
    ) -> Dict:
        """Train model and evaluate performance"""
        try:
            # Train the model
            metrics = self.model.train(
                training_data['X'],
                training_data['y'],
                historical_matches
            )
            
            # Additional validation if provided
            if validation_data:
                val_metrics = self.model.evaluate_model(
                    validation_data['X'],
                    validation_data['y']
                )
                metrics['validation'] = val_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model training and evaluation: {str(e)}")
            raise

    async def automated_training_pipeline(
        self,
        training_data: Dict,
        validation_data: Optional[Dict] = None,
        historical_matches: Optional[List[Dict]] = None
    ) -> Dict:
        """Run automated model training pipeline"""
        try:
            # Train and evaluate model
            metrics = await self.train_and_evaluate(
                training_data,
                validation_data,
                historical_matches
            )
            
            # Check if we should deploy new model
            if self.registry.should_retrain(metrics):
                # Save new model
                version = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = os.path.join(self.registry.base_path, f"model_{version}.joblib")
                self.model.save_model(model_path)
                
                # Register new model
                version = self.registry.register_model(model_path, metrics, version)
                
                # Activate if it's the first model or if it performs better
                if not self.registry.get_active_model():
                    self.registry.activate_model(version)
                    logger.info(f"Activated new model version: {version}")
                
                return {
                    'status': 'success',
                    'message': 'New model trained and registered',
                    'version': version,
                    'metrics': metrics
                }
            
            return {
                'status': 'success',
                'message': 'Current model performing adequately',
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error in automated training pipeline: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def load_best_model(self) -> None:
        """Load the currently active model"""
        try:
            active_model = self.registry.get_active_model()
            if active_model:
                self.model.load_model(active_model['path'])
                logger.info(f"Loaded active model version: {active_model['version']}")
            else:
                logger.warning("No active model found")
                
        except Exception as e:
            logger.error(f"Error loading best model: {str(e)}")
            raise 