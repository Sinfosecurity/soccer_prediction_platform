from typing import Dict, List, Optional
import random
from datetime import datetime
import numpy as np
from app.core.logging_config import logger
from .model_registry import ModelRegistry

class ABTestingManager:
    def __init__(self, registry: ModelRegistry, traffic_split: float = 0.1):
        self.registry = registry
        self.traffic_split = traffic_split
        self.experiment_metrics = {}
        
    def should_use_candidate(self) -> bool:
        """Determine if candidate model should be used based on traffic split"""
        return random.random() < self.traffic_split
        
    def get_model_for_prediction(self) -> Optional[Dict]:
        """Get model to use for prediction based on A/B test rules"""
        active_model = self.registry.get_active_model()
        if not active_model:
            return None
            
        # Get latest candidate model
        candidate = self.registry.get_latest_candidate()
        if not candidate or not self.should_use_candidate():
            return active_model
            
        return candidate
        
    def record_prediction_outcome(
        self,
        model_version: str,
        prediction: int,
        confidence: float,
        actual_outcome: Optional[int] = None
    ):
        """Record prediction outcome for A/B testing analysis"""
        if model_version not in self.experiment_metrics:
            self.experiment_metrics[model_version] = {
                'predictions': [],
                'confidences': [],
                'actuals': [],
                'timestamps': []
            }
            
        metrics = self.experiment_metrics[model_version]
        metrics['predictions'].append(prediction)
        metrics['confidences'].append(confidence)
        metrics['actuals'].append(actual_outcome)
        metrics['timestamps'].append(datetime.now().isoformat())
        
    def analyze_experiment(self, min_samples: int = 100) -> Dict:
        """Analyze A/B test results"""
        results = {}
        active_model = self.registry.get_active_model()
        candidate = self.registry.get_latest_candidate()
        
        if not active_model or not candidate:
            return results
            
        for model_version, metrics in self.experiment_metrics.items():
            if len(metrics['predictions']) < min_samples:
                continue
                
            # Calculate metrics
            actuals = np.array([a for a in metrics['actuals'] if a is not None])
            predictions = np.array(metrics['predictions'])[:len(actuals)]
            confidences = np.array(metrics['confidences'])
            
            if len(actuals) > 0:
                accuracy = np.mean(predictions == actuals)
                avg_confidence = np.mean(confidences)
                calibration_error = abs(accuracy - avg_confidence)
                
                results[model_version] = {
                    'accuracy': float(accuracy),
                    'avg_confidence': float(avg_confidence),
                    'calibration_error': float(calibration_error),
                    'sample_size': len(actuals)
                }
                
        return results 