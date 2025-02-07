import os
import json
from datetime import datetime
from typing import Dict, Optional
from app.core.logging_config import logger
from app.core.config import get_settings

settings = get_settings()

class ModelRegistry:
    def __init__(self, base_path: str = "models"):
        self.base_path = base_path
        self.metadata_file = os.path.join(base_path, "model_metadata.json")
        self._ensure_directory()
        self.metadata = self._load_metadata()

    def _ensure_directory(self):
        """Ensure the model directory exists"""
        os.makedirs(self.base_path, exist_ok=True)

    def _load_metadata(self) -> Dict:
        """Load model metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'models': {}}

    def _save_metadata(self):
        """Save model metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def register_model(self, model_path: str, metrics: Dict, version: Optional[str] = None) -> str:
        """Register a new model version with its metrics"""
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_info = {
            'path': model_path,
            'metrics': metrics,
            'created_at': datetime.now().isoformat(),
            'is_active': False
        }

        self.metadata['models'][version] = model_info
        self._save_metadata()
        return version

    def activate_model(self, version: str):
        """Set a model version as active"""
        if version not in self.metadata['models']:
            raise ValueError(f"Model version {version} not found")

        # Deactivate current active model
        for v in self.metadata['models']:
            self.metadata['models'][v]['is_active'] = False

        self.metadata['models'][version]['is_active'] = True
        self._save_metadata()

    def get_active_model(self) -> Optional[Dict]:
        """Get the currently active model information"""
        for version, info in self.metadata['models'].items():
            if info['is_active']:
                return {'version': version, **info}
        return None

    def get_model_path(self, version: str) -> str:
        """Get the file path for a specific model version"""
        if version not in self.metadata['models']:
            raise ValueError(f"Model version {version} not found")
        return self.metadata['models'][version]['path']

    def should_retrain(self, current_metrics: Dict, threshold: float = 0.02) -> bool:
        """Determine if model should be retrained based on performance degradation"""
        active_model = self.get_active_model()
        if not active_model:
            return True

        active_metrics = active_model['metrics']
        
        # Compare key metrics
        for metric in ['accuracy', 'f1']:
            if active_metrics[metric] - current_metrics[metric] > threshold:
                logger.info(f"Performance degradation detected in {metric}")
                return True

        return False 