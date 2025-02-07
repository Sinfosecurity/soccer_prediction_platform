from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict
from datetime import datetime
import numpy as np
from app.models.database import get_db
from app.models.match import Match
from app.ml.model import SoccerPredictionModel
from app.core.logging_config import logger

router = APIRouter()
model = SoccerPredictionModel()

@router.get("/test")
def test_endpoint():
    return {"message": "API is working"}

@router.get("/matches", response_model=List[Dict])
def get_matches(db: Session = Depends(get_db)):
    """Get all upcoming matches"""
    try:
        matches = db.query(Match).filter(
            Match.start_time > datetime.utcnow()
        ).all()
        return [match.to_dict() for match in matches]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/matches/{match_id}")
def get_match(match_id: str, db: Session = Depends(get_db)):
    """Get specific match details"""
    match = db.query(Match).filter(Match.match_id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match.to_dict()

@router.post("/train-model")
async def train_model(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Train model using historical data"""
    try:
        # Get historical matches with known outcomes
        historical_matches = db.query(Match).filter(
            Match.start_time < datetime.utcnow(),
            Match.prediction is not None
        ).all()
        
        if len(historical_matches) < 100:  # Minimum matches required for training
            raise HTTPException(
                status_code=400,
                detail="Insufficient historical data for training"
            )
            
        # Prepare training data
        X = np.array([match.odds_data for match in historical_matches])
        y = np.array([match.prediction for match in historical_matches])
        
        # Train model in background
        background_tasks.add_task(
            model.train,
            X=X,
            y=y,
            historical_matches=[m.to_dict() for m in historical_matches]
        )
        
        return {"message": "Model training initiated"}
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-prediction")
async def batch_prediction(db: Session = Depends(get_db)):
    """Make predictions for all upcoming matches"""
    try:
        # Get upcoming matches without predictions
        upcoming_matches = db.query(Match).filter(
            Match.start_time > datetime.utcnow(),
            Match.prediction.is_(None)
        ).all()
        
        if not upcoming_matches:
            return {"message": "No new matches to predict"}
            
        # Get historical matches for feature engineering
        historical_matches = db.query(Match).filter(
            Match.start_time < datetime.utcnow()
        ).all()
        
        results = []
        for match in upcoming_matches:
            try:
                prediction, confidence = model.predict(
                    match.odds_data,
                    [m.to_dict() for m in historical_matches]
                )
                
                # Update match with prediction
                match.prediction = prediction
                match.confidence = confidence
                results.append({
                    'match_id': match.match_id,
                    'prediction': prediction,
                    'confidence': confidence
                })
                
            except Exception as e:
                logger.error(f"Error predicting match {match.match_id}: {str(e)}")
                continue
                
        db.commit()
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-performance-metrics")
async def model_performance_metrics(db: Session = Depends(get_db)):
    """Get model performance metrics"""
    try:
        # Get historical predictions with known outcomes
        historical_predictions = db.query(Match).filter(
            Match.start_time < datetime.utcnow(),
            Match.prediction.isnot(None)
        ).all()
        
        if not historical_predictions:
            return {"message": "No historical predictions available"}
            
        # Calculate metrics
        predictions = np.array([m.prediction for m in historical_predictions])
        actuals = np.array([m.actual_result for m in historical_predictions])
        confidences = np.array([m.confidence for m in historical_predictions])
        
        metrics = {
            'accuracy': float(np.mean(predictions == actuals)),
            'avg_confidence': float(np.mean(confidences)),
            'total_predictions': len(predictions),
            'prediction_distribution': {
                'home_win': int(np.sum(predictions == 0)),
                'draw': int(np.sum(predictions == 1)),
                'away_win': int(np.sum(predictions == 2))
            },
            'last_updated': datetime.utcnow().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical-predictions")
async def historical_predictions(
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get historical predictions and their outcomes"""
    try:
        predictions = db.query(Match).filter(
            Match.prediction.isnot(None)
        ).order_by(
            Match.start_time.desc()
        ).limit(limit).all()
        
        return [match.to_dict() for match in predictions]
        
    except Exception as e:
        logger.error(f"Error getting historical predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 