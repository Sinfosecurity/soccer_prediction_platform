from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.core.logging_config import logger
from app.core.config import get_settings
from app.models.database import get_db
from sqlalchemy.orm import Session
from app.ml.model_registry import ModelRegistry
from app.ml.ab_testing import ABTestingManager
from app.models.match import Match

settings = get_settings()
router = APIRouter()

class MetricsSummary(BaseModel):
    version: str
    recent_accuracy: Optional[float] = Field(None, ge=0, le=1)
    avg_confidence: float = Field(..., ge=0, le=1)
    total_predictions: int = Field(..., ge=0)
    last_updated: datetime
    calibration_error: Optional[float] = None
    prediction_distribution: Optional[Dict[str, int]] = None
    high_confidence_accuracy: Optional[float] = None

class ModelMonitoringDashboard:
    def __init__(self, registry: ModelRegistry, ab_testing: ABTestingManager):
        self.registry = registry
        self.ab_testing = ab_testing
        self.metrics_cache: Dict[str, Any] = {}
        self.cache_timeout = timedelta(minutes=5)
        self.last_cache_update: Optional[datetime] = None

    def _should_update_cache(self) -> bool:
        """Check if cache needs updating"""
        if not self.last_cache_update:
            return True
        return datetime.now() - self.last_cache_update > self.cache_timeout

    def _prepare_metrics_dataframe(self, metrics_history: Dict) -> pd.DataFrame:
        """Convert metrics history to DataFrame with proper formatting"""
        try:
            if not metrics_history:
                return pd.DataFrame()

            data = []
            for version, metrics in metrics_history.items():
                for timestamp, values in metrics.items():
                    row = {
                        'version': version,
                        'timestamp': pd.to_datetime(timestamp),
                        **values
                    }
                    data.append(row)

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            return df.sort_values('timestamp')

        except Exception as e:
            logger.error(f"Error preparing metrics DataFrame: {str(e)}")
            return pd.DataFrame()

    def _create_accuracy_confidence_plot(self, df: pd.DataFrame, fig: go.Figure, row: int, col: int):
        """Create accuracy and confidence over time plot"""
        try:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['accuracy'],
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='blue')
                ),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['avg_confidence'],
                    mode='lines+markers',
                    name='Avg Confidence',
                    line=dict(color='green')
                ),
                row=row, col=col
            )
        except Exception as e:
            logger.error(f"Error creating accuracy confidence plot: {str(e)}")

    def _create_confidence_distribution(self, df: pd.DataFrame, fig: go.Figure, row: int, col: int):
        """Create confidence distribution histogram"""
        try:
            fig.add_trace(
                go.Histogram(
                    x=df['confidence'],
                    name='Confidence Dist',
                    nbinsx=20,
                    histnorm='probability',
                    marker_color='rgb(55, 83, 109)'
                ),
                row=row, col=col
            )
            fig.update_xaxes(title_text='Confidence', row=row, col=col)
            fig.update_yaxes(title_text='Frequency', row=row, col=col)
        except Exception as e:
            logger.error(f"Error creating confidence distribution plot: {str(e)}")

    def _create_calibration_plot(self, df: pd.DataFrame, fig: go.Figure, row: int, col: int):
        """Create calibration error plot"""
        try:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['calibration_error'],
                    mode='lines+markers',
                    name='Calibration Error',
                    line=dict(color='red'),
                    hovertemplate='%{y:.3f}'
                ),
                row=row, col=col
            )
            fig.update_xaxes(title_text='Time', row=row, col=col)
            fig.update_yaxes(title_text='Calibration Error', row=row, col=col)
        except Exception as e:
            logger.error(f"Error creating calibration plot: {str(e)}")

    def _create_roc_curve(self, df: pd.DataFrame, fig: go.Figure, row: int, col: int):
        """Create ROC curve plot"""
        try:
            if 'true_positive_rate' in df.columns and 'false_positive_rate' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['false_positive_rate'],
                        y=df['true_positive_rate'],
                        mode='lines',
                        name='ROC Curve',
                        line=dict(color='orange')
                    ),
                    row=row, col=col
                )
                # Add diagonal reference line
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        name='Reference',
                        line=dict(color='gray', dash='dash')
                    ),
                    row=row, col=col
                )
                fig.update_xaxes(title_text='False Positive Rate', row=row, col=col)
                fig.update_yaxes(title_text='True Positive Rate', row=row, col=col)
        except Exception as e:
            logger.error(f"Error creating ROC curve: {str(e)}")

    def _create_confusion_matrix(self, df: pd.DataFrame, fig: go.Figure, row: int, col: int):
        """Create confusion matrix heatmap"""
        try:
            if 'actual' in df.columns and 'predicted' in df.columns:
                # Calculate confusion matrix
                labels = ['Home Win', 'Draw', 'Away Win']
                confusion = pd.crosstab(df['actual'], df['predicted'])
                
                # Normalize confusion matrix
                confusion_norm = confusion.div(confusion.sum(axis=1), axis=0)
                
                fig.add_trace(
                    go.Heatmap(
                        z=confusion_norm.values,
                        x=labels,
                        y=labels,
                        colorscale='Viridis',
                        showscale=True,
                        text=confusion.values,
                        texttemplate='%{text}',
                        textfont={"size": 12},
                        name='Confusion Matrix'
                    ),
                    row=row, col=col
                )
                fig.update_xaxes(title_text='Predicted', row=row, col=col)
                fig.update_yaxes(title_text='Actual', row=row, col=col)
        except Exception as e:
            logger.error(f"Error creating confusion matrix: {str(e)}")

    def _create_prediction_distribution(self, df: pd.DataFrame, fig: go.Figure, row: int, col: int):
        """Create prediction distribution bar plot"""
        try:
            if 'prediction' in df.columns:
                prediction_counts = df['prediction'].value_counts().sort_index()
                labels = ['Home Win', 'Draw', 'Away Win']
                
                fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=[prediction_counts.get(i, 0) for i in range(3)],
                        name='Prediction Dist',
                        marker_color='rgb(158,202,225)',
                        text=[prediction_counts.get(i, 0) for i in range(3)],
                        textposition='auto'
                    ),
                    row=row, col=col
                )
                fig.update_xaxes(title_text='Prediction', row=row, col=col)
                fig.update_yaxes(title_text='Count', row=row, col=col)
        except Exception as e:
            logger.error(f"Error creating prediction distribution plot: {str(e)}")

    def _add_plot_annotations(self, fig: go.Figure):
        """Add annotations and styling to the plot"""
        try:
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                font=dict(
                    family="Arial, sans-serif",
                    size=12
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=100, b=50, l=50, r=50)
            )
            
            # Update grid styling
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
            
        except Exception as e:
            logger.error(f"Error adding plot annotations: {str(e)}")

    def create_performance_plot(self, metrics_history: Dict) -> Dict:
        """Create performance visualization with enhanced metrics"""
        try:
            if self._should_update_cache():
                df = self._prepare_metrics_dataframe(metrics_history)
                if df.empty:
                    return {}

                fig = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=(
                        'Accuracy & Confidence Over Time',
                        'Confidence Distribution',
                        'Calibration Error',
                        'ROC Curve',
                        'Confusion Matrix',
                        'Prediction Distribution'
                    ),
                    specs=[
                        [{"type": "scatter"}, {"type": "histogram"}],
                        [{"type": "scatter"}, {"type": "scatter"}],
                        [{"type": "heatmap"}, {"type": "bar"}]
                    ]
                )

                # Create individual plots
                self._create_accuracy_confidence_plot(df, fig, 1, 1)
                self._create_confidence_distribution(df, fig, 1, 2)
                self._create_calibration_plot(df, fig, 2, 1)
                self._create_roc_curve(df, fig, 2, 2)
                self._create_confusion_matrix(df, fig, 3, 1)
                self._create_prediction_distribution(df, fig, 3, 2)

                # Add styling and annotations
                self._add_plot_annotations(fig)

                # Update layout with theme
                fig.update_layout(
                    height=1200,
                    title_text="Model Performance Dashboard",
                    title_x=0.5,
                    template='plotly_dark' if settings.DARK_MODE else 'plotly_white'
                )

                self.metrics_cache['plot'] = fig.to_dict()
                self.last_cache_update = datetime.now()

            return self.metrics_cache.get('plot', {})

        except Exception as e:
            logger.error(f"Error creating performance plot: {str(e)}")
            return {}

    def get_model_metrics_summary(self, db: Session) -> MetricsSummary:
        """Get comprehensive model performance metrics"""
        try:
            if self._should_update_cache():
                active_model = self.registry.get_active_model()
                if not active_model:
                    raise HTTPException(status_code=404, detail="No active model found")

                recent_metrics = self.ab_testing.experiment_metrics.get(
                    active_model['version'], {}
                )

                if not recent_metrics:
                    raise HTTPException(status_code=404, detail="No metrics available")

                recent_window = 100
                metrics = self._calculate_metrics(recent_metrics, recent_window)
                
                self.metrics_cache['summary'] = metrics
                self.last_cache_update = datetime.now()

            return MetricsSummary(**self.metrics_cache.get('summary', {}))

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting metrics summary: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    def _calculate_metrics(self, recent_metrics: Dict, window: int) -> Dict:
        """Calculate comprehensive metrics from recent predictions"""
        confidences = recent_metrics['confidences'][-window:]
        predictions = recent_metrics['predictions'][-window:]
        actuals = [a for a in recent_metrics['actuals'][-window:] if a is not None]

        base_metrics = {
            'version': recent_metrics.get('version', 'unknown'),
            'total_predictions': len(recent_metrics['predictions']),
            'avg_confidence': float(np.mean(confidences)),
            'last_updated': datetime.now()
        }

        if not actuals:
            return base_metrics

        predictions_array = np.array(predictions[:len(actuals)])
        actuals_array = np.array(actuals)
        
        return {
            **base_metrics,
            'recent_accuracy': float(np.mean(predictions_array == actuals_array)),
            'calibration_error': float(abs(
                np.mean(confidences) - np.mean(predictions_array == actuals_array)
            )),
            'prediction_distribution': dict(pd.Series(predictions).value_counts()),
            'high_confidence_accuracy': float(self._calculate_high_confidence_accuracy(
                predictions_array, actuals_array, confidences[:len(actuals)]
            ))
        }

    def _calculate_high_confidence_accuracy(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidences: List[float]
    ) -> float:
        """Calculate accuracy for high confidence predictions"""
        high_conf_mask = np.array(confidences) >= settings.MIN_CONFIDENCE_THRESHOLD
        if not high_conf_mask.any():
            return 0.0
        return float(np.mean(predictions[high_conf_mask] == actuals[high_conf_mask]))

class DashboardMetrics:
    @staticmethod
    def calculate_metrics(matches: List[Match]) -> Dict:
        """Calculate comprehensive model performance metrics"""
        try:
            if not matches:
                return {}
                
            predictions = np.array([m.prediction for m in matches])
            actuals = np.array([m.actual_result for m in matches if m.actual_result is not None])
            confidences = np.array([m.confidence for m in matches])
            
            metrics = {
                'total_predictions': len(predictions),
                'accuracy': float(np.mean(predictions[:len(actuals)] == actuals)),
                'avg_confidence': float(np.mean(confidences)),
                'high_confidence_accuracy': float(np.mean(
                    predictions[confidences > 0.7][:len(actuals)] == actuals
                )),
                'prediction_distribution': {
                    'home_win': int(np.sum(predictions == 0)),
                    'draw': int(np.sum(predictions == 1)),
                    'away_win': int(np.sum(predictions == 2))
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    @staticmethod
    def create_performance_plots(matches: List[Match]) -> Dict:
        """Create visualization of model performance"""
        try:
            if not matches:
                return {}
                
            df = pd.DataFrame([
                {
                    'prediction': m.prediction,
                    'actual': m.actual_result,
                    'confidence': m.confidence,
                    'timestamp': m.created_at
                }
                for m in matches
            ])
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Accuracy Over Time',
                    'Confidence Distribution',
                    'Prediction Distribution',
                    'Confusion Matrix'
                )
            )
            
            # Accuracy over time
            window_size = 50
            df['rolling_accuracy'] = (
                df['prediction'] == df['actual']
            ).rolling(window_size).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['rolling_accuracy'],
                    mode='lines',
                    name=f'Accuracy ({window_size}-match rolling)'
                ),
                row=1, col=1
            )
            
            # Confidence distribution
            fig.add_trace(
                go.Histogram(
                    x=df['confidence'],
                    name='Confidence Distribution'
                ),
                row=1, col=2
            )
            
            # Prediction distribution
            pred_dist = df['prediction'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=['Home Win', 'Draw', 'Away Win'],
                    y=[pred_dist.get(0, 0), pred_dist.get(1, 0), pred_dist.get(2, 0)],
                    name='Predictions'
                ),
                row=2, col=1
            )
            
            # Confusion matrix
            confusion_df = pd.crosstab(df['actual'], df['prediction'])
            fig.add_trace(
                go.Heatmap(
                    z=confusion_df.values,
                    x=['Home Win', 'Draw', 'Away Win'],
                    y=['Home Win', 'Draw', 'Away Win'],
                    name='Confusion Matrix'
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True)
            return fig.to_dict()
            
        except Exception as e:
            logger.error(f"Error creating plots: {str(e)}")
            return {}

@router.get("/dashboard/metrics", response_model=MetricsSummary)
async def get_dashboard_metrics(
    db: Session = Depends(get_db),
    refresh_cache: bool = Query(False, description="Force refresh the metrics cache")
):
    """Get current model performance metrics"""
    try:
        dashboard = ModelMonitoringDashboard(
            registry=ModelRegistry(),
            ab_testing=ABTestingManager(ModelRegistry())
        )
        if refresh_cache:
            dashboard.last_cache_update = None
        return dashboard.get_model_metrics_summary(db)
    except Exception as e:
        logger.error(f"Error in dashboard metrics endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/performance-plot")
async def get_performance_plot(
    db: Session = Depends(get_db),
    refresh_cache: bool = Query(False, description="Force refresh the plot cache")
):
    """Get performance visualization data"""
    try:
        dashboard = ModelMonitoringDashboard(
            registry=ModelRegistry(),
            ab_testing=ABTestingManager(ModelRegistry())
        )
        if refresh_cache:
            dashboard.last_cache_update = None
        plot_data = dashboard.create_performance_plot(
            dashboard.ab_testing.experiment_metrics
        )
        if not plot_data:
            raise HTTPException(status_code=404, detail="No performance data available")
        return plot_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in performance plot endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_dashboard_metrics_from_matches(db: Session = Depends(get_db)):
    """Get current model performance metrics"""
    try:
        matches = db.query(Match).filter(
            Match.prediction.isnot(None),
            Match.start_time < datetime.utcnow()
        ).all()
        
        return DashboardMetrics.calculate_metrics(matches)
        
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/plots")
async def get_dashboard_plots(db: Session = Depends(get_db)):
    """Get performance visualization plots"""
    try:
        matches = db.query(Match).filter(
            Match.prediction.isnot(None)
        ).order_by(Match.start_time.desc()).all()
        
        return DashboardMetrics.create_performance_plots(matches)
        
    except Exception as e:
        logger.error(f"Error getting dashboard plots: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 