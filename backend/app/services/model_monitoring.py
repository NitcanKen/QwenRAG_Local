"""
Model performance monitoring and auto-retraining system.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from app.core.logging import get_logger
from app.core.database import get_supabase_client
from app.services.ml_pipeline import ChurnPredictor, MLPipelineService

logger = get_logger(__name__)


class ModelStatus(Enum):
    """Model status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RETRAINING = "retraining"
    DEPLOYING = "deploying"


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    sample_count: int = 0
    timestamp: str = ""
    model_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RetrainingEvent:
    """Retraining event record."""
    event_id: str
    trigger_reason: str
    triggered_at: str
    old_model_version: str
    new_model_version: str
    old_metrics: PerformanceMetrics
    new_metrics: Optional[PerformanceMetrics] = None
    status: str = "initiated"
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['old_metrics'] = self.old_metrics.to_dict()
        if self.new_metrics:
            data['new_metrics'] = self.new_metrics.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrainingEvent':
        """Create from dictionary."""
        old_metrics = PerformanceMetrics.from_dict(data['old_metrics'])
        new_metrics = None
        if data.get('new_metrics'):
            new_metrics = PerformanceMetrics.from_dict(data['new_metrics'])
        
        return cls(
            event_id=data['event_id'],
            trigger_reason=data['trigger_reason'],
            triggered_at=data['triggered_at'],
            old_model_version=data['old_model_version'],
            new_model_version=data['new_model_version'],
            old_metrics=old_metrics,
            new_metrics=new_metrics,
            status=data.get('status', 'initiated'),
            completed_at=data.get('completed_at'),
            error_message=data.get('error_message')
        )


class ModelMonitor:
    """Model performance monitoring and auto-retraining system."""

    def __init__(self, 
                 ml_service: MLPipelineService,
                 monitoring_dir: str = "data/monitoring"):
        self.ml_service = ml_service
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        self.supabase = get_supabase_client()
        
        # Monitoring thresholds
        self.accuracy_threshold = 0.75  # Retrain if accuracy drops below 75%
        self.drift_threshold = 0.05     # Retrain if accuracy drops by 5%
        self.sample_threshold = 100     # Minimum samples before evaluation
        self.evaluation_window_days = 7 # Evaluate performance weekly
        
        # Performance history
        self.metrics_history: List[PerformanceMetrics] = []
        self.retraining_history: List[RetrainingEvent] = []
        
        # Load existing history
        self._load_monitoring_data()

    def _load_monitoring_data(self) -> None:
        """Load existing monitoring data from disk."""
        try:
            # Load metrics history
            metrics_file = self.monitoring_dir / "metrics_history.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics_history = [
                        PerformanceMetrics.from_dict(item) for item in data
                    ]
                    
            # Load retraining history
            retraining_file = self.monitoring_dir / "retraining_history.json"
            if retraining_file.exists():
                with open(retraining_file, 'r') as f:
                    data = json.load(f)
                    self.retraining_history = [
                        RetrainingEvent.from_dict(item) for item in data
                    ]
                    
            logger.info(f"Loaded {len(self.metrics_history)} metrics and "
                       f"{len(self.retraining_history)} retraining events")
                       
        except Exception as e:
            logger.error(f"Failed to load monitoring data: {e}")
            self.metrics_history = []
            self.retraining_history = []

    def _save_monitoring_data(self) -> None:
        """Save monitoring data to disk."""
        try:
            # Save metrics history
            metrics_file = self.monitoring_dir / "metrics_history.json"
            with open(metrics_file, 'w') as f:
                json.dump([m.to_dict() for m in self.metrics_history], f, indent=2)
                
            # Save retraining history
            retraining_file = self.monitoring_dir / "retraining_history.json"
            with open(retraining_file, 'w') as f:
                json.dump([r.to_dict() for r in self.retraining_history], f, indent=2)
                
            logger.info("Monitoring data saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save monitoring data: {e}")

    async def evaluate_model_performance(self, 
                                       test_data: Optional[pd.DataFrame] = None) -> PerformanceMetrics:
        """
        Evaluate current model performance.
        
        Args:
            test_data: Optional test data. If None, loads recent data from database.
            
        Returns:
            Performance metrics
        """
        try:
            if test_data is None:
                # Get recent data for evaluation
                test_data = await self._get_recent_data_for_evaluation()
                
            if len(test_data) < self.sample_threshold:
                raise ValueError(f"Insufficient data for evaluation: {len(test_data)} < {self.sample_threshold}")
            
            # Evaluate model
            evaluation_results = self.ml_service.predictor.evaluate(test_data)
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                accuracy=evaluation_results['accuracy'],
                precision=evaluation_results['precision'],
                recall=evaluation_results['recall'],
                f1_score=evaluation_results['f1_score'],
                confusion_matrix=evaluation_results.get('confusion_matrix'),
                sample_count=len(test_data),
                timestamp=datetime.now().isoformat(),
                model_version=self.ml_service.predictor.model_version
            )
            
            # Add to history
            self.metrics_history.append(metrics)
            
            # Keep only recent history (last 50 evaluations)
            if len(self.metrics_history) > 50:
                self.metrics_history = self.metrics_history[-50:]
            
            # Save monitoring data
            self._save_monitoring_data()
            
            logger.info(f"Model evaluation completed: accuracy={metrics.accuracy:.3f}, "
                       f"f1_score={metrics.f1_score:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise

    async def _get_recent_data_for_evaluation(self) -> pd.DataFrame:
        """Get recent customer data for model evaluation."""
        try:
            # Get data from the last evaluation window
            cutoff_date = datetime.now() - timedelta(days=self.evaluation_window_days)
            
            # Query recent customer data
            response = self.supabase.table('customers').select('*').gte(
                'created_at', cutoff_date.isoformat()
            ).execute()
            
            if not response.data:
                # If no recent data, get a sample of all data
                response = self.supabase.table('customers').select('*').limit(1000).execute()
                
            if not response.data:
                raise ValueError("No customer data available for evaluation")
            
            df = pd.DataFrame(response.data)
            logger.info(f"Retrieved {len(df)} records for evaluation")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get evaluation data: {e}")
            raise

    def check_retraining_triggers(self, current_metrics: PerformanceMetrics) -> Tuple[bool, str]:
        """
        Check if retraining should be triggered.
        
        Args:
            current_metrics: Current model performance metrics
            
        Returns:
            Tuple of (should_retrain, reason)
        """
        # Check absolute accuracy threshold
        if current_metrics.accuracy < self.accuracy_threshold:
            return True, f"Accuracy below threshold: {current_metrics.accuracy:.3f} < {self.accuracy_threshold}"
        
        # Check performance drift
        if len(self.metrics_history) >= 2:
            # Compare with baseline (median of last 5 evaluations or all if less than 5)
            recent_metrics = self.metrics_history[-5:] if len(self.metrics_history) >= 5 else self.metrics_history
            baseline_accuracy = np.median([m.accuracy for m in recent_metrics[:-1]])
            
            accuracy_drop = baseline_accuracy - current_metrics.accuracy
            if accuracy_drop > self.drift_threshold:
                return True, f"Performance drift detected: accuracy dropped by {accuracy_drop:.3f}"
        
        # Check if last retraining was too long ago (30 days)
        if self.retraining_history:
            last_retraining = datetime.fromisoformat(self.retraining_history[-1].triggered_at)
            days_since_retrain = (datetime.now() - last_retraining).days
            
            if days_since_retrain > 30:
                return True, f"Scheduled retraining: {days_since_retrain} days since last retrain"
        elif len(self.metrics_history) > 10:
            # If no retraining history but we have metrics, schedule first retrain
            return True, "Initial scheduled retraining"
        
        # Check F1 score degradation
        if len(self.metrics_history) >= 2:
            recent_f1 = np.median([m.f1_score for m in self.metrics_history[-5:]])
            f1_drop = recent_f1 - current_metrics.f1_score
            
            if f1_drop > self.drift_threshold:
                return True, f"F1 score drift detected: dropped by {f1_drop:.3f}"
        
        return False, "No retraining triggers met"

    async def trigger_retraining(self, reason: str, 
                               current_metrics: PerformanceMetrics) -> RetrainingEvent:
        """
        Trigger model retraining process.
        
        Args:
            reason: Reason for retraining
            current_metrics: Current model performance metrics
            
        Returns:
            Retraining event record
        """
        event_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        old_version = self.ml_service.predictor.model_version
        new_version = f"{old_version.split('.')[0]}.{int(old_version.split('.')[1]) + 1}.0"
        
        # Create retraining event
        event = RetrainingEvent(
            event_id=event_id,
            trigger_reason=reason,
            triggered_at=datetime.now().isoformat(),
            old_model_version=old_version,
            new_model_version=new_version,
            old_metrics=current_metrics,
            status="initiated"
        )
        
        logger.info(f"Triggering retraining: {reason}")
        
        try:
            # Update model version for new training
            self.ml_service.predictor.model_version = new_version
            
            # Perform retraining
            training_metrics = await self.ml_service.train_model()
            
            # Evaluate new model
            new_metrics = await self.evaluate_model_performance()
            
            # Compare models and decide deployment
            deploy_new_model = self._should_deploy_new_model(current_metrics, new_metrics)
            
            if deploy_new_model:
                event.status = "completed"
                event.new_metrics = new_metrics
                event.completed_at = datetime.now().isoformat()
                logger.info(f"Retraining completed successfully. New accuracy: {new_metrics.accuracy:.3f}")
            else:
                # Rollback to old model
                await self._rollback_model(old_version)
                event.status = "rollback"
                event.error_message = "New model performance not better than current model"
                logger.warning("New model performance not improved, rolling back")
            
        except Exception as e:
            # Handle retraining failure
            event.status = "failed"
            event.error_message = str(e)
            event.completed_at = datetime.now().isoformat()
            
            # Try to rollback to old model
            try:
                await self._rollback_model(old_version)
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")
            
            logger.error(f"Retraining failed: {e}")
        
        # Save event
        self.retraining_history.append(event)
        self._save_monitoring_data()
        
        return event

    def _should_deploy_new_model(self, 
                               old_metrics: PerformanceMetrics, 
                               new_metrics: PerformanceMetrics) -> bool:
        """
        Decide whether to deploy the new model.
        
        Args:
            old_metrics: Current model metrics
            new_metrics: New model metrics
            
        Returns:
            True if new model should be deployed
        """
        # New model should be significantly better
        accuracy_improvement = new_metrics.accuracy - old_metrics.accuracy
        f1_improvement = new_metrics.f1_score - old_metrics.f1_score
        
        # Require at least 1% improvement in accuracy or F1
        min_improvement = 0.01
        
        if accuracy_improvement >= min_improvement or f1_improvement >= min_improvement:
            return True
        
        # If performance is similar, prefer the new model if it's not significantly worse
        if accuracy_improvement >= -0.005 and f1_improvement >= -0.005:
            return True
            
        return False

    async def _rollback_model(self, target_version: str) -> None:
        """
        Rollback to a previous model version.
        
        Args:
            target_version: Version to rollback to
        """
        try:
            # Find the model file for the target version
            model_files = list(self.ml_service.predictor.model_dir.glob("churn_model_*.joblib"))
            
            # Try to find the specific version or use the latest available
            target_file = None
            for model_file in sorted(model_files, reverse=True):
                if target_version in model_file.name:
                    target_file = model_file
                    break
            
            if not target_file and model_files:
                target_file = model_files[0]  # Use most recent
            
            if target_file:
                self.ml_service.predictor.load_model(str(target_file))
                logger.info(f"Successfully rolled back to model: {target_file.name}")
            else:
                logger.error("No model files found for rollback")
                
        except Exception as e:
            logger.error(f"Model rollback failed: {e}")
            raise

    async def run_monitoring_cycle(self) -> Dict[str, Any]:
        """
        Run a complete monitoring cycle.
        
        Returns:
            Monitoring cycle results
        """
        try:
            logger.info("Starting monitoring cycle")
            
            # Check if model is trained
            if not self.ml_service.predictor.is_trained:
                # Try to load latest model
                success = await self.ml_service.load_latest_model()
                if not success:
                    return {
                        "status": "error",
                        "message": "No trained model available",
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Evaluate current performance
            current_metrics = await self.evaluate_model_performance()
            
            # Check retraining triggers
            should_retrain, reason = self.check_retraining_triggers(current_metrics)
            
            result = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "current_metrics": current_metrics.to_dict(),
                "should_retrain": should_retrain,
                "trigger_reason": reason if should_retrain else None,
                "model_status": self._get_model_status(current_metrics).value
            }
            
            # Trigger retraining if needed
            if should_retrain:
                retraining_event = await self.trigger_retraining(reason, current_metrics)
                result["retraining_event"] = retraining_event.to_dict()
            
            logger.info(f"Monitoring cycle completed: {result['model_status']}")
            return result
            
        except Exception as e:
            logger.error(f"Monitoring cycle failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _get_model_status(self, metrics: PerformanceMetrics) -> ModelStatus:
        """
        Determine model status based on metrics.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Model status
        """
        if metrics.accuracy < 0.6:
            return ModelStatus.FAILED
        elif metrics.accuracy < self.accuracy_threshold:
            return ModelStatus.DEGRADED
        else:
            return ModelStatus.HEALTHY

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring system summary."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        latest_retraining = self.retraining_history[-1] if self.retraining_history else None
        
        return {
            "monitoring_config": {
                "accuracy_threshold": self.accuracy_threshold,
                "drift_threshold": self.drift_threshold,
                "sample_threshold": self.sample_threshold,
                "evaluation_window_days": self.evaluation_window_days
            },
            "current_status": {
                "model_status": self._get_model_status(latest_metrics).value if latest_metrics else "unknown",
                "last_evaluation": latest_metrics.to_dict() if latest_metrics else None,
                "last_retraining": latest_retraining.to_dict() if latest_retraining else None
            },
            "history_summary": {
                "total_evaluations": len(self.metrics_history),
                "total_retrainings": len(self.retraining_history),
                "successful_retrainings": len([r for r in self.retraining_history if r.status == "completed"]),
                "failed_retrainings": len([r for r in self.retraining_history if r.status == "failed"])
            }
        }


# Global monitoring instance
model_monitor = None

def get_model_monitor() -> ModelMonitor:
    """Get or create model monitor instance."""
    global model_monitor
    if model_monitor is None:
        from app.services.ml_pipeline import ml_service
        model_monitor = ModelMonitor(ml_service)
    return model_monitor