"""
Machine Learning endpoints for churn prediction.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.services.ml_pipeline import ml_service
from app.services.model_monitoring import get_model_monitor
from app.services.monitoring_scheduler import get_monitoring_scheduler
from app.services.prediction_service import prediction_service
from app.models.customer import CustomerBase
from app.models.prediction import (
    PredictionRequest, BatchPredictionRequest, PredictionResponse, 
    BatchPredictionResponse, PredictionStats, PredictionHistoryRequest,
    BatchHistoryRequest
)

logger = get_logger(__name__)
router = APIRouter()


@router.post("/predict", response_model=Dict[str, Any])
async def predict_churn(customer_data: CustomerBase, use_cache: bool = True) -> Dict[str, Any]:
    """
    Enhanced single customer churn prediction with confidence scoring and explanations.
    
    Args:
        customer_data: Customer features for prediction
        use_cache: Whether to use prediction caching
        
    Returns:
        Enhanced prediction results with confidence factors and feature contributions
    """
    try:
        logger.info(f"Enhanced prediction for customer: {customer_data.customer_id}")
        
        # Make enhanced prediction
        result = await prediction_service.predict_single(customer_data, use_cache=use_cache)
        
        return JSONResponse(
            content={
                "success": True,
                "prediction": result.to_dict()
            }
        )
        
    except ValueError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@router.post("/predict/batch", response_model=Dict[str, Any])
async def predict_batch(request: BatchPredictionRequest) -> Dict[str, Any]:
    """
    Batch prediction for multiple customers with enhanced features.
    
    Args:
        request: Batch prediction request with customer data and options
        
    Returns:
        Batch prediction results with summary statistics
    """
    try:
        logger.info(f"Batch prediction for {len(request.customers_data)} customers")
        
        # Make batch prediction
        result = await prediction_service.predict_batch(
            customers_data=request.customers_data,
            use_cache=request.use_cache,
            max_parallel=request.max_parallel
        )
        
        return JSONResponse(
            content={
                "success": True,
                "batch_result": result.to_dict()
            }
        )
        
    except ValueError as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during batch prediction")


@router.get("/model-status")
async def get_model_status() -> Dict[str, Any]:
    """
    Get current ML model performance metrics.
    
    Returns:
        Dictionary containing model status and metrics
    """
    try:
        logger.info("Getting ML model status")
        
        status = await ml_service.get_model_status()
        
        return JSONResponse(
            content={
                "success": True,
                "model_status": status
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model status")


@router.post("/retrain")
async def trigger_model_retraining() -> Dict[str, Any]:
    """
    Trigger ML model retraining process.
    
    Returns:
        Dictionary containing retraining job status
    """
    try:
        logger.info("Triggering ML model retraining")
        
        # Train model with latest data
        training_metrics = await ml_service.train_model()
        
        return JSONResponse(
            content={
                "success": True,
                "message": "Model retraining completed successfully",
                "training_metrics": training_metrics
            }
        )
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model retraining failed: {str(e)}")


@router.get("/feature-importance")
async def get_feature_importance() -> Dict[str, Any]:
    """
    Get feature importance from the current model.
    
    Returns:
        Dictionary containing feature importance rankings
    """
    try:
        logger.info("Getting feature importance")
        
        model_info = await ml_service.get_model_status()
        
        if not model_info.get('is_trained', False):
            raise HTTPException(status_code=400, detail="Model is not trained yet")
        
        return JSONResponse(
            content={
                "success": True,
                "feature_importance": model_info.get('top_features', {}),
                "model_version": model_info.get('model_version', 'unknown')
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get feature importance")


@router.post("/evaluate")
async def evaluate_model_performance() -> Dict[str, Any]:
    """
    Evaluate current model performance on recent data.
    
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        logger.info("Evaluating model performance")
        
        monitor = get_model_monitor()
        metrics = await monitor.evaluate_model_performance()
        
        return JSONResponse(
            content={
                "success": True,
                "evaluation_metrics": metrics.to_dict(),
                "message": "Model evaluation completed successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")


@router.post("/monitoring/run-cycle")
async def run_monitoring_cycle() -> Dict[str, Any]:
    """
    Run a complete monitoring cycle (evaluation + auto-retraining if needed).
    
    Returns:
        Dictionary containing monitoring cycle results
    """
    try:
        logger.info("Running monitoring cycle")
        
        monitor = get_model_monitor()
        results = await monitor.run_monitoring_cycle()
        
        return JSONResponse(
            content={
                "success": True,
                "monitoring_results": results
            }
        )
        
    except Exception as e:
        logger.error(f"Monitoring cycle failed: {e}")
        raise HTTPException(status_code=500, detail=f"Monitoring cycle failed: {str(e)}")


@router.get("/monitoring/summary")
async def get_monitoring_summary() -> Dict[str, Any]:
    """
    Get monitoring system summary and configuration.
    
    Returns:
        Dictionary containing monitoring summary
    """
    try:
        logger.info("Getting monitoring summary")
        
        monitor = get_model_monitor()
        summary = monitor.get_monitoring_summary()
        
        return JSONResponse(
            content={
                "success": True,
                "monitoring_summary": summary
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get monitoring summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring summary")


@router.get("/monitoring/history")
async def get_monitoring_history() -> Dict[str, Any]:
    """
    Get model performance and retraining history.
    
    Returns:
        Dictionary containing monitoring history
    """
    try:
        logger.info("Getting monitoring history")
        
        monitor = get_model_monitor()
        
        # Get recent metrics history (last 20)
        metrics_history = [m.to_dict() for m in monitor.metrics_history[-20:]]
        
        # Get retraining history (last 10)
        retraining_history = [r.to_dict() for r in monitor.retraining_history[-10:]]
        
        return JSONResponse(
            content={
                "success": True,
                "metrics_history": metrics_history,
                "retraining_history": retraining_history,
                "total_evaluations": len(monitor.metrics_history),
                "total_retrainings": len(monitor.retraining_history)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get monitoring history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring history")


@router.post("/monitoring/configure")
async def configure_monitoring(config: Dict[str, float]) -> Dict[str, Any]:
    """
    Configure monitoring thresholds.
    
    Args:
        config: Dictionary with threshold values
        
    Returns:
        Dictionary containing updated configuration
    """
    try:
        logger.info(f"Configuring monitoring thresholds: {config}")
        
        monitor = get_model_monitor()
        
        # Update thresholds if provided
        if 'accuracy_threshold' in config:
            monitor.accuracy_threshold = config['accuracy_threshold']
        if 'drift_threshold' in config:
            monitor.drift_threshold = config['drift_threshold']
        if 'sample_threshold' in config:
            monitor.sample_threshold = int(config['sample_threshold'])
        if 'evaluation_window_days' in config:
            monitor.evaluation_window_days = int(config['evaluation_window_days'])
        
        return JSONResponse(
            content={
                "success": True,
                "message": "Monitoring configuration updated",
                "current_config": {
                    "accuracy_threshold": monitor.accuracy_threshold,
                    "drift_threshold": monitor.drift_threshold,
                    "sample_threshold": monitor.sample_threshold,
                    "evaluation_window_days": monitor.evaluation_window_days
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to configure monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@router.post("/monitoring/scheduler/start")
async def start_monitoring_scheduler() -> Dict[str, Any]:
    """
    Start the automatic monitoring scheduler.
    
    Returns:
        Dictionary containing scheduler status
    """
    try:
        logger.info("Starting monitoring scheduler")
        
        scheduler = get_monitoring_scheduler()
        scheduler.start()
        
        return JSONResponse(
            content={
                "success": True,
                "message": "Monitoring scheduler started",
                "scheduler_status": scheduler.get_status()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start monitoring scheduler: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start scheduler: {str(e)}")


@router.post("/monitoring/scheduler/stop")
async def stop_monitoring_scheduler() -> Dict[str, Any]:
    """
    Stop the automatic monitoring scheduler.
    
    Returns:
        Dictionary containing scheduler status
    """
    try:
        logger.info("Stopping monitoring scheduler")
        
        scheduler = get_monitoring_scheduler()
        scheduler.stop()
        
        return JSONResponse(
            content={
                "success": True,
                "message": "Monitoring scheduler stopped",
                "scheduler_status": scheduler.get_status()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring scheduler: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop scheduler: {str(e)}")


@router.get("/monitoring/scheduler/status")
async def get_scheduler_status() -> Dict[str, Any]:
    """
    Get monitoring scheduler status.
    
    Returns:
        Dictionary containing scheduler status
    """
    try:
        scheduler = get_monitoring_scheduler()
        status = scheduler.get_status()
        
        return JSONResponse(
            content={
                "success": True,
                "scheduler_status": status
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get scheduler status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get scheduler status")


# Prediction Service Endpoints

@router.get("/predictions/stats")
async def get_prediction_stats() -> Dict[str, Any]:
    """
    Get prediction service statistics and performance metrics.
    
    Returns:
        Dictionary containing prediction service statistics
    """
    try:
        stats = prediction_service.get_prediction_stats()
        
        return JSONResponse(
            content={
                "success": True,
                "prediction_stats": stats
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get prediction stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get prediction statistics")


@router.get("/predictions/history")
async def get_prediction_history(
    limit: int = 50, 
    customer_id: str = None,
    risk_level: str = None
) -> Dict[str, Any]:
    """
    Get recent prediction history with optional filtering.
    
    Args:
        limit: Number of recent predictions to return (max 500)
        customer_id: Optional filter by customer ID
        risk_level: Optional filter by risk level (low/medium/high)
        
    Returns:
        Dictionary containing recent predictions
    """
    try:
        if limit > 500:
            limit = 500
            
        predictions = prediction_service.get_recent_predictions(limit=limit)
        
        # Apply filters if provided
        if customer_id:
            predictions = [p for p in predictions if p.get('customer_id') == customer_id]
        
        if risk_level:
            predictions = [p for p in predictions if p.get('risk_level') == risk_level]
        
        return JSONResponse(
            content={
                "success": True,
                "predictions": predictions,
                "total_returned": len(predictions),
                "filters": {
                    "customer_id": customer_id,
                    "risk_level": risk_level,
                    "limit": limit
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get prediction history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get prediction history")


@router.get("/predictions/batches")
async def get_batch_history(limit: int = 10, include_predictions: bool = False) -> Dict[str, Any]:
    """
    Get recent batch prediction history.
    
    Args:
        limit: Number of recent batches to return (max 100)
        include_predictions: Whether to include individual predictions in response
        
    Returns:
        Dictionary containing recent batch results
    """
    try:
        if limit > 100:
            limit = 100
            
        batches = prediction_service.get_recent_batches(limit=limit)
        
        # Optionally exclude individual predictions to reduce response size
        if not include_predictions:
            for batch in batches:
                if 'predictions' in batch:
                    batch['predictions'] = f"[{len(batch['predictions'])} predictions - use include_predictions=true to see details]"
        
        return JSONResponse(
            content={
                "success": True,
                "batches": batches,
                "total_returned": len(batches),
                "include_predictions": include_predictions
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get batch history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get batch history")


@router.post("/predictions/cache/clear")
async def clear_prediction_cache() -> Dict[str, Any]:
    """
    Clear the prediction cache.
    
    Returns:
        Dictionary containing cache clear status
    """
    try:
        prediction_service.clear_cache()
        
        return JSONResponse(
            content={
                "success": True,
                "message": "Prediction cache cleared successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to clear prediction cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear prediction cache")


@router.get("/predictions/cache/stats")
async def get_cache_stats() -> Dict[str, Any]:
    """
    Get prediction cache statistics.
    
    Returns:
        Dictionary containing cache statistics
    """
    try:
        stats = prediction_service.get_prediction_stats()
        cache_stats = stats.get('cache_stats', {})
        
        return JSONResponse(
            content={
                "success": True,
                "cache_stats": cache_stats,
                "cache_hit_rate": stats.get('cache_hit_rate', 0)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache statistics")


@router.post("/predictions/explain/{prediction_id}")
async def explain_prediction(prediction_id: str) -> Dict[str, Any]:
    """
    Get detailed explanation for a specific prediction.
    
    Args:
        prediction_id: Unique prediction identifier
        
    Returns:
        Dictionary containing detailed prediction explanation
    """
    try:
        # Find the prediction in history
        predictions = prediction_service.get_recent_predictions(limit=1000)
        prediction = next((p for p in predictions if p.get('prediction_id') == prediction_id), None)
        
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        # Generate human-readable explanation
        explanation = _generate_prediction_explanation(prediction)
        
        return JSONResponse(
            content={
                "success": True,
                "explanation": explanation
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to explain prediction {prediction_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate prediction explanation")


@router.get("/predictions/summary")
async def get_predictions_summary() -> Dict[str, Any]:
    """
    Get summary of recent prediction activity.
    
    Returns:
        Dictionary containing prediction activity summary
    """
    try:
        stats = prediction_service.get_prediction_stats()
        recent_predictions = prediction_service.get_recent_predictions(limit=100)
        recent_batches = prediction_service.get_recent_batches(limit=10)
        
        # Calculate summary metrics
        if recent_predictions:
            risk_distribution = {}
            confidence_scores = []
            processing_times = []
            
            for pred in recent_predictions:
                risk_level = pred.get('risk_level', 'unknown')
                risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
                confidence_scores.append(pred.get('confidence', 0))
                processing_times.append(pred.get('processing_time_ms', 0))
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            avg_processing_time = sum(processing_times) / len(processing_times)
        else:
            risk_distribution = {}
            avg_confidence = 0
            avg_processing_time = 0
        
        summary = {
            "total_predictions": stats.get('total_predictions', 0),
            "recent_predictions_count": len(recent_predictions),
            "recent_batches_count": len(recent_batches),
            "cache_hit_rate": stats.get('cache_hit_rate', 0),
            "avg_confidence": avg_confidence,
            "avg_processing_time_ms": avg_processing_time,
            "risk_distribution": risk_distribution,
            "service_status": "healthy" if stats.get('total_predictions', 0) > 0 else "inactive"
        }
        
        return JSONResponse(
            content={
                "success": True,
                "summary": summary
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get predictions summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get predictions summary")


def _generate_prediction_explanation(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Generate human-readable explanation for a prediction."""
    customer_id = prediction.get('customer_id', 'unknown')
    churn_prob = prediction.get('churn_probability', 0)
    risk_level = prediction.get('risk_level', 'unknown')
    confidence = prediction.get('confidence', 0)
    
    # Generate explanation summary
    if churn_prob >= 0.7:
        summary = f"Customer {customer_id} has a HIGH probability ({churn_prob:.1%}) of churning."
    elif churn_prob >= 0.4:
        summary = f"Customer {customer_id} has a MEDIUM probability ({churn_prob:.1%}) of churning."
    else:
        summary = f"Customer {customer_id} has a LOW probability ({churn_prob:.1%}) of churning."
    
    # Key factors
    confidence_factors = prediction.get('confidence_factors', {})
    feature_contributions = prediction.get('feature_contributions', {})
    
    key_factors = []
    risk_factors = []
    protective_factors = []
    
    # Analyze confidence factors
    contract_stability = confidence_factors.get('contract_stability', 0)
    if contract_stability < 0.5:
        risk_factors.append("Short-term or month-to-month contract")
        key_factors.append("Contract type")
    else:
        protective_factors.append("Long-term contract commitment")
    
    tenure_stability = confidence_factors.get('tenure_stability', 0)
    if tenure_stability < 0.5:
        risk_factors.append("Low customer tenure")
        key_factors.append("Customer tenure")
    else:
        protective_factors.append("Established customer relationship")
    
    # Analyze feature contributions
    if feature_contributions.get('monthly_charges', 0) > 0.1:
        risk_factors.append("High monthly charges")
        key_factors.append("Monthly charges")
    
    if feature_contributions.get('internet_service', 0) > 0.1:
        risk_factors.append("Internet service type")
        key_factors.append("Internet service")
    
    # Confidence explanation
    if confidence >= 0.8:
        confidence_explanation = "Very high confidence - strong signal from multiple factors"
    elif confidence >= 0.6:
        confidence_explanation = "High confidence - clear signal from key factors"
    elif confidence >= 0.4:
        confidence_explanation = "Medium confidence - some uncertainty in prediction"
    else:
        confidence_explanation = "Low confidence - prediction has significant uncertainty"
    
    # Recommendations
    recommendations = []
    if risk_level == 'high':
        recommendations.extend([
            "Immediate retention intervention recommended",
            "Consider personalized offers or incentives",
            "Reach out proactively to address concerns"
        ])
    elif risk_level == 'medium':
        recommendations.extend([
            "Monitor customer engagement closely",
            "Consider preventive retention measures",
            "Ensure excellent customer service"
        ])
    else:
        recommendations.extend([
            "Continue standard service delivery",
            "Maintain regular customer touchpoints",
            "Consider upselling opportunities"
        ])
    
    return {
        "prediction_id": prediction.get('prediction_id'),
        "customer_id": customer_id,
        "explanation_summary": summary,
        "key_factors": key_factors or ["General customer profile"],
        "risk_factors": risk_factors or ["No significant risk factors identified"],
        "protective_factors": protective_factors or ["No specific protective factors identified"],
        "confidence_explanation": confidence_explanation,
        "recommendations": recommendations
    }