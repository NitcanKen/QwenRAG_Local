"""
Prediction-related Pydantic models for input validation and API responses.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime

from app.models.customer import CustomerBase


class PredictionRequest(BaseModel):
    """Single prediction request model."""
    customer_data: CustomerBase = Field(..., description="Customer data for prediction")
    use_cache: bool = Field(True, description="Whether to use prediction caching")
    include_explanations: bool = Field(True, description="Include confidence factors and feature contributions")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    customers_data: List[CustomerBase] = Field(..., description="List of customer data for batch prediction")
    use_cache: bool = Field(True, description="Whether to use prediction caching")
    include_explanations: bool = Field(True, description="Include confidence factors and feature contributions")
    max_parallel: int = Field(10, ge=1, le=50, description="Maximum parallel predictions")
    
    @validator('customers_data')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError("Batch must contain at least one customer")
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 customers")
        return v


class ConfidenceFactors(BaseModel):
    """Confidence factors contributing to prediction."""
    probability_spread: float = Field(..., description="Confidence based on probability spread")
    contract_stability: float = Field(..., description="Confidence based on contract type")
    tenure_stability: float = Field(..., description="Confidence based on customer tenure")
    service_complexity: float = Field(..., description="Confidence based on service usage")


class FeatureContributions(BaseModel):
    """Feature contributions to the prediction."""
    contract: Optional[float] = Field(None, description="Contract type contribution")
    tenure: Optional[float] = Field(None, description="Tenure contribution")
    monthly_charges: Optional[float] = Field(None, description="Monthly charges contribution")
    internet_service: Optional[float] = Field(None, description="Internet service contribution")


class PredictionResponse(BaseModel):
    """Enhanced prediction response model."""
    customer_id: str = Field(..., description="Customer identifier")
    churn_prediction: int = Field(..., description="Churn prediction (0 or 1)")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    no_churn_probability: float = Field(..., ge=0, le=1, description="Probability of no churn")
    confidence: float = Field(..., ge=0, le=1, description="Overall prediction confidence")
    risk_level: str = Field(..., description="Risk level (low/medium/high)")
    confidence_factors: Dict[str, float] = Field(..., description="Factors contributing to confidence")
    feature_contributions: Dict[str, float] = Field(..., description="Feature contributions to prediction")
    model_version: str = Field(..., description="Model version used for prediction")
    predicted_at: str = Field(..., description="Prediction timestamp")
    prediction_id: str = Field(..., description="Unique prediction identifier")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchSummary(BaseModel):
    """Batch prediction summary statistics."""
    avg_churn_probability: float = Field(..., description="Average churn probability")
    median_churn_probability: float = Field(..., description="Median churn probability")
    avg_confidence: float = Field(..., description="Average confidence score")
    avg_processing_time_ms: float = Field(..., description="Average processing time")
    risk_distribution: Dict[str, int] = Field(..., description="Distribution of risk levels")
    high_risk_customers: int = Field(..., description="Number of high-risk customers")
    churn_predictions: int = Field(..., description="Number of churn predictions")
    no_churn_predictions: int = Field(..., description="Number of no-churn predictions")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    batch_id: str = Field(..., description="Unique batch identifier")
    predictions: List[PredictionResponse] = Field(..., description="Individual predictions")
    batch_summary: BatchSummary = Field(..., description="Batch summary statistics")
    total_predictions: int = Field(..., description="Total number of predictions requested")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    processed_at: str = Field(..., description="Batch processing timestamp")


class PredictionStats(BaseModel):
    """Prediction service statistics."""
    total_predictions: int = Field(..., description="Total predictions made")
    cache_hit_rate: float = Field(..., ge=0, le=1, description="Cache hit rate")
    cache_stats: Dict[str, Any] = Field(..., description="Cache statistics")
    recent_performance: Dict[str, Any] = Field(..., description="Recent performance metrics")
    history_size: Dict[str, int] = Field(..., description="History size information")
    recent_batches: int = Field(..., description="Number of recent batches")


class CacheStats(BaseModel):
    """Cache statistics model."""
    total_items: int = Field(..., description="Total cached items")
    expired_items: int = Field(..., description="Number of expired items")
    active_items: int = Field(..., description="Number of active items")
    max_size: int = Field(..., description="Maximum cache size")
    ttl_seconds: int = Field(..., description="Time to live in seconds")
    cache_utilization: float = Field(..., ge=0, le=1, description="Cache utilization percentage")


class PredictionHistoryRequest(BaseModel):
    """Request model for prediction history."""
    limit: int = Field(50, ge=1, le=500, description="Number of recent predictions to return")
    customer_id: Optional[str] = Field(None, description="Filter by customer ID")
    risk_level: Optional[str] = Field(None, description="Filter by risk level")
    start_date: Optional[str] = Field(None, description="Start date for filtering (ISO format)")
    end_date: Optional[str] = Field(None, description="End date for filtering (ISO format)")


class BatchHistoryRequest(BaseModel):
    """Request model for batch history."""
    limit: int = Field(10, ge=1, le=100, description="Number of recent batches to return")
    include_predictions: bool = Field(False, description="Include individual predictions in response")


class PredictionExplanation(BaseModel):
    """Detailed prediction explanation."""
    prediction_id: str = Field(..., description="Prediction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    explanation_summary: str = Field(..., description="Human-readable explanation summary")
    key_factors: List[str] = Field(..., description="Key factors influencing the prediction")
    risk_factors: List[str] = Field(..., description="Factors increasing churn risk")
    protective_factors: List[str] = Field(..., description="Factors reducing churn risk")
    confidence_explanation: str = Field(..., description="Explanation of confidence level")
    recommendations: List[str] = Field(..., description="Recommendations based on prediction")


class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics for API responses."""
    model_version: str = Field(..., description="Model version")
    accuracy: float = Field(..., ge=0, le=1, description="Model accuracy")
    precision: float = Field(..., ge=0, le=1, description="Model precision")
    recall: float = Field(..., ge=0, le=1, description="Model recall")
    f1_score: float = Field(..., ge=0, le=1, description="Model F1 score")
    last_trained: str = Field(..., description="Last training timestamp")
    evaluation_date: str = Field(..., description="Last evaluation timestamp")
    sample_count: int = Field(..., description="Evaluation sample count")


class PredictionServiceStatus(BaseModel):
    """Prediction service status model."""
    service_status: str = Field(..., description="Service status (healthy/degraded/failed)")
    model_status: str = Field(..., description="Model status")
    cache_status: str = Field(..., description="Cache status")
    total_predictions_today: int = Field(..., description="Total predictions made today")
    avg_response_time_ms: float = Field(..., description="Average response time")
    error_rate: float = Field(..., ge=0, le=1, description="Error rate percentage")
    last_health_check: str = Field(..., description="Last health check timestamp")


class BulkCustomerData(BaseModel):
    """Model for bulk customer data upload."""
    customers: List[Dict[str, Any]] = Field(..., description="List of customer data dictionaries")
    validate_schema: bool = Field(True, description="Whether to validate customer data schema")
    prediction_options: Optional[PredictionRequest] = Field(None, description="Prediction options to apply")
    
    @validator('customers')
    def validate_customers_list(cls, v):
        if len(v) == 0:
            raise ValueError("Must provide at least one customer")
        if len(v) > 5000:
            raise ValueError("Cannot process more than 5000 customers in one request")
        return v