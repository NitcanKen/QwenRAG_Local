"""
Enhanced prediction service with batch processing, confidence scoring, and caching.
"""

import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

from app.core.logging import get_logger
from app.services.ml_pipeline import ml_service
from app.models.customer import CustomerBase

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Enhanced prediction result with confidence and explanations."""
    customer_id: str
    churn_prediction: int
    churn_probability: float
    no_churn_probability: float
    confidence: float
    risk_level: str
    confidence_factors: Dict[str, float]
    feature_contributions: Dict[str, float]
    model_version: str
    predicted_at: str
    prediction_id: str
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionResult':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class BatchPredictionResult:
    """Batch prediction result container."""
    batch_id: str
    predictions: List[PredictionResult]
    batch_summary: Dict[str, Any]
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    total_processing_time_ms: float
    processed_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "predictions": [p.to_dict() for p in self.predictions],
            "batch_summary": self.batch_summary,
            "total_predictions": self.total_predictions,
            "successful_predictions": self.successful_predictions,
            "failed_predictions": self.failed_predictions,
            "total_processing_time_ms": self.total_processing_time_ms,
            "processed_at": self.processed_at
        }


class PredictionCache:
    """Simple in-memory prediction cache with TTL."""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 10000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        
    def _get_cache_key(self, customer_data: Dict[str, Any], model_version: str) -> str:
        """Generate cache key from customer data and model version."""
        # Create a deterministic hash of the customer data
        data_str = json.dumps(customer_data, sort_keys=True)
        return hashlib.md5(f"{data_str}_{model_version}".encode()).hexdigest()
    
    def get(self, customer_data: Dict[str, Any], model_version: str) -> Optional[PredictionResult]:
        """Get cached prediction if available and not expired."""
        cache_key = self._get_cache_key(customer_data, model_version)
        
        if cache_key not in self.cache:
            return None
            
        cached_item = self.cache[cache_key]
        cached_time = self.access_times[cache_key]
        
        # Check if expired
        if time.time() - cached_time > self.ttl_seconds:
            del self.cache[cache_key]
            del self.access_times[cache_key]
            return None
            
        # Update access time
        self.access_times[cache_key] = time.time()
        return PredictionResult.from_dict(cached_item)
    
    def put(self, customer_data: Dict[str, Any], model_version: str, 
            result: PredictionResult) -> None:
        """Cache a prediction result."""
        cache_key = self._get_cache_key(customer_data, model_version)
        
        # Evict oldest items if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
            
        self.cache[cache_key] = result.to_dict()
        self.access_times[cache_key] = time.time()
    
    def _evict_oldest(self) -> None:
        """Evict the oldest cached items."""
        # Remove 10% of oldest items
        num_to_remove = max(1, self.max_size // 10)
        oldest_keys = sorted(self.access_times.keys(), 
                           key=lambda k: self.access_times[k])[:num_to_remove]
        
        for key in oldest_keys:
            del self.cache[key]
            del self.access_times[key]
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        expired_count = sum(1 for t in self.access_times.values() 
                          if current_time - t > self.ttl_seconds)
        
        return {
            "total_items": len(self.cache),
            "expired_items": expired_count,
            "active_items": len(self.cache) - expired_count,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "cache_utilization": len(self.cache) / self.max_size
        }


class EnhancedPredictionService:
    """Enhanced prediction service with batch processing and caching."""
    
    def __init__(self):
        self.cache = PredictionCache(ttl_seconds=3600, max_size=10000)  # 1 hour TTL
        self.prediction_history: List[PredictionResult] = []
        self.batch_history: List[BatchPredictionResult] = []
        
        # Statistics
        self.total_predictions = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
    async def predict_single(self, customer_data: Union[CustomerBase, Dict[str, Any]], 
                           use_cache: bool = True) -> PredictionResult:
        """
        Make a single customer churn prediction with enhanced features.
        
        Args:
            customer_data: Customer data (Pydantic model or dict)
            use_cache: Whether to use prediction caching
            
        Returns:
            Enhanced prediction result
        """
        start_time = time.time()
        
        # Convert to dict if Pydantic model
        if isinstance(customer_data, CustomerBase):
            customer_dict = customer_data.model_dump(exclude={'id', 'created_at', 'updated_at'})
            customer_id = customer_data.customer_id
        else:
            customer_dict = customer_data.copy()
            customer_id = customer_dict.get('customer_id', 'unknown')
        
        model_version = ml_service.predictor.model_version
        
        # Check cache first
        if use_cache:
            cached_result = self.cache.get(customer_dict, model_version)
            if cached_result:
                self.cache_hits += 1
                logger.info(f"Cache hit for customer {customer_id}")
                return cached_result
            else:
                self.cache_misses += 1
        
        # Make prediction
        try:
            base_prediction = await ml_service.predict_customer_churn(customer_dict)
            
            # Calculate enhanced confidence and explanations
            confidence_factors = self._calculate_confidence_factors(customer_dict, base_prediction)
            feature_contributions = self._get_feature_contributions(customer_dict)
            
            # Create enhanced result
            prediction_id = self._generate_prediction_id(customer_id)
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            result = PredictionResult(
                customer_id=customer_id,
                churn_prediction=base_prediction['churn_prediction'],
                churn_probability=base_prediction['churn_probability'],
                no_churn_probability=base_prediction['no_churn_probability'],
                confidence=base_prediction['confidence'],
                risk_level=base_prediction['risk_level'],
                confidence_factors=confidence_factors,
                feature_contributions=feature_contributions,
                model_version=model_version,
                predicted_at=datetime.now().isoformat(),
                prediction_id=prediction_id,
                processing_time_ms=processing_time
            )
            
            # Cache the result
            if use_cache:
                self.cache.put(customer_dict, model_version, result)
            
            # Update statistics
            self.total_predictions += 1
            self.prediction_history.append(result)
            
            # Keep only recent history (last 1000 predictions)
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            logger.info(f"Single prediction completed for {customer_id}: "
                       f"risk={result.risk_level}, time={processing_time:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Single prediction failed for {customer_id}: {e}")
            raise
    
    async def predict_batch(self, customers_data: List[Union[CustomerBase, Dict[str, Any]]], 
                          use_cache: bool = True, 
                          max_parallel: int = 10) -> BatchPredictionResult:
        """
        Make batch predictions for multiple customers.
        
        Args:
            customers_data: List of customer data
            use_cache: Whether to use prediction caching
            max_parallel: Maximum parallel predictions (for future async implementation)
            
        Returns:
            Batch prediction result
        """
        start_time = time.time()
        batch_id = self._generate_batch_id()
        
        logger.info(f"Starting batch prediction {batch_id} for {len(customers_data)} customers")
        
        predictions = []
        failed_count = 0
        
        for i, customer_data in enumerate(customers_data):
            try:
                result = await self.predict_single(customer_data, use_cache=use_cache)
                predictions.append(result)
                
                # Log progress for large batches
                if (i + 1) % 100 == 0:
                    logger.info(f"Batch {batch_id}: Processed {i + 1}/{len(customers_data)} predictions")
                    
            except Exception as e:
                failed_count += 1
                customer_id = getattr(customer_data, 'customer_id', 'unknown')
                logger.error(f"Batch {batch_id}: Failed prediction for {customer_id}: {e}")
        
        # Calculate batch summary
        total_processing_time = (time.time() - start_time) * 1000
        batch_summary = self._calculate_batch_summary(predictions)
        
        result = BatchPredictionResult(
            batch_id=batch_id,
            predictions=predictions,
            batch_summary=batch_summary,
            total_predictions=len(customers_data),
            successful_predictions=len(predictions),
            failed_predictions=failed_count,
            total_processing_time_ms=total_processing_time,
            processed_at=datetime.now().isoformat()
        )
        
        self.batch_history.append(result)
        
        # Keep only recent batch history (last 100 batches)
        if len(self.batch_history) > 100:
            self.batch_history = self.batch_history[-100:]
        
        logger.info(f"Batch prediction {batch_id} completed: "
                   f"{len(predictions)}/{len(customers_data)} successful, "
                   f"time={total_processing_time:.1f}ms")
        
        return result
    
    def _calculate_confidence_factors(self, customer_data: Dict[str, Any], 
                                    prediction: Dict[str, Any]) -> Dict[str, float]:
        """Calculate factors contributing to prediction confidence."""
        confidence_factors = {}
        
        # Probability spread (higher spread = higher confidence)
        prob_spread = abs(prediction['churn_probability'] - 0.5) * 2
        confidence_factors['probability_spread'] = prob_spread
        
        # Contract type confidence
        contract = customer_data.get('contract', '')
        if contract == 'Two year':
            confidence_factors['contract_stability'] = 0.8
        elif contract == 'One year':
            confidence_factors['contract_stability'] = 0.6
        else:
            confidence_factors['contract_stability'] = 0.3
        
        # Tenure confidence
        tenure = customer_data.get('tenure', 0)
        if tenure > 60:
            confidence_factors['tenure_stability'] = 0.9
        elif tenure > 24:
            confidence_factors['tenure_stability'] = 0.7
        elif tenure > 12:
            confidence_factors['tenure_stability'] = 0.5
        else:
            confidence_factors['tenure_stability'] = 0.2
        
        # Service complexity (more services = higher confidence)
        services_count = sum([
            customer_data.get('phone_service') == 'Yes',
            customer_data.get('internet_service') in ['DSL', 'Fiber optic'],
            customer_data.get('online_security') == 'Yes',
            customer_data.get('online_backup') == 'Yes',
            customer_data.get('device_protection') == 'Yes',
            customer_data.get('tech_support') == 'Yes',
            customer_data.get('streaming_tv') == 'Yes',
            customer_data.get('streaming_movies') == 'Yes'
        ])
        confidence_factors['service_complexity'] = min(services_count / 8.0, 1.0)
        
        return confidence_factors
    
    def _get_feature_contributions(self, customer_data: Dict[str, Any]) -> Dict[str, float]:
        """Get feature contributions to the prediction."""
        # Get feature importance from the model
        feature_importance = ml_service.predictor.get_feature_importance(top_n=15)
        
        # Map customer data to contribution scores
        contributions = {}
        
        # Contract contribution
        contract = customer_data.get('contract', '')
        if 'contract' in feature_importance:
            base_score = feature_importance['contract']
            if contract == 'Month-to-month':
                contributions['contract'] = base_score * 1.0  # High churn risk
            elif contract == 'One year':
                contributions['contract'] = base_score * 0.5  # Medium risk
            else:
                contributions['contract'] = base_score * 0.1  # Low risk
        
        # Tenure contribution
        tenure = customer_data.get('tenure', 0)
        if 'tenure' in feature_importance:
            base_score = feature_importance['tenure']
            # Normalize tenure impact (lower tenure = higher churn risk)
            tenure_factor = max(0, (72 - tenure) / 72)
            contributions['tenure'] = base_score * tenure_factor
        
        # Monthly charges contribution
        monthly_charges = customer_data.get('monthly_charges', 0)
        if 'monthly_charges' in feature_importance:
            base_score = feature_importance['monthly_charges']
            # Higher charges often correlate with churn
            charge_factor = min(monthly_charges / 100.0, 1.0)
            contributions['monthly_charges'] = base_score * charge_factor
        
        # Internet service contribution
        internet_service = customer_data.get('internet_service', '')
        if 'internet_service' in feature_importance:
            base_score = feature_importance['internet_service']
            if internet_service == 'Fiber optic':
                contributions['internet_service'] = base_score * 0.8
            elif internet_service == 'DSL':
                contributions['internet_service'] = base_score * 0.4
            else:
                contributions['internet_service'] = base_score * 0.1
        
        return contributions
    
    def _calculate_batch_summary(self, predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Calculate summary statistics for a batch of predictions."""
        if not predictions:
            return {}
        
        churn_probabilities = [p.churn_probability for p in predictions]
        confidences = [p.confidence for p in predictions]
        processing_times = [p.processing_time_ms for p in predictions]
        
        risk_counts = {}
        for p in predictions:
            risk_counts[p.risk_level] = risk_counts.get(p.risk_level, 0) + 1
        
        return {
            "avg_churn_probability": np.mean(churn_probabilities),
            "median_churn_probability": np.median(churn_probabilities),
            "avg_confidence": np.mean(confidences),
            "avg_processing_time_ms": np.mean(processing_times),
            "risk_distribution": risk_counts,
            "high_risk_customers": len([p for p in predictions if p.risk_level == 'high']),
            "churn_predictions": len([p for p in predictions if p.churn_prediction == 1]),
            "no_churn_predictions": len([p for p in predictions if p.churn_prediction == 0])
        }
    
    def _generate_prediction_id(self, customer_id: str) -> str:
        """Generate unique prediction ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"pred_{customer_id}_{timestamp}_{self.total_predictions}"
    
    def _generate_batch_id(self) -> str:
        """Generate unique batch ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"batch_{timestamp}_{len(self.batch_history)}"
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction service statistics."""
        cache_stats = self.cache.get_stats()
        
        recent_predictions = self.prediction_history[-100:] if self.prediction_history else []
        recent_batches = self.batch_history[-10:] if self.batch_history else []
        
        if recent_predictions:
            avg_processing_time = np.mean([p.processing_time_ms for p in recent_predictions])
            avg_confidence = np.mean([p.confidence for p in recent_predictions])
            
            risk_distribution = {}
            for p in recent_predictions:
                risk_distribution[p.risk_level] = risk_distribution.get(p.risk_level, 0) + 1
        else:
            avg_processing_time = 0
            avg_confidence = 0
            risk_distribution = {}
        
        return {
            "total_predictions": self.total_predictions,
            "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            "cache_stats": cache_stats,
            "recent_performance": {
                "avg_processing_time_ms": avg_processing_time,
                "avg_confidence": avg_confidence,
                "risk_distribution": risk_distribution
            },
            "history_size": {
                "predictions": len(self.prediction_history),
                "batches": len(self.batch_history)
            },
            "recent_batches": len(recent_batches)
        }
    
    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self.cache.clear()
        logger.info("Prediction cache cleared")
    
    def get_recent_predictions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent predictions."""
        recent = self.prediction_history[-limit:] if self.prediction_history else []
        return [p.to_dict() for p in recent]
    
    def get_recent_batches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent batch results."""
        recent = self.batch_history[-limit:] if self.batch_history else []
        return [b.to_dict() for b in recent]


# Global prediction service instance
prediction_service = EnhancedPredictionService()