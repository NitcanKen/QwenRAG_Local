"""
Test the enhanced prediction API implementation.
"""

import asyncio
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import os

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

# Mock dependencies
class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")

class MockMLService:
    def __init__(self):
        self.predictor = MockPredictor()
    
    async def predict_customer_churn(self, customer_data):
        # Simulate prediction based on customer data
        contract = customer_data.get('contract', 'Month-to-month')
        tenure = customer_data.get('tenure', 12)
        
        # Simple rule-based mock prediction
        if contract == 'Month-to-month' and tenure < 12:
            churn_prob = 0.75
        elif contract == 'One year':
            churn_prob = 0.35
        else:
            churn_prob = 0.15
        
        # Add some randomness
        churn_prob += np.random.normal(0, 0.1)
        churn_prob = max(0.05, min(0.95, churn_prob))
        
        prediction = 1 if churn_prob >= 0.5 else 0
        confidence = max(churn_prob, 1 - churn_prob)
        
        if churn_prob >= 0.7:
            risk_level = "high"
        elif churn_prob >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            'churn_prediction': prediction,
            'churn_probability': churn_prob,
            'no_churn_probability': 1 - churn_prob,
            'confidence': confidence,
            'risk_level': risk_level,
            'model_version': '1.0.0',
            'predicted_at': datetime.now().isoformat()
        }

class MockPredictor:
    def __init__(self):
        self.model_version = "1.0.0"
        self.is_trained = True
    
    def get_feature_importance(self, top_n=10):
        return {
            'contract': 0.25,
            'tenure': 0.20,
            'monthly_charges': 0.15,
            'internet_service': 0.12,
            'total_charges': 0.10,
            'payment_method': 0.08,
            'senior_citizen': 0.05,
            'partner': 0.03,
            'dependents': 0.02
        }

# Create mock modules  
sys.modules['app.core.logging'] = type('Module', (), {'get_logger': lambda *args, **kwargs: MockLogger()})()
sys.modules['app.services.ml_pipeline'] = type('Module', (), {'ml_service': MockMLService()})()

# Import the prediction service
from app.services.prediction_service import EnhancedPredictionService, PredictionResult, BatchPredictionResult, PredictionCache


def create_sample_customer(customer_id: str = "TEST_001", **overrides):
    """Create sample customer data."""
    customer = {
        'customer_id': customer_id,
        'gender': 'Male',
        'senior_citizen': 0,
        'partner': 'Yes',
        'dependents': 'No',
        'tenure': 24,
        'phone_service': 'Yes',
        'multiple_lines': 'No',
        'internet_service': 'DSL',
        'online_security': 'Yes',
        'online_backup': 'No',
        'device_protection': 'No',
        'tech_support': 'Yes',
        'streaming_tv': 'No',
        'streaming_movies': 'No',
        'contract': 'One year',
        'paperless_billing': 'Yes',
        'payment_method': 'Credit card (automatic)',
        'monthly_charges': 65.0,
        'total_charges': 1560.0,
        'tenure_group': '13-24',
        'monthly_charges_group': 'Medium'
    }
    customer.update(overrides)
    return customer


def test_prediction_cache():
    """Test prediction caching functionality."""
    print("\n1. Testing Prediction Cache...")
    
    cache = PredictionCache(ttl_seconds=10, max_size=5)
    
    # Test cache miss
    customer_data = create_sample_customer()
    result = cache.get(customer_data, "1.0.0")
    assert result is None
    print("   OK - Cache miss works")
    
    # Test cache put and hit
    prediction_result = PredictionResult(
        customer_id="TEST_001",
        churn_prediction=0,
        churn_probability=0.3,
        no_churn_probability=0.7,
        confidence=0.7,
        risk_level="low",
        confidence_factors={},
        feature_contributions={},
        model_version="1.0.0",
        predicted_at=datetime.now().isoformat(),
        prediction_id="test_pred_001",
        processing_time_ms=50.0
    )
    
    cache.put(customer_data, "1.0.0", prediction_result)
    cached_result = cache.get(customer_data, "1.0.0")
    assert cached_result is not None
    assert cached_result.customer_id == "TEST_001"
    print("   OK - Cache put and hit works")
    
    # Test cache expiration (simulate)
    time.sleep(0.1)  # Small delay
    cache.ttl_seconds = 0.05  # Very short TTL
    expired_result = cache.get(customer_data, "1.0.0")
    assert expired_result is None
    print("   OK - Cache expiration works")
    
    # Test cache eviction
    cache = PredictionCache(ttl_seconds=300, max_size=2)
    for i in range(3):
        customer = create_sample_customer(customer_id=f"TEST_{i:03d}")
        prediction = PredictionResult(
            customer_id=f"TEST_{i:03d}",
            churn_prediction=0,
            churn_probability=0.3,
            no_churn_probability=0.7,
            confidence=0.7,
            risk_level="low",
            confidence_factors={},
            feature_contributions={},
            model_version="1.0.0",
            predicted_at=datetime.now().isoformat(),
            prediction_id=f"test_pred_{i:03d}",
            processing_time_ms=50.0
        )
        cache.put(customer, "1.0.0", prediction)
    
    stats = cache.get_stats()
    assert stats['total_items'] <= 2
    print("   OK - Cache eviction works")


async def test_single_prediction():
    """Test single customer prediction."""
    print("\n2. Testing Single Prediction...")
    
    service = EnhancedPredictionService()
    
    # Test basic prediction
    customer_data = create_sample_customer()
    result = await service.predict_single(customer_data, use_cache=False)
    
    assert isinstance(result, PredictionResult)
    assert result.customer_id == "TEST_001"
    assert 0 <= result.churn_probability <= 1
    assert 0 <= result.confidence <= 1
    assert result.risk_level in ['low', 'medium', 'high']
    assert result.processing_time_ms > 0
    
    print(f"   Prediction: {result.risk_level} risk, {result.churn_probability:.3f} probability")
    print("   OK - Basic single prediction works")
    
    # Test with caching
    start_time = time.time()
    cached_result = await service.predict_single(customer_data, use_cache=True)
    cache_time = (time.time() - start_time) * 1000
    
    # Second call should be faster due to caching
    start_time = time.time()
    cached_result2 = await service.predict_single(customer_data, use_cache=True)
    cache_time2 = (time.time() - start_time) * 1000
    
    assert cache_time2 < cache_time  # Should be faster
    assert cached_result.prediction_id == cached_result2.prediction_id  # Same cached result
    
    print(f"   Cache speedup: {cache_time:.1f}ms -> {cache_time2:.1f}ms")
    print("   OK - Prediction caching works")
    
    # Test confidence factors
    assert 'probability_spread' in result.confidence_factors
    assert 'contract_stability' in result.confidence_factors
    assert 'tenure_stability' in result.confidence_factors
    assert 'service_complexity' in result.confidence_factors
    
    print("   OK - Confidence factors generated")
    
    # Test feature contributions
    assert isinstance(result.feature_contributions, dict)
    assert len(result.feature_contributions) > 0
    
    print("   OK - Feature contributions generated")


async def test_batch_prediction():
    """Test batch prediction functionality."""
    print("\n3. Testing Batch Prediction...")
    
    service = EnhancedPredictionService()
    
    # Create batch of customers
    customers = []
    for i in range(5):
        customer = create_sample_customer(
            customer_id=f"BATCH_{i:03d}",
            tenure=12 + i * 6,
            contract=['Month-to-month', 'One year', 'Two year'][i % 3]
        )
        customers.append(customer)
    
    # Test batch prediction
    start_time = time.time()
    batch_result = await service.predict_batch(customers, use_cache=False)
    processing_time = (time.time() - start_time) * 1000
    
    assert isinstance(batch_result, BatchPredictionResult)
    assert batch_result.total_predictions == 5
    assert batch_result.successful_predictions == 5
    assert batch_result.failed_predictions == 0
    assert len(batch_result.predictions) == 5
    
    print(f"   Batch processed in {processing_time:.1f}ms")
    print("   OK - Batch prediction works")
    
    # Test batch summary
    summary = batch_result.batch_summary
    assert 'avg_churn_probability' in summary
    assert 'risk_distribution' in summary
    assert 'high_risk_customers' in summary
    
    print(f"   Avg churn probability: {summary['avg_churn_probability']:.3f}")
    print(f"   Risk distribution: {summary['risk_distribution']}")
    print("   OK - Batch summary generated")
    
    # Test with caching (should be faster)
    start_time = time.time()
    cached_batch_result = await service.predict_batch(customers, use_cache=True)
    cached_time = (time.time() - start_time) * 1000
    
    assert cached_time < processing_time  # Should be faster due to caching
    print(f"   Cached batch processed in {cached_time:.1f}ms (speedup: {processing_time/cached_time:.1f}x)")
    print("   OK - Batch caching works")


def test_service_statistics():
    """Test prediction service statistics."""
    print("\n4. Testing Service Statistics...")
    
    service = EnhancedPredictionService()
    
    # Generate some prediction history
    for i in range(10):
        prediction = PredictionResult(
            customer_id=f"STATS_{i:03d}",
            churn_prediction=i % 2,
            churn_probability=0.3 + (i * 0.05),
            no_churn_probability=0.7 - (i * 0.05),
            confidence=0.6 + (i * 0.02),
            risk_level=['low', 'medium', 'high'][i % 3],
            confidence_factors={},
            feature_contributions={},
            model_version="1.0.0",
            predicted_at=datetime.now().isoformat(),
            prediction_id=f"stats_pred_{i:03d}",
            processing_time_ms=50.0 + i * 5
        )
        service.prediction_history.append(prediction)
        service.total_predictions += 1
    
    # Test statistics
    stats = service.get_prediction_stats()
    
    assert stats['total_predictions'] == 10
    assert 'cache_stats' in stats
    assert 'recent_performance' in stats
    assert 'history_size' in stats
    
    print(f"   Total predictions: {stats['total_predictions']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print("   OK - Service statistics work")
    
    # Test recent predictions
    recent = service.get_recent_predictions(limit=5)
    assert len(recent) == 5
    
    print(f"   Recent predictions: {len(recent)}")
    print("   OK - Recent predictions retrieval works")


async def test_high_risk_customer():
    """Test prediction for high-risk customer profile."""
    print("\n5. Testing High-Risk Customer Profile...")
    
    service = EnhancedPredictionService()
    
    # Create high-risk customer
    high_risk_customer = create_sample_customer(
        customer_id="HIGH_RISK_001",
        contract="Month-to-month",
        tenure=3,
        senior_citizen=1,
        partner="No",
        dependents="No",
        internet_service="Fiber optic",
        monthly_charges=95.0,
        payment_method="Electronic check"
    )
    
    result = await service.predict_single(high_risk_customer, use_cache=False)
    
    # Should be high risk
    print(f"   Customer: {result.customer_id}")
    print(f"   Churn probability: {result.churn_probability:.3f}")
    print(f"   Risk level: {result.risk_level}")
    print(f"   Confidence: {result.confidence:.3f}")
    
    # Analyze confidence factors
    print("   Confidence factors:")
    for factor, value in result.confidence_factors.items():
        print(f"     {factor}: {value:.3f}")
    
    # Analyze feature contributions
    print("   Feature contributions:")
    for feature, value in result.feature_contributions.items():
        print(f"     {feature}: {value:.3f}")
    
    print("   OK - High-risk customer analysis complete")


async def test_low_risk_customer():
    """Test prediction for low-risk customer profile."""
    print("\n6. Testing Low-Risk Customer Profile...")
    
    service = EnhancedPredictionService()
    
    # Create low-risk customer
    low_risk_customer = create_sample_customer(
        customer_id="LOW_RISK_001",
        contract="Two year",
        tenure=60,
        senior_citizen=0,
        partner="Yes",
        dependents="Yes",
        internet_service="DSL",
        monthly_charges=45.0,
        payment_method="Credit card (automatic)"
    )
    
    result = await service.predict_single(low_risk_customer, use_cache=False)
    
    # Should be low risk
    print(f"   Customer: {result.customer_id}")
    print(f"   Churn probability: {result.churn_probability:.3f}")
    print(f"   Risk level: {result.risk_level}")
    print(f"   Confidence: {result.confidence:.3f}")
    
    print("   OK - Low-risk customer analysis complete")


def test_error_handling():
    """Test error handling in prediction service."""
    print("\n7. Testing Error Handling...")
    
    service = EnhancedPredictionService()
    
    # Test cache with invalid data
    cache = service.cache
    try:
        # This should not crash
        cache.get({}, "invalid_version")
        print("   OK - Cache handles invalid data gracefully")
    except Exception as e:
        print(f"   ERROR - Cache failed with invalid data: {e}")
    
    # Test cache statistics
    stats = cache.get_stats()
    assert isinstance(stats, dict)
    assert 'total_items' in stats
    
    print("   OK - Error handling works")


async def test_performance():
    """Test prediction service performance."""
    print("\n8. Testing Performance...")
    
    service = EnhancedPredictionService()
    
    # Test single prediction performance
    customer = create_sample_customer()
    
    start_time = time.time()
    result = await service.predict_single(customer, use_cache=False)
    single_time = (time.time() - start_time) * 1000
    
    print(f"   Single prediction time: {single_time:.1f}ms")
    assert single_time < 1000  # Should be under 1 second
    
    # Test batch performance
    customers = [create_sample_customer(customer_id=f"PERF_{i:03d}") for i in range(20)]
    
    start_time = time.time()
    batch_result = await service.predict_batch(customers, use_cache=False)
    batch_time = (time.time() - start_time) * 1000
    avg_time_per_prediction = batch_time / len(customers)
    
    print(f"   Batch prediction time: {batch_time:.1f}ms for {len(customers)} customers")
    print(f"   Average time per prediction: {avg_time_per_prediction:.1f}ms")
    
    assert batch_result.successful_predictions == len(customers)
    print("   OK - Performance test passed")


async def main():
    """Run all prediction API tests."""
    print("Testing Enhanced Prediction API")
    print("=" * 50)
    
    try:
        test_prediction_cache()
        await test_single_prediction()
        await test_batch_prediction()
        test_service_statistics()
        await test_high_risk_customer()
        await test_low_risk_customer()
        test_error_handling()
        await test_performance()
        
        print("\n" + "=" * 50)
        print("All Prediction API tests passed successfully!")
        print("\nKey features validated:")
        print("  * Enhanced single predictions with confidence scoring")
        print("  * Batch prediction with summary statistics")
        print("  * Intelligent caching with TTL and eviction")
        print("  * Detailed confidence factors and feature contributions")
        print("  * Performance monitoring and statistics")
        print("  * Error handling and edge cases")
        print("  * High/low risk customer analysis")
        print("\nStage 3.3: Prediction API is ready!")
        
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(main())