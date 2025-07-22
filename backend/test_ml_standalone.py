"""
Standalone test for the ML pipeline without backend dependencies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import logging

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the app directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

# Mock the logger and database dependencies
class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")

class MockSupabaseClient:
    def table(self, name):
        return self
    
    def select(self, columns):
        return self
    
    def execute(self):
        # Return mock data
        data = create_sample_data().to_dict('records')
        return type('Response', (), {'data': data})()

# Create mock modules
sys.modules['app.core.logging'] = type('Module', (), {'get_logger': lambda name: MockLogger()})()
sys.modules['app.core.database'] = type('Module', (), {'get_supabase_client': lambda: MockSupabaseClient()})()

# Now import the ML pipeline
from app.services.ml_pipeline import ChurnPredictor, FeaturePreprocessor


def create_sample_data():
    """Create sample customer data for testing."""
    np.random.seed(42)
    
    # Create synthetic data
    n_samples = 1000
    
    data = {
        'customer_id': [f'CUST_{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'partner': np.random.choice(['Yes', 'No'], n_samples),
        'dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(1, 73, n_samples),
        'phone_service': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'monthly_charges': np.random.uniform(18.0, 120.0, n_samples),
        'total_charges': np.random.uniform(18.0, 8000.0, n_samples),
        'tenure_group': np.random.choice(['1-12', '13-24', '25-36', '37-48', '49-60', '61-72'], n_samples),
        'monthly_charges_group': np.random.choice(['Low', 'Medium', 'High'], n_samples)
    }
    
    # Create realistic churn labels based on some patterns
    churn_prob = np.zeros(n_samples)
    
    # Higher churn for month-to-month contracts
    churn_prob += (np.array(data['contract']) == 'Month-to-month') * 0.3
    
    # Higher churn for fiber optic customers (as per real telco data)
    churn_prob += (np.array(data['internet_service']) == 'Fiber optic') * 0.2
    
    # Higher churn for senior citizens
    churn_prob += np.array(data['senior_citizen']) * 0.1
    
    # Lower churn for customers with partners
    churn_prob -= (np.array(data['partner']) == 'Yes') * 0.1
    
    # Generate churn labels
    data['churn'] = np.random.binomial(1, np.clip(churn_prob, 0, 1), n_samples)
    
    return pd.DataFrame(data)


def test_feature_preprocessing():
    """Test the feature preprocessing."""
    print("\n🧪 Testing Feature Preprocessing")
    
    # Create sample data
    df = create_sample_data()
    print(f"Created sample dataset with {len(df)} records")
    
    # Initialize preprocessor
    preprocessor = FeaturePreprocessor()
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(df)
    
    # Check output shape
    assert X_processed.shape[0] == len(df)
    assert X_processed.shape[1] > 0
    
    # Check that all values are numeric
    assert np.isfinite(X_processed).all()
    
    print(f"✅ Preprocessing test passed. Output shape: {X_processed.shape}")
    return True


def test_model_training():
    """Test model training."""
    print("\n🧪 Testing Model Training")
    
    # Create sample data
    df = create_sample_data()
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Train model
    metrics = predictor.train(df)
    
    # Check that training completed
    assert predictor.is_trained
    
    # Check metrics are reasonable
    assert 0 <= metrics['val_accuracy'] <= 1
    assert 0 <= metrics['val_precision'] <= 1
    assert 0 <= metrics['val_recall'] <= 1
    assert 0 <= metrics['val_f1'] <= 1
    
    print(f"✅ Model training test passed")
    print(f"   Validation accuracy: {metrics['val_accuracy']:.3f}")
    print(f"   Validation precision: {metrics['val_precision']:.3f}")
    print(f"   Validation recall: {metrics['val_recall']:.3f}")
    print(f"   Validation F1: {metrics['val_f1']:.3f}")
    
    return predictor


def test_prediction():
    """Test model prediction."""
    print("\n🧪 Testing Model Prediction")
    
    # Train model first
    predictor = test_model_training()
    
    # Create sample customer (high churn risk profile)
    high_risk_customer = {
        'customer_id': 'TEST_HIGH_RISK',
        'gender': 'Female',
        'senior_citizen': 1,
        'partner': 'No',
        'dependents': 'No',
        'tenure': 2,
        'phone_service': 'Yes',
        'multiple_lines': 'No',
        'internet_service': 'Fiber optic',
        'online_security': 'No',
        'online_backup': 'No',
        'device_protection': 'No',
        'tech_support': 'No',
        'streaming_tv': 'Yes',
        'streaming_movies': 'Yes',
        'contract': 'Month-to-month',
        'paperless_billing': 'Yes',
        'payment_method': 'Electronic check',
        'monthly_charges': 95.0,
        'total_charges': 190.0,
        'tenure_group': '1-12',
        'monthly_charges_group': 'High'
    }
    
    # Create sample customer (low churn risk profile)
    low_risk_customer = {
        'customer_id': 'TEST_LOW_RISK',
        'gender': 'Male',
        'senior_citizen': 0,
        'partner': 'Yes',
        'dependents': 'Yes',
        'tenure': 60,
        'phone_service': 'Yes',
        'multiple_lines': 'Yes',
        'internet_service': 'DSL',
        'online_security': 'Yes',
        'online_backup': 'Yes',
        'device_protection': 'Yes',
        'tech_support': 'Yes',
        'streaming_tv': 'No',
        'streaming_movies': 'No',
        'contract': 'Two year',
        'paperless_billing': 'No',
        'payment_method': 'Credit card (automatic)',
        'monthly_charges': 45.0,
        'total_charges': 2700.0,
        'tenure_group': '49-60',
        'monthly_charges_group': 'Low'
    }
    
    # Make predictions
    high_risk_result = predictor.predict(high_risk_customer)
    low_risk_result = predictor.predict(low_risk_customer)
    
    # Check prediction format
    for result in [high_risk_result, low_risk_result]:
        assert 'churn_prediction' in result
        assert 'churn_probability' in result
        assert 'confidence' in result
        assert 'risk_level' in result
        
        # Check value ranges
        assert result['churn_prediction'] in [0, 1]
        assert 0 <= result['churn_probability'] <= 1
        assert 0 <= result['confidence'] <= 1
        assert result['risk_level'] in ['low', 'medium', 'high']
    
    print(f"✅ Prediction test passed")
    print(f"   High-risk customer churn probability: {high_risk_result['churn_probability']:.3f} ({high_risk_result['risk_level']} risk)")
    print(f"   Low-risk customer churn probability: {low_risk_result['churn_probability']:.3f} ({low_risk_result['risk_level']} risk)")
    
    return True


def test_feature_importance():
    """Test feature importance."""
    print("\n🧪 Testing Feature Importance")
    
    # Train model
    predictor = test_model_training()
    
    # Get feature importance
    importance = predictor.get_feature_importance(top_n=10)
    
    # Check format
    assert isinstance(importance, dict)
    assert len(importance) <= 10
    
    # Check values are between 0 and 1
    for feature, score in importance.items():
        assert 0 <= score <= 1
    
    print(f"✅ Feature importance test passed")
    print("   Top 5 most important features:")
    for i, (feature, score) in enumerate(list(importance.items())[:5]):
        print(f"     {i+1}. {feature}: {score:.3f}")
    
    return True


def test_model_save_load():
    """Test model save and load."""
    print("\n🧪 Testing Model Save/Load")
    
    # Train model
    predictor = ChurnPredictor()
    df = create_sample_data()
    predictor.train(df)
    
    # Save model
    model_path = predictor.save_model("test_model.joblib")
    assert Path(model_path).exists()
    
    # Create new predictor and load model
    new_predictor = ChurnPredictor()
    new_predictor.load_model(model_path)
    
    # Check that model was loaded
    assert new_predictor.is_trained
    assert new_predictor.model_version == predictor.model_version
    
    # Test that loaded model can make predictions
    test_customer = {
        'customer_id': 'LOAD_TEST',
        'gender': 'Male',
        'senior_citizen': 0,
        'partner': 'Yes',
        'dependents': 'No',
        'tenure': 12,
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
        'total_charges': 780.0,
        'tenure_group': '1-12',
        'monthly_charges_group': 'Medium'
    }
    
    result = new_predictor.predict(test_customer)
    assert 'churn_probability' in result
    
    # Clean up
    Path(model_path).unlink()
    
    print(f"✅ Model save/load test passed")
    return True


def main():
    """Run all tests."""
    print("🚀 Starting ML Pipeline Standalone Tests")
    print("=" * 50)
    
    try:
        # Run all tests
        test_feature_preprocessing()
        test_model_training()
        test_prediction()
        test_feature_importance()
        test_model_save_load()
        
        print("\n" + "=" * 50)
        print("🎉 All ML Pipeline tests passed successfully!")
        print("\nThe ML pipeline is ready for Stage 3.1 completion.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()