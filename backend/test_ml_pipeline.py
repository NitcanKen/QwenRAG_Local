"""
Test the ML pipeline implementation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.ml_pipeline import ChurnPredictor, FeaturePreprocessor, MLPipelineService


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
    churn_prob += (data['contract'] == 'Month-to-month') * 0.3
    
    # Higher churn for fiber optic customers (as per real telco data)
    churn_prob += (data['internet_service'] == 'Fiber optic') * 0.2
    
    # Higher churn for senior citizens
    churn_prob += data['senior_citizen'] * 0.1
    
    # Lower churn for customers with partners
    churn_prob -= (data['partner'] == 'Yes') * 0.1
    
    # Generate churn labels
    data['churn'] = np.random.binomial(1, np.clip(churn_prob, 0, 1), n_samples)
    
    return pd.DataFrame(data)


class TestFeaturePreprocessor:
    """Test the FeaturePreprocessor class."""
    
    def test_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline."""
        # Create sample data
        df = create_sample_data()
        
        # Initialize preprocessor
        preprocessor = FeaturePreprocessor()
        
        # Fit and transform
        X_processed = preprocessor.fit_transform(df)
        
        # Check output shape
        assert X_processed.shape[0] == len(df)
        assert X_processed.shape[1] > 0
        
        # Check that all values are numeric
        assert np.isfinite(X_processed).all()
        
        print(f"✅ Preprocessing pipeline test passed. Output shape: {X_processed.shape}")
    
    def test_missing_value_handling(self):
        """Test handling of missing values."""
        # Create data with missing values
        df = create_sample_data().head(100)
        df.loc[10:20, 'total_charges'] = None
        df.loc[30:40, 'gender'] = None
        
        # Initialize and fit preprocessor
        preprocessor = FeaturePreprocessor()
        X_processed = preprocessor.fit_transform(df)
        
        # Check no NaN values remain
        assert not np.isnan(X_processed).any()
        
        print("✅ Missing value handling test passed")
    
    def test_unseen_categories(self):
        """Test handling of unseen categorical values."""
        # Create training data
        train_df = create_sample_data().head(500)
        
        # Create test data with unseen category
        test_df = train_df.head(50).copy()
        test_df.loc[0, 'gender'] = 'Other'  # Unseen category
        
        # Fit on training data
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(train_df)
        
        # Transform test data (should not fail)
        X_test = preprocessor.transform(test_df)
        
        # Check output is valid
        assert X_test.shape[0] == len(test_df)
        assert np.isfinite(X_test).all()
        
        print("✅ Unseen categories handling test passed")


class TestChurnPredictor:
    """Test the ChurnPredictor class."""
    
    def test_model_training(self):
        """Test model training process."""
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
        
        print(f"✅ Model training test passed. Validation accuracy: {metrics['val_accuracy']:.3f}")
        
        return predictor
    
    def test_single_prediction(self):
        """Test single customer prediction."""
        # Train model first
        predictor = self.test_model_training()
        
        # Create sample customer
        customer = {
            'customer_id': 'TEST_001',
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
        
        # Make prediction
        result = predictor.predict(customer)
        
        # Check prediction format
        assert 'churn_prediction' in result
        assert 'churn_probability' in result
        assert 'confidence' in result
        assert 'risk_level' in result
        
        # Check value ranges
        assert result['churn_prediction'] in [0, 1]
        assert 0 <= result['churn_probability'] <= 1
        assert 0 <= result['confidence'] <= 1
        assert result['risk_level'] in ['low', 'medium', 'high']
        
        print(f"✅ Single prediction test passed. Churn probability: {result['churn_probability']:.3f}")
    
    def test_batch_prediction(self):
        """Test batch prediction."""
        # Train model first
        predictor = self.test_model_training()
        
        # Create sample customers
        customers = []
        for i in range(5):
            customer = {
                'customer_id': f'TEST_{i:03d}',
                'gender': 'Male',
                'senior_citizen': 0,
                'partner': 'Yes',
                'dependents': 'No',
                'tenure': 12 + i,
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
                'monthly_charges': 65.0 + i * 5,
                'total_charges': 780.0 + i * 100,
                'tenure_group': '1-12',
                'monthly_charges_group': 'Medium'
            }
            customers.append(customer)
        
        # Make batch prediction
        results = predictor.predict_batch(customers)
        
        # Check results
        assert len(results) == len(customers)
        
        for result in results:
            assert 'churn_prediction' in result
            assert 'churn_probability' in result
            assert 'confidence' in result
            assert 'risk_level' in result
        
        print(f"✅ Batch prediction test passed. Processed {len(results)} customers")
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        # Create separate training and test data
        train_df = create_sample_data()
        test_df = create_sample_data().head(200)  # Smaller test set
        
        # Train model
        predictor = ChurnPredictor()
        predictor.train(train_df)
        
        # Evaluate model
        eval_metrics = predictor.evaluate(test_df)
        
        # Check evaluation metrics
        assert 'accuracy' in eval_metrics
        assert 'precision' in eval_metrics
        assert 'recall' in eval_metrics
        assert 'f1_score' in eval_metrics
        
        # Check metric ranges
        assert 0 <= eval_metrics['accuracy'] <= 1
        assert 0 <= eval_metrics['precision'] <= 1
        assert 0 <= eval_metrics['recall'] <= 1
        assert 0 <= eval_metrics['f1_score'] <= 1
        
        print(f"✅ Model evaluation test passed. Test accuracy: {eval_metrics['accuracy']:.3f}")
    
    def test_model_save_load(self):
        """Test model saving and loading."""
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
        
        # Clean up
        Path(model_path).unlink()
        
        print("✅ Model save/load test passed")
    
    def test_feature_importance(self):
        """Test feature importance functionality."""
        # Train model
        predictor = ChurnPredictor()
        df = create_sample_data()
        predictor.train(df)
        
        # Get feature importance
        importance = predictor.get_feature_importance(top_n=5)
        
        # Check format
        assert isinstance(importance, dict)
        assert len(importance) <= 5
        
        # Check values are between 0 and 1
        for feature, score in importance.items():
            assert 0 <= score <= 1
        
        print(f"✅ Feature importance test passed. Top features: {list(importance.keys())[:3]}")


def test_integration():
    """Test integration of all components."""
    print("\n🧪 Running ML Pipeline Integration Test")
    
    # Create sample data
    df = create_sample_data()
    print(f"Created sample dataset with {len(df)} records")
    
    # Test preprocessing
    print("Testing feature preprocessing...")
    preprocessor = FeaturePreprocessor()
    X_processed = preprocessor.fit_transform(df)
    print(f"✅ Preprocessing completed. Features shape: {X_processed.shape}")
    
    # Test model training
    print("Testing model training...")
    predictor = ChurnPredictor()
    metrics = predictor.train(df)
    print(f"✅ Model training completed. Accuracy: {metrics['val_accuracy']:.3f}")
    
    # Test prediction
    print("Testing prediction...")
    sample_customer = {
        'customer_id': 'INTEGRATION_TEST',
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
    
    result = predictor.predict(sample_customer)
    print(f"✅ Prediction completed. Churn probability: {result['churn_probability']:.3f}, Risk: {result['risk_level']}")
    
    # Test feature importance
    print("Testing feature importance...")
    importance = predictor.get_feature_importance(top_n=5)
    print(f"✅ Feature importance: {list(importance.keys())}")
    
    print("\n🎉 All ML Pipeline tests passed successfully!")


if __name__ == "__main__":
    test_integration()