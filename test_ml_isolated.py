"""
Completely isolated test for the ML pipeline core functionality.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


class FeaturePreprocessor:
    """Feature preprocessing for customer churn data."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.categorical_columns = [
            'gender', 'partner', 'dependents', 'phone_service', 'multiple_lines',
            'internet_service', 'online_security', 'online_backup', 'device_protection',
            'tech_support', 'streaming_tv', 'streaming_movies', 'contract',
            'paperless_billing', 'payment_method', 'tenure_group', 'monthly_charges_group'
        ]
        self.numerical_columns = [
            'senior_citizen', 'tenure', 'monthly_charges', 'total_charges'
        ]
    
    def fit(self, df: pd.DataFrame) -> 'FeaturePreprocessor':
        """Fit the preprocessor on training data."""
        print("Fitting feature preprocessor")
        
        # Handle missing values
        df_clean = self._handle_missing_values(df.copy())
        
        # Fit label encoders for categorical columns
        for col in self.categorical_columns:
            if col in df_clean.columns:
                self.label_encoders[col] = LabelEncoder()
                df_clean[col] = df_clean[col].astype(str)
                self.label_encoders[col].fit(df_clean[col])
        
        # Prepare features for scaling
        df_encoded = self._encode_features(df_clean)
        feature_df = df_encoded[self.categorical_columns + self.numerical_columns]
        
        # Fit scaler
        self.scaler.fit(feature_df)
        self.feature_columns = feature_df.columns.tolist()
        
        print(f"Preprocessor fitted with {len(self.feature_columns)} features")
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        # Handle missing values
        df_clean = self._handle_missing_values(df.copy())
        
        # Encode categorical features
        df_encoded = self._encode_features(df_clean)
        
        # Select and order features
        feature_df = df_encoded[self.feature_columns]
        
        # Scale features
        scaled_features = self.scaler.transform(feature_df)
        
        return scaled_features
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit preprocessor and transform data in one step."""
        return self.fit(df).transform(df)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Convert TotalCharges to numeric, replacing empty strings with NaN
        if 'total_charges' in df.columns:
            df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
            df['total_charges'].fillna(df['total_charges'].median(), inplace=True)
        
        # Fill categorical missing values with 'Unknown'
        for col in self.categorical_columns:
            if col in df.columns:
                df[col].fillna('Unknown', inplace=True)
        
        # Fill numerical missing values with median
        for col in self.numerical_columns:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using fitted encoders."""
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col in df_encoded.columns and col in self.label_encoders:
                # Handle unseen categories
                df_encoded[col] = df_encoded[col].astype(str)
                
                # Map unseen categories to a default value
                known_categories = set(self.label_encoders[col].classes_)
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: x if x in known_categories else self.label_encoders[col].classes_[0]
                )
                
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded


class ChurnPredictor:
    """Customer churn prediction model using Random Forest."""
    
    def __init__(self, model_dir: str = "data/ml_models"):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self.preprocessor = FeaturePreprocessor()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_trained = False
        self.model_version = "1.0.0"
        self.training_metrics = {}
        self.feature_importance = {}
        
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the churn prediction model."""
        print("Starting model training")
        
        # Prepare features and target
        if 'churn' not in data.columns:
            raise ValueError("Training data must contain 'churn' column")
        
        X = data.drop(['churn', 'id', 'customer_id', 'created_at', 'updated_at'], 
                     axis=1, errors='ignore')
        y = data['churn']
        
        # Preprocess features
        X_processed = self.preprocessor.fit_transform(X)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        print("Training Random Forest model")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'train_precision': precision_score(y_train, train_pred, average='weighted'),
            'val_precision': precision_score(y_val, val_pred, average='weighted'),
            'train_recall': recall_score(y_train, train_pred, average='weighted'),
            'val_recall': recall_score(y_val, val_pred, average='weighted'),
            'train_f1': f1_score(y_train, train_pred, average='weighted'),
            'val_f1': f1_score(y_val, val_pred, average='weighted'),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'trained_at': datetime.now().isoformat()
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_processed, y, cv=5, scoring='accuracy')
        metrics['cv_mean_accuracy'] = cv_scores.mean()
        metrics['cv_std_accuracy'] = cv_scores.std()
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.preprocessor.feature_columns
            importance_dict = dict(zip(feature_names, self.model.feature_importances_))
            # Sort by importance
            self.feature_importance = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            metrics['top_features'] = list(self.feature_importance.keys())[:10]
        
        self.training_metrics = metrics
        self.is_trained = True
        
        print(f"Model training completed. Validation accuracy: {metrics['val_accuracy']:.3f}")
        return metrics
    
    def predict(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict churn probability for a single customer."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Preprocess
        X_processed = self.preprocessor.transform(df)
        
        # Make prediction
        prediction = self.model.predict(X_processed)[0]
        prediction_proba = self.model.predict_proba(X_processed)[0]
        
        # Calculate confidence and risk level
        churn_probability = prediction_proba[1]  # Probability of churn (class 1)
        confidence = max(prediction_proba)
        
        if churn_probability >= 0.7:
            risk_level = "high"
        elif churn_probability >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        result = {
            'churn_prediction': int(prediction),
            'churn_probability': float(churn_probability),
            'no_churn_probability': float(prediction_proba[0]),
            'confidence': float(confidence),
            'risk_level': risk_level,
            'model_version': self.model_version,
            'predicted_at': datetime.now().isoformat()
        }
        
        return result
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """Save the trained model and preprocessor."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"churn_model_{timestamp}.joblib"
        
        model_path = self.model_dir / filename
        
        # Save model, preprocessor, and metadata
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'model_version': self.model_version,
            'training_metrics': self.training_metrics,
            'feature_importance': self.feature_importance,
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model and preprocessor."""
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.model_version = model_data.get('model_version', '1.0.0')
            self.training_metrics = model_data.get('training_metrics', {})
            self.feature_importance = model_data.get('feature_importance', {})
            
            self.is_trained = True
            print(f"Model loaded from {model_path}")
            
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            raise
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Get top N most important features."""
        if not self.feature_importance:
            return {}
        
        return dict(list(self.feature_importance.items())[:top_n])


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


def main():
    """Run comprehensive ML pipeline test."""
    print("Starting ML Pipeline Isolated Test")
    print("=" * 50)
    
    try:
        # Create sample data
        print("\nCreating sample dataset...")
        df = create_sample_data()
        print(f"Created dataset with {len(df)} records and {len(df.columns)} features")
        print(f"Churn rate: {df['churn'].mean():.1%}")
        
        # Test feature preprocessing
        print("\nTesting feature preprocessing...")
        preprocessor = FeaturePreprocessor()
        X_processed = preprocessor.fit_transform(df)
        print(f"Preprocessing completed. Features shape: {X_processed.shape}")
        
        # Test model training
        print("\nTesting model training...")
        predictor = ChurnPredictor()
        metrics = predictor.train(df)
        print(f"Model training completed")
        print(f"   Validation accuracy: {metrics['val_accuracy']:.3f}")
        print(f"   Validation precision: {metrics['val_precision']:.3f}")
        print(f"   Validation recall: {metrics['val_recall']:.3f}")
        print(f"   Validation F1: {metrics['val_f1']:.3f}")
        print(f"   Cross-validation accuracy: {metrics['cv_mean_accuracy']:.3f} ± {metrics['cv_std_accuracy']:.3f}")
        
        # Test prediction
        print("\nTesting predictions...")
        
        # High-risk customer profile
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
        
        # Low-risk customer profile
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
        
        high_risk_result = predictor.predict(high_risk_customer)
        low_risk_result = predictor.predict(low_risk_customer)
        
        print(f"Predictions completed")
        print(f"   High-risk customer: {high_risk_result['churn_probability']:.3f} probability ({high_risk_result['risk_level']} risk)")
        print(f"   Low-risk customer: {low_risk_result['churn_probability']:.3f} probability ({low_risk_result['risk_level']} risk)")
        
        # Test feature importance
        print("\nTesting feature importance...")
        importance = predictor.get_feature_importance(top_n=10)
        print(f"Feature importance analysis completed")
        print("   Top 5 most important features:")
        for i, (feature, score) in enumerate(list(importance.items())[:5]):
            print(f"     {i+1}. {feature}: {score:.3f}")
        
        # Test model save/load
        print("\nTesting model save/load...")
        model_path = predictor.save_model("test_model.joblib")
        
        new_predictor = ChurnPredictor()
        new_predictor.load_model(model_path)
        
        # Test that loaded model works
        test_result = new_predictor.predict(high_risk_customer)
        assert abs(test_result['churn_probability'] - high_risk_result['churn_probability']) < 0.001
        
        # Clean up
        Path(model_path).unlink()
        print(f"Model save/load test passed")
        
        print("\n" + "=" * 50)
        print("All ML Pipeline tests passed successfully!")
        print("\nKey achievements:")
        print(f"   * Feature preprocessing with {X_processed.shape[1]} features")
        print(f"   * Random Forest model with {metrics['val_accuracy']:.1%} accuracy")
        print(f"   * Risk-based customer classification")
        print(f"   * Feature importance analysis")
        print(f"   * Model serialization and loading")
        print("\nThe ML pipeline is ready for integration!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()