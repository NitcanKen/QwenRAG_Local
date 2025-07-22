"""
Machine Learning pipeline for customer churn prediction.
"""

import os
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from app.core.logging import get_logger
from app.core.database import get_supabase_client

logger = get_logger(__name__)


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
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training dataframe
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting feature preprocessor")
        
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
        
        logger.info(f"Preprocessor fitted with {len(self.feature_columns)} features")
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: Dataframe to transform
            
        Returns:
            Transformed feature matrix
        """
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
        """
        Train the churn prediction model.
        
        Args:
            data: Training dataframe with churn labels
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info("Starting model training")
        
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
        logger.info("Training Random Forest model")
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
        
        logger.info(f"Model training completed. Validation accuracy: {metrics['val_accuracy']:.3f}")
        return metrics
    
    def predict(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict churn probability for a single customer.
        
        Args:
            customer_data: Customer features dictionary
            
        Returns:
            Dictionary containing prediction results
        """
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
    
    def predict_batch(self, customers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict churn for multiple customers.
        
        Args:
            customers_data: List of customer feature dictionaries
            
        Returns:
            List of prediction result dictionaries
        """
        return [self.predict(customer) for customer in customers_data]
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test dataframe with churn labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        X = test_data.drop(['churn', 'id', 'customer_id', 'created_at', 'updated_at'], 
                          axis=1, errors='ignore')
        y = test_data['churn']
        
        # Preprocess and predict
        X_processed = self.preprocessor.transform(X)
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted'),
            'recall': recall_score(y, predictions, average='weighted'),
            'f1_score': f1_score(y, predictions, average='weighted'),
            'test_samples': len(X),
            'evaluated_at': datetime.now().isoformat()
        }
        
        # Classification report
        report = classification_report(y, predictions, output_dict=True)
        metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save the trained model and preprocessor.
        
        Args:
            filename: Optional filename, defaults to timestamped name
            
        Returns:
            Path to saved model file
        """
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
        logger.info(f"Model saved to {model_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model and preprocessor.
        
        Args:
            model_path: Path to saved model file
        """
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.model_version = model_data.get('model_version', '1.0.0')
            self.training_metrics = model_data.get('training_metrics', {})
            self.feature_importance = model_data.get('feature_importance', {})
            
            self.is_trained = True
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and importance scores
        """
        if not self.feature_importance:
            return {}
        
        return dict(list(self.feature_importance.items())[:top_n])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status."""
        return {
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'feature_count': len(self.preprocessor.feature_columns) if self.preprocessor.feature_columns else 0,
            'model_type': 'RandomForestClassifier',
            'parameters': self.model.get_params() if self.is_trained else {}
        }


class MLPipelineService:
    """Service for managing the ML pipeline."""
    
    def __init__(self):
        self.predictor = ChurnPredictor()
        self.supabase = None  # Initialize lazily
        
    def _ensure_supabase(self):
        """Ensure Supabase client is initialized."""
        if self.supabase is None:
            try:
                self.supabase = get_supabase_client()
            except Exception:
                logger.warning("Supabase not available, using fallback data")
                self.supabase = None
    
    def _get_sample_training_data(self) -> pd.DataFrame:
        """Generate sample training data when database is not available."""
        import numpy as np
        
        # Generate sample data similar to telco dataset
        n_samples = 1000
        np.random.seed(42)
        
        data = {
            'customer_id': [f'C{i:04d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
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
            'total_charges': np.random.uniform(18.0, 8500.0, n_samples),
            'churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])  # 27% churn rate
        }
        
        return pd.DataFrame(data)
        
    async def load_training_data(self) -> pd.DataFrame:
        """Load training data from Supabase."""
        logger.info("Loading training data from Supabase")
        
        # Ensure Supabase is initialized
        self._ensure_supabase()
        
        if self.supabase is None:
            # Fallback to sample data if Supabase not available
            logger.warning("Using fallback sample data for ML training")
            return self._get_sample_training_data()
        
        try:
            # Query all customer data
            response = self.supabase.table('customers').select('*').execute()
            
            if not response.data:
                raise ValueError("No training data found in database")
            
            df = pd.DataFrame(response.data)
            logger.info(f"Loaded {len(df)} records for training")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise
    
    async def train_model(self) -> Dict[str, Any]:
        """Train the churn prediction model with latest data."""
        try:
            # Load data
            data = await self.load_training_data()
            
            # Train model
            metrics = self.predictor.train(data)
            
            # Save model
            model_path = self.predictor.save_model()
            metrics['model_path'] = model_path
            
            logger.info("Model training completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    async def predict_customer_churn(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict churn for a single customer."""
        try:
            return self.predictor.predict(customer_data)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and metrics."""
        model_info = self.predictor.get_model_info()
        feature_importance = self.predictor.get_feature_importance(top_n=10)
        
        return {
            **model_info,
            'top_features': feature_importance,
            'status': 'healthy' if self.predictor.is_trained else 'not_trained'
        }
    
    async def load_latest_model(self) -> bool:
        """Load the latest trained model from disk."""
        try:
            model_files = list(self.predictor.model_dir.glob("churn_model_*.joblib"))
            
            if not model_files:
                logger.warning("No saved models found")
                return False
            
            # Load the most recent model
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            self.predictor.load_model(str(latest_model))
            
            logger.info(f"Loaded latest model: {latest_model.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load latest model: {e}")
            return False


# Global instance
ml_service = MLPipelineService()