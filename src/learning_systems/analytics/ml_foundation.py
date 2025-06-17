"""
ALL-USE Learning Systems - Machine Learning Foundation

This module implements the basic machine learning foundation for the ALL-USE Learning Systems,
providing fundamental ML capabilities for pattern recognition, prediction, and optimization.

The ML foundation is designed to:
- Provide basic machine learning algorithms
- Support supervised and unsupervised learning
- Enable model training and evaluation
- Support feature engineering and selection
- Provide model persistence and versioning

Classes:
- MLFoundation: Core machine learning foundation
- ModelTrainer: Trains machine learning models
- ModelEvaluator: Evaluates model performance
- FeatureEngineer: Handles feature engineering
- ModelRegistry: Manages model versions and persistence

Version: 1.0.0
"""

import time
import logging
import threading
import pickle
import json
import os
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import statistics
from collections import defaultdict
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of machine learning models."""
    LINEAR_REGRESSION = 1
    LOGISTIC_REGRESSION = 2
    DECISION_TREE = 3
    RANDOM_FOREST = 4
    NEURAL_NETWORK = 5
    CLUSTERING = 6
    ANOMALY_DETECTION = 7
    TIME_SERIES = 8

class LearningType(Enum):
    """Types of machine learning."""
    SUPERVISED = 1
    UNSUPERVISED = 2
    REINFORCEMENT = 3
    SEMI_SUPERVISED = 4

@dataclass
class MLConfig:
    """Configuration for machine learning operations."""
    model_storage_path: str = "/tmp/ml_models"
    max_training_time: int = 3600  # seconds
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    enable_feature_selection: bool = True
    enable_hyperparameter_tuning: bool = True
    cross_validation_folds: int = 5

@dataclass
class TrainingData:
    """Represents training data for machine learning."""
    features: np.ndarray
    targets: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelMetrics:
    """Metrics for evaluating model performance."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class MLModel:
    """Represents a machine learning model."""
    model_id: str
    model_type: ModelType
    learning_type: LearningType
    model_object: Any
    feature_names: List[str]
    target_names: List[str]
    training_metrics: ModelMetrics
    validation_metrics: ModelMetrics
    created_at: float
    updated_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class SimpleLinearRegression:
    """Simple linear regression implementation."""
    
    def __init__(self):
        self.weights = None
        self.bias = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the linear regression model."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Add bias term
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Calculate weights using normal equation
        try:
            weights = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
            self.bias = weights[0]
            self.weights = weights[1:]
            self.is_fitted = True
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            weights = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
            self.bias = weights[0]
            self.weights = weights[1:]
            self.is_fitted = True
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        return X @ self.weights + self.bias

class SimpleLogisticRegression:
    """Simple logistic regression implementation."""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.is_fitted = False
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the logistic regression model."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            predictions = self._sigmoid(z)
            
            # Calculate gradients
            dw = (1 / n_samples) * X.T @ (predictions - y)
            db = (1 / n_samples) * np.sum(predictions - y)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
        self.is_fitted = True
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        z = X @ self.weights + self.bias
        return self._sigmoid(z)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions."""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

class SimpleDecisionTree:
    """Simple decision tree implementation."""
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.is_fitted = False
        
    def _gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0
            
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
        
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        """Find the best split for the data."""
        best_gini = float('inf')
        best_feature = 0
        best_threshold = 0
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                    
                left_gini = self._gini_impurity(y[left_mask])
                right_gini = self._gini_impurity(y[right_mask])
                
                weighted_gini = (np.sum(left_mask) * left_gini + np.sum(right_mask) * right_gini) / len(y)
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold, best_gini
        
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict[str, Any]:
        """Recursively build the decision tree."""
        # Check stopping criteria
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            len(np.unique(y)) == 1):
            
            # Return leaf node
            unique_classes, counts = np.unique(y, return_counts=True)
            majority_class = unique_classes[np.argmax(counts)]
            return {'type': 'leaf', 'class': majority_class}
            
        # Find best split
        feature, threshold, gini = self._best_split(X, y)
        
        # Split the data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            # Can't split further, return leaf
            unique_classes, counts = np.unique(y, return_counts=True)
            majority_class = unique_classes[np.argmax(counts)]
            return {'type': 'leaf', 'class': majority_class}
            
        # Recursively build subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'type': 'split',
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the decision tree model."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        self.tree = self._build_tree(X, y)
        self.is_fitted = True
        
    def _predict_sample(self, x: np.ndarray, tree: Dict[str, Any]) -> Any:
        """Predict a single sample using the tree."""
        if tree['type'] == 'leaf':
            return tree['class']
            
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        predictions = []
        for x in X:
            predictions.append(self._predict_sample(x, self.tree))
            
        return np.array(predictions)

class FeatureEngineer:
    """Handles feature engineering operations."""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.feature_stats = {}
        self.selected_features = []
        
    def normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize features to zero mean and unit variance."""
        if fit:
            self.feature_stats['mean'] = np.mean(X, axis=0)
            self.feature_stats['std'] = np.std(X, axis=0)
            
        mean = self.feature_stats.get('mean', 0)
        std = self.feature_stats.get('std', 1)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        return (X - mean) / std
        
    def select_features(self, X: np.ndarray, y: np.ndarray, 
                       feature_names: List[str], k: int = 10) -> Tuple[np.ndarray, List[str]]:
        """Select top k features based on correlation with target."""
        if not self.config.enable_feature_selection or k >= X.shape[1]:
            return X, feature_names
            
        # Calculate correlation with target
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)
            
        # Select top k features
        feature_indices = np.argsort(correlations)[-k:]
        selected_features = [feature_names[i] for i in feature_indices]
        
        self.selected_features = feature_indices
        
        return X[:, feature_indices], selected_features
        
    def create_polynomial_features(self, X: np.ndarray, degree: int = 2) -> np.ndarray:
        """Create polynomial features."""
        if degree == 1:
            return X
            
        n_samples, n_features = X.shape
        poly_features = [X]
        
        # Add polynomial terms
        for d in range(2, degree + 1):
            for i in range(n_features):
                poly_features.append((X[:, i] ** d).reshape(-1, 1))
                
        return np.column_stack(poly_features)

class ModelEvaluator:
    """Evaluates machine learning model performance."""
    
    def __init__(self):
        pass
        
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """Evaluate regression model performance."""
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return ModelMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2_score=r2
        )
        
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """Evaluate classification model performance."""
        # Convert to binary if needed
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        
        if len(unique_classes) == 2:
            # Binary classification
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
        else:
            # Multi-class classification
            accuracy = np.mean(y_true == y_pred)
            precision = accuracy  # Simplified for multi-class
            recall = accuracy
            f1 = accuracy
            
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1
        )

class ModelRegistry:
    """Manages model versions and persistence."""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.models = {}
        
        # Create storage directory
        os.makedirs(config.model_storage_path, exist_ok=True)
        
    def save_model(self, model: MLModel) -> str:
        """Save a model to disk."""
        model_path = os.path.join(self.config.model_storage_path, f"{model.model_id}.pkl")
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
            self.models[model.model_id] = model
            logger.info(f"Model {model.model_id} saved successfully")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model {model.model_id}: {e}")
            raise
            
    def load_model(self, model_id: str) -> Optional[MLModel]:
        """Load a model from disk."""
        if model_id in self.models:
            return self.models[model_id]
            
        model_path = os.path.join(self.config.model_storage_path, f"{model_id}.pkl")
        
        if not os.path.exists(model_path):
            return None
            
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            self.models[model_id] = model
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None
            
    def list_models(self) -> List[str]:
        """List all available models."""
        model_files = [f for f in os.listdir(self.config.model_storage_path) if f.endswith('.pkl')]
        return [f.replace('.pkl', '') for f in model_files]
        
    def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        model_path = os.path.join(self.config.model_storage_path, f"{model_id}.pkl")
        
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
                
            if model_id in self.models:
                del self.models[model_id]
                
            logger.info(f"Model {model_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False

class ModelTrainer:
    """Trains machine learning models."""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.evaluator = ModelEvaluator()
        self.registry = ModelRegistry(config)
        
    def train_model(self, model_type: ModelType, training_data: TrainingData,
                   model_params: Optional[Dict[str, Any]] = None) -> MLModel:
        """Train a machine learning model."""
        model_params = model_params or {}
        
        # Prepare data
        X = training_data.features
        y = training_data.targets
        
        if y is None and model_type not in [ModelType.CLUSTERING, ModelType.ANOMALY_DETECTION]:
            raise ValueError("Supervised learning requires target values")
            
        # Feature engineering
        X_normalized = self.feature_engineer.normalize_features(X, fit=True)
        
        if self.config.enable_feature_selection and y is not None:
            X_selected, selected_feature_names = self.feature_engineer.select_features(
                X_normalized, y, training_data.feature_names
            )
        else:
            X_selected = X_normalized
            selected_feature_names = training_data.feature_names
            
        # Split data
        n_samples = X_selected.shape[0]
        n_train = int(n_samples * (1 - self.config.validation_split - self.config.test_split))
        n_val = int(n_samples * self.config.validation_split)
        
        # Shuffle indices
        np.random.seed(self.config.random_seed)
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        
        if y is not None:
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            y_train, y_val = None, None
            
        # Create and train model
        model_object = self._create_model(model_type, model_params)
        
        start_time = time.time()
        
        if y_train is not None:
            model_object.fit(X_train, y_train)
        else:
            model_object.fit(X_train)
            
        training_time = time.time() - start_time
        
        # Evaluate model
        training_metrics = self._evaluate_model(model_object, X_train, y_train, model_type)
        validation_metrics = self._evaluate_model(model_object, X_val, y_val, model_type)
        
        # Create ML model
        model_id = str(uuid.uuid4())
        current_time = time.time()
        
        ml_model = MLModel(
            model_id=model_id,
            model_type=model_type,
            learning_type=self._get_learning_type(model_type),
            model_object=model_object,
            feature_names=selected_feature_names,
            target_names=training_data.target_names,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            created_at=current_time,
            updated_at=current_time,
            metadata={
                'training_time': training_time,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'model_params': model_params
            }
        )
        
        # Save model
        self.registry.save_model(ml_model)
        
        logger.info(f"Model {model_id} trained successfully in {training_time:.2f} seconds")
        return ml_model
        
    def _create_model(self, model_type: ModelType, params: Dict[str, Any]) -> Any:
        """Create a model instance based on type."""
        if model_type == ModelType.LINEAR_REGRESSION:
            return SimpleLinearRegression()
        elif model_type == ModelType.LOGISTIC_REGRESSION:
            return SimpleLogisticRegression(
                learning_rate=params.get('learning_rate', 0.01),
                max_iterations=params.get('max_iterations', 1000)
            )
        elif model_type == ModelType.DECISION_TREE:
            return SimpleDecisionTree(
                max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 2)
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def _evaluate_model(self, model: Any, X: np.ndarray, y: Optional[np.ndarray], 
                       model_type: ModelType) -> ModelMetrics:
        """Evaluate a trained model."""
        if y is None:
            return ModelMetrics()  # Can't evaluate without targets
            
        try:
            y_pred = model.predict(X)
            
            if model_type in [ModelType.LINEAR_REGRESSION]:
                return self.evaluator.evaluate_regression(y, y_pred)
            else:
                return self.evaluator.evaluate_classification(y, y_pred)
                
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return ModelMetrics()
            
    def _get_learning_type(self, model_type: ModelType) -> LearningType:
        """Get the learning type for a model type."""
        if model_type in [ModelType.LINEAR_REGRESSION, ModelType.LOGISTIC_REGRESSION, 
                         ModelType.DECISION_TREE, ModelType.RANDOM_FOREST]:
            return LearningType.SUPERVISED
        elif model_type in [ModelType.CLUSTERING, ModelType.ANOMALY_DETECTION]:
            return LearningType.UNSUPERVISED
        else:
            return LearningType.SUPERVISED

class MLFoundation:
    """Main machine learning foundation that coordinates all ML operations."""
    
    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self.trainer = ModelTrainer(self.config)
        self.registry = ModelRegistry(self.config)
        self.active_models = {}
        
        logger.info("Machine learning foundation initialized")
        
    def create_training_data(self, features: np.ndarray, targets: Optional[np.ndarray] = None,
                           feature_names: Optional[List[str]] = None,
                           target_names: Optional[List[str]] = None) -> TrainingData:
        """Create training data object."""
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
            
        if target_names is None and targets is not None:
            if targets.ndim == 1:
                target_names = ["target"]
            else:
                target_names = [f"target_{i}" for i in range(targets.shape[1])]
                
        return TrainingData(
            features=features,
            targets=targets,
            feature_names=feature_names,
            target_names=target_names or []
        )
        
    def train_model(self, model_type: ModelType, training_data: TrainingData,
                   model_params: Optional[Dict[str, Any]] = None) -> str:
        """Train a new model and return its ID."""
        model = self.trainer.train_model(model_type, training_data, model_params)
        self.active_models[model.model_id] = model
        return model.model_id
        
    def predict(self, model_id: str, features: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model."""
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")
            
        # Apply same feature engineering as during training
        X_normalized = self.trainer.feature_engineer.normalize_features(features, fit=False)
        
        if hasattr(self.trainer.feature_engineer, 'selected_features') and len(self.trainer.feature_engineer.selected_features) > 0:
            X_selected = X_normalized[:, self.trainer.feature_engineer.selected_features]
        else:
            X_selected = X_normalized
            
        return model.model_object.predict(X_selected)
        
    def get_model(self, model_id: str) -> Optional[MLModel]:
        """Get a model by ID."""
        if model_id in self.active_models:
            return self.active_models[model_id]
            
        # Try to load from registry
        model = self.registry.load_model(model_id)
        if model:
            self.active_models[model_id] = model
            
        return model
        
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with their metadata."""
        model_ids = self.registry.list_models()
        models_info = []
        
        for model_id in model_ids:
            model = self.get_model(model_id)
            if model:
                models_info.append({
                    'model_id': model.model_id,
                    'model_type': model.model_type.name,
                    'learning_type': model.learning_type.name,
                    'created_at': model.created_at,
                    'training_accuracy': model.training_metrics.accuracy,
                    'validation_accuracy': model.validation_metrics.accuracy,
                    'feature_count': len(model.feature_names)
                })
                
        return models_info
        
    def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        if model_id in self.active_models:
            del self.active_models[model_id]
            
        return self.registry.delete_model(model_id)
        
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get detailed performance metrics for a model."""
        model = self.get_model(model_id)
        if model is None:
            return {}
            
        return {
            'model_id': model.model_id,
            'model_type': model.model_type.name,
            'training_metrics': {
                'accuracy': model.training_metrics.accuracy,
                'precision': model.training_metrics.precision,
                'recall': model.training_metrics.recall,
                'f1_score': model.training_metrics.f1_score,
                'mse': model.training_metrics.mse,
                'rmse': model.training_metrics.rmse,
                'mae': model.training_metrics.mae,
                'r2_score': model.training_metrics.r2_score
            },
            'validation_metrics': {
                'accuracy': model.validation_metrics.accuracy,
                'precision': model.validation_metrics.precision,
                'recall': model.validation_metrics.recall,
                'f1_score': model.validation_metrics.f1_score,
                'mse': model.validation_metrics.mse,
                'rmse': model.validation_metrics.rmse,
                'mae': model.validation_metrics.mae,
                'r2_score': model.validation_metrics.r2_score
            },
            'metadata': model.metadata
        }

# Example usage and testing
if __name__ == "__main__":
    # Create ML foundation
    config = MLConfig(
        model_storage_path="/tmp/test_ml_models",
        validation_split=0.2,
        enable_feature_selection=True
    )
    
    ml_foundation = MLFoundation(config)
    
    # Generate some test data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Regression data
    X_reg = np.random.randn(n_samples, n_features)
    y_reg = X_reg[:, 0] * 2 + X_reg[:, 1] * -1 + np.random.randn(n_samples) * 0.1
    
    # Classification data
    X_clf = np.random.randn(n_samples, n_features)
    y_clf = (X_clf[:, 0] + X_clf[:, 1] > 0).astype(int)
    
    try:
        # Train regression model
        reg_data = ml_foundation.create_training_data(X_reg, y_reg)
        reg_model_id = ml_foundation.train_model(ModelType.LINEAR_REGRESSION, reg_data)
        
        print(f"Trained regression model: {reg_model_id}")
        
        # Train classification model
        clf_data = ml_foundation.create_training_data(X_clf, y_clf)
        clf_model_id = ml_foundation.train_model(ModelType.LOGISTIC_REGRESSION, clf_data)
        
        print(f"Trained classification model: {clf_model_id}")
        
        # Make predictions
        test_X = np.random.randn(10, n_features)
        
        reg_predictions = ml_foundation.predict(reg_model_id, test_X)
        clf_predictions = ml_foundation.predict(clf_model_id, test_X)
        
        print(f"Regression predictions: {reg_predictions[:5]}")
        print(f"Classification predictions: {clf_predictions[:5]}")
        
        # Get model performance
        reg_performance = ml_foundation.get_model_performance(reg_model_id)
        clf_performance = ml_foundation.get_model_performance(clf_model_id)
        
        print(f"Regression R²: {reg_performance['validation_metrics']['r2_score']:.3f}")
        print(f"Classification accuracy: {clf_performance['validation_metrics']['accuracy']:.3f}")
        
        # List all models
        models = ml_foundation.list_models()
        print(f"Total models: {len(models)}")
        
    except Exception as e:
        logger.error(f"Error in ML foundation testing: {e}")
        raise

