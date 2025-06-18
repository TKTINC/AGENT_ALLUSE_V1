"""
WS5-P5: Advanced Analytics and Predictive Optimization
Predictive analytics and forecasting for proactive performance optimization.

This module provides advanced analytics capabilities including:
- Performance forecasting and trend prediction
- Bottleneck prediction and early warning systems
- Capacity planning and resource requirement forecasting
- Optimization opportunity identification
- Performance impact assessment and modeling
"""

import time
import threading
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3
import os
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Advanced analytics imports
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Represents a prediction result."""
    metric_name: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    prediction_horizon: timedelta
    model_accuracy: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary format."""
        return {
            'metric_name': self.metric_name,
            'predicted_value': self.predicted_value,
            'confidence_interval': self.confidence_interval,
            'prediction_horizon_seconds': self.prediction_horizon.total_seconds(),
            'model_accuracy': self.model_accuracy,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }

@dataclass
class AnomalyPrediction:
    """Represents an anomaly prediction."""
    metric_name: str
    anomaly_probability: float
    expected_time: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float
    contributing_factors: List[str]
    recommended_actions: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert anomaly prediction to dictionary format."""
        return {
            'metric_name': self.metric_name,
            'anomaly_probability': self.anomaly_probability,
            'expected_time': self.expected_time.isoformat(),
            'severity': self.severity,
            'confidence': self.confidence,
            'contributing_factors': self.contributing_factors,
            'recommended_actions': self.recommended_actions,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class CapacityForecast:
    """Represents a capacity planning forecast."""
    resource_type: str
    current_utilization: float
    predicted_utilization: float
    capacity_threshold: float
    time_to_threshold: Optional[timedelta]
    recommended_capacity: float
    confidence: float
    forecast_horizon: timedelta
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert capacity forecast to dictionary format."""
        return {
            'resource_type': self.resource_type,
            'current_utilization': self.current_utilization,
            'predicted_utilization': self.predicted_utilization,
            'capacity_threshold': self.capacity_threshold,
            'time_to_threshold_seconds': self.time_to_threshold.total_seconds() if self.time_to_threshold else None,
            'recommended_capacity': self.recommended_capacity,
            'confidence': self.confidence,
            'forecast_horizon_seconds': self.forecast_horizon.total_seconds(),
            'timestamp': self.timestamp.isoformat()
        }

class TimeSeriesForecaster:
    """Advanced time series forecasting for performance metrics."""
    
    def __init__(self, model_type: str = 'auto'):
        """
        Initialize time series forecaster.
        
        Args:
            model_type: Type of forecasting model ('linear', 'random_forest', 'auto')
        """
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.model_accuracies = {}
        self.training_history = defaultdict(list)
        
    def prepare_time_series_data(self, data: List[Dict[str, Any]], 
                                metric_name: str, 
                                window_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for training."""
        # Extract metric values and timestamps
        metric_data = []
        for point in data:
            if point.get('name') == metric_name:
                metric_data.append({
                    'timestamp': datetime.fromisoformat(point['timestamp']) if isinstance(point['timestamp'], str) else point['timestamp'],
                    'value': point['value']
                })
        
        # Sort by timestamp
        metric_data.sort(key=lambda x: x['timestamp'])
        
        if len(metric_data) < window_size + 1:
            raise ValueError(f"Insufficient data for metric {metric_name}. Need at least {window_size + 1} points.")
        
        # Create sliding windows
        values = [d['value'] for d in metric_data]
        X, y = [], []
        
        for i in range(len(values) - window_size):
            X.append(values[i:i + window_size])
            y.append(values[i + window_size])
        
        return np.array(X), np.array(y)
    
    def train_forecasting_model(self, data: List[Dict[str, Any]], 
                               metric_name: str, 
                               window_size: int = 10) -> Dict[str, Any]:
        """Train forecasting model for specific metric."""
        try:
            X, y = self.prepare_time_series_data(data, metric_name, window_size)
            
            if len(X) == 0:
                raise ValueError(f"No training data available for {metric_name}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select and train model
            if self.model_type == 'auto':
                model_type = self._select_best_model_type(X_train_scaled, y_train)
            else:
                model_type = self.model_type
            
            if model_type == 'linear':
                model = Ridge(alpha=1.0)
            elif model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = LinearRegression()
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model and scaler
            self.models[metric_name] = model
            self.scalers[metric_name] = scaler
            self.model_accuracies[metric_name] = r2
            
            # Store training history
            self.training_history[metric_name].append({
                'timestamp': datetime.now(),
                'model_type': model_type,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'training_samples': len(X_train)
            })
            
            logger.info(f"Trained {model_type} model for {metric_name}: R² = {r2:.3f}")
            
            return {
                'metric_name': metric_name,
                'model_type': model_type,
                'accuracy': r2,
                'mse': mse,
                'mae': mae,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Error training model for {metric_name}: {e}")
            return {'error': str(e)}
    
    def _select_best_model_type(self, X: np.ndarray, y: np.ndarray) -> str:
        """Automatically select best model type based on data characteristics."""
        # Simple heuristics for model selection
        n_samples, n_features = X.shape
        
        if n_samples < 50:
            return 'linear'  # Linear for small datasets
        elif n_features > 20:
            return 'random_forest'  # Random forest for high-dimensional data
        else:
            # Test both and select better one
            try:
                # Quick cross-validation
                linear_model = LinearRegression()
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
                
                # Simple train-test split
                split_idx = int(0.8 * len(X))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Test linear model
                linear_model.fit(X_train, y_train)
                linear_pred = linear_model.predict(X_test)
                linear_r2 = r2_score(y_test, linear_pred)
                
                # Test random forest
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                rf_r2 = r2_score(y_test, rf_pred)
                
                return 'random_forest' if rf_r2 > linear_r2 else 'linear'
                
            except Exception:
                return 'linear'  # Default fallback
    
    def forecast_metric(self, metric_name: str, 
                       recent_data: List[float], 
                       horizon_steps: int = 1) -> PredictionResult:
        """Forecast future values for a metric."""
        if metric_name not in self.models:
            raise ValueError(f"No trained model available for {metric_name}")
        
        model = self.models[metric_name]
        scaler = self.scalers[metric_name]
        accuracy = self.model_accuracies.get(metric_name, 0.0)
        
        try:
            # Prepare input data
            if len(recent_data) < model.n_features_in_:
                # Pad with last value if insufficient data
                recent_data = recent_data + [recent_data[-1]] * (model.n_features_in_ - len(recent_data))
            elif len(recent_data) > model.n_features_in_:
                # Take last n values
                recent_data = recent_data[-model.n_features_in_:]
            
            # Scale input
            input_data = np.array(recent_data).reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Calculate confidence interval (simple approach)
            std_error = np.std(recent_data) * (1 - accuracy)
            confidence_interval = (
                prediction - 1.96 * std_error,
                prediction + 1.96 * std_error
            )
            
            return PredictionResult(
                metric_name=metric_name,
                predicted_value=prediction,
                confidence_interval=confidence_interval,
                prediction_horizon=timedelta(minutes=horizon_steps),
                model_accuracy=accuracy,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error forecasting {metric_name}: {e}")
            raise
    
    def forecast_multiple_steps(self, metric_name: str, 
                               recent_data: List[float], 
                               horizon_steps: int = 5) -> List[PredictionResult]:
        """Forecast multiple steps ahead."""
        predictions = []
        current_data = recent_data.copy()
        
        for step in range(1, horizon_steps + 1):
            try:
                prediction = self.forecast_metric(metric_name, current_data, step)
                predictions.append(prediction)
                
                # Update data with prediction for next step
                current_data.append(prediction.predicted_value)
                if len(current_data) > 20:  # Keep reasonable window
                    current_data = current_data[-20:]
                    
            except Exception as e:
                logger.error(f"Error in multi-step forecasting at step {step}: {e}")
                break
        
        return predictions

class AnomalyPredictor:
    """Predicts potential anomalies in performance metrics."""
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize anomaly predictor.
        
        Args:
            contamination: Expected proportion of anomalies in data
        """
        self.contamination = contamination
        self.anomaly_models = {}
        self.anomaly_thresholds = {}
        self.prediction_history = defaultdict(list)
        
    def train_anomaly_detector(self, data: List[Dict[str, Any]], metric_name: str) -> Dict[str, Any]:
        """Train anomaly detection model for specific metric."""
        try:
            # Extract metric values
            metric_values = []
            for point in data:
                if point.get('name') == metric_name:
                    metric_values.append(point['value'])
            
            if len(metric_values) < 10:
                raise ValueError(f"Insufficient data for anomaly detection: {len(metric_values)} points")
            
            # Prepare features (value, moving average, trend)
            features = self._prepare_anomaly_features(metric_values)
            
            # Train Isolation Forest
            model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            model.fit(features)
            
            # Calculate anomaly scores for threshold setting
            anomaly_scores = model.decision_function(features)
            threshold = np.percentile(anomaly_scores, self.contamination * 100)
            
            # Store model and threshold
            self.anomaly_models[metric_name] = model
            self.anomaly_thresholds[metric_name] = threshold
            
            # Evaluate model
            predictions = model.predict(features)
            anomaly_count = np.sum(predictions == -1)
            
            logger.info(f"Trained anomaly detector for {metric_name}: {anomaly_count} anomalies detected")
            
            return {
                'metric_name': metric_name,
                'training_samples': len(features),
                'anomalies_detected': anomaly_count,
                'contamination_rate': anomaly_count / len(features),
                'threshold': threshold
            }
            
        except Exception as e:
            logger.error(f"Error training anomaly detector for {metric_name}: {e}")
            return {'error': str(e)}
    
    def _prepare_anomaly_features(self, values: List[float]) -> np.ndarray:
        """Prepare features for anomaly detection."""
        features = []
        
        for i in range(len(values)):
            feature_vector = [values[i]]  # Current value
            
            # Moving average (last 5 points)
            start_idx = max(0, i - 4)
            moving_avg = np.mean(values[start_idx:i+1])
            feature_vector.append(moving_avg)
            
            # Trend (slope of last 3 points)
            if i >= 2:
                x = np.arange(3)
                y = values[i-2:i+1]
                trend = np.polyfit(x, y, 1)[0]
            else:
                trend = 0
            feature_vector.append(trend)
            
            # Deviation from moving average
            deviation = values[i] - moving_avg
            feature_vector.append(deviation)
            
            # Rate of change
            if i > 0:
                rate_of_change = values[i] - values[i-1]
            else:
                rate_of_change = 0
            feature_vector.append(rate_of_change)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def predict_anomaly(self, metric_name: str, 
                       recent_values: List[float], 
                       forecast_horizon: timedelta = timedelta(hours=1)) -> AnomalyPrediction:
        """Predict potential anomalies."""
        if metric_name not in self.anomaly_models:
            raise ValueError(f"No trained anomaly model for {metric_name}")
        
        model = self.anomaly_models[metric_name]
        threshold = self.anomaly_thresholds[metric_name]
        
        try:
            # Prepare features for recent data
            features = self._prepare_anomaly_features(recent_values)
            
            if len(features) == 0:
                raise ValueError("Insufficient data for anomaly prediction")
            
            # Get anomaly score for latest point
            latest_features = features[-1].reshape(1, -1)
            anomaly_score = model.decision_function(latest_features)[0]
            
            # Calculate anomaly probability
            anomaly_probability = max(0, (threshold - anomaly_score) / abs(threshold))
            
            # Determine severity
            if anomaly_probability > 0.8:
                severity = 'critical'
            elif anomaly_probability > 0.6:
                severity = 'high'
            elif anomaly_probability > 0.4:
                severity = 'medium'
            else:
                severity = 'low'
            
            # Analyze contributing factors
            contributing_factors = self._analyze_contributing_factors(recent_values, features[-1])
            
            # Generate recommendations
            recommended_actions = self._generate_anomaly_recommendations(metric_name, severity, contributing_factors)
            
            # Estimate expected time (simple heuristic)
            expected_time = datetime.now() + forecast_horizon
            
            prediction = AnomalyPrediction(
                metric_name=metric_name,
                anomaly_probability=anomaly_probability,
                expected_time=expected_time,
                severity=severity,
                confidence=min(0.95, 0.5 + anomaly_probability * 0.5),
                contributing_factors=contributing_factors,
                recommended_actions=recommended_actions,
                timestamp=datetime.now()
            )
            
            # Store prediction history
            self.prediction_history[metric_name].append(prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting anomaly for {metric_name}: {e}")
            raise
    
    def _analyze_contributing_factors(self, values: List[float], features: np.ndarray) -> List[str]:
        """Analyze factors contributing to potential anomaly."""
        factors = []
        
        if len(values) < 2:
            return factors
        
        current_value = values[-1]
        previous_value = values[-2] if len(values) > 1 else current_value
        
        # Check for sudden spikes
        if current_value > previous_value * 1.5:
            factors.append("Sudden spike in metric value")
        elif current_value < previous_value * 0.5:
            factors.append("Sudden drop in metric value")
        
        # Check trend
        if len(features) >= 3:
            trend = features[2]  # Trend feature
            if abs(trend) > np.std(values) * 2:
                factors.append("Unusual trend pattern detected")
        
        # Check deviation from moving average
        if len(features) >= 4:
            deviation = features[3]  # Deviation feature
            if abs(deviation) > np.std(values):
                factors.append("Significant deviation from moving average")
        
        # Check rate of change
        if len(features) >= 5:
            rate_of_change = features[4]  # Rate of change feature
            if abs(rate_of_change) > np.std(np.diff(values)) * 2:
                factors.append("Unusual rate of change")
        
        return factors if factors else ["No specific factors identified"]
    
    def _generate_anomaly_recommendations(self, metric_name: str, severity: str, factors: List[str]) -> List[str]:
        """Generate recommendations based on anomaly prediction."""
        recommendations = []
        
        if severity in ['critical', 'high']:
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Prepare contingency plans")
            
        if 'spike' in ' '.join(factors).lower():
            recommendations.append("Check for resource bottlenecks")
            recommendations.append("Review recent configuration changes")
            
        if 'drop' in ' '.join(factors).lower():
            recommendations.append("Verify system health")
            recommendations.append("Check for service interruptions")
            
        if 'trend' in ' '.join(factors).lower():
            recommendations.append("Analyze long-term patterns")
            recommendations.append("Consider capacity adjustments")
        
        # Metric-specific recommendations
        if 'cpu' in metric_name.lower():
            recommendations.append("Monitor CPU-intensive processes")
            recommendations.append("Consider load balancing")
        elif 'memory' in metric_name.lower():
            recommendations.append("Check for memory leaks")
            recommendations.append("Review memory allocation")
        elif 'disk' in metric_name.lower():
            recommendations.append("Monitor disk space")
            recommendations.append("Check I/O patterns")
        
        return recommendations if recommendations else ["Monitor closely and investigate if anomaly occurs"]

class CapacityPlanner:
    """Plans capacity requirements based on usage forecasts."""
    
    def __init__(self):
        """Initialize capacity planner."""
        self.capacity_models = {}
        self.utilization_history = defaultdict(list)
        self.capacity_forecasts = defaultdict(list)
        
    def analyze_capacity_trends(self, data: List[Dict[str, Any]], 
                               resource_type: str,
                               capacity_threshold: float = 0.8) -> Dict[str, Any]:
        """Analyze capacity trends for a resource type."""
        try:
            # Extract utilization data
            utilization_data = []
            for point in data:
                if resource_type in point.get('name', '').lower():
                    utilization_data.append({
                        'timestamp': datetime.fromisoformat(point['timestamp']) if isinstance(point['timestamp'], str) else point['timestamp'],
                        'utilization': point['value'] / 100.0  # Convert percentage to ratio
                    })
            
            if len(utilization_data) < 10:
                raise ValueError(f"Insufficient data for capacity analysis: {len(utilization_data)} points")
            
            # Sort by timestamp
            utilization_data.sort(key=lambda x: x['timestamp'])
            
            # Calculate trends
            utilizations = [d['utilization'] for d in utilization_data]
            timestamps = [d['timestamp'] for d in utilization_data]
            
            # Linear trend analysis
            x = np.arange(len(utilizations))
            trend_coeffs = np.polyfit(x, utilizations, 1)
            trend_slope = trend_coeffs[0]
            
            # Calculate statistics
            current_utilization = utilizations[-1]
            mean_utilization = np.mean(utilizations)
            max_utilization = np.max(utilizations)
            std_utilization = np.std(utilizations)
            
            # Estimate time to threshold
            time_to_threshold = None
            if trend_slope > 0 and current_utilization < capacity_threshold:
                steps_to_threshold = (capacity_threshold - current_utilization) / trend_slope
                if steps_to_threshold > 0:
                    # Assume each step represents the average time interval
                    if len(timestamps) > 1:
                        avg_interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
                        time_to_threshold = avg_interval * steps_to_threshold
            
            # Store analysis
            analysis = {
                'resource_type': resource_type,
                'current_utilization': current_utilization,
                'mean_utilization': mean_utilization,
                'max_utilization': max_utilization,
                'std_utilization': std_utilization,
                'trend_slope': trend_slope,
                'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing' if trend_slope < 0 else 'stable',
                'capacity_threshold': capacity_threshold,
                'time_to_threshold': time_to_threshold,
                'data_points': len(utilization_data),
                'analysis_timestamp': datetime.now()
            }
            
            self.utilization_history[resource_type].append(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing capacity trends for {resource_type}: {e}")
            return {'error': str(e)}
    
    def forecast_capacity_requirements(self, resource_type: str, 
                                     current_utilization: float,
                                     trend_slope: float,
                                     forecast_horizon: timedelta = timedelta(days=30),
                                     capacity_threshold: float = 0.8) -> CapacityForecast:
        """Forecast capacity requirements."""
        try:
            # Calculate predicted utilization
            # Simple linear extrapolation
            horizon_steps = forecast_horizon.total_seconds() / 3600  # Convert to hours
            predicted_utilization = current_utilization + (trend_slope * horizon_steps)
            
            # Calculate time to threshold
            time_to_threshold = None
            if trend_slope > 0 and current_utilization < capacity_threshold:
                hours_to_threshold = (capacity_threshold - current_utilization) / trend_slope
                if hours_to_threshold > 0:
                    time_to_threshold = timedelta(hours=hours_to_threshold)
            
            # Calculate recommended capacity
            if predicted_utilization > capacity_threshold:
                # Need to increase capacity
                required_capacity_ratio = predicted_utilization / capacity_threshold
                recommended_capacity = required_capacity_ratio * 1.2  # 20% buffer
            else:
                recommended_capacity = 1.0  # Current capacity is sufficient
            
            # Calculate confidence based on trend stability
            confidence = max(0.5, 1.0 - abs(trend_slope) * 10)  # Lower confidence for high volatility
            
            forecast = CapacityForecast(
                resource_type=resource_type,
                current_utilization=current_utilization,
                predicted_utilization=predicted_utilization,
                capacity_threshold=capacity_threshold,
                time_to_threshold=time_to_threshold,
                recommended_capacity=recommended_capacity,
                confidence=confidence,
                forecast_horizon=forecast_horizon,
                timestamp=datetime.now()
            )
            
            # Store forecast
            self.capacity_forecasts[resource_type].append(forecast)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error forecasting capacity for {resource_type}: {e}")
            raise
    
    def generate_capacity_recommendations(self, forecast: CapacityForecast) -> List[str]:
        """Generate capacity planning recommendations."""
        recommendations = []
        
        if forecast.predicted_utilization > forecast.capacity_threshold:
            recommendations.append(f"Increase {forecast.resource_type} capacity by {(forecast.recommended_capacity - 1) * 100:.1f}%")
            
            if forecast.time_to_threshold and forecast.time_to_threshold < timedelta(days=7):
                recommendations.append("URGENT: Capacity threshold will be reached within 7 days")
            elif forecast.time_to_threshold and forecast.time_to_threshold < timedelta(days=30):
                recommendations.append("Plan capacity increase within 30 days")
            
        elif forecast.predicted_utilization < 0.3:
            recommendations.append(f"Consider reducing {forecast.resource_type} capacity to optimize costs")
        
        # Resource-specific recommendations
        if 'cpu' in forecast.resource_type.lower():
            recommendations.append("Consider horizontal scaling or load balancing")
            recommendations.append("Review CPU-intensive processes for optimization")
        elif 'memory' in forecast.resource_type.lower():
            recommendations.append("Analyze memory usage patterns")
            recommendations.append("Consider memory optimization techniques")
        elif 'disk' in forecast.resource_type.lower():
            recommendations.append("Plan for additional storage")
            recommendations.append("Implement data archiving strategies")
        elif 'network' in forecast.resource_type.lower():
            recommendations.append("Consider bandwidth upgrades")
            recommendations.append("Implement traffic optimization")
        
        return recommendations

class OptimizationOpportunityIdentifier:
    """Identifies optimization opportunities based on performance analysis."""
    
    def __init__(self):
        """Initialize optimization opportunity identifier."""
        self.opportunity_history = deque(maxlen=1000)
        self.optimization_patterns = {}
        
    def identify_opportunities(self, performance_data: Dict[str, Any], 
                             forecasts: List[PredictionResult],
                             anomaly_predictions: List[AnomalyPrediction]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities."""
        opportunities = []
        
        try:
            # Analyze performance bottlenecks
            bottleneck_opportunities = self._identify_bottleneck_opportunities(performance_data)
            opportunities.extend(bottleneck_opportunities)
            
            # Analyze forecast-based opportunities
            forecast_opportunities = self._identify_forecast_opportunities(forecasts)
            opportunities.extend(forecast_opportunities)
            
            # Analyze anomaly-based opportunities
            anomaly_opportunities = self._identify_anomaly_opportunities(anomaly_predictions)
            opportunities.extend(anomaly_opportunities)
            
            # Analyze resource utilization opportunities
            resource_opportunities = self._identify_resource_opportunities(performance_data)
            opportunities.extend(resource_opportunities)
            
            # Rank opportunities by impact and feasibility
            ranked_opportunities = self._rank_opportunities(opportunities)
            
            # Store opportunities
            for opportunity in ranked_opportunities:
                self.opportunity_history.append(opportunity)
            
            logger.info(f"Identified {len(ranked_opportunities)} optimization opportunities")
            
            return ranked_opportunities
            
        except Exception as e:
            logger.error(f"Error identifying optimization opportunities: {e}")
            return []
    
    def _identify_bottleneck_opportunities(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify bottleneck-based optimization opportunities."""
        opportunities = []
        
        # Check for high resource utilization
        for metric_name, value in performance_data.items():
            if isinstance(value, (int, float)):
                if 'cpu' in metric_name.lower() and value > 80:
                    opportunities.append({
                        'type': 'bottleneck',
                        'category': 'cpu_optimization',
                        'metric': metric_name,
                        'current_value': value,
                        'severity': 'high' if value > 90 else 'medium',
                        'description': f"High CPU utilization detected: {value:.1f}%",
                        'recommendations': [
                            "Optimize CPU-intensive algorithms",
                            "Consider horizontal scaling",
                            "Review process priorities"
                        ],
                        'estimated_impact': 'high',
                        'implementation_effort': 'medium',
                        'timestamp': datetime.now()
                    })
                
                elif 'memory' in metric_name.lower() and value > 85:
                    opportunities.append({
                        'type': 'bottleneck',
                        'category': 'memory_optimization',
                        'metric': metric_name,
                        'current_value': value,
                        'severity': 'high' if value > 95 else 'medium',
                        'description': f"High memory utilization detected: {value:.1f}%",
                        'recommendations': [
                            "Optimize memory usage patterns",
                            "Implement memory caching strategies",
                            "Review memory leaks"
                        ],
                        'estimated_impact': 'high',
                        'implementation_effort': 'medium',
                        'timestamp': datetime.now()
                    })
                
                elif 'disk' in metric_name.lower() and value > 90:
                    opportunities.append({
                        'type': 'bottleneck',
                        'category': 'storage_optimization',
                        'metric': metric_name,
                        'current_value': value,
                        'severity': 'critical' if value > 95 else 'high',
                        'description': f"High disk utilization detected: {value:.1f}%",
                        'recommendations': [
                            "Implement data archiving",
                            "Optimize storage allocation",
                            "Consider additional storage"
                        ],
                        'estimated_impact': 'high',
                        'implementation_effort': 'low',
                        'timestamp': datetime.now()
                    })
        
        return opportunities
    
    def _identify_forecast_opportunities(self, forecasts: List[PredictionResult]) -> List[Dict[str, Any]]:
        """Identify forecast-based optimization opportunities."""
        opportunities = []
        
        for forecast in forecasts:
            # Check for predicted performance degradation
            if forecast.model_accuracy > 0.7:  # Only consider reliable forecasts
                if 'response_time' in forecast.metric_name.lower():
                    if forecast.predicted_value > forecast.confidence_interval[1] * 1.2:
                        opportunities.append({
                            'type': 'forecast',
                            'category': 'performance_degradation',
                            'metric': forecast.metric_name,
                            'predicted_value': forecast.predicted_value,
                            'confidence': forecast.model_accuracy,
                            'severity': 'medium',
                            'description': f"Predicted performance degradation in {forecast.metric_name}",
                            'recommendations': [
                                "Proactive performance optimization",
                                "Review system configuration",
                                "Monitor closely for early intervention"
                            ],
                            'estimated_impact': 'medium',
                            'implementation_effort': 'low',
                            'timestamp': datetime.now()
                        })
                
                elif 'throughput' in forecast.metric_name.lower():
                    if forecast.predicted_value < forecast.confidence_interval[0] * 0.8:
                        opportunities.append({
                            'type': 'forecast',
                            'category': 'throughput_optimization',
                            'metric': forecast.metric_name,
                            'predicted_value': forecast.predicted_value,
                            'confidence': forecast.model_accuracy,
                            'severity': 'medium',
                            'description': f"Predicted throughput decline in {forecast.metric_name}",
                            'recommendations': [
                                "Optimize processing algorithms",
                                "Review resource allocation",
                                "Consider performance tuning"
                            ],
                            'estimated_impact': 'medium',
                            'implementation_effort': 'medium',
                            'timestamp': datetime.now()
                        })
        
        return opportunities
    
    def _identify_anomaly_opportunities(self, anomaly_predictions: List[AnomalyPrediction]) -> List[Dict[str, Any]]:
        """Identify anomaly-based optimization opportunities."""
        opportunities = []
        
        for prediction in anomaly_predictions:
            if prediction.anomaly_probability > 0.6:  # High probability anomalies
                opportunities.append({
                    'type': 'anomaly_prevention',
                    'category': 'proactive_optimization',
                    'metric': prediction.metric_name,
                    'anomaly_probability': prediction.anomaly_probability,
                    'severity': prediction.severity,
                    'description': f"High probability anomaly predicted in {prediction.metric_name}",
                    'recommendations': prediction.recommended_actions,
                    'estimated_impact': 'high' if prediction.severity in ['critical', 'high'] else 'medium',
                    'implementation_effort': 'low',
                    'timestamp': datetime.now()
                })
        
        return opportunities
    
    def _identify_resource_opportunities(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify resource optimization opportunities."""
        opportunities = []
        
        # Check for underutilized resources
        for metric_name, value in performance_data.items():
            if isinstance(value, (int, float)):
                if any(resource in metric_name.lower() for resource in ['cpu', 'memory', 'disk']) and value < 20:
                    opportunities.append({
                        'type': 'resource_optimization',
                        'category': 'underutilization',
                        'metric': metric_name,
                        'current_value': value,
                        'severity': 'low',
                        'description': f"Underutilized resource detected: {metric_name} at {value:.1f}%",
                        'recommendations': [
                            "Consider resource reallocation",
                            "Optimize resource provisioning",
                            "Review capacity planning"
                        ],
                        'estimated_impact': 'medium',
                        'implementation_effort': 'low',
                        'timestamp': datetime.now()
                    })
        
        return opportunities
    
    def _rank_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank opportunities by impact and feasibility."""
        def calculate_priority_score(opportunity):
            # Impact scoring
            impact_scores = {'high': 3, 'medium': 2, 'low': 1}
            impact_score = impact_scores.get(opportunity.get('estimated_impact', 'low'), 1)
            
            # Effort scoring (inverse - lower effort = higher score)
            effort_scores = {'low': 3, 'medium': 2, 'high': 1}
            effort_score = effort_scores.get(opportunity.get('implementation_effort', 'high'), 1)
            
            # Severity scoring
            severity_scores = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            severity_score = severity_scores.get(opportunity.get('severity', 'low'), 1)
            
            # Combined score
            return (impact_score * 0.4) + (effort_score * 0.3) + (severity_score * 0.3)
        
        # Add priority scores and sort
        for opportunity in opportunities:
            opportunity['priority_score'] = calculate_priority_score(opportunity)
        
        return sorted(opportunities, key=lambda x: x['priority_score'], reverse=True)

class PredictiveAnalyzer:
    """Main predictive analytics coordinator."""
    
    def __init__(self):
        """Initialize predictive analyzer."""
        self.forecaster = TimeSeriesForecaster()
        self.anomaly_predictor = AnomalyPredictor()
        self.capacity_planner = CapacityPlanner()
        self.opportunity_identifier = OptimizationOpportunityIdentifier()
        
        self.analysis_history = deque(maxlen=500)
        self.is_running = False
        self.analysis_thread = None
        
        logger.info("Predictive analyzer initialized")
    
    def train_predictive_models(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train all predictive models with historical data."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'forecasting_models': {},
            'anomaly_models': {},
            'training_status': 'success'
        }
        
        try:
            # Get unique metric names
            metric_names = set()
            for point in historical_data:
                if 'name' in point:
                    metric_names.add(point['name'])
            
            logger.info(f"Training models for {len(metric_names)} metrics")
            
            # Train forecasting models
            for metric_name in metric_names:
                try:
                    forecast_result = self.forecaster.train_forecasting_model(historical_data, metric_name)
                    results['forecasting_models'][metric_name] = forecast_result
                except Exception as e:
                    logger.warning(f"Failed to train forecasting model for {metric_name}: {e}")
                    results['forecasting_models'][metric_name] = {'error': str(e)}
            
            # Train anomaly detection models
            for metric_name in metric_names:
                try:
                    anomaly_result = self.anomaly_predictor.train_anomaly_detector(historical_data, metric_name)
                    results['anomaly_models'][metric_name] = anomaly_result
                except Exception as e:
                    logger.warning(f"Failed to train anomaly model for {metric_name}: {e}")
                    results['anomaly_models'][metric_name] = {'error': str(e)}
            
            logger.info("Predictive model training completed")
            
        except Exception as e:
            logger.error(f"Error in predictive model training: {e}")
            results['training_status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def run_comprehensive_analysis(self, recent_data: List[Dict[str, Any]], 
                                 performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Run comprehensive predictive analysis."""
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'forecasts': [],
            'anomaly_predictions': [],
            'capacity_forecasts': [],
            'optimization_opportunities': [],
            'analysis_summary': {}
        }
        
        try:
            # Group data by metric
            metrics_data = defaultdict(list)
            for point in recent_data:
                if 'name' in point and 'value' in point:
                    metrics_data[point['name']].append(point['value'])
            
            # Generate forecasts
            for metric_name, values in metrics_data.items():
                if len(values) >= 10:  # Minimum data for forecasting
                    try:
                        forecast = self.forecaster.forecast_metric(metric_name, values[-20:])  # Use last 20 points
                        analysis_result['forecasts'].append(forecast.to_dict())
                    except Exception as e:
                        logger.warning(f"Forecasting failed for {metric_name}: {e}")
            
            # Generate anomaly predictions
            for metric_name, values in metrics_data.items():
                if len(values) >= 10:  # Minimum data for anomaly detection
                    try:
                        anomaly_pred = self.anomaly_predictor.predict_anomaly(metric_name, values[-20:])
                        analysis_result['anomaly_predictions'].append(anomaly_pred.to_dict())
                    except Exception as e:
                        logger.warning(f"Anomaly prediction failed for {metric_name}: {e}")
            
            # Generate capacity forecasts
            resource_types = ['cpu', 'memory', 'disk', 'network']
            for resource_type in resource_types:
                matching_metrics = [name for name in metrics_data.keys() if resource_type in name.lower()]
                if matching_metrics:
                    metric_name = matching_metrics[0]  # Use first matching metric
                    values = metrics_data[metric_name]
                    if len(values) >= 10:
                        try:
                            # Analyze capacity trends
                            trend_analysis = self.capacity_planner.analyze_capacity_trends(recent_data, resource_type)
                            if 'error' not in trend_analysis:
                                # Generate capacity forecast
                                capacity_forecast = self.capacity_planner.forecast_capacity_requirements(
                                    resource_type,
                                    trend_analysis['current_utilization'],
                                    trend_analysis['trend_slope']
                                )
                                analysis_result['capacity_forecasts'].append(capacity_forecast.to_dict())
                        except Exception as e:
                            logger.warning(f"Capacity forecasting failed for {resource_type}: {e}")
            
            # Identify optimization opportunities
            forecasts = [PredictionResult(**f) for f in analysis_result['forecasts']]
            anomaly_predictions = [AnomalyPrediction(**a) for a in analysis_result['anomaly_predictions']]
            
            opportunities = self.opportunity_identifier.identify_opportunities(
                performance_metrics, forecasts, anomaly_predictions
            )
            analysis_result['optimization_opportunities'] = opportunities
            
            # Generate analysis summary
            analysis_result['analysis_summary'] = {
                'total_forecasts': len(analysis_result['forecasts']),
                'high_risk_anomalies': len([a for a in analysis_result['anomaly_predictions'] 
                                          if a['severity'] in ['high', 'critical']]),
                'capacity_alerts': len([c for c in analysis_result['capacity_forecasts'] 
                                      if c['time_to_threshold_seconds'] and c['time_to_threshold_seconds'] < 7*24*3600]),
                'high_priority_opportunities': len([o for o in opportunities if o.get('priority_score', 0) > 2.5]),
                'analysis_quality': 'good'  # Could be calculated based on model accuracies
            }
            
            # Store analysis
            self.analysis_history.append(analysis_result)
            
            logger.info(f"Comprehensive analysis completed: {len(analysis_result['forecasts'])} forecasts, "
                       f"{len(analysis_result['anomaly_predictions'])} anomaly predictions, "
                       f"{len(opportunities)} optimization opportunities")
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            analysis_result['error'] = str(e)
        
        return analysis_result
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of recent analyses."""
        if not self.analysis_history:
            return {'message': 'No analysis history available'}
        
        recent_analyses = list(self.analysis_history)[-10:]  # Last 10 analyses
        
        summary = {
            'total_analyses': len(self.analysis_history),
            'recent_analyses_count': len(recent_analyses),
            'average_forecasts_per_analysis': np.mean([len(a.get('forecasts', [])) for a in recent_analyses]),
            'average_anomaly_predictions': np.mean([len(a.get('anomaly_predictions', [])) for a in recent_analyses]),
            'total_opportunities_identified': sum(len(a.get('optimization_opportunities', [])) for a in recent_analyses),
            'model_performance': {
                'forecasting_models': len(self.forecaster.models),
                'anomaly_models': len(self.anomaly_predictor.anomaly_models),
                'average_forecast_accuracy': np.mean(list(self.forecaster.model_accuracies.values())) if self.forecaster.model_accuracies else 0
            }
        }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Create predictive analyzer
    analyzer = PredictiveAnalyzer()
    
    # Generate sample historical data
    def generate_sample_data(metric_name: str, num_points: int = 100) -> List[Dict[str, Any]]:
        data = []
        base_time = datetime.now() - timedelta(hours=num_points)
        
        for i in range(num_points):
            # Generate realistic performance data with trends and noise
            if 'cpu' in metric_name:
                base_value = 50 + 20 * np.sin(i * 0.1) + np.random.normal(0, 5)
            elif 'memory' in metric_name:
                base_value = 60 + 10 * np.sin(i * 0.05) + np.random.normal(0, 3)
            else:
                base_value = 40 + 15 * np.sin(i * 0.08) + np.random.normal(0, 4)
            
            data.append({
                'name': metric_name,
                'value': max(0, min(100, base_value)),
                'timestamp': (base_time + timedelta(hours=i)).isoformat()
            })
        
        return data
    
    # Generate sample data for multiple metrics
    print("Generating sample data...")
    historical_data = []
    metrics = ['cpu_usage_percent', 'memory_usage_percent', 'disk_usage_percent', 'network_throughput']
    
    for metric in metrics:
        historical_data.extend(generate_sample_data(metric, 200))
    
    # Train predictive models
    print("Training predictive models...")
    training_results = analyzer.train_predictive_models(historical_data)
    print(f"Training completed: {training_results['training_status']}")
    
    # Generate recent data for analysis
    recent_data = []
    for metric in metrics:
        recent_data.extend(generate_sample_data(metric, 50))
    
    # Performance metrics
    performance_metrics = {
        'cpu_usage_percent': 75.5,
        'memory_usage_percent': 68.2,
        'disk_usage_percent': 45.8,
        'network_throughput': 85.3,
        'response_time_ms': 150.2
    }
    
    # Run comprehensive analysis
    print("Running comprehensive analysis...")
    analysis_result = analyzer.run_comprehensive_analysis(recent_data, performance_metrics)
    
    print(f"Analysis completed:")
    print(f"- Forecasts: {len(analysis_result['forecasts'])}")
    print(f"- Anomaly predictions: {len(analysis_result['anomaly_predictions'])}")
    print(f"- Capacity forecasts: {len(analysis_result['capacity_forecasts'])}")
    print(f"- Optimization opportunities: {len(analysis_result['optimization_opportunities'])}")
    
    # Display some results
    if analysis_result['optimization_opportunities']:
        print("\nTop optimization opportunities:")
        for i, opp in enumerate(analysis_result['optimization_opportunities'][:3]):
            print(f"{i+1}. {opp['description']} (Priority: {opp.get('priority_score', 0):.2f})")
    
    # Get analysis summary
    summary = analyzer.get_analysis_summary()
    print(f"\nAnalysis summary: {summary}")

