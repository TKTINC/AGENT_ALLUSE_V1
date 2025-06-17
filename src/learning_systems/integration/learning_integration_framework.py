"""
ALL-USE Learning Systems - Learning Integration Framework

This module implements the integration framework for the ALL-USE Learning Systems,
providing seamless integration between data collection, storage, analytics, and ML components.

The integration framework is designed to:
- Coordinate data flow between components
- Manage learning pipelines
- Handle real-time and batch processing
- Provide unified interfaces for learning operations
- Enable system-wide learning coordination

Classes:
- LearningIntegrationFramework: Main integration coordinator
- LearningPipeline: Manages learning workflows
- DataFlowManager: Handles data flow between components
- LearningCoordinator: Coordinates learning across system components

Version: 1.0.0
"""

import time
import logging
import threading
import queue
import json
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque

# Import learning system components
from .real_time_analytics import RealTimeAnalyticsEngine, DataPoint, AnalyticsConfig
from .ml_foundation import MLFoundation, MLConfig, ModelType, TrainingData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    """Status of learning pipelines."""
    IDLE = 1
    RUNNING = 2
    PAUSED = 3
    COMPLETED = 4
    FAILED = 5

class LearningMode(Enum):
    """Learning modes for the system."""
    REAL_TIME = 1
    BATCH = 2
    HYBRID = 3
    ON_DEMAND = 4

@dataclass
class LearningConfig:
    """Configuration for learning operations."""
    learning_mode: LearningMode = LearningMode.HYBRID
    real_time_window_size: int = 1000
    batch_processing_interval: int = 3600  # seconds
    model_retraining_threshold: float = 0.1  # performance degradation threshold
    enable_auto_retraining: bool = True
    enable_model_ensemble: bool = True
    max_concurrent_pipelines: int = 5
    data_retention_days: int = 30

@dataclass
class LearningTask:
    """Represents a learning task."""
    task_id: str
    task_type: str
    model_type: ModelType
    data_source: str
    target_metric: str
    parameters: Dict[str, Any]
    priority: int = 5  # 1-10, lower is higher priority
    created_at: float = field(default_factory=time.time)
    status: PipelineStatus = PipelineStatus.IDLE
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class LearningResult:
    """Result of a learning operation."""
    task_id: str
    model_id: Optional[str]
    performance_metrics: Dict[str, float]
    insights: List[str]
    recommendations: List[str]
    confidence: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataFlowManager:
    """Manages data flow between learning system components."""
    
    def __init__(self, config: LearningConfig):
        self.config = config
        self.data_streams = defaultdict(deque)
        self.stream_processors = {}
        self.data_subscribers = defaultdict(list)
        self.lock = threading.Lock()
        
    def register_data_stream(self, stream_name: str, max_size: int = 10000) -> None:
        """Register a new data stream."""
        with self.lock:
            self.data_streams[stream_name] = deque(maxlen=max_size)
            
    def publish_data(self, stream_name: str, data: Any) -> None:
        """Publish data to a stream."""
        with self.lock:
            if stream_name in self.data_streams:
                self.data_streams[stream_name].append({
                    'timestamp': time.time(),
                    'data': data
                })
                
                # Notify subscribers
                for callback in self.data_subscribers[stream_name]:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Error in data subscriber callback: {e}")
                        
    def subscribe_to_stream(self, stream_name: str, callback: Callable) -> None:
        """Subscribe to a data stream."""
        with self.lock:
            self.data_subscribers[stream_name].append(callback)
            
    def get_stream_data(self, stream_name: str, limit: Optional[int] = None) -> List[Any]:
        """Get data from a stream."""
        with self.lock:
            if stream_name not in self.data_streams:
                return []
                
            data = list(self.data_streams[stream_name])
            if limit:
                data = data[-limit:]
                
            return [item['data'] for item in data]
            
    def get_stream_stats(self, stream_name: str) -> Dict[str, Any]:
        """Get statistics for a data stream."""
        with self.lock:
            if stream_name not in self.data_streams:
                return {}
                
            stream_data = self.data_streams[stream_name]
            if not stream_data:
                return {'count': 0}
                
            timestamps = [item['timestamp'] for item in stream_data]
            
            return {
                'count': len(stream_data),
                'oldest_timestamp': min(timestamps),
                'newest_timestamp': max(timestamps),
                'data_rate': len(stream_data) / (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0,
                'subscribers': len(self.data_subscribers[stream_name])
            }

class LearningPipeline:
    """Manages learning workflows and pipelines."""
    
    def __init__(self, config: LearningConfig, analytics_engine: RealTimeAnalyticsEngine, 
                 ml_foundation: MLFoundation, data_flow_manager: DataFlowManager):
        self.config = config
        self.analytics_engine = analytics_engine
        self.ml_foundation = ml_foundation
        self.data_flow_manager = data_flow_manager
        
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_queue = queue.PriorityQueue()
        self.worker_threads = []
        self.is_running = False
        self.lock = threading.Lock()
        
    def start(self) -> None:
        """Start the learning pipeline."""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start worker threads
        for i in range(self.config.max_concurrent_pipelines):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.worker_threads.append(worker)
            
        logger.info(f"Learning pipeline started with {len(self.worker_threads)} workers")
        
    def stop(self) -> None:
        """Stop the learning pipeline."""
        self.is_running = False
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
            
        logger.info("Learning pipeline stopped")
        
    def submit_task(self, task: LearningTask) -> str:
        """Submit a learning task for execution."""
        with self.lock:
            self.active_tasks[task.task_id] = task
            
        # Add to priority queue (lower priority number = higher priority)
        self.task_queue.put((task.priority, time.time(), task))
        
        logger.info(f"Learning task {task.task_id} submitted")
        return task.task_id
        
    def get_task_status(self, task_id: str) -> Optional[LearningTask]:
        """Get the status of a learning task."""
        with self.lock:
            if task_id in self.active_tasks:
                return self.active_tasks[task_id]
            elif task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
                
        return None
        
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a learning task."""
        with self.lock:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = PipelineStatus.FAILED
                task.error_message = "Task cancelled by user"
                return True
                
        return False
        
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        with self.lock:
            active_count = len(self.active_tasks)
            completed_count = len(self.completed_tasks)
            
            status_counts = defaultdict(int)
            for task in self.active_tasks.values():
                status_counts[task.status.name] += 1
                
        return {
            'active_tasks': active_count,
            'completed_tasks': completed_count,
            'queue_size': self.task_queue.qsize(),
            'worker_threads': len(self.worker_threads),
            'status_breakdown': dict(status_counts),
            'is_running': self.is_running
        }
        
    def _worker_loop(self) -> None:
        """Main worker loop for processing learning tasks."""
        while self.is_running:
            try:
                # Get next task from queue (with timeout)
                try:
                    priority, timestamp, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                # Process the task
                self._process_task(task)
                
            except Exception as e:
                logger.error(f"Error in learning pipeline worker: {e}")
                time.sleep(1.0)
                
    def _process_task(self, task: LearningTask) -> None:
        """Process a single learning task."""
        start_time = time.time()
        
        try:
            with self.lock:
                task.status = PipelineStatus.RUNNING
                task.progress = 0.0
                
            logger.info(f"Processing learning task {task.task_id}")
            
            # Execute the task based on type
            if task.task_type == "model_training":
                result = self._execute_model_training(task)
            elif task.task_type == "performance_analysis":
                result = self._execute_performance_analysis(task)
            elif task.task_type == "pattern_detection":
                result = self._execute_pattern_detection(task)
            elif task.task_type == "anomaly_detection":
                result = self._execute_anomaly_detection(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
                
            # Update task status
            with self.lock:
                task.status = PipelineStatus.COMPLETED
                task.progress = 1.0
                task.result = result
                
                # Move to completed tasks
                self.completed_tasks[task.task_id] = task
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                    
            execution_time = time.time() - start_time
            logger.info(f"Learning task {task.task_id} completed in {execution_time:.2f} seconds")
            
        except Exception as e:
            with self.lock:
                task.status = PipelineStatus.FAILED
                task.error_message = str(e)
                
                # Move to completed tasks
                self.completed_tasks[task.task_id] = task
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                    
            logger.error(f"Learning task {task.task_id} failed: {e}")
            
    def _execute_model_training(self, task: LearningTask) -> Dict[str, Any]:
        """Execute model training task."""
        # Get training data from data stream
        training_data_raw = self.data_flow_manager.get_stream_data(
            task.data_source, 
            limit=task.parameters.get('training_samples', 1000)
        )
        
        if not training_data_raw:
            raise ValueError(f"No training data available from source: {task.data_source}")
            
        # Convert to training format
        features = []
        targets = []
        
        for data_point in training_data_raw:
            if isinstance(data_point, dict) and 'features' in data_point:
                features.append(data_point['features'])
                if 'target' in data_point:
                    targets.append(data_point['target'])
                    
        if not features:
            raise ValueError("No valid feature data found")
            
        features = np.array(features)
        targets = np.array(targets) if targets else None
        
        # Create training data
        training_data = self.ml_foundation.create_training_data(features, targets)
        
        # Update progress
        task.progress = 0.3
        
        # Train model
        model_id = self.ml_foundation.train_model(
            task.model_type, 
            training_data, 
            task.parameters.get('model_params', {})
        )
        
        # Update progress
        task.progress = 0.8
        
        # Get model performance
        performance = self.ml_foundation.get_model_performance(model_id)
        
        return {
            'model_id': model_id,
            'performance': performance,
            'training_samples': len(features),
            'feature_count': features.shape[1]
        }
        
    def _execute_performance_analysis(self, task: LearningTask) -> Dict[str, Any]:
        """Execute performance analysis task."""
        # Get performance data
        performance_data = self.data_flow_manager.get_stream_data(
            task.data_source,
            limit=task.parameters.get('analysis_samples', 1000)
        )
        
        if not performance_data:
            raise ValueError(f"No performance data available from source: {task.data_source}")
            
        # Extract metrics
        metrics = []
        timestamps = []
        
        for data_point in performance_data:
            if isinstance(data_point, dict):
                if task.target_metric in data_point:
                    metrics.append(data_point[task.target_metric])
                    timestamps.append(data_point.get('timestamp', time.time()))
                    
        if not metrics:
            raise ValueError(f"No data found for target metric: {task.target_metric}")
            
        # Update progress
        task.progress = 0.5
        
        # Perform statistical analysis
        metrics = np.array(metrics)
        
        analysis_result = {
            'metric_name': task.target_metric,
            'sample_count': len(metrics),
            'mean': float(np.mean(metrics)),
            'std': float(np.std(metrics)),
            'min': float(np.min(metrics)),
            'max': float(np.max(metrics)),
            'median': float(np.median(metrics)),
            'percentile_95': float(np.percentile(metrics, 95)),
            'percentile_99': float(np.percentile(metrics, 99))
        }
        
        # Trend analysis
        if len(metrics) > 10:
            x = np.arange(len(metrics))
            slope = np.polyfit(x, metrics, 1)[0]
            correlation = np.corrcoef(x, metrics)[0, 1]
            
            analysis_result.update({
                'trend_slope': float(slope),
                'trend_correlation': float(correlation),
                'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            })
            
        return analysis_result
        
    def _execute_pattern_detection(self, task: LearningTask) -> Dict[str, Any]:
        """Execute pattern detection task."""
        # Get data for pattern analysis
        pattern_data = self.data_flow_manager.get_stream_data(
            task.data_source,
            limit=task.parameters.get('pattern_samples', 500)
        )
        
        if not pattern_data:
            raise ValueError(f"No pattern data available from source: {task.data_source}")
            
        # Extract values for pattern analysis
        values = []
        for data_point in pattern_data:
            if isinstance(data_point, dict) and task.target_metric in data_point:
                values.append(data_point[task.target_metric])
                
        if len(values) < 10:
            raise ValueError("Insufficient data for pattern detection")
            
        # Update progress
        task.progress = 0.4
        
        # Simple pattern detection
        values = np.array(values)
        
        # Detect cyclical patterns using autocorrelation
        autocorr = np.correlate(values, values, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find peaks
        peaks = []
        for i in range(1, min(len(autocorr) - 1, 50)):  # Check first 50 lags
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
                peaks.append((i, autocorr[i]))
                
        # Update progress
        task.progress = 0.8
        
        patterns_found = []
        
        if peaks:
            # Sort by correlation strength
            peaks.sort(key=lambda x: x[1], reverse=True)
            
            for lag, correlation in peaks[:3]:  # Top 3 patterns
                patterns_found.append({
                    'type': 'cyclical',
                    'period': lag,
                    'strength': float(correlation),
                    'confidence': min(float(correlation), 1.0)
                })
                
        # Detect trend patterns
        if len(values) > 20:
            # Split into segments and analyze trends
            segment_size = len(values) // 4
            segments = [values[i:i+segment_size] for i in range(0, len(values), segment_size)]
            
            trend_changes = 0
            for i in range(len(segments) - 1):
                if len(segments[i]) > 2 and len(segments[i+1]) > 2:
                    trend1 = np.polyfit(range(len(segments[i])), segments[i], 1)[0]
                    trend2 = np.polyfit(range(len(segments[i+1])), segments[i+1], 1)[0]
                    
                    if (trend1 > 0) != (trend2 > 0):  # Trend direction change
                        trend_changes += 1
                        
            if trend_changes > 0:
                patterns_found.append({
                    'type': 'trend_changes',
                    'change_count': trend_changes,
                    'confidence': min(trend_changes / (len(segments) - 1), 1.0)
                })
                
        return {
            'patterns_found': patterns_found,
            'total_patterns': len(patterns_found),
            'data_points_analyzed': len(values)
        }
        
    def _execute_anomaly_detection(self, task: LearningTask) -> Dict[str, Any]:
        """Execute anomaly detection task."""
        # Get data for anomaly detection
        anomaly_data = self.data_flow_manager.get_stream_data(
            task.data_source,
            limit=task.parameters.get('anomaly_samples', 1000)
        )
        
        if not anomaly_data:
            raise ValueError(f"No anomaly data available from source: {task.data_source}")
            
        # Extract values
        values = []
        timestamps = []
        
        for data_point in anomaly_data:
            if isinstance(data_point, dict) and task.target_metric in data_point:
                values.append(data_point[task.target_metric])
                timestamps.append(data_point.get('timestamp', time.time()))
                
        if len(values) < 10:
            raise ValueError("Insufficient data for anomaly detection")
            
        # Update progress
        task.progress = 0.3
        
        values = np.array(values)
        
        # Statistical anomaly detection
        mean_val = np.mean(values)
        std_val = np.std(values)
        threshold = task.parameters.get('anomaly_threshold', 2.0)
        
        anomalies = []
        for i, (value, timestamp) in enumerate(zip(values, timestamps)):
            z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
            
            if z_score > threshold:
                anomalies.append({
                    'index': i,
                    'value': float(value),
                    'timestamp': timestamp,
                    'z_score': float(z_score),
                    'severity': 'high' if z_score > 3.0 else 'medium'
                })
                
        # Update progress
        task.progress = 0.8
        
        # Additional anomaly detection using IQR method
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        iqr_anomalies = 0
        for value in values:
            if value < lower_bound or value > upper_bound:
                iqr_anomalies += 1
                
        return {
            'anomalies_found': anomalies,
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(values),
            'iqr_anomalies': iqr_anomalies,
            'statistical_summary': {
                'mean': float(mean_val),
                'std': float(std_val),
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr)
            }
        }

class LearningCoordinator:
    """Coordinates learning across system components."""
    
    def __init__(self, config: LearningConfig):
        self.config = config
        self.learning_strategies = {}
        self.performance_baselines = {}
        self.model_performance_history = defaultdict(list)
        self.lock = threading.Lock()
        
    def register_learning_strategy(self, component_name: str, strategy: Dict[str, Any]) -> None:
        """Register a learning strategy for a system component."""
        with self.lock:
            self.learning_strategies[component_name] = strategy
            
    def update_performance_baseline(self, component_name: str, metrics: Dict[str, float]) -> None:
        """Update performance baseline for a component."""
        with self.lock:
            self.performance_baselines[component_name] = {
                'metrics': metrics,
                'updated_at': time.time()
            }
            
    def check_retraining_needed(self, component_name: str, current_metrics: Dict[str, float]) -> bool:
        """Check if model retraining is needed based on performance degradation."""
        with self.lock:
            if component_name not in self.performance_baselines:
                return False
                
            baseline = self.performance_baselines[component_name]['metrics']
            
            for metric_name, current_value in current_metrics.items():
                if metric_name in baseline:
                    baseline_value = baseline[metric_name]
                    degradation = (baseline_value - current_value) / baseline_value if baseline_value != 0 else 0
                    
                    if degradation > self.config.model_retraining_threshold:
                        return True
                        
        return False
        
    def get_learning_recommendations(self, component_name: str) -> List[str]:
        """Get learning recommendations for a component."""
        recommendations = []
        
        with self.lock:
            if component_name in self.performance_baselines:
                baseline_age = time.time() - self.performance_baselines[component_name]['updated_at']
                
                if baseline_age > 86400:  # 24 hours
                    recommendations.append("Update performance baseline - data is over 24 hours old")
                    
            if component_name in self.model_performance_history:
                history = self.model_performance_history[component_name]
                
                if len(history) > 5:
                    recent_performance = [h['accuracy'] for h in history[-5:] if 'accuracy' in h]
                    if recent_performance and len(recent_performance) > 2:
                        trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
                        
                        if trend < -0.01:  # Declining performance
                            recommendations.append("Consider model retraining - performance is declining")
                            
        return recommendations
        
    def record_model_performance(self, component_name: str, model_id: str, 
                                performance_metrics: Dict[str, float]) -> None:
        """Record model performance for tracking."""
        with self.lock:
            self.model_performance_history[component_name].append({
                'timestamp': time.time(),
                'model_id': model_id,
                **performance_metrics
            })
            
            # Keep only last 100 records
            if len(self.model_performance_history[component_name]) > 100:
                self.model_performance_history[component_name] = self.model_performance_history[component_name][-100:]

class LearningIntegrationFramework:
    """Main integration framework that coordinates all learning system components."""
    
    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()
        
        # Initialize components
        analytics_config = AnalyticsConfig(
            window_size=self.config.real_time_window_size,
            enable_forecasting=True,
            enable_pattern_matching=True,
            enable_anomaly_detection=True
        )
        
        ml_config = MLConfig(
            validation_split=0.2,
            enable_feature_selection=True,
            enable_hyperparameter_tuning=True
        )
        
        self.analytics_engine = RealTimeAnalyticsEngine(analytics_config)
        self.ml_foundation = MLFoundation(ml_config)
        self.data_flow_manager = DataFlowManager(self.config)
        self.learning_pipeline = LearningPipeline(
            self.config, self.analytics_engine, self.ml_foundation, self.data_flow_manager
        )
        self.learning_coordinator = LearningCoordinator(self.config)
        
        self.is_running = False
        
        # Register default data streams
        self._setup_default_streams()
        
        logger.info("Learning integration framework initialized")
        
    def start(self) -> None:
        """Start the learning integration framework."""
        if self.is_running:
            return
            
        self.analytics_engine.start()
        self.learning_pipeline.start()
        self.is_running = True
        
        logger.info("Learning integration framework started")
        
    def stop(self) -> None:
        """Stop the learning integration framework."""
        if not self.is_running:
            return
            
        self.analytics_engine.stop()
        self.learning_pipeline.stop()
        self.is_running = False
        
        logger.info("Learning integration framework stopped")
        
    def submit_learning_task(self, task_type: str, model_type: ModelType, 
                           data_source: str, target_metric: str,
                           parameters: Optional[Dict[str, Any]] = None,
                           priority: int = 5) -> str:
        """Submit a learning task for execution."""
        task = LearningTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            model_type=model_type,
            data_source=data_source,
            target_metric=target_metric,
            parameters=parameters or {},
            priority=priority
        )
        
        return self.learning_pipeline.submit_task(task)
        
    def add_performance_data(self, component_name: str, metrics: Dict[str, float],
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add performance data to the learning system."""
        data_point = {
            'component': component_name,
            'timestamp': time.time(),
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        # Publish to data stream
        self.data_flow_manager.publish_data(f"performance_{component_name}", data_point)
        
        # Add to analytics engine for real-time analysis
        for metric_name, value in metrics.items():
            analytics_data_point = DataPoint(
                timestamp=time.time(),
                value=value,
                metric_name=f"{component_name}_{metric_name}",
                tags={'component': component_name},
                metadata=metadata or {}
            )
            self.analytics_engine.add_data_point(analytics_data_point)
            
    def get_learning_insights(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """Get learning insights and recommendations."""
        insights = {
            'timestamp': time.time(),
            'analytics_results': [],
            'learning_recommendations': [],
            'system_status': self.get_system_status()
        }
        
        # Get analytics results
        if component_name:
            analytics_results = self.analytics_engine.get_analytics_results(
                metric_name=f"{component_name}_*"
            )
        else:
            analytics_results = self.analytics_engine.get_analytics_results()
            
        insights['analytics_results'] = [
            {
                'type': result.analytics_type.name,
                'metric': result.metric_name,
                'result': result.result,
                'confidence': result.confidence,
                'timestamp': result.timestamp
            }
            for result in analytics_results[-10:]  # Last 10 results
        ]
        
        # Get learning recommendations
        if component_name:
            recommendations = self.learning_coordinator.get_learning_recommendations(component_name)
            insights['learning_recommendations'] = recommendations
        else:
            all_recommendations = []
            for comp_name in self.learning_coordinator.performance_baselines.keys():
                comp_recommendations = self.learning_coordinator.get_learning_recommendations(comp_name)
                all_recommendations.extend([f"{comp_name}: {rec}" for rec in comp_recommendations])
            insights['learning_recommendations'] = all_recommendations
            
        return insights
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            'is_running': self.is_running,
            'analytics_engine': {
                'active_metrics': len(self.analytics_engine.stream_processor.data_windows),
                'total_results': len(self.analytics_engine.analytics_results),
                'active_alerts': len(self.analytics_engine.alert_manager.get_active_alerts())
            },
            'learning_pipeline': self.learning_pipeline.get_pipeline_stats(),
            'data_streams': {
                stream_name: self.data_flow_manager.get_stream_stats(stream_name)
                for stream_name in self.data_flow_manager.data_streams.keys()
            },
            'ml_models': len(self.ml_foundation.list_models())
        }
        
    def _setup_default_streams(self) -> None:
        """Setup default data streams."""
        default_streams = [
            'performance_account_management',
            'performance_analytics',
            'performance_optimization',
            'performance_security',
            'system_metrics',
            'user_behavior',
            'market_data'
        ]
        
        for stream_name in default_streams:
            self.data_flow_manager.register_data_stream(stream_name)

# Example usage and testing
if __name__ == "__main__":
    # Create learning integration framework
    config = LearningConfig(
        learning_mode=LearningMode.HYBRID,
        real_time_window_size=100,
        batch_processing_interval=60,
        enable_auto_retraining=True
    )
    
    framework = LearningIntegrationFramework(config)
    framework.start()
    
    try:
        # Simulate adding performance data
        import random
        
        for i in range(50):
            # Simulate account management performance data
            metrics = {
                'response_time': random.uniform(10, 100),
                'throughput': random.uniform(50, 200),
                'error_rate': random.uniform(0, 0.1),
                'cpu_usage': random.uniform(20, 80)
            }
            
            framework.add_performance_data('account_management', metrics)
            time.sleep(0.1)
            
        # Submit learning tasks
        task1_id = framework.submit_learning_task(
            task_type="performance_analysis",
            model_type=ModelType.LINEAR_REGRESSION,
            data_source="performance_account_management",
            target_metric="response_time",
            priority=1
        )
        
        task2_id = framework.submit_learning_task(
            task_type="anomaly_detection",
            model_type=ModelType.ANOMALY_DETECTION,
            data_source="performance_account_management",
            target_metric="error_rate",
            priority=2
        )
        
        # Wait for tasks to complete
        time.sleep(5)
        
        # Get insights
        insights = framework.get_learning_insights('account_management')
        print(f"Learning insights: {json.dumps(insights, indent=2, default=str)}")
        
        # Get system status
        status = framework.get_system_status()
        print(f"System status: {json.dumps(status, indent=2, default=str)}")
        
    finally:
        framework.stop()

