"""
ALL-USE Learning Systems - Integration Testing Framework

This module implements comprehensive testing for the ALL-USE Learning Systems,
ensuring all components work together seamlessly and meet performance requirements.

The testing framework covers:
- Unit testing of individual components
- Integration testing between components
- Performance testing under various loads
- End-to-end workflow testing
- Data flow validation

Classes:
- LearningSystemTestFramework: Main testing coordinator
- ComponentTester: Tests individual learning components
- IntegrationTester: Tests component interactions
- PerformanceTester: Tests system performance
- DataFlowTester: Tests data flow between components

Version: 1.0.0
"""

import time
import logging
import threading
import unittest
import numpy as np
import json
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
from collections import defaultdict
import statistics

# Import learning system components
from ..data_collection.collection_agent import CollectionAgent, CollectionConfig
from ..data_collection.metrics_collector import MetricsCollector
from ..data_collection.streaming_pipeline import StreamingPipeline
from ..data_storage.time_series_db import TimeSeriesDB
from ..analytics.real_time_analytics import RealTimeAnalyticsEngine, DataPoint, AnalyticsConfig
from ..analytics.ml_foundation import MLFoundation, MLConfig, ModelType, TrainingData
from ..integration.learning_integration_framework import LearningIntegrationFramework, LearningConfig, LearningMode, PipelineStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """Status of test execution."""
    PENDING = 1
    RUNNING = 2
    PASSED = 3
    FAILED = 4
    SKIPPED = 5

class TestCategory(Enum):
    """Categories of tests."""
    UNIT = 1
    INTEGRATION = 2
    PERFORMANCE = 3
    END_TO_END = 4
    STRESS = 5

@dataclass
class TestResult:
    """Result of a test execution."""
    test_id: str
    test_name: str
    category: TestCategory
    status: TestStatus
    execution_time: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestSuite:
    """Collection of related tests."""
    suite_id: str
    suite_name: str
    tests: List[str] = field(default_factory=list)
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None

class ComponentTester:
    """Tests individual learning system components."""
    
    def __init__(self):
        self.test_results = []
        
    def test_collection_agent(self) -> TestResult:
        """Test the collection agent functionality."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Create collection agent
            config = CollectionConfig(
                collection_interval=0.1,
                batch_size=10,
                max_retries=3
            )
            
            agent = CollectionAgent(config)
            
            # Test data collection
            collected_data = []
            
            def data_callback(data):
                collected_data.append(data)
                
            agent.set_data_callback(data_callback)
            agent.start()
            
            # Generate test data
            for i in range(20):
                test_data = {'value': i, 'timestamp': time.time()}
                agent.collect_data(test_data)
                time.sleep(0.05)
                
            time.sleep(0.5)  # Allow processing
            agent.stop()
            
            # Validate results
            assert len(collected_data) > 0, "No data was collected"
            assert len(collected_data) <= 20, "Too much data collected"
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="Collection Agent Test",
                category=TestCategory.UNIT,
                status=TestStatus.PASSED,
                execution_time=execution_time,
                metrics={
                    'data_points_collected': len(collected_data),
                    'collection_rate': len(collected_data) / execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="Collection Agent Test",
                category=TestCategory.UNIT,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
    def test_metrics_collector(self) -> TestResult:
        """Test the metrics collector functionality."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            collector = MetricsCollector()
            
            # Test metric collection
            collector.record_counter('test_counter', 1)
            collector.record_gauge('test_gauge', 42.5)
            collector.record_histogram('test_histogram', [1, 2, 3, 4, 5])
            
            # Get metrics
            metrics = collector.get_metrics()
            
            # Validate results
            assert 'test_counter' in metrics, "Counter metric not found"
            assert 'test_gauge' in metrics, "Gauge metric not found"
            assert 'test_histogram' in metrics, "Histogram metric not found"
            assert metrics['test_gauge'] == 42.5, "Gauge value incorrect"
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="Metrics Collector Test",
                category=TestCategory.UNIT,
                status=TestStatus.PASSED,
                execution_time=execution_time,
                metrics={
                    'metrics_collected': len(metrics),
                    'collection_time': execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="Metrics Collector Test",
                category=TestCategory.UNIT,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
    def test_time_series_db(self) -> TestResult:
        """Test the time series database functionality."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            db = TimeSeriesDB()
            
            # Test data insertion
            current_time = time.time()
            test_data = []
            
            for i in range(100):
                timestamp = current_time + i
                value = 100 + i * 0.5 + np.random.normal(0, 5)
                test_data.append((timestamp, value))
                db.insert_metric('test_metric', timestamp, value, {'source': 'test'})
                
            # Test data retrieval
            retrieved_data = db.query_metrics(
                'test_metric',
                start_time=current_time,
                end_time=current_time + 100
            )
            
            # Validate results
            assert len(retrieved_data) == 100, f"Expected 100 records, got {len(retrieved_data)}"
            
            # Test aggregation
            aggregated = db.aggregate_metrics(
                'test_metric',
                start_time=current_time,
                end_time=current_time + 100,
                aggregation='avg',
                interval=10
            )
            
            assert len(aggregated) > 0, "No aggregated data returned"
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="Time Series DB Test",
                category=TestCategory.UNIT,
                status=TestStatus.PASSED,
                execution_time=execution_time,
                metrics={
                    'records_inserted': 100,
                    'records_retrieved': len(retrieved_data),
                    'aggregated_points': len(aggregated),
                    'insertion_rate': 100 / execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="Time Series DB Test",
                category=TestCategory.UNIT,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
    def test_analytics_engine(self) -> TestResult:
        """Test the real-time analytics engine."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            config = AnalyticsConfig(
                window_size=50,
                anomaly_threshold=2.0,
                enable_anomaly_detection=True,
                enable_pattern_matching=True
            )
            
            engine = RealTimeAnalyticsEngine(config)
            engine.start()
            
            # Generate test data with trend and anomalies
            for i in range(100):
                value = 100 + i * 0.5  # Upward trend
                
                # Add occasional anomalies
                if i % 20 == 0:
                    value += 50  # Spike
                    
                data_point = DataPoint(
                    timestamp=time.time(),
                    value=value,
                    metric_name='test_metric',
                    tags={'source': 'test'}
                )
                
                engine.add_data_point(data_point)
                time.sleep(0.01)
                
            time.sleep(1.0)  # Allow processing
            
            # Get results
            dashboard = engine.get_real_time_dashboard()
            analytics_results = engine.get_analytics_results('test_metric')
            
            engine.stop()
            
            # Validate results
            assert 'test_metric' in dashboard['metrics_summary'], "Metric not found in dashboard"
            assert len(analytics_results) > 0, "No analytics results generated"
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="Analytics Engine Test",
                category=TestCategory.UNIT,
                status=TestStatus.PASSED,
                execution_time=execution_time,
                metrics={
                    'data_points_processed': 100,
                    'analytics_results': len(analytics_results),
                    'processing_rate': 100 / execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="Analytics Engine Test",
                category=TestCategory.UNIT,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
    def test_ml_foundation(self) -> TestResult:
        """Test the machine learning foundation."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            config = MLConfig(
                validation_split=0.2,
                enable_feature_selection=True
            )
            
            ml_foundation = MLFoundation(config)
            
            # Generate test data
            np.random.seed(42)
            n_samples = 200
            n_features = 5
            
            X = np.random.randn(n_samples, n_features)
            y = X[:, 0] * 2 + X[:, 1] * -1 + np.random.randn(n_samples) * 0.1
            
            # Create training data
            training_data = ml_foundation.create_training_data(X, y)
            
            # Train model
            model_id = ml_foundation.train_model(ModelType.LINEAR_REGRESSION, training_data)
            
            # Make predictions
            test_X = np.random.randn(10, n_features)
            predictions = ml_foundation.predict(model_id, test_X)
            
            # Get model performance
            performance = ml_foundation.get_model_performance(model_id)
            
            # Validate results
            assert model_id is not None, "Model ID is None"
            assert len(predictions) == 10, f"Expected 10 predictions, got {len(predictions)}"
            assert performance['validation_metrics']['r2_score'] is not None, "R² score is None"
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="ML Foundation Test",
                category=TestCategory.UNIT,
                status=TestStatus.PASSED,
                execution_time=execution_time,
                metrics={
                    'training_samples': n_samples,
                    'features': n_features,
                    'r2_score': performance['validation_metrics']['r2_score'],
                    'training_time': execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="ML Foundation Test",
                category=TestCategory.UNIT,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )

class IntegrationTester:
    """Tests integration between learning system components."""
    
    def __init__(self):
        self.test_results = []
        
    def test_data_collection_to_storage(self) -> TestResult:
        """Test integration between data collection and storage."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Setup components
            db = TimeSeriesDB()
            collector = MetricsCollector()
            
            # Collect and store data
            for i in range(50):
                timestamp = time.time()
                value = 100 + i + np.random.normal(0, 5)
                
                # Collect metric
                collector.record_gauge('integration_test', value)
                
                # Store in database
                db.insert_metric('integration_test', timestamp, value, {'source': 'integration_test'})
                
                time.sleep(0.01)
                
            # Retrieve and validate
            metrics = collector.get_metrics()
            stored_data = db.query_metrics('integration_test')
            
            assert 'integration_test' in metrics, "Metric not found in collector"
            assert len(stored_data) == 50, f"Expected 50 stored records, got {len(stored_data)}"
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="Data Collection to Storage Integration",
                category=TestCategory.INTEGRATION,
                status=TestStatus.PASSED,
                execution_time=execution_time,
                metrics={
                    'data_points': 50,
                    'storage_success_rate': len(stored_data) / 50,
                    'integration_time': execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="Data Collection to Storage Integration",
                category=TestCategory.INTEGRATION,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
    def test_analytics_to_ml_integration(self) -> TestResult:
        """Test integration between analytics and ML components."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Setup components
            analytics_config = AnalyticsConfig(window_size=100)
            analytics_engine = RealTimeAnalyticsEngine(analytics_config)
            analytics_engine.start()
            
            ml_config = MLConfig(validation_split=0.2)
            ml_foundation = MLFoundation(ml_config)
            
            # Generate analytics data
            analytics_data = []
            for i in range(200):
                value = 100 + i * 0.3 + np.random.normal(0, 10)
                
                data_point = DataPoint(
                    timestamp=time.time(),
                    value=value,
                    metric_name='ml_integration_test',
                    tags={'source': 'integration'}
                )
                
                analytics_engine.add_data_point(data_point)
                analytics_data.append([value, i])  # Features: value, index
                
                time.sleep(0.005)
                
            time.sleep(0.5)  # Allow analytics processing
            
            # Get analytics results
            analytics_results = analytics_engine.get_analytics_results('ml_integration_test')
            
            # Prepare ML training data from analytics
            if len(analytics_data) > 50:
                features = np.array(analytics_data)
                targets = features[:, 0] + np.random.normal(0, 5, len(features))  # Synthetic targets
                
                training_data = ml_foundation.create_training_data(features, targets)
                model_id = ml_foundation.train_model(ModelType.LINEAR_REGRESSION, training_data)
                
                # Test predictions
                test_features = np.array([[150, 100], [200, 150]])
                predictions = ml_foundation.predict(model_id, test_features)
                
                analytics_engine.stop()
                
                # Validate integration
                assert len(analytics_results) > 0, "No analytics results generated"
                assert model_id is not None, "Model training failed"
                assert len(predictions) == 2, "Prediction failed"
                
                execution_time = time.time() - start_time
                
                return TestResult(
                    test_id=test_id,
                    test_name="Analytics to ML Integration",
                    category=TestCategory.INTEGRATION,
                    status=TestStatus.PASSED,
                    execution_time=execution_time,
                    metrics={
                        'analytics_results': len(analytics_results),
                        'training_samples': len(features),
                        'model_trained': model_id is not None,
                        'predictions_made': len(predictions)
                    }
                )
            else:
                raise ValueError("Insufficient data for ML training")
                
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="Analytics to ML Integration",
                category=TestCategory.INTEGRATION,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
    def test_end_to_end_learning_pipeline(self) -> TestResult:
        """Test complete end-to-end learning pipeline."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Setup learning integration framework
            config = LearningConfig(
                learning_mode=LearningMode.HYBRID,
                real_time_window_size=100,
                max_concurrent_pipelines=2
            )
            
            framework = LearningIntegrationFramework(config)
            framework.start()
            
            # Add performance data
            for i in range(100):
                metrics = {
                    'response_time': 50 + i * 0.5 + np.random.normal(0, 5),
                    'throughput': 100 + i * 0.3 + np.random.normal(0, 10),
                    'error_rate': 0.01 + np.random.uniform(0, 0.05)
                }
                
                framework.add_performance_data('test_component', metrics)
                time.sleep(0.01)
                
            # Submit learning tasks
            task1_id = framework.submit_learning_task(
                task_type="performance_analysis",
                model_type=ModelType.LINEAR_REGRESSION,
                data_source="performance_test_component",
                target_metric="response_time",
                priority=1
            )
            
            task2_id = framework.submit_learning_task(
                task_type="anomaly_detection",
                model_type=ModelType.ANOMALY_DETECTION,
                data_source="performance_test_component",
                target_metric="error_rate",
                priority=2
            )
            
            # Wait for tasks to complete
            max_wait = 10
            wait_time = 0
            
            while wait_time < max_wait:
                task1_status = framework.learning_pipeline.get_task_status(task1_id)
                task2_status = framework.learning_pipeline.get_task_status(task2_id)
                
                if (task1_status and task1_status.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED] and
                    task2_status and task2_status.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]):
                    break
                    
                time.sleep(0.5)
                wait_time += 0.5
                
            # Get results
            insights = framework.get_learning_insights('test_component')
            system_status = framework.get_system_status()
            
            framework.stop()
            
            # Validate end-to-end pipeline
            assert task1_status is not None, "Task 1 status not found"
            assert task2_status is not None, "Task 2 status not found"
            assert len(insights['analytics_results']) > 0, "No analytics results in insights"
            assert system_status['is_running'] == False, "Framework not stopped properly"
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="End-to-End Learning Pipeline",
                category=TestCategory.END_TO_END,
                status=TestStatus.PASSED,
                execution_time=execution_time,
                metrics={
                    'data_points_processed': 100,
                    'tasks_submitted': 2,
                    'task1_status': task1_status.status.name,
                    'task2_status': task2_status.status.name,
                    'analytics_results': len(insights['analytics_results']),
                    'pipeline_time': execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="End-to-End Learning Pipeline",
                category=TestCategory.END_TO_END,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )

class PerformanceTester:
    """Tests system performance under various loads."""
    
    def __init__(self):
        self.test_results = []
        
    def test_high_volume_data_collection(self) -> TestResult:
        """Test data collection under high volume."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            config = CollectionConfig(
                collection_interval=0.001,  # Very fast collection
                batch_size=100
            )
            
            agent = CollectionAgent(config)
            collected_count = 0
            
            def count_callback(data):
                nonlocal collected_count
                collected_count += 1
                
            agent.set_data_callback(count_callback)
            agent.start()
            
            # Generate high volume data
            target_count = 10000
            for i in range(target_count):
                test_data = {'value': i, 'timestamp': time.time()}
                agent.collect_data(test_data)
                
                if i % 1000 == 0:
                    time.sleep(0.001)  # Brief pause every 1000 items
                    
            time.sleep(1.0)  # Allow processing
            agent.stop()
            
            execution_time = time.time() - start_time
            collection_rate = collected_count / execution_time
            
            # Performance criteria
            min_rate = 1000  # Minimum 1000 items/second
            success = collection_rate >= min_rate
            
            return TestResult(
                test_id=test_id,
                test_name="High Volume Data Collection",
                category=TestCategory.PERFORMANCE,
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                execution_time=execution_time,
                metrics={
                    'target_count': target_count,
                    'collected_count': collected_count,
                    'collection_rate': collection_rate,
                    'success_rate': collected_count / target_count,
                    'meets_performance_criteria': success
                },
                error_message=None if success else f"Collection rate {collection_rate:.1f}/s below minimum {min_rate}/s"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="High Volume Data Collection",
                category=TestCategory.PERFORMANCE,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
    def test_concurrent_analytics_processing(self) -> TestResult:
        """Test analytics processing with concurrent data streams."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            config = AnalyticsConfig(
                window_size=200,
                enable_anomaly_detection=True,
                enable_pattern_matching=True
            )
            
            engine = RealTimeAnalyticsEngine(config)
            engine.start()
            
            # Create multiple concurrent data streams
            num_streams = 5
            points_per_stream = 500
            
            def generate_stream_data(stream_id):
                for i in range(points_per_stream):
                    value = 100 + i * 0.1 + np.random.normal(0, 10)
                    
                    data_point = DataPoint(
                        timestamp=time.time(),
                        value=value,
                        metric_name=f'stream_{stream_id}',
                        tags={'stream_id': str(stream_id)}
                    )
                    
                    engine.add_data_point(data_point)
                    time.sleep(0.001)
                    
            # Start concurrent threads
            threads = []
            for stream_id in range(num_streams):
                thread = threading.Thread(target=generate_stream_data, args=(stream_id,))
                thread.start()
                threads.append(thread)
                
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
                
            time.sleep(2.0)  # Allow processing
            
            # Get results
            dashboard = engine.get_real_time_dashboard()
            total_results = 0
            
            for stream_id in range(num_streams):
                results = engine.get_analytics_results(f'stream_{stream_id}')
                total_results += len(results)
                
            engine.stop()
            
            execution_time = time.time() - start_time
            processing_rate = (num_streams * points_per_stream) / execution_time
            
            # Performance criteria
            min_rate = 1000  # Minimum 1000 points/second
            success = processing_rate >= min_rate and total_results > 0
            
            return TestResult(
                test_id=test_id,
                test_name="Concurrent Analytics Processing",
                category=TestCategory.PERFORMANCE,
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                execution_time=execution_time,
                metrics={
                    'concurrent_streams': num_streams,
                    'points_per_stream': points_per_stream,
                    'total_points': num_streams * points_per_stream,
                    'processing_rate': processing_rate,
                    'analytics_results': total_results,
                    'meets_performance_criteria': success
                },
                error_message=None if success else f"Processing rate {processing_rate:.1f}/s below minimum {min_rate}/s"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="Concurrent Analytics Processing",
                category=TestCategory.PERFORMANCE,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )

class LearningSystemTestFramework:
    """Main testing framework that coordinates all learning system tests."""
    
    def __init__(self):
        self.component_tester = ComponentTester()
        self.integration_tester = IntegrationTester()
        self.performance_tester = PerformanceTester()
        
        self.test_suites = {}
        self.test_results = []
        self.test_execution_order = []
        
        self._setup_default_test_suites()
        
    def _setup_default_test_suites(self) -> None:
        """Setup default test suites."""
        # Unit test suite
        unit_suite = TestSuite(
            suite_id="unit_tests",
            suite_name="Unit Tests",
            tests=[
                "test_collection_agent",
                "test_metrics_collector", 
                "test_time_series_db",
                "test_analytics_engine",
                "test_ml_foundation"
            ]
        )
        
        # Integration test suite
        integration_suite = TestSuite(
            suite_id="integration_tests",
            suite_name="Integration Tests",
            tests=[
                "test_data_collection_to_storage",
                "test_analytics_to_ml_integration",
                "test_end_to_end_learning_pipeline"
            ]
        )
        
        # Performance test suite
        performance_suite = TestSuite(
            suite_id="performance_tests",
            suite_name="Performance Tests",
            tests=[
                "test_high_volume_data_collection",
                "test_concurrent_analytics_processing"
            ]
        )
        
        self.test_suites = {
            "unit_tests": unit_suite,
            "integration_tests": integration_suite,
            "performance_tests": performance_suite
        }
        
    def run_test_suite(self, suite_id: str) -> List[TestResult]:
        """Run a specific test suite."""
        if suite_id not in self.test_suites:
            raise ValueError(f"Test suite {suite_id} not found")
            
        suite = self.test_suites[suite_id]
        suite_results = []
        
        logger.info(f"Running test suite: {suite.suite_name}")
        
        # Run setup if available
        if suite.setup_function:
            suite.setup_function()
            
        try:
            for test_name in suite.tests:
                logger.info(f"Running test: {test_name}")
                
                # Execute test based on name
                if hasattr(self.component_tester, test_name):
                    result = getattr(self.component_tester, test_name)()
                elif hasattr(self.integration_tester, test_name):
                    result = getattr(self.integration_tester, test_name)()
                elif hasattr(self.performance_tester, test_name):
                    result = getattr(self.performance_tester, test_name)()
                else:
                    # Create failed result for unknown test
                    result = TestResult(
                        test_id=str(uuid.uuid4()),
                        test_name=test_name,
                        category=TestCategory.UNIT,
                        status=TestStatus.FAILED,
                        execution_time=0.0,
                        error_message=f"Test method {test_name} not found"
                    )
                    
                suite_results.append(result)
                self.test_results.append(result)
                
                logger.info(f"Test {test_name} completed: {result.status.name}")
                
        finally:
            # Run teardown if available
            if suite.teardown_function:
                suite.teardown_function()
                
        logger.info(f"Test suite {suite.suite_name} completed")
        return suite_results
        
    def run_all_tests(self) -> Dict[str, List[TestResult]]:
        """Run all test suites."""
        all_results = {}
        
        # Run in order: unit -> integration -> performance
        execution_order = ["unit_tests", "integration_tests", "performance_tests"]
        
        for suite_id in execution_order:
            if suite_id in self.test_suites:
                results = self.run_test_suite(suite_id)
                all_results[suite_id] = results
                
        return all_results
        
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all test results."""
        if not self.test_results:
            return {'message': 'No tests have been run'}
            
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == TestStatus.PASSED])
        failed_tests = len([r for r in self.test_results if r.status == TestStatus.FAILED])
        
        # Category breakdown
        category_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0})
        
        for result in self.test_results:
            category = result.category.name
            category_stats[category]['total'] += 1
            
            if result.status == TestStatus.PASSED:
                category_stats[category]['passed'] += 1
            elif result.status == TestStatus.FAILED:
                category_stats[category]['failed'] += 1
                
        # Execution time stats
        execution_times = [r.execution_time for r in self.test_results]
        total_execution_time = sum(execution_times)
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'total_execution_time': total_execution_time,
            'average_execution_time': avg_execution_time,
            'category_breakdown': dict(category_stats),
            'failed_test_details': [
                {
                    'test_name': r.test_name,
                    'category': r.category.name,
                    'error_message': r.error_message
                }
                for r in self.test_results if r.status == TestStatus.FAILED
            ]
        }
        
    def generate_test_report(self) -> str:
        """Generate a comprehensive test report."""
        summary = self.get_test_summary()
        
        if 'message' in summary:
            return summary['message']
            
        report = []
        report.append("=" * 60)
        report.append("ALL-USE Learning Systems Test Report")
        report.append("=" * 60)
        report.append("")
        
        # Overall summary
        report.append(f"Total Tests: {summary['total_tests']}")
        report.append(f"Passed: {summary['passed_tests']}")
        report.append(f"Failed: {summary['failed_tests']}")
        report.append(f"Success Rate: {summary['success_rate']:.1%}")
        report.append(f"Total Execution Time: {summary['total_execution_time']:.2f} seconds")
        report.append(f"Average Test Time: {summary['average_execution_time']:.2f} seconds")
        report.append("")
        
        # Category breakdown
        report.append("Test Category Breakdown:")
        report.append("-" * 30)
        
        for category, stats in summary['category_breakdown'].items():
            success_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            report.append(f"{category}: {stats['passed']}/{stats['total']} ({success_rate:.1%})")
            
        report.append("")
        
        # Failed tests
        if summary['failed_test_details']:
            report.append("Failed Tests:")
            report.append("-" * 20)
            
            for failed_test in summary['failed_test_details']:
                report.append(f"• {failed_test['test_name']} ({failed_test['category']})")
                report.append(f"  Error: {failed_test['error_message']}")
                report.append("")
                
        # Individual test results
        report.append("Detailed Test Results:")
        report.append("-" * 30)
        
        for result in self.test_results:
            status_symbol = "✓" if result.status == TestStatus.PASSED else "✗"
            report.append(f"{status_symbol} {result.test_name} ({result.category.name})")
            report.append(f"  Execution Time: {result.execution_time:.2f}s")
            
            if result.metrics:
                report.append("  Metrics:")
                for key, value in result.metrics.items():
                    report.append(f"    {key}: {value}")
                    
            if result.error_message:
                report.append(f"  Error: {result.error_message}")
                
            report.append("")
            
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    # Create test framework
    test_framework = LearningSystemTestFramework()
    
    try:
        # Run all tests
        logger.info("Starting comprehensive learning systems testing")
        
        all_results = test_framework.run_all_tests()
        
        # Generate and print report
        report = test_framework.generate_test_report()
        print(report)
        
        # Save report to file
        with open('/tmp/learning_systems_test_report.txt', 'w') as f:
            f.write(report)
            
        logger.info("Testing completed. Report saved to /tmp/learning_systems_test_report.txt")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise

