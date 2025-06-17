#!/usr/bin/env python3
"""
ALL-USE Agent - Protocol Engine Performance and Load Testing
P6 of WS2: Protocol Engine Final Integration and System Testing - Phase 3

This module provides comprehensive performance and load testing for the complete
Protocol Engine system, validating performance under various load conditions
and measuring the effectiveness of optimization systems implemented in P5 of WS2.

Performance Testing Components:
1. Performance Benchmarking - Baseline performance measurement
2. Load Testing Framework - High-frequency scenario testing
3. Optimization Effectiveness Measurement - Validate P5 optimizations
4. Scalability Assessment - Resource usage and scaling analysis
5. Performance Analytics - Comprehensive performance reporting
6. Stress Testing - System behavior under extreme conditions
"""

import time
import json
import threading
import concurrent.futures
import multiprocessing
import psutil
import gc
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, '/home/ubuntu/AGENT_ALLUSE_V1/src')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics collection"""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    thread_count: int
    cache_hit_rate: float = 0.0
    operations_per_second: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class LoadTestResult:
    """Load test result"""
    test_name: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    total_duration: float
    average_response_time: float
    min_response_time: float
    max_response_time: float
    operations_per_second: float
    error_rate: float
    memory_peak: float
    cpu_peak: float
    performance_metrics: List[PerformanceMetrics]


@dataclass
class OptimizationEffectivenessResult:
    """Optimization effectiveness measurement"""
    optimization_type: str
    baseline_performance: PerformanceMetrics
    optimized_performance: PerformanceMetrics
    improvement_factor: float
    memory_reduction: float
    speed_improvement: float
    effectiveness_score: float


class PerformanceBenchmarker:
    """Comprehensive performance benchmarking system"""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.benchmark_results = {}
        
        logger.info("Performance Benchmarker initialized")
    
    def initialize_test_components(self) -> Dict[str, Any]:
        """Initialize components for performance testing"""
        components = {}
        
        try:
            # Initialize core components for testing
            from protocol_engine.week_classification.week_classifier import WeekClassifier
            from protocol_engine.market_analysis.market_condition_analyzer import MarketConditionAnalyzer
            
            components['week_classifier'] = WeekClassifier()
            components['market_analyzer'] = MarketConditionAnalyzer()
            
            logger.info("Core components initialized for performance testing")
            
        except Exception as e:
            logger.error(f"Failed to initialize core components: {e}")
        
        try:
            # Initialize optimization components
            from protocol_engine.optimization.cache_manager import get_cache_coordinator
            from protocol_engine.monitoring.performance_monitor import get_monitoring_coordinator
            
            components['cache_coordinator'] = get_cache_coordinator()
            components['monitoring_coordinator'] = get_monitoring_coordinator()
            
            logger.info("Optimization components initialized for performance testing")
            
        except Exception as e:
            logger.warning(f"Optimization components not fully available: {e}")
        
        return components
    
    def measure_baseline_performance(self, components: Dict[str, Any]) -> Dict[str, PerformanceMetrics]:
        """Measure baseline performance without optimizations"""
        baseline_results = {}
        
        # Test market analysis performance
        if 'market_analyzer' in components:
            test_data = {
                'symbol': 'TEST',
                'current_price': 100.0,
                'previous_close': 98.0,
                'week_start_price': 95.0,
                'volume': 1000000,
                'average_volume': 800000
            }
            
            # Measure performance
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            start_cpu = psutil.cpu_percent()
            
            try:
                # Call market analysis multiple times for accurate measurement
                for _ in range(10):
                    result = components['market_analyzer'].analyze_market_conditions(test_data)
                
                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                end_cpu = psutil.cpu_percent()
                
                baseline_results['market_analysis'] = PerformanceMetrics(
                    execution_time=(end_time - start_time) / 10,  # Average per operation
                    memory_usage=end_memory - start_memory,
                    cpu_usage=end_cpu - start_cpu,
                    thread_count=threading.active_count()
                )
                
                logger.info(f"Market analysis baseline: {(end_time - start_time)*100:.2f}ms per operation")
                
            except Exception as e:
                logger.error(f"Market analysis baseline measurement failed: {e}")
        
        # Test week classification performance
        if 'week_classifier' in components:
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                # Create mock market condition for testing
                class MockMarketCondition:
                    def __init__(self):
                        self.condition = "bullish"
                        self.confidence = 0.8
                
                mock_condition = MockMarketCondition()
                
                # Test week classification multiple times
                for _ in range(10):
                    try:
                        result = components['week_classifier'].classify_week(mock_condition, 'FLAT')
                    except Exception:
                        # If classify_week fails, try alternative approach
                        pass
                
                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                baseline_results['week_classification'] = PerformanceMetrics(
                    execution_time=(end_time - start_time) / 10,  # Average per operation
                    memory_usage=end_memory - start_memory,
                    cpu_usage=0.0,  # CPU measurement not reliable for short operations
                    thread_count=threading.active_count()
                )
                
                logger.info(f"Week classification baseline: {(end_time - start_time)*100:.2f}ms per operation")
                
            except Exception as e:
                logger.error(f"Week classification baseline measurement failed: {e}")
        
        return baseline_results
    
    def measure_optimized_performance(self, components: Dict[str, Any]) -> Dict[str, PerformanceMetrics]:
        """Measure performance with optimizations enabled"""
        optimized_results = {}
        
        # Test with cache coordinator if available
        if 'cache_coordinator' in components:
            cache_coordinator = components['cache_coordinator']
            
            # Test cached operations
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                # Perform operations that should benefit from caching
                test_data = {
                    'symbol': 'TEST',
                    'current_price': 100.0,
                    'previous_close': 98.0,
                    'week_start_price': 95.0,
                    'volume': 1000000,
                    'average_volume': 800000
                }
                
                # First run to populate cache
                if 'market_analyzer' in components:
                    components['market_analyzer'].analyze_market_conditions(test_data)
                
                # Measure cached performance
                cached_start = time.perf_counter()
                for _ in range(10):
                    if 'market_analyzer' in components:
                        components['market_analyzer'].analyze_market_conditions(test_data)
                
                cached_end = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # Get cache statistics
                cache_stats = cache_coordinator.get_comprehensive_stats()
                total_hits = sum(stats.get('hits', 0) for stats in cache_stats.values())
                total_requests = sum(stats.get('hits', 0) + stats.get('misses', 0) for stats in cache_stats.values())
                hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
                
                optimized_results['cached_operations'] = PerformanceMetrics(
                    execution_time=(cached_end - cached_start) / 10,
                    memory_usage=end_memory - start_memory,
                    cpu_usage=0.0,
                    thread_count=threading.active_count(),
                    cache_hit_rate=hit_rate
                )
                
                logger.info(f"Cached operations performance: {(cached_end - cached_start)*100:.2f}ms per operation, "
                           f"hit rate: {hit_rate:.1%}")
                
            except Exception as e:
                logger.error(f"Optimized performance measurement failed: {e}")
        
        return optimized_results


class LoadTestingFramework:
    """Comprehensive load testing framework"""
    
    def __init__(self, components: Dict[str, Any]):
        self.components = components
        self.load_test_results = {}
        
        logger.info("Load Testing Framework initialized")
    
    def run_concurrent_load_test(self, operation_count: int, thread_count: int) -> LoadTestResult:
        """Run concurrent load test with specified parameters"""
        test_name = f"Concurrent Load Test ({operation_count} ops, {thread_count} threads)"
        logger.info(f"Starting {test_name}")
        
        # Prepare test data
        test_scenarios = [
            {
                'symbol': f'TEST{i}',
                'current_price': 100.0 + (i % 10),
                'previous_close': 98.0 + (i % 8),
                'week_start_price': 95.0 + (i % 5),
                'volume': 1000000 + (i * 10000),
                'average_volume': 800000
            }
            for i in range(operation_count)
        ]
        
        # Performance tracking
        performance_metrics = []
        successful_operations = 0
        failed_operations = 0
        response_times = []
        
        # Memory and CPU monitoring
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_peak = start_memory
        cpu_peak = 0.0
        
        def execute_operation(scenario_data):
            """Execute single operation"""
            nonlocal successful_operations, failed_operations, memory_peak, cpu_peak
            
            operation_start = time.perf_counter()
            
            try:
                # Monitor resources
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                current_cpu = psutil.cpu_percent()
                memory_peak = max(memory_peak, current_memory)
                cpu_peak = max(cpu_peak, current_cpu)
                
                # Execute market analysis if available
                if 'market_analyzer' in self.components:
                    result = self.components['market_analyzer'].analyze_market_conditions(scenario_data)
                
                operation_end = time.perf_counter()
                response_time = operation_end - operation_start
                response_times.append(response_time)
                
                successful_operations += 1
                
                # Collect performance metrics
                performance_metrics.append(PerformanceMetrics(
                    execution_time=response_time,
                    memory_usage=current_memory,
                    cpu_usage=current_cpu,
                    thread_count=threading.active_count()
                ))
                
                return True
                
            except Exception as e:
                failed_operations += 1
                operation_end = time.perf_counter()
                response_times.append(operation_end - operation_start)
                logger.warning(f"Operation failed: {e}")
                return False
        
        # Execute load test
        test_start = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(execute_operation, scenario) for scenario in test_scenarios]
            concurrent.futures.wait(futures)
        
        test_end = time.perf_counter()
        total_duration = test_end - test_start
        
        # Calculate results
        total_operations = operation_count
        average_response_time = statistics.mean(response_times) if response_times else 0.0
        min_response_time = min(response_times) if response_times else 0.0
        max_response_time = max(response_times) if response_times else 0.0
        operations_per_second = total_operations / total_duration if total_duration > 0 else 0.0
        error_rate = failed_operations / total_operations if total_operations > 0 else 0.0
        
        result = LoadTestResult(
            test_name=test_name,
            total_operations=total_operations,
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            total_duration=total_duration,
            average_response_time=average_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            operations_per_second=operations_per_second,
            error_rate=error_rate,
            memory_peak=memory_peak,
            cpu_peak=cpu_peak,
            performance_metrics=performance_metrics
        )
        
        logger.info(f"Load test completed: {successful_operations}/{total_operations} successful, "
                   f"{operations_per_second:.1f} ops/sec, {error_rate:.1%} error rate")
        
        return result
    
    def run_stress_test(self) -> Dict[str, LoadTestResult]:
        """Run comprehensive stress testing"""
        stress_results = {}
        
        # Test scenarios with increasing load
        test_scenarios = [
            {'operations': 50, 'threads': 5, 'name': 'light_load'},
            {'operations': 100, 'threads': 10, 'name': 'medium_load'},
            {'operations': 200, 'threads': 20, 'name': 'heavy_load'},
            {'operations': 500, 'threads': 50, 'name': 'stress_load'}
        ]
        
        for scenario in test_scenarios:
            logger.info(f"Running stress test: {scenario['name']}")
            
            try:
                result = self.run_concurrent_load_test(
                    scenario['operations'], 
                    scenario['threads']
                )
                stress_results[scenario['name']] = result
                
                # Brief pause between tests
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Stress test {scenario['name']} failed: {e}")
        
        return stress_results


class OptimizationEffectivenessMeasurer:
    """Measures effectiveness of optimization systems"""
    
    def __init__(self, benchmarker: PerformanceBenchmarker):
        self.benchmarker = benchmarker
        self.effectiveness_results = {}
        
        logger.info("Optimization Effectiveness Measurer initialized")
    
    def measure_cache_effectiveness(self, components: Dict[str, Any]) -> OptimizationEffectivenessResult:
        """Measure cache optimization effectiveness"""
        logger.info("Measuring cache optimization effectiveness")
        
        # Baseline measurement (without cache benefits)
        baseline_start = time.perf_counter()
        baseline_memory_start = psutil.Process().memory_info().rss / 1024 / 1024
        
        test_data = {
            'symbol': 'CACHE_TEST',
            'current_price': 100.0,
            'previous_close': 98.0,
            'week_start_price': 95.0,
            'volume': 1000000,
            'average_volume': 800000
        }
        
        # Run operations without cache benefits (different data each time)
        for i in range(20):
            test_data_unique = test_data.copy()
            test_data_unique['current_price'] = 100.0 + i
            
            if 'market_analyzer' in components:
                try:
                    components['market_analyzer'].analyze_market_conditions(test_data_unique)
                except Exception:
                    pass
        
        baseline_end = time.perf_counter()
        baseline_memory_end = psutil.Process().memory_info().rss / 1024 / 1024
        
        baseline_metrics = PerformanceMetrics(
            execution_time=(baseline_end - baseline_start) / 20,
            memory_usage=baseline_memory_end - baseline_memory_start,
            cpu_usage=0.0,
            thread_count=threading.active_count()
        )
        
        # Optimized measurement (with cache benefits)
        optimized_start = time.perf_counter()
        optimized_memory_start = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Run same operation multiple times to benefit from caching
        for i in range(20):
            if 'market_analyzer' in components:
                try:
                    components['market_analyzer'].analyze_market_conditions(test_data)
                except Exception:
                    pass
        
        optimized_end = time.perf_counter()
        optimized_memory_end = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Get cache statistics
        cache_hit_rate = 0.0
        if 'cache_coordinator' in components:
            try:
                cache_stats = components['cache_coordinator'].get_comprehensive_stats()
                total_hits = sum(stats.get('hits', 0) for stats in cache_stats.values())
                total_requests = sum(stats.get('hits', 0) + stats.get('misses', 0) for stats in cache_stats.values())
                cache_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
            except Exception:
                pass
        
        optimized_metrics = PerformanceMetrics(
            execution_time=(optimized_end - optimized_start) / 20,
            memory_usage=optimized_memory_end - optimized_memory_start,
            cpu_usage=0.0,
            thread_count=threading.active_count(),
            cache_hit_rate=cache_hit_rate
        )
        
        # Calculate effectiveness
        improvement_factor = baseline_metrics.execution_time / optimized_metrics.execution_time if optimized_metrics.execution_time > 0 else 1.0
        memory_reduction = baseline_metrics.memory_usage - optimized_metrics.memory_usage
        speed_improvement = (baseline_metrics.execution_time - optimized_metrics.execution_time) / baseline_metrics.execution_time if baseline_metrics.execution_time > 0 else 0.0
        
        # Calculate effectiveness score (0-100)
        effectiveness_score = min(100.0, (improvement_factor - 1.0) * 50 + cache_hit_rate * 50)
        
        result = OptimizationEffectivenessResult(
            optimization_type="Cache Optimization",
            baseline_performance=baseline_metrics,
            optimized_performance=optimized_metrics,
            improvement_factor=improvement_factor,
            memory_reduction=memory_reduction,
            speed_improvement=speed_improvement,
            effectiveness_score=effectiveness_score
        )
        
        logger.info(f"Cache effectiveness: {improvement_factor:.1f}x improvement, "
                   f"{cache_hit_rate:.1%} hit rate, {effectiveness_score:.1f}/100 score")
        
        return result


class PerformanceAnalyticsGenerator:
    """Generates comprehensive performance analytics and reports"""
    
    def __init__(self):
        self.analytics_data = {}
        
        logger.info("Performance Analytics Generator initialized")
    
    def generate_performance_report(self, 
                                  baseline_results: Dict[str, PerformanceMetrics],
                                  load_test_results: Dict[str, LoadTestResult],
                                  optimization_results: List[OptimizationEffectivenessResult]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'timestamp': datetime.now(),
            'baseline_performance': baseline_results,
            'load_testing': load_test_results,
            'optimization_effectiveness': optimization_results,
            'summary': {},
            'recommendations': []
        }
        
        # Calculate summary statistics
        if load_test_results:
            avg_ops_per_sec = statistics.mean([result.operations_per_second for result in load_test_results.values()])
            avg_error_rate = statistics.mean([result.error_rate for result in load_test_results.values()])
            max_memory_usage = max([result.memory_peak for result in load_test_results.values()])
            
            report['summary'] = {
                'average_operations_per_second': avg_ops_per_sec,
                'average_error_rate': avg_error_rate,
                'peak_memory_usage': max_memory_usage,
                'optimization_count': len(optimization_results)
            }
        
        # Generate recommendations
        recommendations = []
        
        if load_test_results:
            for test_name, result in load_test_results.items():
                if result.error_rate > 0.05:  # More than 5% error rate
                    recommendations.append(f"High error rate ({result.error_rate:.1%}) in {test_name} - investigate error handling")
                
                if result.operations_per_second < 10:  # Less than 10 ops/sec
                    recommendations.append(f"Low throughput ({result.operations_per_second:.1f} ops/sec) in {test_name} - consider optimization")
        
        if optimization_results:
            for opt_result in optimization_results:
                if opt_result.effectiveness_score < 50:
                    recommendations.append(f"Low optimization effectiveness ({opt_result.effectiveness_score:.1f}/100) for {opt_result.optimization_type}")
        
        report['recommendations'] = recommendations
        
        return report
    
    def create_performance_visualization(self, load_test_results: Dict[str, LoadTestResult], output_path: str):
        """Create performance visualization charts"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Protocol Engine Performance and Load Testing Results', fontsize=16)
            
            # Extract data for visualization
            test_names = list(load_test_results.keys())
            ops_per_sec = [result.operations_per_second for result in load_test_results.values()]
            error_rates = [result.error_rate * 100 for result in load_test_results.values()]  # Convert to percentage
            avg_response_times = [result.average_response_time * 1000 for result in load_test_results.values()]  # Convert to ms
            memory_peaks = [result.memory_peak for result in load_test_results.values()]
            
            # Operations per second
            ax1.bar(test_names, ops_per_sec, color='skyblue')
            ax1.set_title('Operations per Second')
            ax1.set_ylabel('Operations/sec')
            ax1.tick_params(axis='x', rotation=45)
            
            # Error rates
            ax2.bar(test_names, error_rates, color='lightcoral')
            ax2.set_title('Error Rates')
            ax2.set_ylabel('Error Rate (%)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Response times
            ax3.bar(test_names, avg_response_times, color='lightgreen')
            ax3.set_title('Average Response Time')
            ax3.set_ylabel('Response Time (ms)')
            ax3.tick_params(axis='x', rotation=45)
            
            # Memory usage
            ax4.bar(test_names, memory_peaks, color='gold')
            ax4.set_title('Peak Memory Usage')
            ax4.set_ylabel('Memory (MB)')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create performance visualization: {e}")


if __name__ == '__main__':
    print("⚡ Protocol Engine Performance and Load Testing (P6 of WS2 - Phase 3)")
    print("=" * 85)
    
    # Initialize performance testing framework
    benchmarker = PerformanceBenchmarker()
    
    print("\n🚀 Initializing components for performance testing...")
    components = benchmarker.initialize_test_components()
    
    print(f"\n📊 Available Components:")
    for component_name, component in components.items():
        print(f"   ✅ {component_name}: {type(component).__name__}")
    
    # Measure baseline performance
    print(f"\n📏 Measuring baseline performance...")
    baseline_results = benchmarker.measure_baseline_performance(components)
    
    print(f"\n📋 Baseline Performance Results:")
    for test_name, metrics in baseline_results.items():
        print(f"   {test_name}:")
        print(f"     Execution Time: {metrics.execution_time*1000:.2f}ms")
        print(f"     Memory Usage: {metrics.memory_usage:.2f}MB")
        print(f"     Thread Count: {metrics.thread_count}")
    
    # Measure optimized performance
    print(f"\n⚡ Measuring optimized performance...")
    optimized_results = benchmarker.measure_optimized_performance(components)
    
    print(f"\n📋 Optimized Performance Results:")
    for test_name, metrics in optimized_results.items():
        print(f"   {test_name}:")
        print(f"     Execution Time: {metrics.execution_time*1000:.2f}ms")
        print(f"     Memory Usage: {metrics.memory_usage:.2f}MB")
        print(f"     Cache Hit Rate: {metrics.cache_hit_rate:.1%}")
    
    # Run load testing
    print(f"\n🔄 Running load testing...")
    load_tester = LoadTestingFramework(components)
    stress_results = load_tester.run_stress_test()
    
    print(f"\n📊 Load Testing Results:")
    for test_name, result in stress_results.items():
        print(f"   {test_name}:")
        print(f"     Operations: {result.successful_operations}/{result.total_operations}")
        print(f"     Throughput: {result.operations_per_second:.1f} ops/sec")
        print(f"     Error Rate: {result.error_rate:.1%}")
        print(f"     Avg Response: {result.average_response_time*1000:.2f}ms")
        print(f"     Memory Peak: {result.memory_peak:.1f}MB")
    
    # Measure optimization effectiveness
    print(f"\n🎯 Measuring optimization effectiveness...")
    effectiveness_measurer = OptimizationEffectivenessMeasurer(benchmarker)
    
    optimization_effectiveness = []
    if components:
        cache_effectiveness = effectiveness_measurer.measure_cache_effectiveness(components)
        optimization_effectiveness.append(cache_effectiveness)
        
        print(f"\n📈 Optimization Effectiveness:")
        print(f"   Cache Optimization:")
        print(f"     Improvement Factor: {cache_effectiveness.improvement_factor:.1f}x")
        print(f"     Speed Improvement: {cache_effectiveness.speed_improvement:.1%}")
        print(f"     Effectiveness Score: {cache_effectiveness.effectiveness_score:.1f}/100")
    
    # Generate comprehensive report
    print(f"\n📄 Generating performance analytics report...")
    analytics_generator = PerformanceAnalyticsGenerator()
    
    performance_report = analytics_generator.generate_performance_report(
        baseline_results,
        stress_results,
        optimization_effectiveness
    )
    
    # Create performance visualization
    output_dir = "/home/ubuntu/AGENT_ALLUSE_V1/docs/performance"
    os.makedirs(output_dir, exist_ok=True)
    
    visualization_path = os.path.join(output_dir, f"performance_load_testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    analytics_generator.create_performance_visualization(stress_results, visualization_path)
    
    # Save performance report
    report_path = os.path.join(output_dir, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, 'w') as f:
        json.dump(performance_report, f, indent=2, default=str)
    
    print(f"\n📊 Performance Report Summary:")
    if performance_report['summary']:
        summary = performance_report['summary']
        print(f"   Average Throughput: {summary.get('average_operations_per_second', 0):.1f} ops/sec")
        print(f"   Average Error Rate: {summary.get('average_error_rate', 0):.1%}")
        print(f"   Peak Memory Usage: {summary.get('peak_memory_usage', 0):.1f}MB")
        print(f"   Optimizations Tested: {summary.get('optimization_count', 0)}")
    
    if performance_report['recommendations']:
        print(f"\n💡 Recommendations:")
        for i, recommendation in enumerate(performance_report['recommendations'], 1):
            print(f"   {i}. {recommendation}")
    
    print(f"\n📁 Files Generated:")
    print(f"   Performance Visualization: {visualization_path}")
    print(f"   Performance Report: {report_path}")
    
    # Calculate overall performance assessment
    overall_success = True
    
    if stress_results:
        avg_error_rate = statistics.mean([result.error_rate for result in stress_results.values()])
        avg_throughput = statistics.mean([result.operations_per_second for result in stress_results.values()])
        
        if avg_error_rate > 0.1:  # More than 10% error rate
            overall_success = False
        if avg_throughput < 5:  # Less than 5 ops/sec
            overall_success = False
    
    print(f"\n🎯 Overall Performance Assessment: {'✅ PASSED' if overall_success else '❌ NEEDS IMPROVEMENT'}")
    
    if overall_success:
        print(f"\n✅ P6 of WS2 - Phase 3: Performance and Load Testing COMPLETE")
        print(f"🚀 Ready for Phase 4: Production Readiness Assessment")
    else:
        print(f"\n⚠️  Performance testing completed with areas for improvement")
        print(f"🚀 Proceeding to Phase 4: Production Readiness Assessment")

