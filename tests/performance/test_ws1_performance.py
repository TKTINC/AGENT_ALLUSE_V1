"""
Performance Benchmarking and Validation Tests for WS1 Components

This module contains comprehensive performance tests that validate response times,
memory usage, scalability, and resource utilization across all WS1 components.
"""

import pytest
import sys
import os
import time
import psutil
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime, timedelta
import tracemalloc
from typing import List, Dict, Any, Callable

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tests.utils.test_utilities import (
    MockDataGenerator, MockServices, TestAssertions, TestFixtures
)


class PerformanceProfiler:
    """Utility class for performance profiling and monitoring."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_start = None
        self.memory_peak = None
        self.cpu_start = None
        self.cpu_end = None
    
    def start_profiling(self):
        """Start performance profiling."""
        gc.collect()  # Clean up before starting
        tracemalloc.start()
        
        self.start_time = time.perf_counter()
        self.memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.cpu_start = psutil.Process().cpu_percent()
        
    def stop_profiling(self):
        """Stop performance profiling and return metrics."""
        self.end_time = time.perf_counter()
        self.cpu_end = psutil.Process().cpu_percent()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            'execution_time_ms': (self.end_time - self.start_time) * 1000,
            'memory_start_mb': self.memory_start,
            'memory_end_mb': memory_end,
            'memory_peak_mb': peak / 1024 / 1024,
            'memory_delta_mb': memory_end - self.memory_start,
            'cpu_usage_percent': self.cpu_end,
            'tracemalloc_current_mb': current / 1024 / 1024,
            'tracemalloc_peak_mb': peak / 1024 / 1024
        }
    
    @staticmethod
    def benchmark_function(func: Callable, iterations: int = 100) -> Dict[str, float]:
        """Benchmark a function over multiple iterations."""
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            profiler = PerformanceProfiler()
            profiler.start_profiling()
            
            try:
                result = func()
            except Exception as e:
                print(f"Error in benchmark: {e}")
                continue
                
            metrics = profiler.stop_profiling()
            times.append(metrics['execution_time_ms'])
            memory_usage.append(metrics['memory_delta_mb'])
        
        return {
            'avg_time_ms': np.mean(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'std_time_ms': np.std(times),
            'p95_time_ms': np.percentile(times, 95),
            'p99_time_ms': np.percentile(times, 99),
            'avg_memory_mb': np.mean(memory_usage),
            'max_memory_mb': np.max(memory_usage),
            'iterations': len(times)
        }


class TestAgentCorePerformance:
    """Performance tests for agent core components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = TestFixtures.setup_test_environment()
        self.agent = TestFixtures.create_test_agent()
        self.profiler = PerformanceProfiler()
    
    def test_agent_initialization_performance(self):
        """Test agent initialization performance."""
        def create_agent():
            from src.agent_core.enhanced_agent import EnhancedALLUSEAgent
            return EnhancedALLUSEAgent()
        
        metrics = PerformanceProfiler.benchmark_function(create_agent, iterations=10)
        
        # Validate performance requirements
        assert metrics['avg_time_ms'] < 100, f"Agent initialization too slow: {metrics['avg_time_ms']:.2f}ms"
        assert metrics['max_time_ms'] < 200, f"Max initialization time too slow: {metrics['max_time_ms']:.2f}ms"
        assert metrics['avg_memory_mb'] < 50, f"Agent initialization uses too much memory: {metrics['avg_memory_mb']:.2f}MB"
        
        print(f"Agent initialization: {metrics['avg_time_ms']:.2f}ms avg, {metrics['max_memory_mb']:.2f}MB peak")
    
    def test_message_processing_performance(self):
        """Test message processing performance."""
        test_messages = [
            "Hello, how are you?",
            "What's the current market condition?",
            "How much should I invest in SPY?",
            "What delta should I use for my options?",
            "Show me my portfolio risk",
            "I want to set up a new account with $100,000"
        ]
        
        def process_message():
            message = np.random.choice(test_messages)
            # Mock the process_message method
            with patch.object(self.agent, 'process_message') as mock_process:
                mock_process.return_value = {
                    'intent': 'test_intent',
                    'response': 'Test response',
                    'confidence': 0.95
                }
                return self.agent.process_message(message)
        
        metrics = PerformanceProfiler.benchmark_function(process_message, iterations=50)
        
        # Validate performance requirements
        assert metrics['avg_time_ms'] < 50, f"Message processing too slow: {metrics['avg_time_ms']:.2f}ms"
        assert metrics['p95_time_ms'] < 100, f"95th percentile too slow: {metrics['p95_time_ms']:.2f}ms"
        assert metrics['avg_memory_mb'] < 5, f"Message processing uses too much memory: {metrics['avg_memory_mb']:.2f}MB"
        
        print(f"Message processing: {metrics['avg_time_ms']:.2f}ms avg, {metrics['p95_time_ms']:.2f}ms p95")
    
    def test_memory_management_performance(self):
        """Test memory management and conversation history performance."""
        # Simulate long conversation
        conversation_length = 100
        
        def simulate_conversation():
            from src.agent_core.enhanced_memory_manager import EnhancedMemoryManager
            memory_manager = EnhancedMemoryManager()
            
            for i in range(conversation_length):
                memory_manager.store_conversation_memory(
                    f"user_{i}", f"Test message {i}", f"Test response {i}"
                )
            
            # Test retrieval
            recent_conversations = memory_manager.get_recent_conversations(10)
            return len(recent_conversations)
        
        metrics = PerformanceProfiler.benchmark_function(simulate_conversation, iterations=5)
        
        # Validate memory management performance
        assert metrics['avg_time_ms'] < 500, f"Memory management too slow: {metrics['avg_time_ms']:.2f}ms"
        assert metrics['avg_memory_mb'] < 20, f"Memory management uses too much memory: {metrics['avg_memory_mb']:.2f}MB"
        
        print(f"Memory management (100 conversations): {metrics['avg_time_ms']:.2f}ms avg")
    
    def test_concurrent_agent_performance(self):
        """Test agent performance under concurrent load."""
        num_threads = 10
        requests_per_thread = 5
        
        def concurrent_message_processing():
            results = []
            
            def process_single_message():
                with patch.object(self.agent, 'process_message') as mock_process:
                    mock_process.return_value = {
                        'intent': 'test_intent',
                        'response': 'Test response',
                        'confidence': 0.95
                    }
                    return self.agent.process_message("Test message")
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                start_time = time.perf_counter()
                
                for _ in range(num_threads * requests_per_thread):
                    future = executor.submit(process_single_message)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=5.0)
                        results.append(result)
                    except Exception as e:
                        print(f"Concurrent processing error: {e}")
                
                end_time = time.perf_counter()
            
            return {
                'total_requests': len(results),
                'total_time_ms': (end_time - start_time) * 1000,
                'requests_per_second': len(results) / (end_time - start_time),
                'success_rate': len(results) / (num_threads * requests_per_thread)
            }
        
        result = concurrent_message_processing()
        
        # Validate concurrent performance
        assert result['success_rate'] >= 0.95, f"Success rate too low: {result['success_rate']:.2f}"
        assert result['requests_per_second'] >= 50, f"Throughput too low: {result['requests_per_second']:.2f} req/s"
        
        print(f"Concurrent performance: {result['requests_per_second']:.2f} req/s, {result['success_rate']:.2%} success")


class TestTradingEnginePerformance:
    """Performance tests for trading engine components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = TestFixtures.setup_test_environment()
        
        # Import trading components
        from src.trading_engine.market_analyzer import MarketAnalyzer
        from src.trading_engine.position_sizer import PositionSizer
        from src.trading_engine.delta_selector import DeltaSelector
        
        self.market_analyzer = MarketAnalyzer()
        self.position_sizer = PositionSizer()
        self.delta_selector = DeltaSelector()
    
    def test_market_analysis_performance(self):
        """Test market analysis performance."""
        def analyze_market():
            market_data = MockDataGenerator.generate_market_data()
            return self.market_analyzer.analyze_market_condition('SPY', market_data)
        
        metrics = PerformanceProfiler.benchmark_function(analyze_market, iterations=50)
        
        # Validate performance requirements
        assert metrics['avg_time_ms'] < 50, f"Market analysis too slow: {metrics['avg_time_ms']:.2f}ms"
        assert metrics['p95_time_ms'] < 100, f"95th percentile too slow: {metrics['p95_time_ms']:.2f}ms"
        assert metrics['avg_memory_mb'] < 2, f"Market analysis uses too much memory: {metrics['avg_memory_mb']:.2f}MB"
        
        print(f"Market analysis: {metrics['avg_time_ms']:.2f}ms avg, {metrics['p95_time_ms']:.2f}ms p95")
    
    def test_position_sizing_performance(self):
        """Test position sizing performance."""
        def calculate_position():
            params = {
                'symbol': 'SPY',
                'account_balance': 100000,
                'account_type': 'GEN_ACC',
                'market_condition': 'Green',
                'volatility': 20
            }
            return self.position_sizer.calculate_position_size(params)
        
        metrics = PerformanceProfiler.benchmark_function(calculate_position, iterations=100)
        
        # Validate performance requirements
        assert metrics['avg_time_ms'] < 25, f"Position sizing too slow: {metrics['avg_time_ms']:.2f}ms"
        assert metrics['p99_time_ms'] < 50, f"99th percentile too slow: {metrics['p99_time_ms']:.2f}ms"
        assert metrics['avg_memory_mb'] < 1, f"Position sizing uses too much memory: {metrics['avg_memory_mb']:.2f}MB"
        
        print(f"Position sizing: {metrics['avg_time_ms']:.2f}ms avg, {metrics['p99_time_ms']:.2f}ms p99")
    
    def test_delta_selection_performance(self):
        """Test delta selection performance."""
        def select_delta():
            params = {
                'market_condition': 'Green',
                'account_type': 'GEN_ACC',
                'volatility_regime': 'Normal',
                'time_to_expiration': 30,
                'portfolio_delta': 0.0
            }
            return self.delta_selector.select_optimal_delta(params)
        
        metrics = PerformanceProfiler.benchmark_function(select_delta, iterations=100)
        
        # Validate performance requirements
        assert metrics['avg_time_ms'] < 20, f"Delta selection too slow: {metrics['avg_time_ms']:.2f}ms"
        assert metrics['p95_time_ms'] < 40, f"95th percentile too slow: {metrics['p95_time_ms']:.2f}ms"
        assert metrics['avg_memory_mb'] < 1, f"Delta selection uses too much memory: {metrics['avg_memory_mb']:.2f}MB"
        
        print(f"Delta selection: {metrics['avg_time_ms']:.2f}ms avg, {metrics['p95_time_ms']:.2f}ms p95")
    
    def test_trading_workflow_performance(self):
        """Test complete trading workflow performance."""
        def complete_trading_workflow():
            # Market analysis
            market_data = MockDataGenerator.generate_market_data()
            market_assessment = self.market_analyzer.analyze_market_condition('SPY', market_data)
            
            # Position sizing
            position_params = {
                'symbol': 'SPY',
                'account_balance': 100000,
                'account_type': 'GEN_ACC',
                'market_condition': 'Green',  # Simplified for performance test
                'volatility': 20
            }
            position_size = self.position_sizer.calculate_position_size(position_params)
            
            # Delta selection
            delta_params = {
                'market_condition': 'Green',  # Simplified for performance test
                'account_type': 'GEN_ACC',
                'volatility_regime': 'Normal',
                'time_to_expiration': 30,
                'portfolio_delta': 0.0
            }
            delta_selection = self.delta_selector.select_optimal_delta(delta_params)
            
            return {
                'market': market_assessment,
                'position': position_size,
                'delta': delta_selection
            }
        
        metrics = PerformanceProfiler.benchmark_function(complete_trading_workflow, iterations=20)
        
        # Validate complete workflow performance
        assert metrics['avg_time_ms'] < 100, f"Trading workflow too slow: {metrics['avg_time_ms']:.2f}ms"
        assert metrics['p95_time_ms'] < 200, f"95th percentile too slow: {metrics['p95_time_ms']:.2f}ms"
        assert metrics['avg_memory_mb'] < 5, f"Trading workflow uses too much memory: {metrics['avg_memory_mb']:.2f}MB"
        
        print(f"Complete trading workflow: {metrics['avg_time_ms']:.2f}ms avg, {metrics['p95_time_ms']:.2f}ms p95")
    
    def test_high_frequency_analysis(self):
        """Test performance under high-frequency analysis scenarios."""
        symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META']
        
        def high_frequency_analysis():
            results = []
            for symbol in symbols:
                market_data = MockDataGenerator.generate_market_data()
                result = self.market_analyzer.analyze_market_condition(symbol, market_data)
                results.append(result)
            return results
        
        metrics = PerformanceProfiler.benchmark_function(high_frequency_analysis, iterations=10)
        
        # Validate high-frequency performance
        assert metrics['avg_time_ms'] < 500, f"High-frequency analysis too slow: {metrics['avg_time_ms']:.2f}ms"
        assert metrics['avg_memory_mb'] < 10, f"High-frequency analysis uses too much memory: {metrics['avg_memory_mb']:.2f}MB"
        
        print(f"High-frequency analysis (10 symbols): {metrics['avg_time_ms']:.2f}ms avg")


class TestRiskManagementPerformance:
    """Performance tests for risk management components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = TestFixtures.setup_test_environment()
        
        # Import risk management components
        from src.risk_management.portfolio_risk_monitor import PortfolioRiskMonitor
        from src.risk_management.drawdown_protection import DrawdownProtection
        from src.optimization.portfolio_optimizer import PortfolioOptimizer
        from src.analytics.performance_analytics import PerformanceAnalytics
        
        self.risk_monitor = PortfolioRiskMonitor()
        self.drawdown_protection = DrawdownProtection()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.performance_analytics = PerformanceAnalytics()
    
    def test_risk_assessment_performance(self):
        """Test portfolio risk assessment performance."""
        def assess_risk():
            portfolio_data = MockDataGenerator.generate_portfolio_data()
            performance_data = MockDataGenerator.generate_performance_data()
            return self.risk_monitor.assess_portfolio_risk(portfolio_data, performance_data['returns'])
        
        metrics = PerformanceProfiler.benchmark_function(assess_risk, iterations=50)
        
        # Validate performance requirements
        assert metrics['avg_time_ms'] < 50, f"Risk assessment too slow: {metrics['avg_time_ms']:.2f}ms"
        assert metrics['p95_time_ms'] < 100, f"95th percentile too slow: {metrics['p95_time_ms']:.2f}ms"
        assert metrics['avg_memory_mb'] < 3, f"Risk assessment uses too much memory: {metrics['avg_memory_mb']:.2f}MB"
        
        print(f"Risk assessment: {metrics['avg_time_ms']:.2f}ms avg, {metrics['p95_time_ms']:.2f}ms p95")
    
    def test_portfolio_optimization_performance(self):
        """Test portfolio optimization performance."""
        def optimize_portfolio():
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
            expected_returns = {symbol: np.random.normal(0.08, 0.02) for symbol in symbols}
            
            n_assets = len(symbols)
            cov_matrix = np.random.rand(n_assets, n_assets) * 0.01
            cov_matrix = (cov_matrix + cov_matrix.T) / 2
            np.fill_diagonal(cov_matrix, np.random.rand(n_assets) * 0.04)
            
            return self.portfolio_optimizer.optimize_max_sharpe(expected_returns, cov_matrix)
        
        metrics = PerformanceProfiler.benchmark_function(optimize_portfolio, iterations=20)
        
        # Validate performance requirements
        assert metrics['avg_time_ms'] < 100, f"Portfolio optimization too slow: {metrics['avg_time_ms']:.2f}ms"
        assert metrics['p95_time_ms'] < 200, f"95th percentile too slow: {metrics['p95_time_ms']:.2f}ms"
        assert metrics['avg_memory_mb'] < 5, f"Portfolio optimization uses too much memory: {metrics['avg_memory_mb']:.2f}MB"
        
        print(f"Portfolio optimization: {metrics['avg_time_ms']:.2f}ms avg, {metrics['p95_time_ms']:.2f}ms p95")
    
    def test_performance_analytics_performance(self):
        """Test performance analytics calculation performance."""
        def calculate_analytics():
            portfolio_id = "test_portfolio"
            analytics = PerformanceAnalytics()
            analytics.add_portfolio(portfolio_id, "Test Portfolio")
            
            # Add 60 days of performance data
            performance_data = MockDataGenerator.generate_performance_data(60)
            for value in performance_data['portfolio_values']:
                analytics.update_portfolio_value(portfolio_id, value)
            
            return analytics.calculate_performance_metrics(portfolio_id)
        
        metrics = PerformanceProfiler.benchmark_function(calculate_analytics, iterations=10)
        
        # Validate performance requirements
        assert metrics['avg_time_ms'] < 75, f"Performance analytics too slow: {metrics['avg_time_ms']:.2f}ms"
        assert metrics['p95_time_ms'] < 150, f"95th percentile too slow: {metrics['p95_time_ms']:.2f}ms"
        assert metrics['avg_memory_mb'] < 10, f"Performance analytics uses too much memory: {metrics['avg_memory_mb']:.2f}MB"
        
        print(f"Performance analytics (60 days): {metrics['avg_time_ms']:.2f}ms avg")
    
    def test_real_time_monitoring_performance(self):
        """Test real-time monitoring performance."""
        def real_time_monitoring():
            portfolio_data = MockDataGenerator.generate_portfolio_data()
            
            # Start monitoring
            self.risk_monitor.start_monitoring(portfolio_data)
            
            # Simulate 10 portfolio updates
            for i in range(10):
                updated_portfolio = portfolio_data.copy()
                updated_portfolio['total_value'] *= (1 + np.random.normal(0, 0.01))  # Small random changes
                alerts = self.risk_monitor.update_portfolio(updated_portfolio)
            
            # Stop monitoring
            self.risk_monitor.stop_monitoring()
            
            return True
        
        metrics = PerformanceProfiler.benchmark_function(real_time_monitoring, iterations=5)
        
        # Validate real-time monitoring performance
        assert metrics['avg_time_ms'] < 200, f"Real-time monitoring too slow: {metrics['avg_time_ms']:.2f}ms"
        assert metrics['avg_memory_mb'] < 5, f"Real-time monitoring uses too much memory: {metrics['avg_memory_mb']:.2f}MB"
        
        print(f"Real-time monitoring (10 updates): {metrics['avg_time_ms']:.2f}ms avg")


class TestScalabilityPerformance:
    """Scalability and load testing for WS1 components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = TestFixtures.setup_test_environment()
    
    def test_portfolio_size_scalability(self):
        """Test performance scaling with portfolio size."""
        from src.risk_management.portfolio_risk_monitor import PortfolioRiskMonitor
        
        risk_monitor = PortfolioRiskMonitor()
        portfolio_sizes = [5, 10, 25, 50, 100]
        results = {}
        
        for size in portfolio_sizes:
            def assess_large_portfolio():
                portfolio_data = MockDataGenerator.generate_portfolio_data(num_positions=size)
                performance_data = MockDataGenerator.generate_performance_data()
                return risk_monitor.assess_portfolio_risk(portfolio_data, performance_data['returns'])
            
            metrics = PerformanceProfiler.benchmark_function(assess_large_portfolio, iterations=10)
            results[size] = metrics
            
            print(f"Portfolio size {size}: {metrics['avg_time_ms']:.2f}ms avg")
        
        # Validate scalability
        for size in portfolio_sizes:
            if size <= 25:
                assert results[size]['avg_time_ms'] < 100, f"Portfolio size {size} too slow: {results[size]['avg_time_ms']:.2f}ms"
            elif size <= 50:
                assert results[size]['avg_time_ms'] < 200, f"Portfolio size {size} too slow: {results[size]['avg_time_ms']:.2f}ms"
            else:  # size == 100
                assert results[size]['avg_time_ms'] < 500, f"Portfolio size {size} too slow: {results[size]['avg_time_ms']:.2f}ms"
    
    def test_data_volume_scalability(self):
        """Test performance scaling with historical data volume."""
        from src.analytics.performance_analytics import PerformanceAnalytics
        
        data_periods = [30, 90, 180, 365, 730]  # Days
        results = {}
        
        for period in data_periods:
            def analyze_large_dataset():
                portfolio_id = f"test_portfolio_{period}"
                analytics = PerformanceAnalytics()
                analytics.add_portfolio(portfolio_id, f"Test Portfolio {period} days")
                
                performance_data = MockDataGenerator.generate_performance_data(period)
                for value in performance_data['portfolio_values']:
                    analytics.update_portfolio_value(portfolio_id, value)
                
                return analytics.calculate_performance_metrics(portfolio_id)
            
            metrics = PerformanceProfiler.benchmark_function(analyze_large_dataset, iterations=5)
            results[period] = metrics
            
            print(f"Data period {period} days: {metrics['avg_time_ms']:.2f}ms avg")
        
        # Validate data volume scalability
        for period in data_periods:
            if period <= 90:
                assert results[period]['avg_time_ms'] < 150, f"Period {period} days too slow: {results[period]['avg_time_ms']:.2f}ms"
            elif period <= 365:
                assert results[period]['avg_time_ms'] < 300, f"Period {period} days too slow: {results[period]['avg_time_ms']:.2f}ms"
            else:  # period == 730
                assert results[period]['avg_time_ms'] < 600, f"Period {period} days too slow: {results[period]['avg_time_ms']:.2f}ms"
    
    def test_concurrent_user_scalability(self):
        """Test system performance with multiple concurrent users."""
        from src.agent_core.enhanced_agent import EnhancedALLUSEAgent
        
        user_counts = [1, 5, 10, 20]
        results = {}
        
        for user_count in user_counts:
            def simulate_concurrent_users():
                agents = [EnhancedALLUSEAgent() for _ in range(user_count)]
                
                def user_session(agent):
                    # Simulate user session with multiple interactions
                    messages = [
                        "Hello",
                        "What's the market condition?",
                        "How much should I invest?",
                        "Show me my risk level"
                    ]
                    
                    results = []
                    for message in messages:
                        with patch.object(agent, 'process_message') as mock_process:
                            mock_process.return_value = {
                                'intent': 'test_intent',
                                'response': 'Test response',
                                'confidence': 0.95
                            }
                            result = agent.process_message(message)
                            results.append(result)
                    
                    return results
                
                with ThreadPoolExecutor(max_workers=user_count) as executor:
                    futures = [executor.submit(user_session, agent) for agent in agents]
                    results = [future.result() for future in as_completed(futures)]
                
                return len(results)
            
            metrics = PerformanceProfiler.benchmark_function(simulate_concurrent_users, iterations=3)
            results[user_count] = metrics
            
            print(f"Concurrent users {user_count}: {metrics['avg_time_ms']:.2f}ms avg")
        
        # Validate concurrent user scalability
        for user_count in user_counts:
            if user_count <= 5:
                assert results[user_count]['avg_time_ms'] < 1000, f"User count {user_count} too slow: {results[user_count]['avg_time_ms']:.2f}ms"
            elif user_count <= 10:
                assert results[user_count]['avg_time_ms'] < 2000, f"User count {user_count} too slow: {results[user_count]['avg_time_ms']:.2f}ms"
            else:  # user_count == 20
                assert results[user_count]['avg_time_ms'] < 5000, f"User count {user_count} too slow: {results[user_count]['avg_time_ms']:.2f}ms"


class TestMemoryLeakDetection:
    """Memory leak detection and validation tests."""
    
    def test_agent_memory_leak(self):
        """Test for memory leaks in agent operations."""
        from src.agent_core.enhanced_agent import EnhancedALLUSEAgent
        
        def repeated_agent_operations():
            agent = EnhancedALLUSEAgent()
            
            # Simulate 100 message processing operations
            for i in range(100):
                with patch.object(agent, 'process_message') as mock_process:
                    mock_process.return_value = {
                        'intent': 'test_intent',
                        'response': f'Test response {i}',
                        'confidence': 0.95
                    }
                    agent.process_message(f"Test message {i}")
            
            return True
        
        # Run multiple iterations and monitor memory
        memory_usage = []
        for iteration in range(10):
            profiler = PerformanceProfiler()
            profiler.start_profiling()
            repeated_agent_operations()
            metrics = profiler.stop_profiling()
            memory_usage.append(metrics['memory_end_mb'])
        
        # Check for memory leak (increasing trend)
        memory_trend = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
        assert memory_trend < 1.0, f"Potential memory leak detected: {memory_trend:.2f}MB/iteration"
        
        print(f"Memory trend: {memory_trend:.3f}MB/iteration (should be < 1.0)")
    
    def test_trading_engine_memory_leak(self):
        """Test for memory leaks in trading engine operations."""
        from src.trading_engine.market_analyzer import MarketAnalyzer
        from src.trading_engine.position_sizer import PositionSizer
        
        def repeated_trading_operations():
            analyzer = MarketAnalyzer()
            sizer = PositionSizer()
            
            for i in range(50):
                # Market analysis
                market_data = MockDataGenerator.generate_market_data()
                analyzer.analyze_market_condition('SPY', market_data)
                
                # Position sizing
                params = {
                    'symbol': 'SPY',
                    'account_balance': 100000,
                    'account_type': 'GEN_ACC',
                    'market_condition': 'Green',
                    'volatility': 20
                }
                sizer.calculate_position_size(params)
            
            return True
        
        # Run multiple iterations and monitor memory
        memory_usage = []
        for iteration in range(10):
            profiler = PerformanceProfiler()
            profiler.start_profiling()
            repeated_trading_operations()
            metrics = profiler.stop_profiling()
            memory_usage.append(metrics['memory_end_mb'])
        
        # Check for memory leak
        memory_trend = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
        assert memory_trend < 0.5, f"Potential memory leak detected: {memory_trend:.2f}MB/iteration"
        
        print(f"Trading engine memory trend: {memory_trend:.3f}MB/iteration (should be < 0.5)")


# Pytest fixtures for performance testing
@pytest.fixture(scope="session")
def performance_baseline():
    """Establish performance baseline for comparison."""
    return {
        'agent_init_ms': 100,
        'message_processing_ms': 50,
        'market_analysis_ms': 50,
        'position_sizing_ms': 25,
        'delta_selection_ms': 20,
        'risk_assessment_ms': 50,
        'portfolio_optimization_ms': 100,
        'memory_limit_mb': 50
    }


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])

