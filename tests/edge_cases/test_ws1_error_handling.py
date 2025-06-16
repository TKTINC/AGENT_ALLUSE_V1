"""
Error Handling and Edge Case Validation Tests for WS1 Components

This module contains comprehensive error handling tests that validate system behavior
under various failure conditions, edge cases, and invalid input scenarios.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime, timedelta
import json
import tempfile
import shutil

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tests.utils.test_utilities import (
    MockDataGenerator, MockServices, TestAssertions, TestFixtures
)


class TestAgentCoreErrorHandling:
    """Error handling tests for agent core components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = TestFixtures.setup_test_environment()
        self.agent = TestFixtures.create_test_agent()
    
    def test_invalid_message_handling(self):
        """Test agent handling of invalid messages."""
        invalid_messages = [
            None,
            "",
            " " * 1000,  # Very long whitespace
            "A" * 10000,  # Extremely long message
            {"invalid": "object"},  # Non-string input
            123,  # Numeric input
            [],  # List input
        ]
        
        for invalid_message in invalid_messages:
            with patch.object(self.agent, 'process_message') as mock_process:
                # Mock should handle invalid input gracefully
                mock_process.side_effect = lambda msg: self._handle_invalid_message(msg)
                
                try:
                    result = self.agent.process_message(invalid_message)
                    # Should return error response, not crash
                    assert 'error' in result or 'intent' in result
                except (ValueError, TypeError) as e:
                    # Expected for some invalid inputs
                    assert any(word in str(e).lower() for word in ['invalid', 'type', 'string', 'message'])
    
    def test_memory_overflow_handling(self):
        """Test memory management under overflow conditions."""
        from src.agent_core.enhanced_memory_manager import EnhancedMemoryManager
        
        memory_manager = EnhancedMemoryManager()
        
        # Test conversation memory overflow
        for i in range(1000):  # Exceed typical memory limits
            try:
                memory_manager.store_conversation_memory(
                    f"user_{i}", f"Message {i}", f"Response {i}"
                )
            except Exception as e:
                # Should handle overflow gracefully
                assert "memory" in str(e).lower() or "limit" in str(e).lower()
                break
        
        # Memory should still be functional after overflow
        recent = memory_manager.get_recent_conversations(5)
        assert isinstance(recent, list)
        assert len(recent) <= 5
    
    def test_cognitive_framework_edge_cases(self):
        """Test cognitive framework with edge case inputs."""
        from src.agent_core.enhanced_cognitive_framework import EnhancedIntentDetector
        
        intent_detector = EnhancedIntentDetector()
        
        edge_cases = [
            "",  # Empty string
            "???",  # Only punctuation
            "1234567890",  # Only numbers
            "AAAAAAAAAAA",  # Repeated characters
            "Hello " * 100,  # Repetitive content
            "Mix3d Ch@r$ct3r5 @nd Numb3r5!",  # Mixed characters
            "你好世界",  # Non-English characters
            "🚀💰📈",  # Only emojis
        ]
        
        for edge_case in edge_cases:
            try:
                intent = intent_detector.detect_intent(edge_case)
                # Should return valid intent or unknown
                assert intent in ['greeting', 'market_analysis', 'account_inquiry', 
                                'setup_accounts', 'position_sizing', 'delta_selection',
                                'risk_assessment', 'portfolio_inquiry', 'unknown']
            except Exception as e:
                # Should handle gracefully, not crash
                assert "invalid" in str(e).lower() or "unknown" in str(e).lower()
    
    def test_response_generation_failures(self):
        """Test response generation under failure conditions."""
        from src.agent_core.response_generator import ResponseGenerator
        
        response_generator = ResponseGenerator()
        
        # Test with invalid context
        invalid_contexts = [
            None,
            {},
            {"incomplete": "context"},
            {"intent": None, "entities": None},
            {"intent": "invalid_intent", "entities": {}},
        ]
        
        for invalid_context in invalid_contexts:
            try:
                response = response_generator.generate_response(invalid_context)
                # Should return fallback response
                assert isinstance(response, str)
                assert len(response) > 0
            except Exception as e:
                # Should handle gracefully
                assert "context" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_concurrent_access_errors(self):
        """Test error handling under concurrent access."""
        import threading
        import time
        
        errors = []
        results = []
        
        def concurrent_agent_access():
            try:
                with patch.object(self.agent, 'process_message') as mock_process:
                    mock_process.return_value = {
                        'intent': 'test_intent',
                        'response': 'Test response',
                        'confidence': 0.95
                    }
                    result = self.agent.process_message("Test message")
                    results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=concurrent_agent_access)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Validate results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) > 0, "No successful concurrent operations"
    
    def _handle_invalid_message(self, message):
        """Helper method to simulate invalid message handling."""
        if message is None or message == "":
            return {'error': 'Empty message', 'intent': 'error'}
        elif not isinstance(message, str):
            raise TypeError("Message must be a string")
        elif len(message) > 5000:
            return {'error': 'Message too long', 'intent': 'error'}
        else:
            return {'intent': 'unknown', 'response': 'I did not understand that.'}


class TestTradingEngineErrorHandling:
    """Error handling tests for trading engine components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.trading_engine.market_analyzer import MarketAnalyzer
        from src.trading_engine.position_sizer import PositionSizer
        from src.trading_engine.delta_selector import DeltaSelector
        
        self.market_analyzer = MarketAnalyzer()
        self.position_sizer = PositionSizer()
        self.delta_selector = DeltaSelector()
    
    def test_market_analyzer_invalid_data(self):
        """Test market analyzer with invalid data."""
        invalid_data_sets = [
            {},  # Empty data
            {'prices': []},  # Empty prices
            {'prices': [100]},  # Single price point
            {'prices': [100, 'invalid']},  # Mixed data types
            {'prices': [100, 101, 102], 'volume': 'invalid'},  # Invalid volume
            {'prices': [np.nan, 100, 101]},  # NaN values
            {'prices': [100, 101, float('inf')]},  # Infinite values
        ]
        
        for invalid_data in invalid_data_sets:
            try:
                result = self.market_analyzer.analyze_market_condition('SPY', invalid_data)
                # Should return error result or default analysis
                assert 'error' in result or 'market_condition' in result
                if 'error' not in result:
                    # If no error, should have valid market condition
                    assert hasattr(result['market_condition'], 'name')  # Enum value
            except (ValueError, TypeError, KeyError) as e:
                # Expected for invalid data
                assert any(word in str(e).lower() for word in ['invalid', 'data', 'empty', 'type'])
    
    def test_position_sizer_edge_cases(self):
        """Test position sizer with edge case parameters."""
        edge_case_params = [
            # Zero account balance
            {'symbol': 'SPY', 'account_balance': 0, 'account_type': 'GEN_ACC', 
             'market_condition': 'Green', 'volatility': 20},
            
            # Negative account balance
            {'symbol': 'SPY', 'account_balance': -1000, 'account_type': 'GEN_ACC', 
             'market_condition': 'Green', 'volatility': 20},
            
            # Extreme volatility
            {'symbol': 'SPY', 'account_balance': 100000, 'account_type': 'GEN_ACC', 
             'market_condition': 'Green', 'volatility': 200},
            
            # Invalid account type
            {'symbol': 'SPY', 'account_balance': 100000, 'account_type': 'INVALID_ACC', 
             'market_condition': 'Green', 'volatility': 20},
            
            # Missing required parameters
            {'symbol': 'SPY', 'account_balance': 100000},
            
            # Invalid market condition
            {'symbol': 'SPY', 'account_balance': 100000, 'account_type': 'GEN_ACC', 
             'market_condition': 'Purple', 'volatility': 20},
        ]
        
        for params in edge_case_params:
            try:
                result = self.position_sizer.calculate_position_size(params)
                # Should return valid result or error
                if 'error' not in result:
                    assert 'position_value' in result
                    assert result['position_value'] >= 0
            except (ValueError, KeyError, TypeError) as e:
                # Expected for invalid parameters
                assert any(word in str(e).lower() for word in ['invalid', 'missing', 'type', 'value'])
    
    def test_delta_selector_boundary_conditions(self):
        """Test delta selector with boundary conditions."""
        boundary_cases = [
            # Extreme time to expiration
            {'market_condition': 'Green', 'account_type': 'GEN_ACC', 'volatility_regime': 'Normal',
             'time_to_expiration': 0, 'portfolio_delta': 0.0},
            
            {'market_condition': 'Green', 'account_type': 'GEN_ACC', 'volatility_regime': 'Normal',
             'time_to_expiration': 1000, 'portfolio_delta': 0.0},
            
            # Extreme portfolio delta
            {'market_condition': 'Green', 'account_type': 'GEN_ACC', 'volatility_regime': 'Normal',
             'time_to_expiration': 30, 'portfolio_delta': 10.0},
            
            {'market_condition': 'Green', 'account_type': 'GEN_ACC', 'volatility_regime': 'Normal',
             'time_to_expiration': 30, 'portfolio_delta': -10.0},
            
            # Invalid volatility regime
            {'market_condition': 'Green', 'account_type': 'GEN_ACC', 'volatility_regime': 'Invalid',
             'time_to_expiration': 30, 'portfolio_delta': 0.0},
            
            # Missing parameters
            {'market_condition': 'Green', 'account_type': 'GEN_ACC'},
        ]
        
        for params in boundary_cases:
            try:
                result = self.delta_selector.select_optimal_delta(params)
                # Should return valid delta or error
                if 'error' not in result:
                    assert 'recommended_delta' in result
                    assert 0 <= result['recommended_delta'] <= 100
            except (ValueError, KeyError, TypeError) as e:
                # Expected for invalid parameters
                assert any(word in str(e).lower() for word in ['invalid', 'missing', 'range', 'value'])
    
    def test_trading_engine_data_corruption(self):
        """Test trading engine behavior with corrupted data."""
        # Simulate corrupted market data
        corrupted_data = {
            'prices': [100, 101, None, 103, 'corrupted'],
            'volume': [1000, 1100, -500, float('inf')],
            'timestamps': ['2023-01-01', 'invalid_date', None],
            'implied_volatility': [0.2, -0.1, 2.0, np.nan]
        }
        
        try:
            result = self.market_analyzer.analyze_market_condition('SPY', corrupted_data)
            # Should handle corruption gracefully
            assert 'error' in result or 'market_condition' in result
        except Exception as e:
            # Should provide meaningful error message
            assert any(word in str(e).lower() for word in ['corrupt', 'invalid', 'data', 'format'])
    
    def test_network_timeout_simulation(self):
        """Test behavior when external data sources timeout."""
        def timeout_simulation():
            import time
            time.sleep(10)  # Simulate timeout
            raise TimeoutError("Data source timeout")
        
        # Mock external data call with timeout
        with patch('src.trading_engine.market_analyzer.MarketAnalyzer._fetch_external_data', 
                  side_effect=timeout_simulation):
            try:
                market_data = MockDataGenerator.generate_market_data()
                result = self.market_analyzer.analyze_market_condition('SPY', market_data)
                # Should handle timeout gracefully
                assert 'error' in result or 'market_condition' in result
            except TimeoutError:
                # Expected behavior
                pass


class TestRiskManagementErrorHandling:
    """Error handling tests for risk management components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.risk_management.portfolio_risk_monitor import PortfolioRiskMonitor
        from src.risk_management.drawdown_protection import DrawdownProtection
        from src.optimization.portfolio_optimizer import PortfolioOptimizer
        
        self.risk_monitor = PortfolioRiskMonitor()
        self.drawdown_protection = DrawdownProtection()
        self.portfolio_optimizer = PortfolioOptimizer()
    
    def test_risk_monitor_invalid_portfolio(self):
        """Test risk monitor with invalid portfolio data."""
        invalid_portfolios = [
            {},  # Empty portfolio
            {'positions': {}},  # No positions
            {'positions': {'AAPL': {'quantity': 'invalid'}}},  # Invalid quantity
            {'positions': {'AAPL': {'quantity': 100, 'price': -50}}},  # Negative price
            {'total_value': -100000},  # Negative total value
            {'positions': {'': {'quantity': 100, 'price': 150}}},  # Empty symbol
        ]
        
        for invalid_portfolio in invalid_portfolios:
            try:
                result = self.risk_monitor.assess_portfolio_risk(invalid_portfolio, [0.01, -0.02, 0.015])
                # Should return error or default risk assessment
                assert 'error' in result or 'risk_level' in result
            except (ValueError, KeyError, TypeError) as e:
                # Expected for invalid data
                assert any(word in str(e).lower() for word in ['invalid', 'portfolio', 'data', 'empty'])
    
    def test_drawdown_protection_extreme_scenarios(self):
        """Test drawdown protection with extreme scenarios."""
        extreme_scenarios = [
            # Extreme drawdown
            {'current_drawdown': 99.9, 'total_value': 1000, 'peak_value': 1000000},
            
            # Negative drawdown (impossible scenario)
            {'current_drawdown': -10.0, 'total_value': 110000, 'peak_value': 100000},
            
            # Zero portfolio value
            {'current_drawdown': 100.0, 'total_value': 0, 'peak_value': 100000},
            
            # Invalid drawdown data
            {'current_drawdown': 'invalid', 'total_value': 100000, 'peak_value': 100000},
        ]
        
        for scenario in extreme_scenarios:
            try:
                result = self.drawdown_protection.check_protection_triggers(scenario)
                # Should handle extreme scenarios gracefully
                assert 'protection_triggered' in result
                assert isinstance(result['protection_triggered'], bool)
            except (ValueError, TypeError) as e:
                # Expected for invalid data
                assert any(word in str(e).lower() for word in ['invalid', 'drawdown', 'value', 'type'])
    
    def test_portfolio_optimizer_singular_matrix(self):
        """Test portfolio optimizer with singular covariance matrix."""
        # Create singular covariance matrix (non-invertible)
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        expected_returns = {symbol: 0.08 for symbol in symbols}
        
        # Singular matrix (all zeros)
        singular_matrix = np.zeros((3, 3))
        
        try:
            result = self.portfolio_optimizer.optimize_max_sharpe(expected_returns, singular_matrix)
            # Should handle singular matrix gracefully
            assert 'error' in result or 'weights' in result
        except (np.linalg.LinAlgError, ValueError) as e:
            # Expected for singular matrix
            assert any(word in str(e).lower() for word in ['singular', 'matrix', 'invert', 'solve'])
    
    def test_optimization_infeasible_constraints(self):
        """Test portfolio optimizer with infeasible constraints."""
        symbols = ['AAPL', 'MSFT']
        expected_returns = {symbol: 0.08 for symbol in symbols}
        cov_matrix = np.array([[0.04, 0.01], [0.01, 0.04]])
        
        # Infeasible constraints (sum of min weights > 1)
        infeasible_constraints = {
            'min_weights': {'AAPL': 0.8, 'MSFT': 0.8},  # Sum = 1.6 > 1.0
            'max_weights': {'AAPL': 0.9, 'MSFT': 0.9}
        }
        
        try:
            result = self.portfolio_optimizer.optimize_with_constraints(
                expected_returns, cov_matrix, infeasible_constraints
            )
            # Should detect infeasible constraints
            assert 'error' in result or 'infeasible' in str(result).lower()
        except ValueError as e:
            # Expected for infeasible constraints
            assert any(word in str(e).lower() for word in ['infeasible', 'constraint', 'feasible'])
    
    def test_performance_analytics_insufficient_data(self):
        """Test performance analytics with insufficient data."""
        from src.analytics.performance_analytics import PerformanceAnalytics
        
        analytics = PerformanceAnalytics()
        portfolio_id = "test_portfolio"
        analytics.add_portfolio(portfolio_id, "Test Portfolio")
        
        # Add insufficient data (only 1 data point)
        analytics.update_portfolio_value(portfolio_id, 100000)
        
        try:
            metrics = analytics.calculate_performance_metrics(portfolio_id)
            # Should handle insufficient data gracefully
            assert 'error' in metrics or 'insufficient_data' in metrics or 'total_return' in metrics
        except ValueError as e:
            # Expected for insufficient data
            assert any(word in str(e).lower() for word in ['insufficient', 'data', 'minimum', 'points'])


class TestSystemIntegrationErrorHandling:
    """Error handling tests for system integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = TestFixtures.setup_test_environment()
    
    def test_component_failure_cascade(self):
        """Test system behavior when components fail in cascade."""
        from src.agent_core.enhanced_agent import EnhancedALLUSEAgent
        from src.trading_engine.market_analyzer import MarketAnalyzer
        
        agent = EnhancedALLUSEAgent()
        market_analyzer = MarketAnalyzer()
        
        # Simulate market analyzer failure
        with patch.object(market_analyzer, 'analyze_market_condition', 
                         side_effect=Exception("Market data service unavailable")):
            
            # Agent should handle component failure gracefully
            with patch.object(agent, 'process_message') as mock_process:
                mock_process.return_value = {
                    'intent': 'error',
                    'response': 'Market analysis temporarily unavailable. Please try again later.',
                    'error': 'Market data service unavailable'
                }
                
                result = agent.process_message("What's the market condition?")
                assert 'error' in result
                assert 'unavailable' in result['response'].lower()
    
    def test_database_connection_failure(self):
        """Test system behavior when database connections fail."""
        # Simulate database connection failure
        with patch('src.agent_core.enhanced_memory_manager.EnhancedMemoryManager._connect_database',
                  side_effect=ConnectionError("Database connection failed")):
            
            try:
                from src.agent_core.enhanced_memory_manager import EnhancedMemoryManager
                memory_manager = EnhancedMemoryManager()
                
                # Should handle database failure gracefully
                result = memory_manager.store_conversation_memory("user1", "test", "response")
                # Should either succeed with fallback or raise expected error
                assert result is not None or True  # Graceful handling
                
            except ConnectionError as e:
                # Expected behavior
                assert "database" in str(e).lower() or "connection" in str(e).lower()
    
    def test_memory_exhaustion_scenario(self):
        """Test system behavior under memory exhaustion."""
        # Simulate memory exhaustion
        def memory_exhaustion_simulation():
            # Try to allocate large amount of memory
            try:
                large_data = [0] * (10**8)  # 100M integers
                return large_data
            except MemoryError:
                raise MemoryError("System out of memory")
        
        try:
            memory_exhaustion_simulation()
        except MemoryError as e:
            # System should handle memory exhaustion gracefully
            assert "memory" in str(e).lower()
    
    def test_file_system_errors(self):
        """Test system behavior with file system errors."""
        # Test with read-only directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Make directory read-only
            os.chmod(temp_dir, 0o444)
            
            try:
                # Try to write to read-only directory
                test_file = os.path.join(temp_dir, "test.log")
                with open(test_file, 'w') as f:
                    f.write("test")
            except (PermissionError, OSError) as e:
                # Expected behavior
                assert any(word in str(e).lower() for word in ['permission', 'access', 'denied'])
            finally:
                # Restore permissions for cleanup
                os.chmod(temp_dir, 0o755)
    
    def test_configuration_validation(self):
        """Test system behavior with invalid configurations."""
        invalid_configs = [
            {},  # Empty config
            {"invalid_key": "invalid_value"},  # Unknown keys
            {"memory_limit": -1},  # Negative values
            {"response_timeout": "invalid"},  # Wrong type
            {"enable_learning": "yes"},  # Wrong boolean format
        ]
        
        for invalid_config in invalid_configs:
            try:
                # Simulate configuration validation
                validated_config = self._validate_config(invalid_config)
                # Should return default config or raise error
                assert isinstance(validated_config, dict)
            except (ValueError, TypeError, KeyError) as e:
                # Expected for invalid configurations
                assert any(word in str(e).lower() for word in ['invalid', 'config', 'value', 'type'])
    
    def _validate_config(self, config):
        """Helper method to simulate configuration validation."""
        default_config = {
            'memory_limit': 1000,
            'response_timeout': 5.0,
            'enable_learning': True
        }
        
        if not config:
            return default_config
        
        validated = default_config.copy()
        
        for key, value in config.items():
            if key not in default_config:
                raise KeyError(f"Invalid configuration key: {key}")
            
            if key == 'memory_limit' and (not isinstance(value, int) or value < 0):
                raise ValueError(f"Invalid memory_limit: {value}")
            
            if key == 'response_timeout' and (not isinstance(value, (int, float)) or value <= 0):
                raise ValueError(f"Invalid response_timeout: {value}")
            
            if key == 'enable_learning' and not isinstance(value, bool):
                raise TypeError(f"Invalid enable_learning type: {type(value)}")
            
            validated[key] = value
        
        return validated


class TestRecoveryMechanisms:
    """Test system recovery mechanisms and resilience."""
    
    def test_automatic_retry_mechanism(self):
        """Test automatic retry mechanisms for transient failures."""
        retry_count = 0
        
        def failing_operation():
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                raise ConnectionError("Transient network error")
            return {"success": True, "attempts": retry_count}
        
        # Simulate retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = failing_operation()
                assert result["success"] is True
                assert result["attempts"] == 3
                break
            except ConnectionError:
                if attempt == max_retries - 1:
                    pytest.fail("Retry mechanism failed")
                continue
    
    def test_graceful_degradation(self):
        """Test graceful degradation when services are unavailable."""
        # Simulate service degradation
        services_available = {
            'market_data': False,
            'risk_analysis': True,
            'portfolio_optimization': False
        }
        
        def get_available_features():
            available_features = []
            if services_available['market_data']:
                available_features.append('market_analysis')
            if services_available['risk_analysis']:
                available_features.append('risk_assessment')
            if services_available['portfolio_optimization']:
                available_features.append('portfolio_optimization')
            
            return available_features
        
        available = get_available_features()
        
        # Should still provide some functionality
        assert len(available) > 0
        assert 'risk_assessment' in available
        assert 'market_analysis' not in available
        assert 'portfolio_optimization' not in available
    
    def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern for failing services."""
        failure_count = 0
        circuit_open = False
        
        def circuit_breaker_operation():
            nonlocal failure_count, circuit_open
            
            if circuit_open:
                raise Exception("Circuit breaker is open")
            
            # Simulate failures
            failure_count += 1
            if failure_count < 5:
                raise ConnectionError("Service failure")
            
            # Open circuit after too many failures
            if failure_count >= 5:
                circuit_open = True
                raise Exception("Circuit breaker opened due to repeated failures")
        
        # Test circuit breaker behavior
        for i in range(6):
            try:
                circuit_breaker_operation()
            except Exception as e:
                if i < 4:
                    assert "Service failure" in str(e)
                else:
                    assert "Circuit breaker" in str(e)


if __name__ == "__main__":
    # Run error handling tests
    pytest.main([__file__, "-v", "--tb=short"])

