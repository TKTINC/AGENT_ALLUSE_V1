"""
Unit Tests for Risk Management Components

This module contains comprehensive unit tests for the risk management system,
including portfolio risk monitor, drawdown protection, and portfolio optimizer.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tests.utils.test_utilities import (
    MockDataGenerator, MockServices, TestAssertions, TestFixtures
)


class TestPortfolioRiskMonitor:
    """Test cases for the Portfolio Risk Monitor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.risk_management.portfolio_risk_monitor import PortfolioRiskMonitor
        self.risk_monitor = PortfolioRiskMonitor()
        self.portfolio_data = MockDataGenerator.generate_portfolio_data()
        self.performance_data = MockDataGenerator.generate_performance_data()
    
    def test_var_calculation(self):
        """Test Value at Risk calculation accuracy."""
        pnl_history = self.performance_data['returns']
        
        var_metrics = self.risk_monitor.calculate_var(pnl_history)
        
        # Validate VaR structure
        required_metrics = ['var_95', 'var_99', 'cvar_95']
        for metric in required_metrics:
            assert metric in var_metrics
        
        # Validate VaR ranges (should be positive percentages)
        assert 0 <= var_metrics['var_95'] <= 100
        assert 0 <= var_metrics['var_99'] <= 100
        assert 0 <= var_metrics['cvar_95'] <= 100
        
        # VaR 99% should be higher than VaR 95%
        assert var_metrics['var_99'] >= var_metrics['var_95']
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation accuracy."""
        portfolio_values = self.performance_data['portfolio_values']
        
        drawdown_metrics = self.risk_monitor.calculate_drawdown(portfolio_values)
        
        # Validate drawdown structure
        required_metrics = ['current_drawdown', 'max_drawdown', 'peak_value', 'trough_value']
        for metric in required_metrics:
            assert metric in drawdown_metrics
        
        # Validate drawdown ranges
        assert 0 <= drawdown_metrics['current_drawdown'] <= 100
        assert 0 <= drawdown_metrics['max_drawdown'] <= 100
        assert drawdown_metrics['peak_value'] >= drawdown_metrics['trough_value']
    
    def test_portfolio_risk_assessment(self):
        """Test comprehensive portfolio risk assessment."""
        risk_assessment = self.risk_monitor.assess_portfolio_risk(
            self.portfolio_data, 
            self.performance_data['returns']
        )
        
        # Validate assessment structure
        required_fields = [
            'overall_risk_score', 'risk_level', 'var_metrics', 'drawdown_metrics',
            'concentration_risk', 'liquidity_risk', 'correlation_risk', 'alerts'
        ]
        for field in required_fields:
            assert field in risk_assessment
        
        # Validate risk score and level
        assert 0 <= risk_assessment['overall_risk_score'] <= 100
        assert risk_assessment['risk_level'] in ['Low', 'Moderate', 'High', 'Extreme']
        
        # Validate individual risk components
        assert 0 <= risk_assessment['concentration_risk'] <= 100
        assert 0 <= risk_assessment['liquidity_risk'] <= 100
        assert 0 <= risk_assessment['correlation_risk'] <= 100
    
    def test_concentration_risk_analysis(self):
        """Test concentration risk analysis."""
        concentration_risk = self.risk_monitor.analyze_concentration_risk(self.portfolio_data)
        
        # Validate concentration analysis
        required_fields = ['hhi_index', 'max_position_weight', 'sector_concentration', 'risk_score']
        for field in required_fields:
            assert field in concentration_risk
        
        # Validate ranges
        assert 0 <= concentration_risk['hhi_index'] <= 10000  # HHI ranges from 0 to 10,000
        assert 0 <= concentration_risk['max_position_weight'] <= 100
        assert 0 <= concentration_risk['risk_score'] <= 100
    
    def test_correlation_risk_analysis(self):
        """Test correlation risk analysis."""
        # Create mock correlation matrix
        num_positions = len(self.portfolio_data['positions'])
        correlation_matrix = np.random.rand(num_positions, num_positions)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1.0)  # Diagonal should be 1
        
        correlation_risk = self.risk_monitor.analyze_correlation_risk(
            self.portfolio_data, correlation_matrix
        )
        
        # Validate correlation analysis
        required_fields = ['avg_correlation', 'max_correlation', 'risk_score', 'diversification_ratio']
        for field in required_fields:
            assert field in correlation_risk
        
        # Validate ranges
        assert -1 <= correlation_risk['avg_correlation'] <= 1
        assert -1 <= correlation_risk['max_correlation'] <= 1
        assert 0 <= correlation_risk['risk_score'] <= 100
        assert 0 <= correlation_risk['diversification_ratio'] <= 10
    
    def test_alert_generation(self):
        """Test risk alert generation."""
        # Create high-risk scenario
        high_risk_portfolio = self.portfolio_data.copy()
        high_risk_portfolio['total_value'] = 50000  # Simulate losses
        
        high_risk_returns = [-0.05, -0.03, -0.08, -0.02, -0.06]  # High losses
        
        risk_assessment = self.risk_monitor.assess_portfolio_risk(
            high_risk_portfolio, high_risk_returns
        )
        
        alerts = risk_assessment['alerts']
        assert isinstance(alerts, list)
        
        # Should generate alerts for high-risk scenario
        if risk_assessment['risk_level'] in ['High', 'Extreme']:
            assert len(alerts) > 0
    
    def test_risk_monitoring_performance(self):
        """Test risk monitoring performance requirements."""
        def assess_risk():
            return self.risk_monitor.assess_portfolio_risk(
                self.portfolio_data, 
                self.performance_data['returns']
            )
        
        # Assert response time is under 50ms
        risk_assessment = TestAssertions.assert_response_time(assess_risk, 50)
        assert risk_assessment is not None
    
    def test_real_time_monitoring(self):
        """Test real-time risk monitoring capabilities."""
        # Set up monitoring
        self.risk_monitor.start_monitoring(self.portfolio_data)
        
        # Simulate portfolio update
        updated_portfolio = self.portfolio_data.copy()
        updated_portfolio['total_value'] *= 0.9  # 10% loss
        
        # Update monitoring
        alerts = self.risk_monitor.update_portfolio(updated_portfolio)
        
        # Should detect the loss and potentially generate alerts
        assert isinstance(alerts, list)
        
        # Stop monitoring
        self.risk_monitor.stop_monitoring()


class TestDrawdownProtection:
    """Test cases for the Drawdown Protection system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.risk_management.drawdown_protection import DrawdownProtection
        self.drawdown_protection = DrawdownProtection()
        self.portfolio_data = MockDataGenerator.generate_portfolio_data()
    
    def test_protection_level_determination(self):
        """Test protection level determination based on drawdown."""
        # Test different drawdown levels
        drawdown_scenarios = [
            (3.0, 'None'),
            (6.0, 'Light'),
            (12.0, 'Moderate'),
            (18.0, 'Aggressive'),
            (25.0, 'Emergency')
        ]
        
        for drawdown, expected_level in drawdown_scenarios:
            protection_level = self.drawdown_protection.determine_protection_level(drawdown)
            assert protection_level == expected_level
    
    def test_position_adjustment_calculation(self):
        """Test position adjustment calculations."""
        # Test light protection scenario
        adjustments = self.drawdown_protection.calculate_position_adjustments(
            self.portfolio_data, 'Light'
        )
        
        # Validate adjustment structure
        assert 'total_reduction' in adjustments
        assert 'position_adjustments' in adjustments
        assert 'high_risk_positions' in adjustments
        
        # Light protection should reduce positions by ~20%
        assert 15 <= adjustments['total_reduction'] <= 25
        
        # Test aggressive protection scenario
        aggressive_adjustments = self.drawdown_protection.calculate_position_adjustments(
            self.portfolio_data, 'Aggressive'
        )
        
        # Aggressive protection should reduce more than light
        assert aggressive_adjustments['total_reduction'] > adjustments['total_reduction']
    
    def test_high_risk_position_identification(self):
        """Test identification of high-risk positions."""
        high_risk_positions = self.drawdown_protection.identify_high_risk_positions(self.portfolio_data)
        
        assert isinstance(high_risk_positions, list)
        
        # Each high-risk position should have required fields
        for position in high_risk_positions:
            assert 'symbol' in position
            assert 'risk_score' in position
            assert 'volatility' in position
            assert 0 <= position['risk_score'] <= 100
    
    def test_protection_trigger_logic(self):
        """Test protection trigger logic."""
        # Test no trigger scenario
        low_drawdown_portfolio = self.portfolio_data.copy()
        low_drawdown_portfolio['current_drawdown'] = 2.0
        
        trigger_result = self.drawdown_protection.check_protection_triggers(low_drawdown_portfolio)
        assert trigger_result['protection_triggered'] is False
        
        # Test trigger scenario
        high_drawdown_portfolio = self.portfolio_data.copy()
        high_drawdown_portfolio['current_drawdown'] = 8.0
        
        trigger_result = self.drawdown_protection.check_protection_triggers(high_drawdown_portfolio)
        assert trigger_result['protection_triggered'] is True
        assert trigger_result['protection_level'] == 'Light'
    
    def test_emergency_protection(self):
        """Test emergency protection measures."""
        # Create emergency scenario
        emergency_portfolio = self.portfolio_data.copy()
        emergency_portfolio['current_drawdown'] = 22.0
        
        emergency_response = self.drawdown_protection.execute_emergency_protection(emergency_portfolio)
        
        # Validate emergency response
        assert 'positions_closed' in emergency_response
        assert 'cash_raised' in emergency_response
        assert 'protection_level' in emergency_response
        
        assert emergency_response['protection_level'] == 'Emergency'
        assert emergency_response['cash_raised'] > 0
    
    def test_recovery_detection(self):
        """Test recovery detection and protection adjustment."""
        # Simulate recovery scenario
        recovery_data = {
            'current_drawdown': 3.0,  # Recovered from higher drawdown
            'previous_drawdown': 12.0,
            'recovery_period': 5  # Days
        }
        
        recovery_assessment = self.drawdown_protection.assess_recovery(recovery_data)
        
        assert 'recovery_detected' in recovery_assessment
        assert 'new_protection_level' in recovery_assessment
        assert 'confidence' in recovery_assessment
        
        # Should detect recovery
        assert recovery_assessment['recovery_detected'] is True
    
    def test_protection_performance(self):
        """Test drawdown protection performance requirements."""
        def execute_protection():
            return self.drawdown_protection.check_protection_triggers(self.portfolio_data)
        
        # Assert response time is under 30ms
        protection_result = TestAssertions.assert_response_time(execute_protection, 30)
        assert protection_result is not None


class TestPortfolioOptimizer:
    """Test cases for the Portfolio Optimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.optimization.portfolio_optimizer import PortfolioOptimizer
        self.optimizer = PortfolioOptimizer()
        self.portfolio_data = MockDataGenerator.generate_portfolio_data()
        
        # Create mock expected returns and covariance matrix
        symbols = list(self.portfolio_data['positions'].keys())
        self.expected_returns = {symbol: np.random.normal(0.08, 0.02) for symbol in symbols}
        
        n_assets = len(symbols)
        cov_matrix = np.random.rand(n_assets, n_assets) * 0.01
        cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(cov_matrix, np.random.rand(n_assets) * 0.04)  # Add variance
        self.covariance_matrix = cov_matrix
    
    def test_max_sharpe_optimization(self):
        """Test maximum Sharpe ratio optimization."""
        optimization_result = self.optimizer.optimize_max_sharpe(
            self.expected_returns, self.covariance_matrix
        )
        
        # Validate optimization result
        required_fields = ['weights', 'expected_return', 'volatility', 'sharpe_ratio']
        for field in required_fields:
            assert field in optimization_result
        
        # Validate weights sum to 1
        weights_sum = sum(optimization_result['weights'].values())
        assert abs(weights_sum - 1.0) < 0.01
        
        # Validate Sharpe ratio is reasonable
        assert -5 <= optimization_result['sharpe_ratio'] <= 10
    
    def test_min_variance_optimization(self):
        """Test minimum variance optimization."""
        optimization_result = self.optimizer.optimize_min_variance(self.covariance_matrix)
        
        # Validate optimization result
        required_fields = ['weights', 'volatility', 'expected_return']
        for field in required_fields:
            assert field in optimization_result
        
        # Validate weights sum to 1
        weights_sum = sum(optimization_result['weights'].values())
        assert abs(weights_sum - 1.0) < 0.01
        
        # Validate volatility is positive
        assert optimization_result['volatility'] > 0
    
    def test_risk_parity_optimization(self):
        """Test risk parity optimization."""
        optimization_result = self.optimizer.optimize_risk_parity(self.covariance_matrix)
        
        # Validate optimization result
        required_fields = ['weights', 'risk_contributions', 'volatility']
        for field in required_fields:
            assert field in optimization_result
        
        # Validate weights sum to 1
        weights_sum = sum(optimization_result['weights'].values())
        assert abs(weights_sum - 1.0) < 0.01
        
        # Risk contributions should be approximately equal
        risk_contributions = list(optimization_result['risk_contributions'].values())
        max_contrib = max(risk_contributions)
        min_contrib = min(risk_contributions)
        assert (max_contrib - min_contrib) / max_contrib < 0.5  # Within 50% of each other
    
    def test_efficient_frontier_generation(self):
        """Test efficient frontier generation."""
        efficient_frontier = self.optimizer.generate_efficient_frontier(
            self.expected_returns, self.covariance_matrix, num_points=10
        )
        
        # Validate frontier structure
        assert 'returns' in efficient_frontier
        assert 'volatilities' in efficient_frontier
        assert 'sharpe_ratios' in efficient_frontier
        assert 'weights' in efficient_frontier
        
        # Should have 10 points
        assert len(efficient_frontier['returns']) == 10
        assert len(efficient_frontier['volatilities']) == 10
        
        # Returns should be in ascending order
        returns = efficient_frontier['returns']
        assert all(returns[i] <= returns[i+1] for i in range(len(returns)-1))
    
    def test_portfolio_rebalancing(self):
        """Test portfolio rebalancing recommendations."""
        # Create target weights
        symbols = list(self.portfolio_data['positions'].keys())
        target_weights = {symbol: 1.0/len(symbols) for symbol in symbols}  # Equal weights
        
        rebalancing_result = self.optimizer.calculate_rebalancing(
            self.portfolio_data, target_weights
        )
        
        # Validate rebalancing result
        required_fields = ['weight_changes', 'trade_recommendations', 'rebalancing_cost', 'net_benefit']
        for field in required_fields:
            assert field in rebalancing_result
        
        # Validate trade recommendations
        assert isinstance(rebalancing_result['trade_recommendations'], list)
        assert rebalancing_result['rebalancing_cost'] >= 0
    
    def test_constraint_handling(self):
        """Test constraint handling in optimization."""
        constraints = {
            'max_weight': 0.3,  # No position > 30%
            'min_weight': 0.05,  # No position < 5%
            'sector_limits': {'Technology': 0.5}  # Tech sector < 50%
        }
        
        optimization_result = self.optimizer.optimize_with_constraints(
            self.expected_returns, self.covariance_matrix, constraints
        )
        
        # Validate constraint compliance
        weights = optimization_result['weights']
        
        # Check individual weight constraints
        for weight in weights.values():
            assert constraints['min_weight'] <= weight <= constraints['max_weight']
        
        # Validate weights sum to 1
        weights_sum = sum(weights.values())
        assert abs(weights_sum - 1.0) < 0.01
    
    def test_optimization_performance(self):
        """Test optimization performance requirements."""
        def optimize_portfolio():
            return self.optimizer.optimize_max_sharpe(
                self.expected_returns, self.covariance_matrix
            )
        
        # Assert response time is under 100ms
        optimization_result = TestAssertions.assert_response_time(optimize_portfolio, 100)
        assert optimization_result is not None


class TestPerformanceAnalytics:
    """Test cases for Performance Analytics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.analytics.performance_analytics import PerformanceAnalytics
        self.analytics = PerformanceAnalytics()
        self.performance_data = MockDataGenerator.generate_performance_data(60)  # 60 days
    
    def test_performance_metrics_calculation(self):
        """Test comprehensive performance metrics calculation."""
        portfolio_id = "test_portfolio"
        
        # Add portfolio and performance data
        self.analytics.add_portfolio(portfolio_id, "Test Portfolio")
        
        for i, (value, return_val) in enumerate(zip(
            self.performance_data['portfolio_values'], 
            self.performance_data['returns']
        )):
            self.analytics.update_portfolio_value(portfolio_id, value)
        
        # Calculate metrics
        metrics = self.analytics.calculate_performance_metrics(portfolio_id)
        
        # Validate metrics structure
        TestAssertions.assert_valid_portfolio_metrics(metrics)
        
        # Validate specific metrics
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'volatility' in metrics
        assert 'calmar_ratio' in metrics
    
    def test_risk_adjusted_metrics(self):
        """Test risk-adjusted performance metrics."""
        portfolio_id = "test_portfolio"
        self.analytics.add_portfolio(portfolio_id, "Test Portfolio")
        
        # Add performance data
        for value in self.performance_data['portfolio_values']:
            self.analytics.update_portfolio_value(portfolio_id, value)
        
        risk_metrics = self.analytics.calculate_risk_metrics(portfolio_id)
        
        # Validate risk metrics
        TestAssertions.assert_valid_risk_metrics(risk_metrics)
    
    def test_attribution_analysis(self):
        """Test performance attribution analysis."""
        portfolio_id = "test_portfolio"
        self.analytics.add_portfolio(portfolio_id, "Test Portfolio")
        
        attribution = self.analytics.calculate_attribution_analysis(portfolio_id)
        
        # Validate attribution structure
        required_fields = ['active_return', 'asset_allocation_effect', 'security_selection_effect']
        for field in required_fields:
            assert field in attribution
        
        # Validate attribution values are reasonable
        assert -100 <= attribution['active_return'] <= 100
        assert -50 <= attribution['asset_allocation_effect'] <= 50
        assert -50 <= attribution['security_selection_effect'] <= 50
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison functionality."""
        portfolio_id = "test_portfolio"
        self.analytics.add_portfolio(portfolio_id, "Test Portfolio", benchmark="SPY")
        
        # Add performance data
        for value in self.performance_data['portfolio_values']:
            self.analytics.update_portfolio_value(portfolio_id, value)
        
        comparison = self.analytics.compare_to_benchmark(portfolio_id)
        
        # Validate comparison structure
        required_fields = ['alpha', 'beta', 'tracking_error', 'information_ratio']
        for field in required_fields:
            assert field in comparison
        
        # Validate ranges
        assert -50 <= comparison['alpha'] <= 50
        assert 0 <= comparison['beta'] <= 3
        assert 0 <= comparison['tracking_error'] <= 100
    
    def test_analytics_performance(self):
        """Test analytics performance requirements."""
        portfolio_id = "test_portfolio"
        self.analytics.add_portfolio(portfolio_id, "Test Portfolio")
        
        # Add performance data
        for value in self.performance_data['portfolio_values']:
            self.analytics.update_portfolio_value(portfolio_id, value)
        
        def calculate_metrics():
            return self.analytics.calculate_performance_metrics(portfolio_id)
        
        # Assert response time is under 75ms
        metrics = TestAssertions.assert_response_time(calculate_metrics, 75)
        assert metrics is not None


# Performance benchmarks for risk management
class TestRiskManagementPerformance:
    """Performance benchmark tests for risk management components."""
    
    def test_risk_monitoring_benchmark(self, benchmark):
        """Benchmark risk monitoring performance."""
        from src.risk_management.portfolio_risk_monitor import PortfolioRiskMonitor
        monitor = PortfolioRiskMonitor()
        
        portfolio_data = MockDataGenerator.generate_portfolio_data()
        performance_data = MockDataGenerator.generate_performance_data()
        
        def assess_risk():
            return monitor.assess_portfolio_risk(portfolio_data, performance_data['returns'])
        
        result = benchmark(assess_risk)
        assert result is not None
    
    def test_optimization_benchmark(self, benchmark):
        """Benchmark portfolio optimization performance."""
        from src.optimization.portfolio_optimizer import PortfolioOptimizer
        optimizer = PortfolioOptimizer()
        
        # Create test data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        expected_returns = {symbol: np.random.normal(0.08, 0.02) for symbol in symbols}
        
        n_assets = len(symbols)
        cov_matrix = np.random.rand(n_assets, n_assets) * 0.01
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        np.fill_diagonal(cov_matrix, np.random.rand(n_assets) * 0.04)
        
        def optimize_portfolio():
            return optimizer.optimize_max_sharpe(expected_returns, cov_matrix)
        
        result = benchmark(optimize_portfolio)
        assert result is not None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])

