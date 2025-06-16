"""
Unit Tests for Trading Engine Components

This module contains comprehensive unit tests for the trading engine,
including market analyzer, position sizer, and delta selector.
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


class TestMarketAnalyzer:
    """Test cases for the Market Analyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.trading_engine.market_analyzer import MarketAnalyzer
        self.market_analyzer = MarketAnalyzer()
        self.test_data = MockDataGenerator.generate_market_data()
    
    def test_market_condition_classification(self):
        """Test market condition classification accuracy."""
        # Test with trending up data
        trending_data = self.test_data.copy()
        trending_data['daily_returns'] = [0.01, 0.015, 0.008, 0.012, 0.005]  # Positive trend
        
        condition = self.market_analyzer.classify_market_condition(trending_data)
        assert condition in ['Green', 'Red', 'Chop']
        
        # Test with trending down data
        declining_data = self.test_data.copy()
        declining_data['daily_returns'] = [-0.01, -0.015, -0.008, -0.012, -0.005]  # Negative trend
        
        condition = self.market_analyzer.classify_market_condition(declining_data)
        assert condition in ['Green', 'Red', 'Chop']
        
        # Test with choppy data
        choppy_data = self.test_data.copy()
        choppy_data['daily_returns'] = [0.01, -0.01, 0.008, -0.012, 0.005]  # Mixed
        
        condition = self.market_analyzer.classify_market_condition(choppy_data)
        assert condition in ['Green', 'Red', 'Chop']
    
    def test_volatility_analysis(self):
        """Test volatility analysis calculations."""
        volatility_metrics = self.market_analyzer.analyze_volatility(self.test_data)
        
        # Validate required metrics
        required_metrics = ['historical_volatility', 'implied_volatility', 'volatility_regime', 'iv_hv_ratio']
        for metric in required_metrics:
            assert metric in volatility_metrics
        
        # Validate ranges
        assert 0 <= volatility_metrics['historical_volatility'] <= 200
        assert 0 <= volatility_metrics['implied_volatility'] <= 200
        assert volatility_metrics['volatility_regime'] in ['Low', 'Normal', 'High', 'Extreme']
        assert 0 <= volatility_metrics['iv_hv_ratio'] <= 5
    
    def test_trend_analysis(self):
        """Test trend analysis capabilities."""
        trend_metrics = self.market_analyzer.analyze_trend(self.test_data)
        
        # Validate required metrics
        required_metrics = ['trend_direction', 'trend_strength', 'momentum_score', 'support_resistance']
        for metric in required_metrics:
            assert metric in trend_metrics
        
        # Validate ranges and types
        assert trend_metrics['trend_direction'] in ['Up', 'Down', 'Sideways']
        assert 0 <= trend_metrics['trend_strength'] <= 100
        assert -100 <= trend_metrics['momentum_score'] <= 100
        assert isinstance(trend_metrics['support_resistance'], dict)
    
    def test_market_assessment_performance(self):
        """Test market assessment performance requirements."""
        def assess_market():
            return self.market_analyzer.assess_market_conditions(self.test_data)
        
        # Assert response time is under 50ms
        assessment = TestAssertions.assert_response_time(assess_market, 50)
        
        # Validate assessment structure
        assert 'market_condition' in assessment
        assert 'confidence_score' in assessment
        assert 'volatility_metrics' in assessment
        assert 'trend_metrics' in assessment
        assert 'recommendations' in assessment
    
    @pytest.mark.parametrize("market_condition,expected_recommendation", [
        ("Green", "bullish"),
        ("Red", "bearish"),
        ("Chop", "neutral")
    ])
    def test_recommendation_logic(self, market_condition, expected_recommendation):
        """Test recommendation logic based on market conditions."""
        mock_data = self.test_data.copy()
        
        with patch.object(self.market_analyzer, 'classify_market_condition', return_value=market_condition):
            assessment = self.market_analyzer.assess_market_conditions(mock_data)
            
            recommendations = assessment['recommendations']
            assert any(expected_recommendation in rec.lower() for rec in recommendations)
    
    def test_confidence_scoring(self):
        """Test confidence scoring accuracy."""
        assessment = self.market_analyzer.assess_market_conditions(self.test_data)
        confidence = assessment['confidence_score']
        
        assert 0 <= confidence <= 100
        assert isinstance(confidence, (int, float))
    
    def test_error_handling(self):
        """Test error handling with invalid data."""
        # Test with empty data
        with pytest.raises(ValueError):
            self.market_analyzer.assess_market_conditions({})
        
        # Test with insufficient data
        insufficient_data = {'prices': [100], 'daily_returns': []}
        with pytest.raises(ValueError):
            self.market_analyzer.assess_market_conditions(insufficient_data)


class TestPositionSizer:
    """Test cases for the Position Sizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.trading_engine.position_sizer import PositionSizer
        self.position_sizer = PositionSizer()
        self.portfolio_data = MockDataGenerator.generate_portfolio_data()
        self.market_data = MockDataGenerator.generate_market_data()
    
    def test_kelly_criterion_calculation(self):
        """Test Kelly Criterion position sizing calculation."""
        trade_params = {
            'win_probability': 0.6,
            'avg_win': 0.15,
            'avg_loss': 0.08,
            'account_balance': 100000,
            'risk_free_rate': 0.02
        }
        
        kelly_fraction = self.position_sizer.calculate_kelly_criterion(trade_params)
        
        assert 0 <= kelly_fraction <= 1
        assert isinstance(kelly_fraction, float)
    
    def test_position_sizing_by_account_type(self):
        """Test position sizing varies appropriately by account type."""
        base_params = {
            'symbol': 'SPY',
            'account_balance': 100000,
            'market_condition': 'Green',
            'volatility': 20
        }
        
        # Test different account types
        account_types = ['GEN_ACC', 'REV_ACC', 'COM_ACC']
        position_sizes = {}
        
        for account_type in account_types:
            params = base_params.copy()
            params['account_type'] = account_type
            
            position_size = self.position_sizer.calculate_position_size(params)
            position_sizes[account_type] = position_size
            
            # Validate position size structure
            assert 'position_value' in position_size
            assert 'risk_amount' in position_size
            assert 'max_loss' in position_size
            assert 'position_percentage' in position_size
        
        # COM_ACC should have larger positions than GEN_ACC
        assert position_sizes['COM_ACC']['position_value'] >= position_sizes['GEN_ACC']['position_value']
    
    def test_volatility_adjustment(self):
        """Test position sizing adjusts for volatility."""
        base_params = {
            'symbol': 'SPY',
            'account_balance': 100000,
            'account_type': 'GEN_ACC',
            'market_condition': 'Green'
        }
        
        # Test low volatility
        low_vol_params = base_params.copy()
        low_vol_params['volatility'] = 10
        low_vol_size = self.position_sizer.calculate_position_size(low_vol_params)
        
        # Test high volatility
        high_vol_params = base_params.copy()
        high_vol_params['volatility'] = 40
        high_vol_size = self.position_sizer.calculate_position_size(high_vol_params)
        
        # High volatility should result in smaller position sizes
        assert high_vol_size['position_value'] <= low_vol_size['position_value']
    
    def test_portfolio_constraints(self):
        """Test portfolio-level constraints are respected."""
        portfolio_params = {
            'current_portfolio': self.portfolio_data,
            'new_position': {
                'symbol': 'AAPL',
                'account_balance': 100000,
                'account_type': 'GEN_ACC',
                'market_condition': 'Green',
                'volatility': 25
            }
        }
        
        constrained_size = self.position_sizer.calculate_position_size_with_constraints(portfolio_params)
        
        # Validate constraint compliance
        assert 'position_value' in constrained_size
        assert 'constraint_violations' in constrained_size
        assert 'portfolio_impact' in constrained_size
        
        # Position should not exceed portfolio limits
        total_portfolio_value = self.portfolio_data['total_value']
        max_position_value = total_portfolio_value * 0.4  # 40% max position
        assert constrained_size['position_value'] <= max_position_value
    
    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation accuracy."""
        position_params = {
            'symbol': 'SPY',
            'position_value': 25000,
            'account_balance': 100000,
            'volatility': 20,
            'correlation_matrix': np.array([[1.0, 0.3], [0.3, 1.0]])
        }
        
        risk_metrics = self.position_sizer.calculate_risk_metrics(position_params)
        
        # Validate risk metrics
        required_metrics = ['var_95', 'var_99', 'expected_shortfall', 'portfolio_beta']
        for metric in required_metrics:
            assert metric in risk_metrics
        
        # Validate ranges
        assert 0 <= risk_metrics['var_95'] <= 100
        assert 0 <= risk_metrics['var_99'] <= 100
        assert 0 <= risk_metrics['expected_shortfall'] <= 100
    
    def test_position_sizing_performance(self):
        """Test position sizing performance requirements."""
        params = {
            'symbol': 'SPY',
            'account_balance': 100000,
            'account_type': 'GEN_ACC',
            'market_condition': 'Green',
            'volatility': 20
        }
        
        def calculate_position():
            return self.position_sizer.calculate_position_size(params)
        
        # Assert response time is under 25ms
        position_size = TestAssertions.assert_response_time(calculate_position, 25)
        assert position_size is not None


class TestDeltaSelector:
    """Test cases for the Delta Selector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.trading_engine.delta_selector import DeltaSelector
        self.delta_selector = DeltaSelector()
        self.portfolio_data = MockDataGenerator.generate_portfolio_data()
    
    def test_delta_selection_by_market_condition(self):
        """Test delta selection varies by market condition."""
        base_params = {
            'account_type': 'GEN_ACC',
            'volatility_regime': 'Normal',
            'time_to_expiration': 30,
            'portfolio_delta': 0.0
        }
        
        market_conditions = ['Green', 'Red', 'Chop']
        delta_selections = {}
        
        for condition in market_conditions:
            params = base_params.copy()
            params['market_condition'] = condition
            
            delta_selection = self.delta_selector.select_optimal_delta(params)
            delta_selections[condition] = delta_selection
            
            # Validate delta selection structure
            assert 'recommended_delta' in delta_selection
            assert 'confidence' in delta_selection
            assert 'rationale' in delta_selection
            
            # Validate delta range
            assert 15 <= delta_selection['recommended_delta'] <= 70
        
        # Green markets should prefer higher deltas than Red markets
        assert delta_selections['Green']['recommended_delta'] >= delta_selections['Red']['recommended_delta']
    
    def test_volatility_regime_adjustment(self):
        """Test delta selection adjusts for volatility regime."""
        base_params = {
            'market_condition': 'Green',
            'account_type': 'GEN_ACC',
            'time_to_expiration': 30,
            'portfolio_delta': 0.0
        }
        
        volatility_regimes = ['Low', 'Normal', 'High', 'Extreme']
        delta_selections = {}
        
        for regime in volatility_regimes:
            params = base_params.copy()
            params['volatility_regime'] = regime
            
            delta_selection = self.delta_selector.select_optimal_delta(params)
            delta_selections[regime] = delta_selection
        
        # High volatility should prefer lower deltas
        assert delta_selections['High']['recommended_delta'] <= delta_selections['Low']['recommended_delta']
    
    def test_time_decay_consideration(self):
        """Test delta selection considers time to expiration."""
        base_params = {
            'market_condition': 'Green',
            'account_type': 'GEN_ACC',
            'volatility_regime': 'Normal',
            'portfolio_delta': 0.0
        }
        
        # Test short-term expiration
        short_term_params = base_params.copy()
        short_term_params['time_to_expiration'] = 7
        short_term_delta = self.delta_selector.select_optimal_delta(short_term_params)
        
        # Test long-term expiration
        long_term_params = base_params.copy()
        long_term_params['time_to_expiration'] = 60
        long_term_delta = self.delta_selector.select_optimal_delta(long_term_params)
        
        # Both should be valid deltas
        assert 15 <= short_term_delta['recommended_delta'] <= 70
        assert 15 <= long_term_delta['recommended_delta'] <= 70
    
    def test_portfolio_diversification_analysis(self):
        """Test portfolio diversification analysis."""
        portfolio_analysis = self.delta_selector.analyze_portfolio_diversification(self.portfolio_data)
        
        # Validate analysis structure
        required_fields = ['delta_distribution', 'concentration_risk', 'diversification_score', 'recommendations']
        for field in required_fields:
            assert field in portfolio_analysis
        
        # Validate ranges
        assert 0 <= portfolio_analysis['concentration_risk'] <= 100
        assert 0 <= portfolio_analysis['diversification_score'] <= 100
        assert isinstance(portfolio_analysis['recommendations'], list)
    
    def test_account_type_delta_ranges(self):
        """Test delta ranges are appropriate for account types."""
        params = {
            'market_condition': 'Green',
            'volatility_regime': 'Normal',
            'time_to_expiration': 30,
            'portfolio_delta': 0.0
        }
        
        account_types = ['GEN_ACC', 'REV_ACC', 'COM_ACC']
        
        for account_type in account_types:
            test_params = params.copy()
            test_params['account_type'] = account_type
            
            delta_selection = self.delta_selector.select_optimal_delta(test_params)
            recommended_delta = delta_selection['recommended_delta']
            
            # Validate account-specific ranges
            if account_type == 'GEN_ACC':
                assert 40 <= recommended_delta <= 50
            elif account_type == 'REV_ACC':
                assert 30 <= recommended_delta <= 40
            elif account_type == 'COM_ACC':
                assert 20 <= recommended_delta <= 30
    
    def test_delta_selection_performance(self):
        """Test delta selection performance requirements."""
        params = {
            'market_condition': 'Green',
            'account_type': 'GEN_ACC',
            'volatility_regime': 'Normal',
            'time_to_expiration': 30,
            'portfolio_delta': 0.0
        }
        
        def select_delta():
            return self.delta_selector.select_optimal_delta(params)
        
        # Assert response time is under 20ms
        delta_selection = TestAssertions.assert_response_time(select_delta, 20)
        assert delta_selection is not None
    
    def test_confidence_scoring(self):
        """Test confidence scoring for delta recommendations."""
        params = {
            'market_condition': 'Green',
            'account_type': 'GEN_ACC',
            'volatility_regime': 'Normal',
            'time_to_expiration': 30,
            'portfolio_delta': 0.0
        }
        
        delta_selection = self.delta_selector.select_optimal_delta(params)
        confidence = delta_selection['confidence']
        
        assert 0 <= confidence <= 100
        assert isinstance(confidence, (int, float))


# Performance benchmarks for trading engine
class TestTradingEnginePerformance:
    """Performance benchmark tests for trading engine components."""
    
    def test_market_analysis_benchmark(self, benchmark):
        """Benchmark market analysis performance."""
        from src.trading_engine.market_analyzer import MarketAnalyzer
        analyzer = MarketAnalyzer()
        test_data = MockDataGenerator.generate_market_data()
        
        def analyze_market():
            return analyzer.assess_market_conditions(test_data)
        
        result = benchmark(analyze_market)
        assert result is not None
    
    def test_position_sizing_benchmark(self, benchmark):
        """Benchmark position sizing performance."""
        from src.trading_engine.position_sizer import PositionSizer
        sizer = PositionSizer()
        
        params = {
            'symbol': 'SPY',
            'account_balance': 100000,
            'account_type': 'GEN_ACC',
            'market_condition': 'Green',
            'volatility': 20
        }
        
        def calculate_position():
            return sizer.calculate_position_size(params)
        
        result = benchmark(calculate_position)
        assert result is not None
    
    def test_delta_selection_benchmark(self, benchmark):
        """Benchmark delta selection performance."""
        from src.trading_engine.delta_selector import DeltaSelector
        selector = DeltaSelector()
        
        params = {
            'market_condition': 'Green',
            'account_type': 'GEN_ACC',
            'volatility_regime': 'Normal',
            'time_to_expiration': 30,
            'portfolio_delta': 0.0
        }
        
        def select_delta():
            return selector.select_optimal_delta(params)
        
        result = benchmark(select_delta)
        assert result is not None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])

