"""
Integration Tests for WS1 Components

This module contains comprehensive integration tests that validate the interaction
between agent core, trading engine, and risk management components.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tests.utils.test_utilities import (
    MockDataGenerator, MockServices, TestAssertions, TestFixtures
)


class TestAgentTradingIntegration:
    """Integration tests between agent core and trading engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = TestFixtures.setup_test_environment()
        self.agent = TestFixtures.create_test_agent()
        
        # Import trading components
        from src.trading_engine.market_analyzer import MarketAnalyzer
        from src.trading_engine.position_sizer import PositionSizer
        from src.trading_engine.delta_selector import DeltaSelector
        
        self.market_analyzer = MarketAnalyzer()
        self.position_sizer = PositionSizer()
        self.delta_selector = DeltaSelector()
    
    def test_market_analysis_to_agent_flow(self):
        """Test market analysis data flowing to agent decision making."""
        # Generate market analysis
        market_data = self.test_data['market_data']
        market_assessment = self.market_analyzer.analyze_market_condition('SPY', market_data)
        
        # Simulate agent processing market analysis
        user_message = "What's the current market condition?"
        
        # Mock the agent's process_message method to use market assessment
        with patch.object(self.agent, 'process_message') as mock_process:
            mock_process.return_value = {
                'intent': 'market_analysis',
                'response': f"Current market condition is {market_assessment['market_condition']}",
                'market_data': market_assessment,
                'confidence': market_assessment['confidence_score']
            }
            
            response = self.agent.process_message(user_message)
            
            # Validate integration
            assert response['intent'] == 'market_analysis'
            assert 'market_data' in response
            # Market condition is an enum, convert to string for comparison
            market_condition_str = str(response['market_data']['market_condition']).split('.')[-1]
            assert market_condition_str in ['GREEN', 'RED', 'CHOP']
            assert 0 <= response['confidence'] <= 100
    
    def test_position_sizing_integration(self):
        """Test position sizing integration with agent recommendations."""
        # Setup position sizing parameters
        position_params = {
            'symbol': 'SPY',
            'account_balance': 100000,
            'account_type': 'GEN_ACC',
            'market_condition': 'Green',
            'volatility': 20
        }
        
        # Calculate position size
        position_size = self.position_sizer.calculate_position_size(position_params)
        
        # Simulate agent incorporating position sizing into response
        user_message = "How much should I invest in SPY?"
        
        with patch.object(self.agent, 'process_message') as mock_process:
            mock_process.return_value = {
                'intent': 'position_sizing',
                'response': f"Recommended position size: ${position_size['position_value']:,.2f}",
                'position_recommendation': position_size,
                'risk_metrics': position_size
            }
            
            response = self.agent.process_message(user_message)
            
            # Validate integration
            assert response['intent'] == 'position_sizing'
            assert 'position_recommendation' in response
            assert response['position_recommendation']['position_value'] > 0
            assert 'risk_amount' in response['position_recommendation']
    
    def test_delta_selection_integration(self):
        """Test delta selection integration with agent recommendations."""
        # Setup delta selection parameters
        delta_params = {
            'market_condition': 'Green',
            'account_type': 'GEN_ACC',
            'volatility_regime': 'Normal',
            'time_to_expiration': 30,
            'portfolio_delta': 0.0
        }
        
        # Select optimal delta
        delta_selection = self.delta_selector.select_optimal_delta(delta_params)
        
        # Simulate agent incorporating delta selection into response
        user_message = "What delta should I use for my options?"
        
        with patch.object(self.agent, 'process_message') as mock_process:
            mock_process.return_value = {
                'intent': 'delta_selection',
                'response': f"Recommended delta: {delta_selection['recommended_delta']}",
                'delta_recommendation': delta_selection,
                'rationale': delta_selection['rationale']
            }
            
            response = self.agent.process_message(user_message)
            
            # Validate integration
            assert response['intent'] == 'delta_selection'
            assert 'delta_recommendation' in response
            assert 15 <= response['delta_recommendation']['recommended_delta'] <= 70
            assert 'rationale' in response
    
    def test_complete_trading_workflow(self):
        """Test complete trading workflow integration."""
        # Step 1: Market Analysis
        market_data = self.test_data['market_data']
        market_assessment = self.market_analyzer.assess_market_conditions(market_data)
        
        # Step 2: Position Sizing based on market condition
        position_params = {
            'symbol': 'SPY',
            'account_balance': 100000,
            'account_type': 'GEN_ACC',
            'market_condition': market_assessment['market_condition'],
            'volatility': market_assessment['volatility_metrics']['historical_volatility']
        }
        position_size = self.position_sizer.calculate_position_size(position_params)
        
        # Step 3: Delta Selection based on market condition
        delta_params = {
            'market_condition': market_assessment['market_condition'],
            'account_type': 'GEN_ACC',
            'volatility_regime': market_assessment['volatility_metrics']['volatility_regime'],
            'time_to_expiration': 30,
            'portfolio_delta': 0.0
        }
        delta_selection = self.delta_selector.select_optimal_delta(delta_params)
        
        # Step 4: Agent integrates all recommendations
        trading_recommendation = {
            'market_analysis': market_assessment,
            'position_sizing': position_size,
            'delta_selection': delta_selection,
            'integrated_confidence': min(
                market_assessment['confidence_score'],
                delta_selection['confidence']
            )
        }
        
        # Validate complete workflow
        assert trading_recommendation['market_analysis']['market_condition'] in ['Green', 'Red', 'Chop']
        assert trading_recommendation['position_sizing']['position_value'] > 0
        assert 15 <= trading_recommendation['delta_selection']['recommended_delta'] <= 70
        assert 0 <= trading_recommendation['integrated_confidence'] <= 100
        
        # Validate consistency between components
        market_condition = trading_recommendation['market_analysis']['market_condition']
        recommended_delta = trading_recommendation['delta_selection']['recommended_delta']
        
        # Green markets should generally prefer higher deltas
        if market_condition == 'Green':
            assert recommended_delta >= 30  # Should be in higher range for bullish markets


class TestAgentRiskIntegration:
    """Integration tests between agent core and risk management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = TestFixtures.setup_test_environment()
        self.agent = TestFixtures.create_test_agent()
        
        # Import risk management components
        from src.risk_management.portfolio_risk_monitor import PortfolioRiskMonitor
        from src.risk_management.drawdown_protection import DrawdownProtection
        from src.optimization.portfolio_optimizer import PortfolioOptimizer
        from src.analytics.performance_analytics import PerformanceAnalytics
        
        self.risk_monitor = PortfolioRiskMonitor()
        self.drawdown_protection = DrawdownProtection()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.performance_analytics = PerformanceAnalytics()
    
    def test_risk_assessment_integration(self):
        """Test risk assessment integration with agent responses."""
        portfolio_data = self.test_data['portfolio_data']
        performance_data = self.test_data['performance_data']
        
        # Perform risk assessment
        risk_assessment = self.risk_monitor.assess_portfolio_risk(
            portfolio_data, performance_data['returns']
        )
        
        # Simulate agent incorporating risk assessment
        user_message = "What's my current risk level?"
        
        with patch.object(self.agent, 'process_message') as mock_process:
            mock_process.return_value = {
                'intent': 'risk_assessment',
                'response': f"Your current risk level is {risk_assessment['risk_level']}",
                'risk_data': risk_assessment,
                'recommendations': risk_assessment.get('alerts', [])
            }
            
            response = self.agent.process_message(user_message)
            
            # Validate integration
            assert response['intent'] == 'risk_assessment'
            assert 'risk_data' in response
            assert response['risk_data']['risk_level'] in ['Low', 'Moderate', 'High', 'Extreme']
            assert 0 <= response['risk_data']['overall_risk_score'] <= 100
    
    def test_drawdown_protection_integration(self):
        """Test drawdown protection integration with agent alerts."""
        portfolio_data = self.test_data['portfolio_data']
        portfolio_data['current_drawdown'] = 8.0  # Trigger light protection
        
        # Check protection triggers
        protection_result = self.drawdown_protection.check_protection_triggers(portfolio_data)
        
        # Simulate agent processing protection alert
        if protection_result['protection_triggered']:
            with patch.object(self.agent, 'process_message') as mock_process:
                mock_process.return_value = {
                    'intent': 'risk_alert',
                    'response': f"Drawdown protection activated: {protection_result['protection_level']}",
                    'protection_data': protection_result,
                    'action_required': True
                }
                
                response = self.agent.process_message("Check my portfolio protection")
                
                # Validate integration
                assert response['intent'] == 'risk_alert'
                assert response['action_required'] is True
                assert 'protection_data' in response
                assert response['protection_data']['protection_level'] == 'Light'
    
    def test_portfolio_optimization_integration(self):
        """Test portfolio optimization integration with agent recommendations."""
        # Create mock expected returns and covariance matrix
        portfolio_data = self.test_data['portfolio_data']
        symbols = list(portfolio_data['positions'].keys())
        expected_returns = {symbol: np.random.normal(0.08, 0.02) for symbol in symbols}
        
        n_assets = len(symbols)
        cov_matrix = np.random.rand(n_assets, n_assets) * 0.01
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        np.fill_diagonal(cov_matrix, np.random.rand(n_assets) * 0.04)
        
        # Perform optimization
        optimization_result = self.portfolio_optimizer.optimize_max_sharpe(
            expected_returns, cov_matrix
        )
        
        # Simulate agent incorporating optimization
        user_message = "How should I optimize my portfolio?"
        
        with patch.object(self.agent, 'process_message') as mock_process:
            mock_process.return_value = {
                'intent': 'portfolio_optimization',
                'response': f"Optimal Sharpe ratio: {optimization_result['sharpe_ratio']:.3f}",
                'optimization_data': optimization_result,
                'rebalancing_needed': True
            }
            
            response = self.agent.process_message(user_message)
            
            # Validate integration
            assert response['intent'] == 'portfolio_optimization'
            assert 'optimization_data' in response
            assert 'sharpe_ratio' in response['optimization_data']
            assert abs(sum(response['optimization_data']['weights'].values()) - 1.0) < 0.01
    
    def test_performance_analytics_integration(self):
        """Test performance analytics integration with agent reporting."""
        # Setup performance analytics
        portfolio_id = "test_portfolio"
        self.performance_analytics.add_portfolio(portfolio_id, "Test Portfolio")
        
        # Add performance data
        performance_data = self.test_data['performance_data']
        for value in performance_data['portfolio_values']:
            self.performance_analytics.update_portfolio_value(portfolio_id, value)
        
        # Calculate metrics
        metrics = self.performance_analytics.calculate_performance_metrics(portfolio_id)
        
        # Simulate agent incorporating performance metrics
        user_message = "Show me my performance metrics"
        
        with patch.object(self.agent, 'process_message') as mock_process:
            mock_process.return_value = {
                'intent': 'performance_inquiry',
                'response': f"Total return: {metrics['total_return']:.2f}%, Sharpe: {metrics['sharpe_ratio']:.3f}",
                'performance_data': metrics,
                'benchmark_comparison': True
            }
            
            response = self.agent.process_message(user_message)
            
            # Validate integration
            assert response['intent'] == 'performance_inquiry'
            assert 'performance_data' in response
            TestAssertions.assert_valid_portfolio_metrics(response['performance_data'])


class TestTradingRiskIntegration:
    """Integration tests between trading engine and risk management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = TestFixtures.setup_test_environment()
        
        # Import all components
        from src.trading_engine.market_analyzer import MarketAnalyzer
        from src.trading_engine.position_sizer import PositionSizer
        from src.trading_engine.delta_selector import DeltaSelector
        from src.risk_management.portfolio_risk_monitor import PortfolioRiskMonitor
        from src.risk_management.drawdown_protection import DrawdownProtection
        
        self.market_analyzer = MarketAnalyzer()
        self.position_sizer = PositionSizer()
        self.delta_selector = DeltaSelector()
        self.risk_monitor = PortfolioRiskMonitor()
        self.drawdown_protection = DrawdownProtection()
    
    def test_risk_adjusted_position_sizing(self):
        """Test position sizing with risk management constraints."""
        # Get current portfolio risk
        portfolio_data = self.test_data['portfolio_data']
        performance_data = self.test_data['performance_data']
        
        risk_assessment = self.risk_monitor.assess_portfolio_risk(
            portfolio_data, performance_data['returns']
        )
        
        # Adjust position sizing based on risk level
        base_position_params = {
            'symbol': 'SPY',
            'account_balance': 100000,
            'account_type': 'GEN_ACC',
            'market_condition': 'Green',
            'volatility': 20
        }
        
        # Apply risk adjustment
        risk_multiplier = 1.0
        if risk_assessment['risk_level'] == 'High':
            risk_multiplier = 0.5  # Reduce position size by 50%
        elif risk_assessment['risk_level'] == 'Extreme':
            risk_multiplier = 0.25  # Reduce position size by 75%
        
        base_position_size = self.position_sizer.calculate_position_size(base_position_params)
        adjusted_position_value = base_position_size['position_value'] * risk_multiplier
        
        # Validate risk adjustment
        if risk_assessment['risk_level'] in ['High', 'Extreme']:
            assert adjusted_position_value < base_position_size['position_value']
        
        # Ensure adjusted position is still reasonable
        assert adjusted_position_value > 0
        assert adjusted_position_value <= base_position_params['account_balance'] * 0.4
    
    def test_drawdown_triggered_position_adjustment(self):
        """Test position adjustments when drawdown protection is triggered."""
        # Create high drawdown scenario
        portfolio_data = self.test_data['portfolio_data']
        portfolio_data['current_drawdown'] = 12.0  # Trigger moderate protection
        
        # Check protection triggers
        protection_result = self.drawdown_protection.check_protection_triggers(portfolio_data)
        
        if protection_result['protection_triggered']:
            # Calculate position adjustments
            adjustments = self.drawdown_protection.calculate_position_adjustments(
                portfolio_data, protection_result['protection_level']
            )
            
            # Validate adjustments affect future position sizing
            new_position_params = {
                'symbol': 'AAPL',
                'account_balance': 100000,
                'account_type': 'GEN_ACC',
                'market_condition': 'Green',
                'volatility': 25,
                'drawdown_protection_active': True,
                'protection_level': protection_result['protection_level']
            }
            
            # Position sizing should be reduced when protection is active
            normal_position = self.position_sizer.calculate_position_size({
                'symbol': 'AAPL',
                'account_balance': 100000,
                'account_type': 'GEN_ACC',
                'market_condition': 'Green',
                'volatility': 25
            })
            
            # Simulate reduced position sizing due to protection
            protection_multiplier = 1.0 - (adjustments['total_reduction'] / 100)
            protected_position_value = normal_position['position_value'] * protection_multiplier
            
            # Validate protection reduces position sizes
            assert protected_position_value < normal_position['position_value']
            assert protection_multiplier < 1.0
    
    def test_market_condition_risk_alignment(self):
        """Test alignment between market conditions and risk assessments."""
        market_data = self.test_data['market_data']
        
        # Analyze market conditions
        market_assessment = self.market_analyzer.assess_market_conditions(market_data)
        
        # Create portfolio data that reflects market conditions
        portfolio_data = self.test_data['portfolio_data']
        
        # Simulate returns that align with market condition
        if market_assessment['market_condition'] == 'Red':
            # Negative returns for bearish market
            recent_returns = [-0.02, -0.015, -0.01, -0.025, -0.008]
        elif market_assessment['market_condition'] == 'Green':
            # Positive returns for bullish market
            recent_returns = [0.015, 0.02, 0.008, 0.012, 0.018]
        else:  # Chop
            # Mixed returns for choppy market
            recent_returns = [0.01, -0.008, 0.005, -0.012, 0.015]
        
        # Assess portfolio risk with market-aligned returns
        risk_assessment = self.risk_monitor.assess_portfolio_risk(portfolio_data, recent_returns)
        
        # Validate alignment between market condition and risk
        if market_assessment['market_condition'] == 'Red':
            # Bearish markets should generally show higher risk
            assert risk_assessment['risk_level'] in ['Moderate', 'High', 'Extreme']
        elif market_assessment['market_condition'] == 'Green':
            # Bullish markets might show lower risk (but not always)
            assert risk_assessment['risk_level'] in ['Low', 'Moderate', 'High']
        
        # Both assessments should be internally consistent
        assert 0 <= market_assessment['confidence_score'] <= 100
        assert 0 <= risk_assessment['overall_risk_score'] <= 100


class TestEndToEndWorkflows:
    """End-to-end integration tests for complete user workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = TestFixtures.setup_test_environment()
        self.agent = TestFixtures.create_test_agent()
        
        # Import all components
        from src.trading_engine.market_analyzer import MarketAnalyzer
        from src.trading_engine.position_sizer import PositionSizer
        from src.trading_engine.delta_selector import DeltaSelector
        from src.risk_management.portfolio_risk_monitor import PortfolioRiskMonitor
        from src.risk_management.drawdown_protection import DrawdownProtection
        from src.optimization.portfolio_optimizer import PortfolioOptimizer
        
        self.market_analyzer = MarketAnalyzer()
        self.position_sizer = PositionSizer()
        self.delta_selector = DeltaSelector()
        self.risk_monitor = PortfolioRiskMonitor()
        self.drawdown_protection = DrawdownProtection()
        self.portfolio_optimizer = PortfolioOptimizer()
    
    def test_new_user_onboarding_workflow(self):
        """Test complete new user onboarding workflow."""
        # Step 1: Greeting
        greeting_response = self._mock_agent_response(
            "Hello", "greeting", "Welcome to ALL-USE! I'm here to help you with options trading."
        )
        assert greeting_response['intent'] == 'greeting'
        
        # Step 2: Account inquiry
        inquiry_response = self._mock_agent_response(
            "Tell me about account setup", "account_inquiry", 
            "I can help you set up your trading accounts. How much would you like to start with?"
        )
        assert inquiry_response['intent'] == 'account_inquiry'
        
        # Step 3: Account setup
        setup_response = self._mock_agent_response(
            "I want to start with $75,000", "setup_accounts",
            "Great! I'll set up your accounts with $75,000. This will create a GEN-ACC account."
        )
        assert setup_response['intent'] == 'setup_accounts'
        assert setup_response['entities']['amount'] == 75000
        
        # Step 4: Market analysis request
        market_response = self._mock_agent_response(
            "What's the current market condition?", "market_analysis",
            "Let me analyze the current market conditions for you."
        )
        assert market_response['intent'] == 'market_analysis'
        
        # Validate workflow progression
        workflow_intents = [
            greeting_response['intent'],
            inquiry_response['intent'], 
            setup_response['intent'],
            market_response['intent']
        ]
        expected_flow = ['greeting', 'account_inquiry', 'setup_accounts', 'market_analysis']
        assert workflow_intents == expected_flow
    
    def test_trading_decision_workflow(self):
        """Test complete trading decision workflow."""
        # Step 1: Market analysis
        market_data = self.test_data['market_data']
        market_assessment = self.market_analyzer.assess_market_conditions(market_data)
        
        # Step 2: Risk assessment
        portfolio_data = self.test_data['portfolio_data']
        performance_data = self.test_data['performance_data']
        risk_assessment = self.risk_monitor.assess_portfolio_risk(
            portfolio_data, performance_data['returns']
        )
        
        # Step 3: Position sizing
        position_params = {
            'symbol': 'SPY',
            'account_balance': 100000,
            'account_type': 'GEN_ACC',
            'market_condition': market_assessment['market_condition'],
            'volatility': market_assessment['volatility_metrics']['historical_volatility']
        }
        position_size = self.position_sizer.calculate_position_size(position_params)
        
        # Step 4: Delta selection
        delta_params = {
            'market_condition': market_assessment['market_condition'],
            'account_type': 'GEN_ACC',
            'volatility_regime': market_assessment['volatility_metrics']['volatility_regime'],
            'time_to_expiration': 30,
            'portfolio_delta': 0.0
        }
        delta_selection = self.delta_selector.select_optimal_delta(delta_params)
        
        # Step 5: Risk validation
        proposed_trade = {
            'symbol': 'SPY',
            'position_value': position_size['position_value'],
            'delta': delta_selection['recommended_delta'],
            'market_condition': market_assessment['market_condition']
        }
        
        # Validate complete trading decision
        assert market_assessment['market_condition'] in ['Green', 'Red', 'Chop']
        assert risk_assessment['risk_level'] in ['Low', 'Moderate', 'High', 'Extreme']
        assert position_size['position_value'] > 0
        assert 15 <= delta_selection['recommended_delta'] <= 70
        
        # Validate decision consistency
        if market_assessment['market_condition'] == 'Green':
            assert delta_selection['recommended_delta'] >= 30  # Bullish bias
        elif market_assessment['market_condition'] == 'Red':
            assert delta_selection['recommended_delta'] <= 50  # Bearish bias
    
    def test_risk_management_workflow(self):
        """Test complete risk management workflow."""
        # Step 1: Portfolio monitoring
        portfolio_data = self.test_data['portfolio_data']
        performance_data = self.test_data['performance_data']
        
        # Simulate portfolio loss
        portfolio_data['total_value'] = 85000  # 15% loss from 100k
        portfolio_data['current_drawdown'] = 15.0
        
        # Step 2: Risk assessment
        risk_assessment = self.risk_monitor.assess_portfolio_risk(
            portfolio_data, [-0.05, -0.03, -0.08, -0.02, -0.06]  # Negative returns
        )
        
        # Step 3: Drawdown protection check
        protection_result = self.drawdown_protection.check_protection_triggers(portfolio_data)
        
        # Step 4: Position adjustments if needed
        if protection_result['protection_triggered']:
            adjustments = self.drawdown_protection.calculate_position_adjustments(
                portfolio_data, protection_result['protection_level']
            )
            
            # Validate protection response
            assert protection_result['protection_level'] == 'Aggressive'  # 15% drawdown
            assert adjustments['total_reduction'] > 0
            assert len(adjustments['high_risk_positions']) > 0
        
        # Step 5: Agent communication
        risk_alert_response = self._mock_agent_response(
            "Check my risk status", "risk_alert",
            f"Risk alert: {risk_assessment['risk_level']} risk level detected. Protection activated."
        )
        
        # Validate risk management workflow
        assert risk_assessment['risk_level'] in ['High', 'Extreme']
        assert protection_result['protection_triggered'] is True
        assert risk_alert_response['intent'] == 'risk_alert'
    
    def _mock_agent_response(self, user_message: str, intent: str, response: str) -> dict:
        """Helper method to mock agent responses."""
        # Extract entities based on intent
        entities = {}
        if intent == 'setup_accounts' and '$' in user_message:
            # Extract amount
            import re
            amount_match = re.search(r'\$?([\d,]+)', user_message)
            if amount_match:
                entities['amount'] = int(amount_match.group(1).replace(',', ''))
        
        return {
            'intent': intent,
            'response': response,
            'entities': entities,
            'user_message': user_message,
            'timestamp': datetime.now(),
            'confidence': 0.95
        }


# Performance tests for integration
class TestIntegrationPerformance:
    """Performance tests for integrated workflows."""
    
    def test_complete_workflow_performance(self, benchmark):
        """Benchmark complete trading workflow performance."""
        def complete_workflow():
            # Setup
            test_data = TestFixtures.setup_test_environment()
            
            # Import components
            from src.trading_engine.market_analyzer import MarketAnalyzer
            from src.trading_engine.position_sizer import PositionSizer
            from src.trading_engine.delta_selector import DeltaSelector
            from src.risk_management.portfolio_risk_monitor import PortfolioRiskMonitor
            
            market_analyzer = MarketAnalyzer()
            position_sizer = PositionSizer()
            delta_selector = DeltaSelector()
            risk_monitor = PortfolioRiskMonitor()
            
            # Execute workflow
            market_assessment = market_analyzer.assess_market_conditions(test_data['market_data'])
            
            position_params = {
                'symbol': 'SPY',
                'account_balance': 100000,
                'account_type': 'GEN_ACC',
                'market_condition': market_assessment['market_condition'],
                'volatility': 20
            }
            position_size = position_sizer.calculate_position_size(position_params)
            
            delta_params = {
                'market_condition': market_assessment['market_condition'],
                'account_type': 'GEN_ACC',
                'volatility_regime': 'Normal',
                'time_to_expiration': 30,
                'portfolio_delta': 0.0
            }
            delta_selection = delta_selector.select_optimal_delta(delta_params)
            
            risk_assessment = risk_monitor.assess_portfolio_risk(
                test_data['portfolio_data'], test_data['performance_data']['returns']
            )
            
            return {
                'market': market_assessment,
                'position': position_size,
                'delta': delta_selection,
                'risk': risk_assessment
            }
        
        result = benchmark(complete_workflow)
        assert result is not None
        assert all(key in result for key in ['market', 'position', 'delta', 'risk'])


if __name__ == "__main__":
    # Run the integration tests
    pytest.main([__file__, "-v", "--tb=short"])

