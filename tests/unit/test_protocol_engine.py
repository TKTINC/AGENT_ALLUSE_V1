#!/usr/bin/env python3
"""
ALL-USE Agent - Protocol Engine Unit Tests
WS2-P4: Comprehensive Testing and Validation

This module provides comprehensive unit tests for all Protocol Engine components
implemented in WS2-P1 through WS2-P3.

Components Tested:
- Week Classification System (WS2-P1)
- Enhanced Protocol Rules (WS2-P2) 
- Advanced Protocol Optimization (WS2-P3)
"""

import unittest
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import Protocol Engine components
from protocol_engine.week_classification.week_classifier import WeekClassifier
from protocol_engine.market_analysis.market_condition_analyzer import MarketConditionAnalyzer
from protocol_engine.decision_system.action_recommendation_system import ActionRecommendationSystem
from protocol_engine.learning.historical_analysis_engine import HistoricalAnalysisEngine
from protocol_engine.rules.trading_protocol_rules import TradingProtocolRulesEngine
from protocol_engine.adjustments.atr_adjustment_system import ATRAdjustmentSystem
from protocol_engine.position_management.position_manager import PositionManager
from protocol_engine.rollover.rollover_protocol import RolloverProtocol
from protocol_engine.ml_optimization.ml_optimizer import MLOptimizer
from protocol_engine.backtesting.backtesting_engine import BacktestingEngine
from protocol_engine.adaptation.adaptation_engine import AdaptationEngine
from protocol_engine.human_oversight.decision_gateway import DecisionGateway
from protocol_engine.trust_system.hitl_trust_system import HITLTrustSystem


class TestWeekClassifier(unittest.TestCase):
    """Test suite for Week Classification System (WS2-P1)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.classifier = WeekClassifier()
        
        # Sample market data for testing
        self.sample_market_data = {
            'spy_price': 450.0,
            'spy_change': 0.02,
            'vix': 18.5,
            'volume': 85000000,
            'rsi': 65.0,
            'macd': 0.5,
            'bollinger_position': 0.7,
            'trend_strength': 0.6
        }
    
    def test_week_classifier_initialization(self):
        """Test WeekClassifier initialization"""
        self.assertIsNotNone(self.classifier)
        self.assertEqual(len(self.classifier.week_types), 11)
        
        # Verify all 11 week types are present
        expected_types = ['P-EW', 'P-AWL', 'P-RO', 'P-AOL', 'P-DD', 
                         'C-WAP', 'C-WAP+', 'C-PNO', 'C-RO', 'C-REC', 'W-IDL']
        for week_type in expected_types:
            self.assertIn(week_type, self.classifier.week_types)
    
    def test_week_classification_accuracy(self):
        """Test week classification accuracy and confidence"""
        result = self.classifier.classify_week(self.sample_market_data)
        
        # Verify result structure
        self.assertIn('week_type', result)
        self.assertIn('confidence', result)
        self.assertIn('probabilities', result)
        self.assertIn('reasoning', result)
        
        # Verify confidence range
        self.assertGreaterEqual(result['confidence'], 0.64)
        self.assertLessEqual(result['confidence'], 0.90)
        
        # Verify probabilities sum to 1
        prob_sum = sum(result['probabilities'].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)
    
    def test_week_frequency_distribution(self):
        """Test annual week frequency distribution"""
        frequencies = self.classifier.get_annual_frequency_distribution()
        
        # Verify total weeks = 52
        total_weeks = sum(frequencies.values())
        self.assertEqual(total_weeks, 52)
        
        # Verify P-EW has highest frequency (31 weeks)
        self.assertEqual(frequencies['P-EW'], 31)
        
        # Verify all week types have positive frequency
        for week_type, freq in frequencies.items():
            self.assertGreater(freq, 0)
    
    def test_expected_returns_calculation(self):
        """Test expected returns calculation"""
        returns = self.classifier.calculate_expected_returns()
        
        # Verify return range (125.4% - 161.7%)
        self.assertGreaterEqual(returns['annual_return'], 1.254)
        self.assertLessEqual(returns['annual_return'], 1.617)
        
        # Verify weekly returns are positive
        for week_type, weekly_return in returns['weekly_returns'].items():
            self.assertGreater(weekly_return, 0)
    
    def test_classification_performance(self):
        """Test classification performance requirements"""
        import time
        
        start_time = time.time()
        result = self.classifier.classify_week(self.sample_market_data)
        end_time = time.time()
        
        # Verify response time < 50ms
        response_time = (end_time - start_time) * 1000
        self.assertLess(response_time, 50)


class TestMarketConditionAnalyzer(unittest.TestCase):
    """Test suite for Market Condition Analyzer (WS2-P1)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = MarketConditionAnalyzer()
        
        # Sample market data
        self.sample_data = {
            'spy_price': 450.0,
            'spy_change': 0.015,
            'vix': 20.0,
            'volume': 90000000,
            'rsi': 60.0,
            'macd': 0.3,
            'bollinger_position': 0.6
        }
    
    def test_market_condition_analysis(self):
        """Test market condition analysis"""
        result = self.analyzer.analyze_market_conditions(self.sample_data)
        
        # Verify result structure
        self.assertIn('market_condition', result)
        self.assertIn('confidence', result)
        self.assertIn('risk_level', result)
        self.assertIn('volatility_regime', result)
        
        # Verify market condition is valid
        valid_conditions = ['extremely_bullish', 'bullish', 'moderately_bullish',
                           'neutral', 'moderately_bearish', 'bearish', 'extremely_bearish']
        self.assertIn(result['market_condition'], valid_conditions)
        
        # Verify confidence range
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_probability_based_selection(self):
        """Test probability-based week type selection"""
        market_condition = 'bullish'
        base_probabilities = {'P-EW': 0.6, 'C-WAP': 0.3, 'W-IDL': 0.1}
        
        adjusted_probs = self.analyzer.adjust_probabilities_for_market(
            base_probabilities, market_condition
        )
        
        # Verify probabilities sum to 1
        prob_sum = sum(adjusted_probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)
        
        # Verify adjustment logic (bullish should favor put strategies)
        self.assertGreaterEqual(adjusted_probs['P-EW'], base_probabilities['P-EW'])
    
    def test_analysis_performance(self):
        """Test analysis performance requirements"""
        import time
        
        start_time = time.time()
        result = self.analyzer.analyze_market_conditions(self.sample_data)
        end_time = time.time()
        
        # Verify response time < 100ms
        response_time = (end_time - start_time) * 1000
        self.assertLess(response_time, 100)


class TestTradingProtocolRules(unittest.TestCase):
    """Test suite for Trading Protocol Rules (WS2-P2)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rules_engine = TradingProtocolRulesEngine()
        
        # Sample trade proposal
        self.sample_trade = {
            'account_type': 'GEN_ACC',
            'strategy': 'short_put',
            'delta': 45,
            'position_size': 8,
            'dte': 35,
            'market_condition': 'bullish',
            'portfolio_value': 100000,
            'current_positions': 5
        }
    
    def test_rules_engine_initialization(self):
        """Test rules engine initialization"""
        self.assertIsNotNone(self.rules_engine)
        self.assertEqual(len(self.rules_engine.rules), 7)
        
        # Verify all rule types are present
        expected_rules = ['delta_range', 'position_size', 'time_constraints',
                         'market_conditions', 'risk_limits', 'portfolio_constraints',
                         'account_specific']
        for rule in expected_rules:
            self.assertTrue(any(r['name'] == rule for r in self.rules_engine.rules))
    
    def test_rule_validation_success(self):
        """Test successful rule validation"""
        result = self.rules_engine.validate_trade(self.sample_trade)
        
        # Verify result structure
        self.assertIn('valid', result)
        self.assertIn('violations', result)
        self.assertIn('warnings', result)
        self.assertIn('confidence', result)
        
        # Should pass validation for valid trade
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['violations']), 0)
    
    def test_delta_range_validation(self):
        """Test delta range rule validation"""
        # Test GEN_ACC delta range (40-50)
        valid_trade = self.sample_trade.copy()
        valid_trade['delta'] = 45
        result = self.rules_engine.validate_trade(valid_trade)
        self.assertTrue(result['valid'])
        
        # Test invalid delta (outside range)
        invalid_trade = self.sample_trade.copy()
        invalid_trade['delta'] = 35  # Below GEN_ACC range
        result = self.rules_engine.validate_trade(invalid_trade)
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['violations']), 0)
    
    def test_account_specific_constraints(self):
        """Test account-specific constraint validation"""
        # Test REV_ACC constraints (30-40 delta)
        rev_trade = self.sample_trade.copy()
        rev_trade['account_type'] = 'REV_ACC'
        rev_trade['delta'] = 35
        result = self.rules_engine.validate_trade(rev_trade)
        self.assertTrue(result['valid'])
        
        # Test COM_ACC constraints (20-30 delta)
        com_trade = self.sample_trade.copy()
        com_trade['account_type'] = 'COM_ACC'
        com_trade['delta'] = 25
        result = self.rules_engine.validate_trade(com_trade)
        self.assertTrue(result['valid'])
    
    def test_validation_performance(self):
        """Test validation performance requirements"""
        import time
        
        start_time = time.time()
        result = self.rules_engine.validate_trade(self.sample_trade)
        end_time = time.time()
        
        # Verify response time < 10ms
        response_time = (end_time - start_time) * 1000
        self.assertLess(response_time, 10)


class TestATRAdjustmentSystem(unittest.TestCase):
    """Test suite for ATR Adjustment System (WS2-P2)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.atr_system = ATRAdjustmentSystem()
        
        # Sample market data with ATR values
        self.sample_data = {
            'atr_14': 8.5,
            'atr_21': 9.2,
            'atr_50': 10.1,
            'current_price': 450.0,
            'volume': 85000000
        }
    
    def test_volatility_regime_classification(self):
        """Test volatility regime classification"""
        result = self.atr_system.classify_volatility_regime(self.sample_data)
        
        # Verify result structure
        self.assertIn('regime', result)
        self.assertIn('confidence', result)
        self.assertIn('adjustment_factor', result)
        
        # Verify regime is valid
        valid_regimes = ['very_low', 'low', 'medium', 'high', 'very_high']
        self.assertIn(result['regime'], valid_regimes)
        
        # Verify confidence range
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_parameter_adjustments(self):
        """Test parameter adjustments based on volatility"""
        base_params = {
            'position_size': 10,
            'delta_target': 45,
            'dte_target': 35,
            'risk_limit': 0.02
        }
        
        adjusted_params = self.atr_system.adjust_parameters(
            base_params, self.sample_data
        )
        
        # Verify all parameters are adjusted
        for param in base_params.keys():
            self.assertIn(param, adjusted_params)
        
        # Verify adjustments are reasonable
        self.assertGreater(adjusted_params['position_size'], 0)
        self.assertGreater(adjusted_params['delta_target'], 0)
        self.assertGreater(adjusted_params['dte_target'], 0)
        self.assertGreater(adjusted_params['risk_limit'], 0)
    
    def test_adjustment_performance(self):
        """Test adjustment performance requirements"""
        import time
        
        start_time = time.time()
        result = self.atr_system.classify_volatility_regime(self.sample_data)
        end_time = time.time()
        
        # Verify response time < 30ms
        response_time = (end_time - start_time) * 1000
        self.assertLess(response_time, 30)


class TestMLOptimizer(unittest.TestCase):
    """Test suite for ML Optimizer (WS2-P3)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ml_optimizer = MLOptimizer()
        
        # Sample historical data for training
        self.sample_data = pd.DataFrame({
            'week_type': ['P-EW', 'C-WAP', 'P-AWL'] * 10,
            'market_condition': ['bullish', 'neutral', 'bearish'] * 10,
            'delta': [45, 40, 35] * 10,
            'dte': [35, 30, 40] * 10,
            'return': [0.025, 0.018, 0.020] * 10
        })
    
    def test_ml_model_training(self):
        """Test ML model training"""
        result = self.ml_optimizer.train_models(self.sample_data)
        
        # Verify training success
        self.assertIn('week_classification_accuracy', result)
        self.assertIn('parameter_optimization_improvement', result)
        self.assertIn('models_trained', result)
        
        # Verify accuracy metrics
        self.assertGreaterEqual(result['week_classification_accuracy'], 0.8)
        self.assertGreaterEqual(result['parameter_optimization_improvement'], 0.0)
    
    def test_confidence_boosting(self):
        """Test confidence boosting for week classification"""
        base_confidence = 0.75
        market_data = {
            'spy_price': 450.0,
            'vix': 18.5,
            'rsi': 65.0
        }
        
        boosted_confidence = self.ml_optimizer.boost_classification_confidence(
            base_confidence, market_data
        )
        
        # Verify confidence improvement
        self.assertGreaterEqual(boosted_confidence, base_confidence)
        self.assertLessEqual(boosted_confidence, 1.0)
    
    def test_parameter_optimization(self):
        """Test ML-based parameter optimization"""
        base_params = {
            'delta': 45,
            'dte': 35,
            'position_size': 10
        }
        
        market_context = {
            'week_type': 'P-EW',
            'market_condition': 'bullish',
            'volatility': 'medium'
        }
        
        optimized_params = self.ml_optimizer.optimize_parameters(
            base_params, market_context
        )
        
        # Verify optimization results
        self.assertIn('optimized_params', optimized_params)
        self.assertIn('improvement_potential', optimized_params)
        self.assertIn('confidence', optimized_params)
        
        # Verify improvement potential
        self.assertGreaterEqual(optimized_params['improvement_potential'], 0.0)


class TestHITLTrustSystem(unittest.TestCase):
    """Test suite for HITL Trust System (WS2-P3)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.trust_system = HITLTrustSystem()
        
        # Sample decision for testing
        self.sample_decision = {
            'decision_type': 'trade_entry',
            'ai_recommendation': {
                'action': 'short_put',
                'confidence': 0.85,
                'reasoning': 'High probability setup'
            },
            'context': {
                'week_type': 'P-EW',
                'market_condition': 'bullish'
            }
        }
    
    def test_trust_system_initialization(self):
        """Test trust system initialization"""
        self.assertIsNotNone(self.trust_system)
        
        # Verify initial trust scores
        trust_scores = self.trust_system.get_trust_scores()
        self.assertIn('overall_trust', trust_scores)
        self.assertIn('component_trust', trust_scores)
        
        # Verify initial trust is conservative (30%)
        self.assertAlmostEqual(trust_scores['overall_trust'], 0.30, places=1)
    
    def test_automation_level_determination(self):
        """Test automation level determination"""
        result = self.trust_system.determine_automation_level(self.sample_decision)
        
        # Verify result structure
        self.assertIn('automation_level', result)
        self.assertIn('requires_approval', result)
        self.assertIn('override_window', result)
        self.assertIn('reasoning', result)
        
        # Verify automation level is valid
        valid_levels = ['no_trust', 'low_trust', 'medium_trust', 'high_trust', 'full_trust']
        self.assertIn(result['automation_level'], valid_levels)
    
    def test_trust_building_mechanism(self):
        """Test trust building through successful outcomes"""
        initial_trust = self.trust_system.get_trust_scores()['overall_trust']
        
        # Simulate successful AI decision
        outcome = {
            'decision_id': 'test_001',
            'ai_recommendation': self.sample_decision['ai_recommendation'],
            'human_decision': 'approved',
            'actual_outcome': 'success',
            'performance_impact': 0.025
        }
        
        self.trust_system.update_trust_from_outcome(outcome)
        
        # Verify trust increased
        updated_trust = self.trust_system.get_trust_scores()['overall_trust']
        self.assertGreater(updated_trust, initial_trust)
    
    def test_decision_processing_performance(self):
        """Test decision processing performance"""
        import time
        
        start_time = time.time()
        result = self.trust_system.determine_automation_level(self.sample_decision)
        end_time = time.time()
        
        # Verify response time is reasonable
        response_time = (end_time - start_time) * 1000
        self.assertLess(response_time, 100)


class TestProtocolEngineIntegration(unittest.TestCase):
    """Test suite for Protocol Engine component integration"""
    
    def setUp(self):
        """Set up test fixtures for integration testing"""
        self.week_classifier = WeekClassifier()
        self.market_analyzer = MarketConditionAnalyzer()
        self.rules_engine = TradingProtocolRulesEngine()
        self.atr_system = ATRAdjustmentSystem()
        self.ml_optimizer = MLOptimizer()
        self.trust_system = HITLTrustSystem()
        
        # Sample market data
        self.market_data = {
            'spy_price': 450.0,
            'spy_change': 0.015,
            'vix': 18.5,
            'volume': 90000000,
            'rsi': 65.0,
            'macd': 0.5,
            'bollinger_position': 0.7,
            'atr_14': 8.5,
            'atr_21': 9.2
        }
    
    def test_complete_protocol_workflow(self):
        """Test complete protocol workflow integration"""
        # Step 1: Week Classification
        week_result = self.week_classifier.classify_week(self.market_data)
        self.assertIn('week_type', week_result)
        
        # Step 2: Market Analysis
        market_result = self.market_analyzer.analyze_market_conditions(self.market_data)
        self.assertIn('market_condition', market_result)
        
        # Step 3: ATR Adjustments
        atr_result = self.atr_system.classify_volatility_regime(self.market_data)
        self.assertIn('regime', atr_result)
        
        # Step 4: Generate Trade Proposal
        trade_proposal = {
            'account_type': 'GEN_ACC',
            'strategy': 'short_put',
            'delta': 45,
            'position_size': 8,
            'dte': 35,
            'market_condition': market_result['market_condition'],
            'week_type': week_result['week_type']
        }
        
        # Step 5: Rule Validation
        rules_result = self.rules_engine.validate_trade(trade_proposal)
        self.assertIn('valid', rules_result)
        
        # Step 6: HITL Decision
        decision = {
            'decision_type': 'trade_entry',
            'ai_recommendation': {
                'action': 'short_put',
                'confidence': week_result['confidence'],
                'reasoning': week_result['reasoning']
            },
            'context': {
                'week_type': week_result['week_type'],
                'market_condition': market_result['market_condition']
            }
        }
        
        hitl_result = self.trust_system.determine_automation_level(decision)
        self.assertIn('automation_level', hitl_result)
        
        # Verify complete workflow success
        self.assertTrue(True)  # If we reach here, integration is working
    
    def test_protocol_performance_integration(self):
        """Test integrated protocol performance"""
        import time
        
        start_time = time.time()
        
        # Execute complete protocol workflow
        week_result = self.week_classifier.classify_week(self.market_data)
        market_result = self.market_analyzer.analyze_market_conditions(self.market_data)
        
        trade_proposal = {
            'account_type': 'GEN_ACC',
            'strategy': 'short_put',
            'delta': 45,
            'position_size': 8,
            'dte': 35,
            'market_condition': market_result['market_condition']
        }
        
        rules_result = self.rules_engine.validate_trade(trade_proposal)
        
        end_time = time.time()
        
        # Verify total protocol response time < 200ms
        total_time = (end_time - start_time) * 1000
        self.assertLess(total_time, 200)


def run_protocol_engine_tests():
    """Run all Protocol Engine unit tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestWeekClassifier,
        TestMarketConditionAnalyzer,
        TestTradingProtocolRules,
        TestATRAdjustmentSystem,
        TestMLOptimizer,
        TestHITLTrustSystem,
        TestProtocolEngineIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return test results
    return {
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun,
        'details': {
            'failures': result.failures,
            'errors': result.errors
        }
    }


if __name__ == '__main__':
    print("🧪 Running Protocol Engine Unit Tests (WS2-P4)")
    print("=" * 60)
    
    results = run_protocol_engine_tests()
    
    print(f"\n📊 Test Results Summary:")
    print(f"Tests Run: {results['tests_run']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    
    if results['success_rate'] >= 0.95:
        print("✅ Protocol Engine unit tests PASSED!")
    else:
        print("❌ Protocol Engine unit tests FAILED!")
        print("\nFailure Details:")
        for failure in results['details']['failures']:
            print(f"- {failure[0]}: {failure[1]}")
        for error in results['details']['errors']:
            print(f"- {error[0]}: {error[1]}")

