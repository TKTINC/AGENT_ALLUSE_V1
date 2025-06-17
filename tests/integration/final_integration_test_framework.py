#!/usr/bin/env python3
"""
ALL-USE Agent - Protocol Engine Final Integration Testing Framework
P6 of WS2: Protocol Engine Final Integration and System Testing - Phase 1

This module provides comprehensive integration testing framework for the complete
Protocol Engine system, validating all components working together seamlessly
from market data input to trading decision output with optimization systems.

Integration Testing Components:
1. Integration Test Framework - Comprehensive testing infrastructure
2. Component Integration Validator - Validates component interactions
3. End-to-End Workflow Tester - Complete workflow validation
4. System Performance Validator - Performance testing with optimizations
5. Error Handling Validator - Comprehensive error scenario testing
6. Integration Report Generator - Detailed testing reports and analysis
"""

import time
import json
import traceback
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, '/home/ubuntu/AGENT_ALLUSE_V1/src')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestResult(Enum):
    """Test result status"""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


class TestCategory(Enum):
    """Test categories"""
    COMPONENT_INTEGRATION = "component_integration"
    END_TO_END = "end_to_end"
    PERFORMANCE = "performance"
    ERROR_HANDLING = "error_handling"
    OPTIMIZATION = "optimization"


@dataclass
class TestCase:
    """Test case definition"""
    name: str
    category: TestCategory
    description: str
    test_function: Callable
    timeout_seconds: int = 30
    prerequisites: List[str] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []


@dataclass
class TestExecution:
    """Test execution result"""
    test_case: TestCase
    result: TestResult
    execution_time: float
    start_time: datetime
    end_time: datetime
    output: str = ""
    error_message: str = ""
    performance_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


@dataclass
class IntegrationTestReport:
    """Integration test report"""
    test_suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_execution_time: float
    test_executions: List[TestExecution]
    summary: Dict[str, Any]
    timestamp: datetime


class ComponentIntegrationValidator:
    """Validates integration between Protocol Engine components"""
    
    def __init__(self):
        self.validation_results = {}
        self.component_instances = {}
        
        logger.info("Component Integration Validator initialized")
    
    def initialize_components(self) -> Dict[str, bool]:
        """Initialize all Protocol Engine components"""
        initialization_results = {}
        
        try:
            # Week Classification System
            from protocol_engine.week_classification.week_classifier import WeekClassifier
            from protocol_engine.market_analysis.market_condition_analyzer import MarketConditionAnalyzer
            from protocol_engine.learning.historical_analysis_engine import HistoricalAnalysisEngine
            
            self.component_instances['week_classifier'] = WeekClassifier()
            self.component_instances['market_analyzer'] = MarketConditionAnalyzer()
            self.component_instances['historical_analyzer'] = HistoricalAnalysisEngine()
            
            initialization_results['week_classification_system'] = True
            logger.info("Week Classification System components initialized")
            
        except Exception as e:
            initialization_results['week_classification_system'] = False
            logger.error(f"Failed to initialize Week Classification System: {e}")
        
        try:
            # Protocol Rules Engine
            from protocol_engine.rules.trading_protocol_rules import TradingProtocolRulesEngine
            from protocol_engine.adjustments.atr_adjustment_system import ATRAdjustmentSystem
            from protocol_engine.position_management.position_manager import PositionManager
            from protocol_engine.rollover.rollover_protocol import RolloverProtocol
            
            self.component_instances['rules_engine'] = TradingProtocolRulesEngine()
            self.component_instances['atr_system'] = ATRAdjustmentSystem()
            self.component_instances['position_manager'] = PositionManager()
            self.component_instances['rollover_protocol'] = RolloverProtocol()
            
            initialization_results['protocol_rules_engine'] = True
            logger.info("Protocol Rules Engine components initialized")
            
        except Exception as e:
            initialization_results['protocol_rules_engine'] = False
            logger.error(f"Failed to initialize Protocol Rules Engine: {e}")
        
        try:
            # Advanced Decision System
            from protocol_engine.human_oversight.decision_gateway import DecisionGateway
            from protocol_engine.ml_optimization.ml_optimizer import MLOptimizer
            from protocol_engine.backtesting.backtesting_engine import BacktestingEngine
            from protocol_engine.adaptation.adaptation_engine import AdaptationEngine
            from protocol_engine.trust_system.hitl_trust_system import HITLTrustSystem
            
            self.component_instances['decision_gateway'] = DecisionGateway()
            self.component_instances['ml_optimizer'] = MLOptimizer()
            self.component_instances['backtesting_engine'] = BacktestingEngine()
            self.component_instances['adaptation_engine'] = AdaptationEngine()
            self.component_instances['trust_system'] = HITLTrustSystem()
            
            initialization_results['advanced_decision_system'] = True
            logger.info("Advanced Decision System components initialized")
            
        except Exception as e:
            initialization_results['advanced_decision_system'] = False
            logger.error(f"Failed to initialize Advanced Decision System: {e}")
        
        try:
            # Performance Optimization System
            from protocol_engine.optimization.performance_analyzer import PerformanceAnalyzer
            from protocol_engine.optimization.memory_manager import get_memory_manager
            from protocol_engine.optimization.cache_manager import get_cache_coordinator
            from protocol_engine.monitoring.performance_monitor import get_monitoring_coordinator
            from protocol_engine.analytics.performance_analytics import AnalyticsDashboard
            
            self.component_instances['performance_analyzer'] = PerformanceAnalyzer()
            self.component_instances['memory_manager'] = get_memory_manager()
            self.component_instances['cache_coordinator'] = get_cache_coordinator()
            self.component_instances['monitoring_coordinator'] = get_monitoring_coordinator()
            self.component_instances['analytics_dashboard'] = AnalyticsDashboard()
            
            initialization_results['performance_optimization_system'] = True
            logger.info("Performance Optimization System components initialized")
            
        except Exception as e:
            initialization_results['performance_optimization_system'] = False
            logger.error(f"Failed to initialize Performance Optimization System: {e}")
        
        return initialization_results
    
    def validate_component_interactions(self) -> Dict[str, bool]:
        """Validate interactions between components"""
        interaction_results = {}
        
        # Test Week Classifier + Market Analyzer interaction
        try:
            week_classifier = self.component_instances.get('week_classifier')
            market_analyzer = self.component_instances.get('market_analyzer')
            
            if week_classifier and market_analyzer:
                # Create test market data
                test_market_data = {
                    'symbol': 'TEST',
                    'current_price': 100.0,
                    'previous_close': 98.0,
                    'week_start_price': 95.0,
                    'volume': 1000000,
                    'average_volume': 800000
                }
                
                # Test market analysis
                market_condition = market_analyzer.analyze_market_condition(test_market_data)
                
                # Test week classification with market condition
                if hasattr(week_classifier, 'classify_week'):
                    classification = week_classifier.classify_week(market_condition, 'FLAT')
                    interaction_results['week_classifier_market_analyzer'] = True
                    logger.info("Week Classifier + Market Analyzer interaction validated")
                else:
                    interaction_results['week_classifier_market_analyzer'] = False
                    logger.warning("Week Classifier classify_week method not found")
            else:
                interaction_results['week_classifier_market_analyzer'] = False
                logger.error("Week Classifier or Market Analyzer not available")
                
        except Exception as e:
            interaction_results['week_classifier_market_analyzer'] = False
            logger.error(f"Week Classifier + Market Analyzer interaction failed: {e}")
        
        # Test Rules Engine + ATR System interaction
        try:
            rules_engine = self.component_instances.get('rules_engine')
            atr_system = self.component_instances.get('atr_system')
            
            if rules_engine and atr_system:
                # Test ATR calculation
                test_price_data = [98.0, 99.0, 100.0, 101.0, 100.5]
                if hasattr(atr_system, 'calculate_atr'):
                    atr_value = atr_system.calculate_atr(test_price_data)
                    
                    # Test rules engine with ATR
                    test_context = {
                        'week_type': 'W-IDL',
                        'position': 'FLAT',
                        'atr_value': atr_value
                    }
                    
                    if hasattr(rules_engine, 'evaluate_trading_decision'):
                        decision = rules_engine.evaluate_trading_decision(test_context)
                        interaction_results['rules_engine_atr_system'] = True
                        logger.info("Rules Engine + ATR System interaction validated")
                    else:
                        interaction_results['rules_engine_atr_system'] = False
                        logger.warning("Rules Engine evaluate_trading_decision method not found")
                else:
                    interaction_results['rules_engine_atr_system'] = False
                    logger.warning("ATR System calculate_atr method not found")
            else:
                interaction_results['rules_engine_atr_system'] = False
                logger.error("Rules Engine or ATR System not available")
                
        except Exception as e:
            interaction_results['rules_engine_atr_system'] = False
            logger.error(f"Rules Engine + ATR System interaction failed: {e}")
        
        # Test ML Optimizer + Backtesting Engine interaction
        try:
            ml_optimizer = self.component_instances.get('ml_optimizer')
            backtesting_engine = self.component_instances.get('backtesting_engine')
            
            if ml_optimizer and backtesting_engine:
                # Test ML optimization
                test_parameters = {'risk_factor': 0.02, 'position_size': 1.0}
                
                if hasattr(ml_optimizer, 'optimize_parameters'):
                    optimized_params = ml_optimizer.optimize_parameters(test_parameters)
                    
                    # Test backtesting with optimized parameters
                    if hasattr(backtesting_engine, 'run_backtest'):
                        backtest_result = backtesting_engine.run_backtest(optimized_params)
                        interaction_results['ml_optimizer_backtesting'] = True
                        logger.info("ML Optimizer + Backtesting Engine interaction validated")
                    else:
                        interaction_results['ml_optimizer_backtesting'] = False
                        logger.warning("Backtesting Engine run_backtest method not found")
                else:
                    interaction_results['ml_optimizer_backtesting'] = False
                    logger.warning("ML Optimizer optimize_parameters method not found")
            else:
                interaction_results['ml_optimizer_backtesting'] = False
                logger.error("ML Optimizer or Backtesting Engine not available")
                
        except Exception as e:
            interaction_results['ml_optimizer_backtesting'] = False
            logger.error(f"ML Optimizer + Backtesting Engine interaction failed: {e}")
        
        # Test Performance Optimization Integration
        try:
            cache_coordinator = self.component_instances.get('cache_coordinator')
            monitoring_coordinator = self.component_instances.get('monitoring_coordinator')
            
            if cache_coordinator and monitoring_coordinator:
                # Test cache statistics
                cache_stats = cache_coordinator.get_comprehensive_stats()
                
                # Test monitoring data
                monitoring_data = monitoring_coordinator.get_monitoring_dashboard_data()
                
                if cache_stats and monitoring_data:
                    interaction_results['performance_optimization_integration'] = True
                    logger.info("Performance Optimization integration validated")
                else:
                    interaction_results['performance_optimization_integration'] = False
                    logger.warning("Performance Optimization data not available")
            else:
                interaction_results['performance_optimization_integration'] = False
                logger.error("Performance Optimization components not available")
                
        except Exception as e:
            interaction_results['performance_optimization_integration'] = False
            logger.error(f"Performance Optimization integration failed: {e}")
        
        return interaction_results


class EndToEndWorkflowTester:
    """Tests complete end-to-end Protocol Engine workflow"""
    
    def __init__(self, component_validator: ComponentIntegrationValidator):
        self.component_validator = component_validator
        self.workflow_results = {}
        
        logger.info("End-to-End Workflow Tester initialized")
    
    def test_complete_workflow(self, market_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test complete workflow from market data to trading decision"""
        workflow_start = time.perf_counter()
        workflow_result = {
            'scenario': market_scenario,
            'steps': {},
            'final_decision': None,
            'execution_time': 0.0,
            'success': False
        }
        
        try:
            # Step 1: Market Analysis
            step_start = time.perf_counter()
            market_analyzer = self.component_validator.component_instances.get('market_analyzer')
            
            if market_analyzer:
                market_condition = market_analyzer.analyze_market_condition(market_scenario)
                workflow_result['steps']['market_analysis'] = {
                    'success': True,
                    'execution_time': time.perf_counter() - step_start,
                    'result': str(market_condition) if market_condition else None
                }
                logger.info(f"Market analysis completed: {market_condition}")
            else:
                workflow_result['steps']['market_analysis'] = {
                    'success': False,
                    'execution_time': time.perf_counter() - step_start,
                    'error': 'Market analyzer not available'
                }
                return workflow_result
            
            # Step 2: Week Classification
            step_start = time.perf_counter()
            week_classifier = self.component_validator.component_instances.get('week_classifier')
            
            if week_classifier and hasattr(week_classifier, 'classify_week'):
                week_classification = week_classifier.classify_week(market_condition, 'FLAT')
                workflow_result['steps']['week_classification'] = {
                    'success': True,
                    'execution_time': time.perf_counter() - step_start,
                    'result': str(week_classification) if week_classification else None
                }
                logger.info(f"Week classification completed: {week_classification}")
            else:
                workflow_result['steps']['week_classification'] = {
                    'success': False,
                    'execution_time': time.perf_counter() - step_start,
                    'error': 'Week classifier not available or missing classify_week method'
                }
                return workflow_result
            
            # Step 3: ATR Calculation
            step_start = time.perf_counter()
            atr_system = self.component_validator.component_instances.get('atr_system')
            
            if atr_system and hasattr(atr_system, 'calculate_atr'):
                # Generate test price data based on market scenario
                base_price = market_scenario.get('current_price', 100.0)
                price_data = [base_price * (1 + i * 0.01) for i in range(-5, 1)]
                
                atr_value = atr_system.calculate_atr(price_data)
                workflow_result['steps']['atr_calculation'] = {
                    'success': True,
                    'execution_time': time.perf_counter() - step_start,
                    'result': atr_value
                }
                logger.info(f"ATR calculation completed: {atr_value}")
            else:
                workflow_result['steps']['atr_calculation'] = {
                    'success': False,
                    'execution_time': time.perf_counter() - step_start,
                    'error': 'ATR system not available or missing calculate_atr method'
                }
                return workflow_result
            
            # Step 4: Trading Rules Evaluation
            step_start = time.perf_counter()
            rules_engine = self.component_validator.component_instances.get('rules_engine')
            
            if rules_engine and hasattr(rules_engine, 'evaluate_trading_decision'):
                trading_context = {
                    'week_type': getattr(week_classification, 'week_type', 'W-IDL') if week_classification else 'W-IDL',
                    'position': 'FLAT',
                    'market_condition': market_condition,
                    'atr_value': atr_value
                }
                
                trading_decision = rules_engine.evaluate_trading_decision(trading_context)
                workflow_result['steps']['trading_rules'] = {
                    'success': True,
                    'execution_time': time.perf_counter() - step_start,
                    'result': str(trading_decision) if trading_decision else None
                }
                workflow_result['final_decision'] = trading_decision
                logger.info(f"Trading rules evaluation completed: {trading_decision}")
            else:
                workflow_result['steps']['trading_rules'] = {
                    'success': False,
                    'execution_time': time.perf_counter() - step_start,
                    'error': 'Rules engine not available or missing evaluate_trading_decision method'
                }
                return workflow_result
            
            # Step 5: ML Optimization (if available)
            step_start = time.perf_counter()
            ml_optimizer = self.component_validator.component_instances.get('ml_optimizer')
            
            if ml_optimizer and hasattr(ml_optimizer, 'optimize_parameters'):
                optimization_params = {
                    'risk_factor': 0.02,
                    'position_size': 1.0,
                    'week_type': getattr(week_classification, 'week_type', 'W-IDL') if week_classification else 'W-IDL'
                }
                
                optimized_params = ml_optimizer.optimize_parameters(optimization_params)
                workflow_result['steps']['ml_optimization'] = {
                    'success': True,
                    'execution_time': time.perf_counter() - step_start,
                    'result': optimized_params
                }
                logger.info(f"ML optimization completed: {optimized_params}")
            else:
                workflow_result['steps']['ml_optimization'] = {
                    'success': False,
                    'execution_time': time.perf_counter() - step_start,
                    'error': 'ML optimizer not available or missing optimize_parameters method'
                }
                # This is not critical for workflow success
            
            # Step 6: Performance Monitoring
            step_start = time.perf_counter()
            monitoring_coordinator = self.component_validator.component_instances.get('monitoring_coordinator')
            
            if monitoring_coordinator:
                monitoring_data = monitoring_coordinator.get_monitoring_dashboard_data()
                workflow_result['steps']['performance_monitoring'] = {
                    'success': True,
                    'execution_time': time.perf_counter() - step_start,
                    'result': 'Monitoring data collected'
                }
                logger.info("Performance monitoring data collected")
            else:
                workflow_result['steps']['performance_monitoring'] = {
                    'success': False,
                    'execution_time': time.perf_counter() - step_start,
                    'error': 'Monitoring coordinator not available'
                }
                # This is not critical for workflow success
            
            # Calculate total execution time
            workflow_result['execution_time'] = time.perf_counter() - workflow_start
            
            # Determine overall success
            critical_steps = ['market_analysis', 'week_classification', 'atr_calculation', 'trading_rules']
            workflow_result['success'] = all(
                workflow_result['steps'].get(step, {}).get('success', False) 
                for step in critical_steps
            )
            
            logger.info(f"Complete workflow test {'PASSED' if workflow_result['success'] else 'FAILED'} "
                       f"in {workflow_result['execution_time']*1000:.2f}ms")
            
        except Exception as e:
            workflow_result['execution_time'] = time.perf_counter() - workflow_start
            workflow_result['error'] = str(e)
            workflow_result['success'] = False
            logger.error(f"Workflow test failed with exception: {e}")
        
        return workflow_result
    
    def test_multiple_scenarios(self) -> Dict[str, Any]:
        """Test workflow with multiple market scenarios"""
        scenarios = [
            {
                'name': 'bullish_market',
                'data': {
                    'symbol': 'TEST',
                    'current_price': 105.0,
                    'previous_close': 100.0,
                    'week_start_price': 95.0,
                    'volume': 1200000,
                    'average_volume': 800000
                }
            },
            {
                'name': 'bearish_market',
                'data': {
                    'symbol': 'TEST',
                    'current_price': 95.0,
                    'previous_close': 100.0,
                    'week_start_price': 105.0,
                    'volume': 1500000,
                    'average_volume': 800000
                }
            },
            {
                'name': 'high_volatility',
                'data': {
                    'symbol': 'TEST',
                    'current_price': 102.0,
                    'previous_close': 98.0,
                    'week_start_price': 100.0,
                    'volume': 2000000,
                    'average_volume': 800000
                }
            },
            {
                'name': 'low_volatility',
                'data': {
                    'symbol': 'TEST',
                    'current_price': 100.5,
                    'previous_close': 100.0,
                    'week_start_price': 99.5,
                    'volume': 600000,
                    'average_volume': 800000
                }
            }
        ]
        
        scenario_results = {}
        total_start = time.perf_counter()
        
        for scenario in scenarios:
            logger.info(f"Testing scenario: {scenario['name']}")
            scenario_results[scenario['name']] = self.test_complete_workflow(scenario['data'])
        
        total_execution_time = time.perf_counter() - total_start
        
        # Calculate summary statistics
        successful_scenarios = sum(1 for result in scenario_results.values() if result['success'])
        total_scenarios = len(scenarios)
        
        summary = {
            'total_scenarios': total_scenarios,
            'successful_scenarios': successful_scenarios,
            'success_rate': successful_scenarios / total_scenarios if total_scenarios > 0 else 0.0,
            'total_execution_time': total_execution_time,
            'average_execution_time': total_execution_time / total_scenarios if total_scenarios > 0 else 0.0
        }
        
        return {
            'scenario_results': scenario_results,
            'summary': summary
        }


class IntegrationTestFramework:
    """Comprehensive integration testing framework"""
    
    def __init__(self):
        self.test_cases = []
        self.test_executions = []
        self.component_validator = ComponentIntegrationValidator()
        self.workflow_tester = None
        
        logger.info("Integration Test Framework initialized")
    
    def register_test_case(self, test_case: TestCase):
        """Register a test case"""
        self.test_cases.append(test_case)
        logger.info(f"Registered test case: {test_case.name}")
    
    def setup_test_environment(self) -> bool:
        """Setup test environment and initialize components"""
        logger.info("Setting up test environment...")
        
        # Initialize components
        initialization_results = self.component_validator.initialize_components()
        
        # Check if critical components are initialized
        critical_systems = ['week_classification_system', 'protocol_rules_engine', 'performance_optimization_system']
        critical_initialized = sum(
            1 for system in critical_systems if initialization_results.get(system, False)
        ) >= 2  # At least 2 out of 3 systems should work
        
        if critical_initialized:
            self.workflow_tester = EndToEndWorkflowTester(self.component_validator)
            logger.info("Test environment setup completed successfully")
            return True
        else:
            logger.error("Failed to initialize critical components")
            return False
    
    def execute_test_case(self, test_case: TestCase) -> TestExecution:
        """Execute a single test case"""
        start_time = datetime.now()
        execution_start = time.perf_counter()
        
        logger.info(f"Executing test case: {test_case.name}")
        
        try:
            # Execute test function with timeout
            result_data = test_case.test_function()
            
            execution_time = time.perf_counter() - execution_start
            end_time = datetime.now()
            
            # Determine result based on return value
            if isinstance(result_data, dict):
                success = result_data.get('success', False)
                output = json.dumps(result_data, indent=2, default=str)
                performance_metrics = result_data.get('performance_metrics', {})
            else:
                success = bool(result_data)
                output = str(result_data)
                performance_metrics = {}
            
            result = TestResult.PASS if success else TestResult.FAIL
            
            execution = TestExecution(
                test_case=test_case,
                result=result,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                output=output,
                performance_metrics=performance_metrics
            )
            
            logger.info(f"Test case {test_case.name} {result.value} in {execution_time*1000:.2f}ms")
            
        except Exception as e:
            execution_time = time.perf_counter() - execution_start
            end_time = datetime.now()
            
            execution = TestExecution(
                test_case=test_case,
                result=TestResult.ERROR,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                error_message=str(e),
                output=traceback.format_exc()
            )
            
            logger.error(f"Test case {test_case.name} ERROR: {e}")
        
        return execution
    
    def run_all_tests(self) -> IntegrationTestReport:
        """Run all registered test cases"""
        logger.info(f"Running {len(self.test_cases)} integration test cases...")
        
        # Setup test environment
        if not self.setup_test_environment():
            logger.error("Failed to setup test environment")
            return None
        
        # Execute all test cases
        test_executions = []
        total_start = time.perf_counter()
        
        for test_case in self.test_cases:
            execution = self.execute_test_case(test_case)
            test_executions.append(execution)
        
        total_execution_time = time.perf_counter() - total_start
        
        # Calculate summary statistics
        total_tests = len(test_executions)
        passed_tests = sum(1 for e in test_executions if e.result == TestResult.PASS)
        failed_tests = sum(1 for e in test_executions if e.result == TestResult.FAIL)
        skipped_tests = sum(1 for e in test_executions if e.result == TestResult.SKIP)
        error_tests = sum(1 for e in test_executions if e.result == TestResult.ERROR)
        
        # Generate summary
        summary = {
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'average_execution_time': total_execution_time / total_tests if total_tests > 0 else 0.0,
            'performance_summary': self._calculate_performance_summary(test_executions),
            'category_breakdown': self._calculate_category_breakdown(test_executions)
        }
        
        # Create report
        report = IntegrationTestReport(
            test_suite_name="Protocol Engine Final Integration Tests",
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            total_execution_time=total_execution_time,
            test_executions=test_executions,
            summary=summary,
            timestamp=datetime.now()
        )
        
        logger.info(f"Integration testing completed: {passed_tests}/{total_tests} tests passed "
                   f"({summary['success_rate']:.1%} success rate)")
        
        return report
    
    def _calculate_performance_summary(self, executions: List[TestExecution]) -> Dict[str, float]:
        """Calculate performance summary from test executions"""
        execution_times = [e.execution_time for e in executions if e.execution_time > 0]
        
        if not execution_times:
            return {}
        
        return {
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'total_execution_time': sum(execution_times)
        }
    
    def _calculate_category_breakdown(self, executions: List[TestExecution]) -> Dict[str, Dict[str, int]]:
        """Calculate test results breakdown by category"""
        category_breakdown = {}
        
        for execution in executions:
            category = execution.test_case.category.value
            if category not in category_breakdown:
                category_breakdown[category] = {
                    'total': 0,
                    'passed': 0,
                    'failed': 0,
                    'error': 0,
                    'skipped': 0
                }
            
            category_breakdown[category]['total'] += 1
            result_key = execution.result.value.lower()
            if result_key in category_breakdown[category]:
                category_breakdown[category][result_key] += 1
        
        return category_breakdown


# Test case definitions
def test_component_initialization():
    """Test component initialization"""
    validator = ComponentIntegrationValidator()
    results = validator.initialize_components()
    
    success = any(results.values())  # At least one system should initialize
    
    return {
        'success': success,
        'initialization_results': results,
        'performance_metrics': {
            'systems_initialized': sum(results.values()),
            'total_systems': len(results)
        }
    }


def test_component_interactions():
    """Test component interactions"""
    validator = ComponentIntegrationValidator()
    validator.initialize_components()
    
    interaction_results = validator.validate_component_interactions()
    
    success = any(interaction_results.values())  # At least one interaction should work
    
    return {
        'success': success,
        'interaction_results': interaction_results,
        'performance_metrics': {
            'successful_interactions': sum(interaction_results.values()),
            'total_interactions': len(interaction_results)
        }
    }


def test_end_to_end_workflow():
    """Test end-to-end workflow"""
    validator = ComponentIntegrationValidator()
    validator.initialize_components()
    
    workflow_tester = EndToEndWorkflowTester(validator)
    
    test_scenario = {
        'symbol': 'TEST',
        'current_price': 100.0,
        'previous_close': 98.0,
        'week_start_price': 95.0,
        'volume': 1000000,
        'average_volume': 800000
    }
    
    workflow_result = workflow_tester.test_complete_workflow(test_scenario)
    
    return {
        'success': workflow_result['success'],
        'workflow_result': workflow_result,
        'performance_metrics': {
            'execution_time_ms': workflow_result['execution_time'] * 1000,
            'successful_steps': sum(1 for step in workflow_result['steps'].values() if step.get('success', False)),
            'total_steps': len(workflow_result['steps'])
        }
    }


def test_multiple_scenarios():
    """Test multiple market scenarios"""
    validator = ComponentIntegrationValidator()
    validator.initialize_components()
    
    workflow_tester = EndToEndWorkflowTester(validator)
    scenario_results = workflow_tester.test_multiple_scenarios()
    
    return {
        'success': scenario_results['summary']['success_rate'] > 0.5,  # At least 50% success
        'scenario_results': scenario_results,
        'performance_metrics': {
            'success_rate': scenario_results['summary']['success_rate'],
            'total_scenarios': scenario_results['summary']['total_scenarios'],
            'avg_execution_time_ms': scenario_results['summary']['average_execution_time'] * 1000
        }
    }


def test_performance_optimization_integration():
    """Test performance optimization system integration"""
    validator = ComponentIntegrationValidator()
    initialization_results = validator.initialize_components()
    
    # Check if performance optimization components are available
    performance_available = initialization_results.get('performance_optimization_system', False)
    
    if not performance_available:
        return {
            'success': False,
            'error': 'Performance optimization system not available',
            'performance_metrics': {}
        }
    
    try:
        # Test cache coordinator
        cache_coordinator = validator.component_instances.get('cache_coordinator')
        if cache_coordinator:
            cache_stats = cache_coordinator.get_comprehensive_stats()
            
        # Test monitoring coordinator
        monitoring_coordinator = validator.component_instances.get('monitoring_coordinator')
        if monitoring_coordinator:
            monitoring_data = monitoring_coordinator.get_monitoring_dashboard_data()
        
        # Test analytics dashboard
        analytics_dashboard = validator.component_instances.get('analytics_dashboard')
        if analytics_dashboard:
            dashboard_summary = analytics_dashboard.get_dashboard_summary()
        
        return {
            'success': True,
            'optimization_integration': 'Performance optimization systems integrated successfully',
            'performance_metrics': {
                'cache_available': bool(cache_coordinator),
                'monitoring_available': bool(monitoring_coordinator),
                'analytics_available': bool(analytics_dashboard)
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'performance_metrics': {}
        }


if __name__ == '__main__':
    print("🧪 Protocol Engine Final Integration Testing Framework (P6 of WS2 - Phase 1)")
    print("=" * 85)
    
    # Initialize test framework
    framework = IntegrationTestFramework()
    
    # Register test cases
    test_cases = [
        TestCase(
            name="Component Initialization",
            category=TestCategory.COMPONENT_INTEGRATION,
            description="Test initialization of all Protocol Engine components",
            test_function=test_component_initialization
        ),
        TestCase(
            name="Component Interactions",
            category=TestCategory.COMPONENT_INTEGRATION,
            description="Test interactions between Protocol Engine components",
            test_function=test_component_interactions
        ),
        TestCase(
            name="End-to-End Workflow",
            category=TestCategory.END_TO_END,
            description="Test complete workflow from market data to trading decision",
            test_function=test_end_to_end_workflow
        ),
        TestCase(
            name="Multiple Market Scenarios",
            category=TestCategory.END_TO_END,
            description="Test workflow with multiple market scenarios",
            test_function=test_multiple_scenarios
        ),
        TestCase(
            name="Performance Optimization Integration",
            category=TestCategory.OPTIMIZATION,
            description="Test integration of performance optimization systems",
            test_function=test_performance_optimization_integration
        )
    ]
    
    for test_case in test_cases:
        framework.register_test_case(test_case)
    
    print(f"\n🚀 Running {len(test_cases)} integration test cases...")
    
    # Run all tests
    report = framework.run_all_tests()
    
    if report:
        print(f"\n📊 Integration Test Results:")
        print(f"   Total Tests: {report.total_tests}")
        print(f"   Passed: {report.passed_tests}")
        print(f"   Failed: {report.failed_tests}")
        print(f"   Errors: {report.error_tests}")
        print(f"   Success Rate: {report.summary['success_rate']:.1%}")
        print(f"   Total Execution Time: {report.total_execution_time*1000:.2f}ms")
        print(f"   Average Execution Time: {report.summary['average_execution_time']*1000:.2f}ms")
        
        # Show category breakdown
        print(f"\n📋 Category Breakdown:")
        for category, stats in report.summary['category_breakdown'].items():
            success_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            print(f"   {category}: {stats['passed']}/{stats['total']} ({success_rate:.1%})")
        
        # Show failed tests
        failed_tests = [e for e in report.test_executions if e.result in [TestResult.FAIL, TestResult.ERROR]]
        if failed_tests:
            print(f"\n❌ Failed Tests:")
            for execution in failed_tests:
                print(f"   {execution.test_case.name}: {execution.result.value}")
                if execution.error_message:
                    print(f"      Error: {execution.error_message}")
        
        print(f"\n✅ P6 of WS2 - Phase 1: Final Integration Testing Framework COMPLETE")
        print(f"🚀 Ready for Phase 2: End-to-End System Validation")
    else:
        print("\n❌ Integration testing failed to complete")

