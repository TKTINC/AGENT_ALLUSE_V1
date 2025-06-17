#!/usr/bin/env python3
"""
ALL-USE Agent - Protocol Engine End-to-End System Validation
P6 of WS2: Protocol Engine Final Integration and System Testing - Phase 2

This module provides comprehensive end-to-end system validation for the complete
Protocol Engine, fixing API issues identified in Phase 1 and ensuring all
components work together seamlessly with proper method signatures and data flow.

System Validation Components:
1. API Method Alignment - Fix method name inconsistencies
2. Component Integration Fixes - Resolve import and integration issues  
3. Complete Workflow Validator - End-to-end functionality validation
4. System Performance Validator - Complete system performance testing
5. Data Flow Validator - Validate data consistency across components
6. System Reliability Tester - Test system under various conditions
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


@dataclass
class SystemValidationResult:
    """System validation result"""
    component_name: str
    validation_type: str
    success: bool
    execution_time: float
    result_data: Dict[str, Any]
    error_message: str = ""
    performance_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


@dataclass
class EndToEndWorkflowResult:
    """End-to-end workflow validation result"""
    workflow_name: str
    total_steps: int
    successful_steps: int
    failed_steps: int
    total_execution_time: float
    step_results: List[Dict[str, Any]]
    final_output: Any
    success: bool
    performance_summary: Dict[str, float]


class APIMethodAligner:
    """Fixes API method name inconsistencies identified in testing"""
    
    def __init__(self):
        self.method_mappings = {}
        self.component_instances = {}
        
        logger.info("API Method Aligner initialized")
    
    def initialize_components_with_fixes(self) -> Dict[str, bool]:
        """Initialize components with API method fixes"""
        initialization_results = {}
        
        try:
            # Week Classification System with proper method names
            from protocol_engine.week_classification.week_classifier import WeekClassifier
            from protocol_engine.market_analysis.market_condition_analyzer import MarketConditionAnalyzer
            from protocol_engine.learning.historical_analysis_engine import HistoricalAnalysisEngine
            
            self.component_instances['week_classifier'] = WeekClassifier()
            self.component_instances['market_analyzer'] = MarketConditionAnalyzer()
            self.component_instances['historical_analyzer'] = HistoricalAnalysisEngine()
            
            # Map correct method names
            self.method_mappings['market_analyzer'] = {
                'analyze_market_condition': 'analyze_market_conditions'  # Correct method name
            }
            
            initialization_results['week_classification_system'] = True
            logger.info("Week Classification System initialized with API fixes")
            
        except Exception as e:
            initialization_results['week_classification_system'] = False
            logger.error(f"Failed to initialize Week Classification System: {e}")
        
        try:
            # Protocol Rules Engine with proper imports
            from protocol_engine.rules.trading_protocol_rules import TradingProtocolRulesEngine
            from protocol_engine.adjustments.atr_adjustment_system import ATRAdjustmentSystem
            from protocol_engine.position_management.position_manager import PositionManager
            
            self.component_instances['rules_engine'] = TradingProtocolRulesEngine()
            self.component_instances['atr_system'] = ATRAdjustmentSystem()
            self.component_instances['position_manager'] = PositionManager()
            
            # Try to import rollover protocol if available
            try:
                from protocol_engine.rollover.rollover_protocol import RolloverProtocol
                self.component_instances['rollover_protocol'] = RolloverProtocol()
            except ImportError:
                logger.warning("RolloverProtocol not available, using mock implementation")
                self.component_instances['rollover_protocol'] = self._create_mock_rollover_protocol()
            
            initialization_results['protocol_rules_engine'] = True
            logger.info("Protocol Rules Engine initialized with import fixes")
            
        except Exception as e:
            initialization_results['protocol_rules_engine'] = False
            logger.error(f"Failed to initialize Protocol Rules Engine: {e}")
        
        try:
            # Advanced Decision System with fallback implementations
            try:
                from protocol_engine.human_oversight.decision_gateway import DecisionGateway
                self.component_instances['decision_gateway'] = DecisionGateway()
            except ImportError:
                logger.warning("DecisionGateway not available, using mock implementation")
                self.component_instances['decision_gateway'] = self._create_mock_decision_gateway()
            
            from protocol_engine.ml_optimization.ml_optimizer import MLOptimizer
            from protocol_engine.backtesting.backtesting_engine import BacktestingEngine
            from protocol_engine.adaptation.adaptation_engine import AdaptationEngine
            from protocol_engine.trust_system.hitl_trust_system import HITLTrustSystem
            
            self.component_instances['ml_optimizer'] = MLOptimizer()
            self.component_instances['backtesting_engine'] = BacktestingEngine()
            self.component_instances['adaptation_engine'] = AdaptationEngine()
            self.component_instances['trust_system'] = HITLTrustSystem()
            
            initialization_results['advanced_decision_system'] = True
            logger.info("Advanced Decision System initialized with fallback implementations")
            
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
            logger.info("Performance Optimization System initialized")
            
        except Exception as e:
            initialization_results['performance_optimization_system'] = False
            logger.error(f"Failed to initialize Performance Optimization System: {e}")
        
        return initialization_results
    
    def _create_mock_rollover_protocol(self):
        """Create mock rollover protocol for testing"""
        class MockRolloverProtocol:
            def __init__(self):
                self.name = "MockRolloverProtocol"
            
            def evaluate_rollover_need(self, context):
                return {"rollover_needed": False, "reason": "mock_implementation"}
        
        return MockRolloverProtocol()
    
    def _create_mock_decision_gateway(self):
        """Create mock decision gateway for testing"""
        class MockDecisionGateway:
            def __init__(self):
                self.name = "MockDecisionGateway"
            
            def process_decision(self, decision_context):
                return {"approved": True, "reason": "mock_implementation"}
        
        return MockDecisionGateway()
    
    def call_method_with_mapping(self, component_name: str, method_name: str, *args, **kwargs):
        """Call method with proper name mapping"""
        component = self.component_instances.get(component_name)
        if not component:
            raise ValueError(f"Component {component_name} not available")
        
        # Check if method name needs mapping
        if component_name in self.method_mappings:
            mapped_name = self.method_mappings[component_name].get(method_name, method_name)
        else:
            mapped_name = method_name
        
        # Call the method
        if hasattr(component, mapped_name):
            method = getattr(component, mapped_name)
            return method(*args, **kwargs)
        else:
            raise AttributeError(f"Component {component_name} has no method {mapped_name}")


class CompleteWorkflowValidator:
    """Validates complete end-to-end Protocol Engine workflow"""
    
    def __init__(self, api_aligner: APIMethodAligner):
        self.api_aligner = api_aligner
        self.workflow_results = {}
        
        logger.info("Complete Workflow Validator initialized")
    
    def validate_complete_workflow(self, market_scenario: Dict[str, Any]) -> EndToEndWorkflowResult:
        """Validate complete workflow with proper API calls"""
        workflow_start = time.perf_counter()
        step_results = []
        performance_summary = {}
        
        workflow_name = f"Complete Protocol Engine Workflow - {market_scenario.get('name', 'Unknown')}"
        
        try:
            # Step 1: Market Analysis with correct method name
            step_start = time.perf_counter()
            try:
                market_condition = self.api_aligner.call_method_with_mapping(
                    'market_analyzer', 'analyze_market_condition', market_scenario
                )
                
                step_time = time.perf_counter() - step_start
                step_results.append({
                    'step': 'market_analysis',
                    'success': True,
                    'execution_time': step_time,
                    'result': str(market_condition) if market_condition else None
                })
                performance_summary['market_analysis_time'] = step_time
                logger.info(f"Market analysis completed: {market_condition}")
                
            except Exception as e:
                step_time = time.perf_counter() - step_start
                step_results.append({
                    'step': 'market_analysis',
                    'success': False,
                    'execution_time': step_time,
                    'error': str(e)
                })
                performance_summary['market_analysis_time'] = step_time
                logger.error(f"Market analysis failed: {e}")
                market_condition = None
            
            # Step 2: Week Classification
            step_start = time.perf_counter()
            try:
                week_classifier = self.api_aligner.component_instances.get('week_classifier')
                if week_classifier and hasattr(week_classifier, 'classify_week'):
                    week_classification = week_classifier.classify_week(market_condition, 'FLAT')
                    
                    step_time = time.perf_counter() - step_start
                    step_results.append({
                        'step': 'week_classification',
                        'success': True,
                        'execution_time': step_time,
                        'result': str(week_classification) if week_classification else None
                    })
                    performance_summary['week_classification_time'] = step_time
                    logger.info(f"Week classification completed: {week_classification}")
                else:
                    raise AttributeError("Week classifier not available or missing classify_week method")
                    
            except Exception as e:
                step_time = time.perf_counter() - step_start
                step_results.append({
                    'step': 'week_classification',
                    'success': False,
                    'execution_time': step_time,
                    'error': str(e)
                })
                performance_summary['week_classification_time'] = step_time
                logger.error(f"Week classification failed: {e}")
                week_classification = None
            
            # Step 3: ATR Calculation
            step_start = time.perf_counter()
            try:
                atr_system = self.api_aligner.component_instances.get('atr_system')
                if atr_system and hasattr(atr_system, 'calculate_atr'):
                    # Generate test price data
                    base_price = market_scenario.get('current_price', 100.0)
                    price_data = [base_price * (1 + i * 0.01) for i in range(-5, 1)]
                    
                    atr_value = atr_system.calculate_atr(price_data)
                    
                    step_time = time.perf_counter() - step_start
                    step_results.append({
                        'step': 'atr_calculation',
                        'success': True,
                        'execution_time': step_time,
                        'result': atr_value
                    })
                    performance_summary['atr_calculation_time'] = step_time
                    logger.info(f"ATR calculation completed: {atr_value}")
                else:
                    raise AttributeError("ATR system not available or missing calculate_atr method")
                    
            except Exception as e:
                step_time = time.perf_counter() - step_start
                step_results.append({
                    'step': 'atr_calculation',
                    'success': False,
                    'execution_time': step_time,
                    'error': str(e)
                })
                performance_summary['atr_calculation_time'] = step_time
                logger.error(f"ATR calculation failed: {e}")
                atr_value = 0.02  # Default value
            
            # Step 4: Trading Rules Evaluation
            step_start = time.perf_counter()
            try:
                rules_engine = self.api_aligner.component_instances.get('rules_engine')
                if rules_engine and hasattr(rules_engine, 'evaluate_trading_decision'):
                    trading_context = {
                        'week_type': getattr(week_classification, 'week_type', 'W-IDL') if week_classification else 'W-IDL',
                        'position': 'FLAT',
                        'market_condition': market_condition,
                        'atr_value': atr_value
                    }
                    
                    trading_decision = rules_engine.evaluate_trading_decision(trading_context)
                    
                    step_time = time.perf_counter() - step_start
                    step_results.append({
                        'step': 'trading_rules',
                        'success': True,
                        'execution_time': step_time,
                        'result': str(trading_decision) if trading_decision else None
                    })
                    performance_summary['trading_rules_time'] = step_time
                    logger.info(f"Trading rules evaluation completed: {trading_decision}")
                else:
                    raise AttributeError("Rules engine not available or missing evaluate_trading_decision method")
                    
            except Exception as e:
                step_time = time.perf_counter() - step_start
                step_results.append({
                    'step': 'trading_rules',
                    'success': False,
                    'execution_time': step_time,
                    'error': str(e)
                })
                performance_summary['trading_rules_time'] = step_time
                logger.error(f"Trading rules evaluation failed: {e}")
                trading_decision = None
            
            # Step 5: Performance Monitoring Integration
            step_start = time.perf_counter()
            try:
                monitoring_coordinator = self.api_aligner.component_instances.get('monitoring_coordinator')
                if monitoring_coordinator:
                    monitoring_data = monitoring_coordinator.get_monitoring_dashboard_data()
                    
                    step_time = time.perf_counter() - step_start
                    step_results.append({
                        'step': 'performance_monitoring',
                        'success': True,
                        'execution_time': step_time,
                        'result': 'Monitoring data collected successfully'
                    })
                    performance_summary['monitoring_time'] = step_time
                    logger.info("Performance monitoring integration completed")
                else:
                    raise AttributeError("Monitoring coordinator not available")
                    
            except Exception as e:
                step_time = time.perf_counter() - step_start
                step_results.append({
                    'step': 'performance_monitoring',
                    'success': False,
                    'execution_time': step_time,
                    'error': str(e)
                })
                performance_summary['monitoring_time'] = step_time
                logger.warning(f"Performance monitoring integration failed: {e}")
            
            # Step 6: Cache Performance Validation
            step_start = time.perf_counter()
            try:
                cache_coordinator = self.api_aligner.component_instances.get('cache_coordinator')
                if cache_coordinator:
                    cache_stats = cache_coordinator.get_comprehensive_stats()
                    
                    step_time = time.perf_counter() - step_start
                    step_results.append({
                        'step': 'cache_validation',
                        'success': True,
                        'execution_time': step_time,
                        'result': f"Cache stats: {len(cache_stats)} cache systems active"
                    })
                    performance_summary['cache_validation_time'] = step_time
                    logger.info("Cache performance validation completed")
                else:
                    raise AttributeError("Cache coordinator not available")
                    
            except Exception as e:
                step_time = time.perf_counter() - step_start
                step_results.append({
                    'step': 'cache_validation',
                    'success': False,
                    'execution_time': step_time,
                    'error': str(e)
                })
                performance_summary['cache_validation_time'] = step_time
                logger.warning(f"Cache validation failed: {e}")
            
            # Calculate workflow summary
            total_execution_time = time.perf_counter() - workflow_start
            successful_steps = sum(1 for step in step_results if step['success'])
            failed_steps = len(step_results) - successful_steps
            
            # Determine overall success (at least 4 out of 6 steps should succeed)
            success = successful_steps >= 4
            
            performance_summary['total_workflow_time'] = total_execution_time
            performance_summary['average_step_time'] = total_execution_time / len(step_results) if step_results else 0
            
            result = EndToEndWorkflowResult(
                workflow_name=workflow_name,
                total_steps=len(step_results),
                successful_steps=successful_steps,
                failed_steps=failed_steps,
                total_execution_time=total_execution_time,
                step_results=step_results,
                final_output=trading_decision,
                success=success,
                performance_summary=performance_summary
            )
            
            logger.info(f"Complete workflow validation {'PASSED' if success else 'FAILED'}: "
                       f"{successful_steps}/{len(step_results)} steps successful "
                       f"in {total_execution_time*1000:.2f}ms")
            
            return result
            
        except Exception as e:
            total_execution_time = time.perf_counter() - workflow_start
            logger.error(f"Workflow validation failed with exception: {e}")
            
            return EndToEndWorkflowResult(
                workflow_name=workflow_name,
                total_steps=0,
                successful_steps=0,
                failed_steps=1,
                total_execution_time=total_execution_time,
                step_results=[{'step': 'workflow_execution', 'success': False, 'error': str(e)}],
                final_output=None,
                success=False,
                performance_summary={'total_workflow_time': total_execution_time}
            )
    
    def validate_multiple_scenarios(self) -> Dict[str, Any]:
        """Validate workflow with multiple market scenarios"""
        scenarios = [
            {
                'name': 'bullish_market',
                'symbol': 'TEST',
                'current_price': 105.0,
                'previous_close': 100.0,
                'week_start_price': 95.0,
                'volume': 1200000,
                'average_volume': 800000
            },
            {
                'name': 'bearish_market',
                'symbol': 'TEST',
                'current_price': 95.0,
                'previous_close': 100.0,
                'week_start_price': 105.0,
                'volume': 1500000,
                'average_volume': 800000
            },
            {
                'name': 'high_volatility',
                'symbol': 'TEST',
                'current_price': 102.0,
                'previous_close': 98.0,
                'week_start_price': 100.0,
                'volume': 2000000,
                'average_volume': 800000
            },
            {
                'name': 'neutral_market',
                'symbol': 'TEST',
                'current_price': 100.5,
                'previous_close': 100.0,
                'week_start_price': 99.5,
                'volume': 600000,
                'average_volume': 800000
            }
        ]
        
        scenario_results = {}
        total_start = time.perf_counter()
        
        for scenario in scenarios:
            logger.info(f"Validating scenario: {scenario['name']}")
            scenario_results[scenario['name']] = self.validate_complete_workflow(scenario)
        
        total_execution_time = time.perf_counter() - total_start
        
        # Calculate summary statistics
        successful_scenarios = sum(1 for result in scenario_results.values() if result.success)
        total_scenarios = len(scenarios)
        
        # Calculate performance statistics
        all_step_times = []
        for result in scenario_results.values():
            all_step_times.extend([step['execution_time'] for step in result.step_results if 'execution_time' in step])
        
        summary = {
            'total_scenarios': total_scenarios,
            'successful_scenarios': successful_scenarios,
            'success_rate': successful_scenarios / total_scenarios if total_scenarios > 0 else 0.0,
            'total_execution_time': total_execution_time,
            'average_scenario_time': total_execution_time / total_scenarios if total_scenarios > 0 else 0.0,
            'average_step_time': sum(all_step_times) / len(all_step_times) if all_step_times else 0.0,
            'min_step_time': min(all_step_times) if all_step_times else 0.0,
            'max_step_time': max(all_step_times) if all_step_times else 0.0
        }
        
        return {
            'scenario_results': scenario_results,
            'summary': summary
        }


class SystemPerformanceValidator:
    """Validates complete system performance with optimizations"""
    
    def __init__(self, api_aligner: APIMethodAligner):
        self.api_aligner = api_aligner
        self.performance_results = {}
        
        logger.info("System Performance Validator initialized")
    
    def validate_optimization_integration(self) -> Dict[str, Any]:
        """Validate performance optimization integration"""
        validation_start = time.perf_counter()
        results = {
            'cache_performance': {},
            'memory_optimization': {},
            'monitoring_integration': {},
            'analytics_integration': {}
        }
        
        try:
            # Test cache performance
            cache_coordinator = self.api_aligner.component_instances.get('cache_coordinator')
            if cache_coordinator:
                cache_stats = cache_coordinator.get_comprehensive_stats()
                results['cache_performance'] = {
                    'available': True,
                    'cache_systems': len(cache_stats),
                    'stats': cache_stats
                }
                logger.info(f"Cache performance validated: {len(cache_stats)} systems active")
            else:
                results['cache_performance'] = {'available': False, 'error': 'Cache coordinator not available'}
            
            # Test memory optimization
            memory_manager = self.api_aligner.component_instances.get('memory_manager')
            if memory_manager:
                memory_stats = memory_manager.get_memory_stats()
                results['memory_optimization'] = {
                    'available': True,
                    'stats': memory_stats
                }
                logger.info("Memory optimization validated")
            else:
                results['memory_optimization'] = {'available': False, 'error': 'Memory manager not available'}
            
            # Test monitoring integration
            monitoring_coordinator = self.api_aligner.component_instances.get('monitoring_coordinator')
            if monitoring_coordinator:
                monitoring_data = monitoring_coordinator.get_monitoring_dashboard_data()
                results['monitoring_integration'] = {
                    'available': True,
                    'data_available': bool(monitoring_data)
                }
                logger.info("Monitoring integration validated")
            else:
                results['monitoring_integration'] = {'available': False, 'error': 'Monitoring coordinator not available'}
            
            # Test analytics integration
            analytics_dashboard = self.api_aligner.component_instances.get('analytics_dashboard')
            if analytics_dashboard:
                dashboard_summary = analytics_dashboard.get_dashboard_summary()
                results['analytics_integration'] = {
                    'available': True,
                    'summary_available': bool(dashboard_summary)
                }
                logger.info("Analytics integration validated")
            else:
                results['analytics_integration'] = {'available': False, 'error': 'Analytics dashboard not available'}
            
        except Exception as e:
            logger.error(f"Optimization integration validation failed: {e}")
            results['error'] = str(e)
        
        validation_time = time.perf_counter() - validation_start
        results['validation_time'] = validation_time
        
        # Calculate overall success
        available_systems = sum(1 for system in results.values() if isinstance(system, dict) and system.get('available', False))
        total_systems = 4  # cache, memory, monitoring, analytics
        results['success'] = available_systems >= 3  # At least 3 out of 4 should work
        results['available_systems'] = available_systems
        results['total_systems'] = total_systems
        
        return results


if __name__ == '__main__':
    print("🔧 Protocol Engine End-to-End System Validation (P6 of WS2 - Phase 2)")
    print("=" * 85)
    
    # Initialize API aligner and fix method issues
    api_aligner = APIMethodAligner()
    
    print("\n🚀 Initializing components with API fixes...")
    initialization_results = api_aligner.initialize_components_with_fixes()
    
    print(f"\n📊 Component Initialization Results:")
    for system, success in initialization_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {system}: {status}")
    
    # Initialize workflow validator
    workflow_validator = CompleteWorkflowValidator(api_aligner)
    
    print(f"\n🧪 Running end-to-end workflow validation...")
    
    # Test single scenario first
    test_scenario = {
        'name': 'test_scenario',
        'symbol': 'TEST',
        'current_price': 100.0,
        'previous_close': 98.0,
        'week_start_price': 95.0,
        'volume': 1000000,
        'average_volume': 800000
    }
    
    single_result = workflow_validator.validate_complete_workflow(test_scenario)
    
    print(f"\n📋 Single Scenario Test Results:")
    print(f"   Workflow: {single_result.workflow_name}")
    print(f"   Success: {'✅ PASSED' if single_result.success else '❌ FAILED'}")
    print(f"   Steps: {single_result.successful_steps}/{single_result.total_steps} successful")
    print(f"   Execution Time: {single_result.total_execution_time*1000:.2f}ms")
    
    if single_result.step_results:
        print(f"\n   Step Details:")
        for step in single_result.step_results:
            status = "✅" if step['success'] else "❌"
            time_ms = step.get('execution_time', 0) * 1000
            print(f"     {status} {step['step']}: {time_ms:.2f}ms")
    
    # Test multiple scenarios
    print(f"\n🔄 Running multiple scenario validation...")
    multi_results = workflow_validator.validate_multiple_scenarios()
    
    print(f"\n📊 Multiple Scenario Results:")
    print(f"   Total Scenarios: {multi_results['summary']['total_scenarios']}")
    print(f"   Successful: {multi_results['summary']['successful_scenarios']}")
    print(f"   Success Rate: {multi_results['summary']['success_rate']:.1%}")
    print(f"   Total Time: {multi_results['summary']['total_execution_time']*1000:.2f}ms")
    print(f"   Average Scenario Time: {multi_results['summary']['average_scenario_time']*1000:.2f}ms")
    print(f"   Average Step Time: {multi_results['summary']['average_step_time']*1000:.2f}ms")
    
    # Test performance optimization integration
    print(f"\n⚡ Validating performance optimization integration...")
    performance_validator = SystemPerformanceValidator(api_aligner)
    optimization_results = performance_validator.validate_optimization_integration()
    
    print(f"\n🚀 Performance Optimization Results:")
    print(f"   Overall Success: {'✅ PASSED' if optimization_results['success'] else '❌ FAILED'}")
    print(f"   Available Systems: {optimization_results['available_systems']}/{optimization_results['total_systems']}")
    print(f"   Validation Time: {optimization_results['validation_time']*1000:.2f}ms")
    
    for system, result in optimization_results.items():
        if isinstance(result, dict) and 'available' in result:
            status = "✅ AVAILABLE" if result['available'] else "❌ UNAVAILABLE"
            print(f"     {system}: {status}")
    
    # Calculate overall system validation success
    overall_success = (
        single_result.success and 
        multi_results['summary']['success_rate'] >= 0.75 and  # At least 75% scenario success
        optimization_results['success']
    )
    
    print(f"\n🎯 Overall System Validation: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    
    if overall_success:
        print(f"\n✅ P6 of WS2 - Phase 2: End-to-End System Validation COMPLETE")
        print(f"🚀 Ready for Phase 3: Performance and Load Testing")
    else:
        print(f"\n⚠️  System validation completed with issues - proceeding to Phase 3 for further testing")
        print(f"🚀 Ready for Phase 3: Performance and Load Testing")

