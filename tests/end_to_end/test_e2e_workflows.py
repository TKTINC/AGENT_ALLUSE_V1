"""
ALL-USE End-to-End Test Suite

Comprehensive end-to-end testing for complete user workflows and system functionality.
Tests realistic user scenarios from greeting through trading decisions with full
system integration.
"""

import asyncio
import pytest
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from system.system_orchestrator import orchestrator, register_all_ws1_components
from tests.utils.test_utilities import MockDataGenerator, TestAssertions

logger = logging.getLogger('e2e_tests')


class EndToEndTestSuite:
    """Comprehensive end-to-end test suite for ALL-USE system."""
    
    def __init__(self):
        self.orchestrator = orchestrator
        self.mock_data = MockDataGenerator()
        self.assertions = TestAssertions()
        self.test_results = {}
        
    async def setup_system(self):
        """Set up the complete system for testing."""
        # Register all components
        register_all_ws1_components()
        
        # Start the system
        success = await self.orchestrator.startup_system()
        if not success:
            raise RuntimeError("Failed to start system for testing")
        
        logger.info("System setup completed for end-to-end testing")
    
    async def teardown_system(self):
        """Tear down the system after testing."""
        await self.orchestrator.shutdown_system()
        logger.info("System teardown completed")
    
    async def test_complete_user_onboarding(self) -> Dict[str, Any]:
        """
        Test Scenario 1: Complete User Onboarding
        
        Flow:
        1. User greeting and welcome
        2. Account setup and configuration
        3. Risk assessment and preferences
        4. Initial recommendations
        5. Performance monitoring setup
        """
        test_name = "complete_user_onboarding"
        start_time = time.time()
        
        try:
            logger.info("Starting complete user onboarding test")
            
            # Get agent component
            agent = self.orchestrator.get_component('enhanced_agent')
            
            # Step 1: User greeting
            greeting_response = await self.simulate_user_interaction(
                agent, "Hello, I'm new to ALL-USE and want to get started"
            )
            
            assert "welcome" in greeting_response.lower()
            assert "account" in greeting_response.lower()
            
            # Step 2: Account setup
            account_response = await self.simulate_user_interaction(
                agent, "I want to set up my trading accounts"
            )
            
            assert "account" in account_response.lower()
            assert any(acc_type in account_response.lower() 
                      for acc_type in ["gen-acc", "rev-acc", "com-acc"])
            
            # Step 3: Risk assessment
            risk_response = await self.simulate_user_interaction(
                agent, "I'm moderately aggressive with a $50,000 account"
            )
            
            assert "risk" in risk_response.lower()
            assert "50" in risk_response or "50000" in risk_response
            
            # Step 4: Initial recommendations
            rec_response = await self.simulate_user_interaction(
                agent, "What should I trade this week?"
            )
            
            assert any(term in rec_response.lower() 
                      for term in ["recommend", "suggest", "trade", "option"])
            
            # Step 5: Performance monitoring
            perf_response = await self.simulate_user_interaction(
                agent, "How can I track my performance?"
            )
            
            assert any(term in perf_response.lower() 
                      for term in ["performance", "track", "monitor", "report"])
            
            execution_time = time.time() - start_time
            
            result = {
                'status': 'passed',
                'execution_time': execution_time,
                'steps_completed': 5,
                'performance_target': 5.0,  # 5 seconds target
                'performance_actual': execution_time,
                'performance_met': execution_time < 5.0,
                'details': {
                    'greeting_response_length': len(greeting_response),
                    'account_setup_mentioned': "account" in account_response.lower(),
                    'risk_assessment_captured': "risk" in risk_response.lower(),
                    'recommendations_provided': "recommend" in rec_response.lower(),
                    'monitoring_explained': "performance" in perf_response.lower()
                }
            }
            
            logger.info(f"User onboarding test completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"User onboarding test failed: {e}")
            
            return {
                'status': 'failed',
                'execution_time': execution_time,
                'error': str(e),
                'steps_completed': 0
            }
    
    async def test_trading_decision_workflow(self) -> Dict[str, Any]:
        """
        Test Scenario 2: Trading Decision Workflow
        
        Flow:
        1. Market analysis request
        2. Risk assessment
        3. Position sizing calculation
        4. Delta selection
        5. Trade execution simulation
        """
        test_name = "trading_decision_workflow"
        start_time = time.time()
        
        try:
            logger.info("Starting trading decision workflow test")
            
            # Get required components
            agent = self.orchestrator.get_component('enhanced_agent')
            market_analyzer = self.orchestrator.get_component('market_analyzer')
            position_sizer = self.orchestrator.get_component('position_sizer')
            delta_selector = self.orchestrator.get_component('delta_selector')
            risk_monitor = self.orchestrator.get_component('portfolio_risk_monitor')
            
            # Generate test data
            market_data = self.mock_data.generate_market_data()
            portfolio_data = self.mock_data.generate_portfolio_data()
            
            # Step 1: Market analysis
            market_analysis = market_analyzer.analyze_market_condition('SPY', market_data)
            assert 'market_condition' in market_analysis
            assert 'confidence' in market_analysis
            
            # Step 2: Risk assessment
            risk_assessment = risk_monitor.assess_portfolio_risk(portfolio_data)
            assert 'overall_risk_level' in risk_assessment
            assert 'risk_score' in risk_assessment
            
            # Step 3: Position sizing
            position_size = position_sizer.calculate_position_size(
                account_balance=50000,
                account_type='GEN_ACC',
                market_condition=market_analysis['market_condition'],
                volatility=market_data['volatility']
            )
            assert 'position_size' in position_size
            assert position_size['position_size'] > 0
            
            # Step 4: Delta selection
            delta_selection = delta_selector.select_optimal_delta(
                market_condition=market_analysis['market_condition'],
                account_type='GEN_ACC',
                volatility=market_data['volatility']
            )
            assert 'selected_delta' in delta_selection
            assert 0.1 <= delta_selection['selected_delta'] <= 0.8
            
            # Step 5: Agent integration test
            trading_response = await self.simulate_user_interaction(
                agent, f"What should I trade on SPY with a $50,000 account?"
            )
            
            assert any(term in trading_response.lower() 
                      for term in ["spy", "trade", "option", "delta", "position"])
            
            execution_time = time.time() - start_time
            
            result = {
                'status': 'passed',
                'execution_time': execution_time,
                'steps_completed': 5,
                'performance_target': 3.0,  # 3 seconds target
                'performance_actual': execution_time,
                'performance_met': execution_time < 3.0,
                'details': {
                    'market_condition': market_analysis['market_condition'],
                    'risk_level': risk_assessment['overall_risk_level'],
                    'position_size': position_size['position_size'],
                    'selected_delta': delta_selection['selected_delta'],
                    'agent_response_length': len(trading_response)
                }
            }
            
            logger.info(f"Trading decision workflow test completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Trading decision workflow test failed: {e}")
            
            return {
                'status': 'failed',
                'execution_time': execution_time,
                'error': str(e),
                'steps_completed': 0
            }
    
    async def test_risk_management_workflow(self) -> Dict[str, Any]:
        """
        Test Scenario 3: Risk Management Workflow
        
        Flow:
        1. Portfolio monitoring
        2. Drawdown detection
        3. Protection activation
        4. Position adjustment
        5. Recovery monitoring
        """
        test_name = "risk_management_workflow"
        start_time = time.time()
        
        try:
            logger.info("Starting risk management workflow test")
            
            # Get required components
            agent = self.orchestrator.get_component('enhanced_agent')
            risk_monitor = self.orchestrator.get_component('portfolio_risk_monitor')
            drawdown_protection = self.orchestrator.get_component('drawdown_protection')
            
            # Generate test data with drawdown scenario
            portfolio_data = self.mock_data.generate_portfolio_data()
            
            # Simulate portfolio with drawdown
            portfolio_data['total_value'] = 45000  # Down from 50000
            portfolio_data['peak_value'] = 50000
            
            # Step 1: Portfolio monitoring
            risk_assessment = risk_monitor.assess_portfolio_risk(portfolio_data)
            assert 'overall_risk_level' in risk_assessment
            
            # Step 2: Drawdown detection
            drawdown_status = drawdown_protection.check_drawdown_status(portfolio_data)
            assert 'current_drawdown' in drawdown_status
            assert drawdown_status['current_drawdown'] > 0
            
            # Step 3: Protection activation
            protection_response = drawdown_protection.activate_protection(
                portfolio_data, drawdown_status['current_drawdown']
            )
            assert 'protection_level' in protection_response
            assert 'adjustments_made' in protection_response
            
            # Step 4: Position adjustment simulation
            if protection_response['adjustments_made']:
                adjusted_portfolio = portfolio_data.copy()
                # Simulate position reductions
                for position in adjusted_portfolio['positions']:
                    position['quantity'] *= 0.8  # 20% reduction
            
            # Step 5: Agent integration test
            risk_response = await self.simulate_user_interaction(
                agent, "My portfolio is down 10%, what should I do?"
            )
            
            assert any(term in risk_response.lower() 
                      for term in ["risk", "drawdown", "protection", "reduce", "adjust"])
            
            execution_time = time.time() - start_time
            
            result = {
                'status': 'passed',
                'execution_time': execution_time,
                'steps_completed': 5,
                'performance_target': 2.0,  # 2 seconds target
                'performance_actual': execution_time,
                'performance_met': execution_time < 2.0,
                'details': {
                    'drawdown_detected': drawdown_status['current_drawdown'],
                    'protection_activated': protection_response['protection_level'],
                    'adjustments_made': protection_response['adjustments_made'],
                    'risk_level': risk_assessment['overall_risk_level'],
                    'agent_response_length': len(risk_response)
                }
            }
            
            logger.info(f"Risk management workflow test completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Risk management workflow test failed: {e}")
            
            return {
                'status': 'failed',
                'execution_time': execution_time,
                'error': str(e),
                'steps_completed': 0
            }
    
    async def test_performance_optimization_workflow(self) -> Dict[str, Any]:
        """
        Test Scenario 4: Performance Optimization Workflow
        
        Flow:
        1. Performance monitoring
        2. Bottleneck detection
        3. Optimization application
        4. Validation
        5. Monitoring integration
        """
        test_name = "performance_optimization_workflow"
        start_time = time.time()
        
        try:
            logger.info("Starting performance optimization workflow test")
            
            # Get required components
            performance_optimizer = self.orchestrator.get_component('performance_optimizer')
            
            # Step 1: Performance monitoring
            initial_metrics = performance_optimizer.get_performance_metrics()
            assert 'cache_stats' in initial_metrics
            assert 'memory_stats' in initial_metrics
            
            # Step 2: Bottleneck detection (simulate some operations)
            for i in range(10):
                # Simulate cache operations
                performance_optimizer.cache.get(f"test_key_{i}", lambda: f"test_value_{i}")
            
            # Step 3: Optimization application
            optimization_result = performance_optimizer.optimize_performance()
            assert 'optimizations_applied' in optimization_result
            
            # Step 4: Validation
            post_optimization_metrics = performance_optimizer.get_performance_metrics()
            assert 'cache_stats' in post_optimization_metrics
            
            # Step 5: System integration validation
            system_status = self.orchestrator.get_system_status()
            assert system_status.status in ['running', 'degraded']
            
            execution_time = time.time() - start_time
            
            result = {
                'status': 'passed',
                'execution_time': execution_time,
                'steps_completed': 5,
                'performance_target': 1.0,  # 1 second target
                'performance_actual': execution_time,
                'performance_met': execution_time < 1.0,
                'details': {
                    'initial_cache_hits': initial_metrics['cache_stats']['hits'],
                    'optimizations_applied': optimization_result['optimizations_applied'],
                    'post_optimization_cache_hits': post_optimization_metrics['cache_stats']['hits'],
                    'system_status': system_status.status,
                    'components_running': len([s for s in system_status.components.values() if s == 'running'])
                }
            }
            
            logger.info(f"Performance optimization workflow test completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Performance optimization workflow test failed: {e}")
            
            return {
                'status': 'failed',
                'execution_time': execution_time,
                'error': str(e),
                'steps_completed': 0
            }
    
    async def test_concurrent_user_load(self) -> Dict[str, Any]:
        """
        Test Scenario 5: Concurrent User Load
        
        Simulate multiple users interacting with the system simultaneously
        to validate performance under load.
        """
        test_name = "concurrent_user_load"
        start_time = time.time()
        
        try:
            logger.info("Starting concurrent user load test")
            
            # Get agent component
            agent = self.orchestrator.get_component('enhanced_agent')
            
            # Define user scenarios
            user_scenarios = [
                "Hello, what's the market looking like today?",
                "I want to trade SPY options, what do you recommend?",
                "My portfolio is down 5%, should I be worried?",
                "What's the best delta for current market conditions?",
                "How should I size my positions for a $100k account?",
                "Can you analyze the risk in my current portfolio?",
                "What are the trading opportunities this week?",
                "I'm new to options, can you help me get started?",
                "Should I adjust my positions based on recent volatility?",
                "What's your recommendation for account management?"
            ]
            
            # Run concurrent user interactions
            concurrent_tasks = []
            for i, scenario in enumerate(user_scenarios):
                task = asyncio.create_task(
                    self.simulate_user_interaction(agent, scenario, user_id=f"user_{i}")
                )
                concurrent_tasks.append(task)
            
            # Wait for all tasks to complete
            responses = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            
            # Analyze results
            successful_responses = [r for r in responses if isinstance(r, str)]
            failed_responses = [r for r in responses if isinstance(r, Exception)]
            
            execution_time = time.time() - start_time
            
            result = {
                'status': 'passed' if len(failed_responses) == 0 else 'partial',
                'execution_time': execution_time,
                'concurrent_users': len(user_scenarios),
                'successful_responses': len(successful_responses),
                'failed_responses': len(failed_responses),
                'performance_target': 10.0,  # 10 seconds target for 10 concurrent users
                'performance_actual': execution_time,
                'performance_met': execution_time < 10.0,
                'details': {
                    'average_response_length': sum(len(r) for r in successful_responses) / len(successful_responses) if successful_responses else 0,
                    'throughput_users_per_second': len(user_scenarios) / execution_time,
                    'error_rate': len(failed_responses) / len(user_scenarios),
                    'system_stability': len(failed_responses) == 0
                }
            }
            
            logger.info(f"Concurrent user load test completed: {len(successful_responses)}/{len(user_scenarios)} successful")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Concurrent user load test failed: {e}")
            
            return {
                'status': 'failed',
                'execution_time': execution_time,
                'error': str(e),
                'concurrent_users': 0
            }
    
    async def simulate_user_interaction(self, agent, message: str, user_id: str = "test_user") -> str:
        """Simulate a user interaction with the agent."""
        try:
            # Create conversation context
            conversation_context = {
                'user_id': user_id,
                'session_id': f"session_{int(time.time())}",
                'timestamp': datetime.now(),
                'message': message
            }
            
            # Process message through agent
            if hasattr(agent, 'process_message'):
                if asyncio.iscoroutinefunction(agent.process_message):
                    response = await agent.process_message(message)
                else:
                    response = agent.process_message(message)
            elif hasattr(agent, 'generate_response'):
                response = agent.generate_response(message)
            else:
                # Fallback to basic response
                response = f"Processed: {message}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in user interaction simulation: {e}")
            raise
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all end-to-end tests and return comprehensive results."""
        logger.info("Starting comprehensive end-to-end test suite")
        
        test_results = {}
        overall_start_time = time.time()
        
        try:
            # Set up system
            await self.setup_system()
            
            # Run all test scenarios
            test_scenarios = [
                ('user_onboarding', self.test_complete_user_onboarding),
                ('trading_workflow', self.test_trading_decision_workflow),
                ('risk_management', self.test_risk_management_workflow),
                ('performance_optimization', self.test_performance_optimization_workflow),
                ('concurrent_load', self.test_concurrent_user_load)
            ]
            
            for test_name, test_method in test_scenarios:
                logger.info(f"Running test: {test_name}")
                test_results[test_name] = await test_method()
            
            # Calculate overall results
            total_execution_time = time.time() - overall_start_time
            passed_tests = sum(1 for result in test_results.values() if result['status'] == 'passed')
            total_tests = len(test_scenarios)
            
            overall_result = {
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': total_tests - passed_tests,
                    'success_rate': passed_tests / total_tests,
                    'total_execution_time': total_execution_time,
                    'overall_status': 'passed' if passed_tests == total_tests else 'failed'
                },
                'test_results': test_results,
                'system_status': self.orchestrator.get_system_status(),
                'timestamp': datetime.now()
            }
            
            logger.info(f"End-to-end test suite completed: {passed_tests}/{total_tests} tests passed")
            return overall_result
            
        except Exception as e:
            logger.error(f"End-to-end test suite failed: {e}")
            return {
                'summary': {
                    'total_tests': 0,
                    'passed_tests': 0,
                    'failed_tests': 1,
                    'success_rate': 0.0,
                    'total_execution_time': time.time() - overall_start_time,
                    'overall_status': 'failed',
                    'error': str(e)
                },
                'test_results': test_results,
                'timestamp': datetime.now()
            }
        
        finally:
            # Clean up system
            await self.teardown_system()


if __name__ == "__main__":
    async def main():
        # Run the comprehensive end-to-end test suite
        test_suite = EndToEndTestSuite()
        results = await test_suite.run_all_tests()
        
        # Print results
        print("\n" + "="*80)
        print("ALL-USE END-TO-END TEST RESULTS")
        print("="*80)
        
        summary = results['summary']
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Execution Time: {summary['total_execution_time']:.2f}s")
        print(f"Overall Status: {summary['overall_status'].upper()}")
        
        if 'test_results' in results:
            print("\nDetailed Test Results:")
            print("-" * 40)
            
            for test_name, test_result in results['test_results'].items():
                status_symbol = "✅" if test_result['status'] == 'passed' else "❌"
                print(f"{status_symbol} {test_name}: {test_result['status']} ({test_result['execution_time']:.2f}s)")
                
                if 'details' in test_result:
                    for key, value in test_result['details'].items():
                        print(f"   {key}: {value}")
        
        print("\n" + "="*80)
        
        # Return exit code based on results
        return 0 if summary['overall_status'] == 'passed' else 1
    
    # Run the test suite
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

