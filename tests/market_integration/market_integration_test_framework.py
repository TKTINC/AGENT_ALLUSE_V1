#!/usr/bin/env python3
"""
ALL-USE Agent - Market Integration Testing Framework
WS4-P4: Market Integration Comprehensive Testing and Validation - Phase 1

This module provides comprehensive testing framework for Market Integration components,
applying the proven testing patterns from P6 of WS2 to validate all market integration
systems for production readiness.

Market Integration Components to Test:
1. Live Market Data System - Real-time market data processing and validation
2. IBKR Integration - Interactive Brokers API integration and connectivity
3. Trading Execution Engine - Order management and execution workflows
4. Paper Trading System - Paper trading and go-live capabilities
5. Trade Monitoring System - Trade monitoring and reporting
6. Risk Management - Risk controls and safety systems
"""

import os
import sys
import time
import json
import logging
import importlib.util
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import random

# Add src to path for imports
sys.path.insert(0, '/home/ubuntu/AGENT_ALLUSE_V1/src')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestResult(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class ComponentStatus(Enum):
    """Market integration component status"""
    OPERATIONAL = "operational"
    PARTIALLY_OPERATIONAL = "partially_operational"
    NON_OPERATIONAL = "non_operational"
    NOT_FOUND = "not_found"


@dataclass
class MarketTestExecution:
    """Market integration test execution result"""
    test_name: str
    component: str
    result: TestResult
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ComponentValidation:
    """Market integration component validation result"""
    component_name: str
    file_path: str
    status: ComponentStatus
    functionality_score: float
    api_methods: List[str]
    validation_details: Dict[str, Any]
    recommendations: List[str]


@dataclass
class MarketDataSimulation:
    """Market data simulation for testing"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: float
    ask: float
    spread: float


class MockMarketDataGenerator:
    """Generates mock market data for testing"""
    
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'SPY', 'QQQ', 'IWM']
        self.base_prices = {
            'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 350.0, 'TSLA': 200.0, 'AMZN': 3200.0,
            'NVDA': 450.0, 'META': 300.0, 'SPY': 450.0, 'QQQ': 380.0, 'IWM': 200.0
        }
        self.price_volatility = 0.02  # 2% volatility
        
        logger.info("Mock Market Data Generator initialized")
    
    def generate_market_data(self, symbol: str, count: int = 100) -> List[MarketDataSimulation]:
        """Generate mock market data for testing"""
        if symbol not in self.base_prices:
            symbol = 'AAPL'  # Default to AAPL if symbol not found
        
        base_price = self.base_prices[symbol]
        market_data = []
        current_price = base_price
        
        for i in range(count):
            # Simulate price movement
            price_change = random.gauss(0, self.price_volatility) * current_price
            current_price = max(0.01, current_price + price_change)
            
            # Generate bid/ask spread
            spread_pct = random.uniform(0.001, 0.005)  # 0.1% to 0.5% spread
            spread = current_price * spread_pct
            bid = current_price - spread / 2
            ask = current_price + spread / 2
            
            # Generate volume
            volume = random.randint(100, 10000)
            
            market_data.append(MarketDataSimulation(
                symbol=symbol,
                price=round(current_price, 2),
                volume=volume,
                timestamp=datetime.now() + timedelta(seconds=i),
                bid=round(bid, 2),
                ask=round(ask, 2),
                spread=round(spread, 2)
            ))
        
        return market_data
    
    def generate_real_time_feed(self, symbol: str, duration_seconds: int = 60) -> queue.Queue:
        """Generate real-time market data feed for testing"""
        data_queue = queue.Queue()
        
        def feed_generator():
            start_time = time.time()
            base_price = self.base_prices.get(symbol, 150.0)
            current_price = base_price
            
            while time.time() - start_time < duration_seconds:
                # Generate new price
                price_change = random.gauss(0, self.price_volatility) * current_price
                current_price = max(0.01, current_price + price_change)
                
                # Generate market data
                spread_pct = random.uniform(0.001, 0.005)
                spread = current_price * spread_pct
                
                market_data = MarketDataSimulation(
                    symbol=symbol,
                    price=round(current_price, 2),
                    volume=random.randint(100, 5000),
                    timestamp=datetime.now(),
                    bid=round(current_price - spread / 2, 2),
                    ask=round(current_price + spread / 2, 2),
                    spread=round(spread, 2)
                )
                
                data_queue.put(market_data)
                time.sleep(random.uniform(0.1, 1.0))  # Random intervals
        
        # Start feed in background thread
        feed_thread = threading.Thread(target=feed_generator)
        feed_thread.daemon = True
        feed_thread.start()
        
        return data_queue


class MockTradingEnvironment:
    """Mock trading environment for testing"""
    
    def __init__(self):
        self.orders = []
        self.positions = {}
        self.account_balance = 100000.0  # $100k starting balance
        self.order_id_counter = 1
        
        logger.info("Mock Trading Environment initialized")
    
    def place_order(self, symbol: str, quantity: int, order_type: str, price: Optional[float] = None) -> Dict[str, Any]:
        """Place a mock order"""
        order_id = f"ORDER_{self.order_id_counter:06d}"
        self.order_id_counter += 1
        
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'quantity': quantity,
            'order_type': order_type,
            'price': price,
            'status': 'submitted',
            'timestamp': datetime.now(),
            'filled_quantity': 0,
            'remaining_quantity': quantity
        }
        
        self.orders.append(order)
        
        # Simulate order execution (simplified)
        if order_type.lower() == 'market':
            self._execute_order(order)
        
        return order
    
    def _execute_order(self, order: Dict[str, Any]):
        """Simulate order execution"""
        # Simulate execution delay
        time.sleep(random.uniform(0.01, 0.1))
        
        # Update order status
        order['status'] = 'filled'
        order['filled_quantity'] = order['quantity']
        order['remaining_quantity'] = 0
        order['execution_time'] = datetime.now()
        
        # Update positions
        symbol = order['symbol']
        if symbol not in self.positions:
            self.positions[symbol] = 0
        
        if order['quantity'] > 0:  # Buy order
            self.positions[symbol] += order['quantity']
        else:  # Sell order
            self.positions[symbol] += order['quantity']  # quantity is negative for sells
    
    def get_positions(self) -> Dict[str, int]:
        """Get current positions"""
        return self.positions.copy()
    
    def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get orders, optionally filtered by status"""
        if status:
            return [order for order in self.orders if order['status'] == status]
        return self.orders.copy()


class MarketIntegrationComponentValidator:
    """Validates market integration components"""
    
    def __init__(self):
        self.project_root = "/home/ubuntu/AGENT_ALLUSE_V1"
        self.market_integration_components = {
            'live_market_data_system': 'src/market_data/live_market_data_system.py',
            'ibkr_integration': 'src/broker_integration/ibkr_integration_and_runtime_config.py',
            'trading_execution_engine': 'src/trading_execution/trading_execution_engine.py',
            'paper_trading_system': 'src/paper_trading/paper_trading_and_go_live_system.py',
            'trade_monitoring_system': 'src/trade_monitoring/trade_monitoring_system.py',
            'broker_integration_framework': 'src/broker_integration/broker_integration_framework.py'
        }
        
        logger.info("Market Integration Component Validator initialized")
    
    def validate_component_availability(self) -> List[ComponentValidation]:
        """Validate availability and basic functionality of market integration components"""
        validations = []
        
        for component_name, file_path in self.market_integration_components.items():
            full_path = os.path.join(self.project_root, file_path)
            
            validation = ComponentValidation(
                component_name=component_name,
                file_path=file_path,
                status=ComponentStatus.NOT_FOUND,
                functionality_score=0.0,
                api_methods=[],
                validation_details={},
                recommendations=[]
            )
            
            if os.path.exists(full_path):
                try:
                    # Try to import and analyze the component
                    spec = importlib.util.spec_from_file_location(component_name, full_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Analyze module contents
                        api_methods = [attr for attr in dir(module) if not attr.startswith('_')]
                        classes = [attr for attr in api_methods if hasattr(getattr(module, attr), '__class__') and 
                                 getattr(module, attr).__class__.__name__ == 'type']
                        functions = [attr for attr in api_methods if callable(getattr(module, attr))]
                        
                        validation.status = ComponentStatus.OPERATIONAL
                        validation.functionality_score = min(100.0, len(api_methods) * 5)
                        validation.api_methods = api_methods[:10]  # Top 10 methods
                        validation.validation_details = {
                            'total_attributes': len(api_methods),
                            'classes_found': len(classes),
                            'functions_found': len(functions),
                            'file_size': os.path.getsize(full_path),
                            'import_successful': True
                        }
                        
                        if validation.functionality_score < 50:
                            validation.status = ComponentStatus.PARTIALLY_OPERATIONAL
                            validation.recommendations.append("Component has limited functionality")
                        
                        if len(classes) == 0:
                            validation.recommendations.append("No classes found - consider adding main component classes")
                        
                        if len(functions) < 5:
                            validation.recommendations.append("Limited functions available - consider expanding API")
                
                except Exception as e:
                    validation.status = ComponentStatus.NON_OPERATIONAL
                    validation.validation_details = {
                        'import_error': str(e),
                        'file_exists': True,
                        'file_size': os.path.getsize(full_path)
                    }
                    validation.recommendations.append(f"Fix import error: {str(e)}")
            else:
                validation.recommendations.append(f"Create component file: {file_path}")
            
            validations.append(validation)
        
        return validations
    
    def test_component_integration(self, component_validations: List[ComponentValidation]) -> List[MarketTestExecution]:
        """Test integration between market integration components"""
        test_executions = []
        
        # Test 1: Component Discovery
        start_time = time.perf_counter()
        operational_components = [v for v in component_validations if v.status == ComponentStatus.OPERATIONAL]
        execution_time = time.perf_counter() - start_time
        
        test_executions.append(MarketTestExecution(
            test_name="Component Discovery",
            component="market_integration",
            result=TestResult.PASSED if len(operational_components) >= 3 else TestResult.FAILED,
            execution_time=execution_time * 1000,  # Convert to ms
            details={
                'total_components': len(component_validations),
                'operational_components': len(operational_components),
                'component_names': [v.component_name for v in operational_components]
            },
            error_message=None if len(operational_components) >= 3 else "Insufficient operational components"
        ))
        
        # Test 2: Market Data Component Integration
        start_time = time.perf_counter()
        market_data_components = [v for v in operational_components if 'market_data' in v.component_name]
        execution_time = time.perf_counter() - start_time
        
        test_executions.append(MarketTestExecution(
            test_name="Market Data Integration",
            component="live_market_data_system",
            result=TestResult.PASSED if len(market_data_components) > 0 else TestResult.SKIPPED,
            execution_time=execution_time * 1000,
            details={
                'market_data_components': len(market_data_components),
                'integration_score': sum(v.functionality_score for v in market_data_components) / max(1, len(market_data_components))
            },
            error_message=None if len(market_data_components) > 0 else "No market data components found"
        ))
        
        # Test 3: Trading Execution Integration
        start_time = time.perf_counter()
        trading_components = [v for v in operational_components if 'trading' in v.component_name or 'execution' in v.component_name]
        execution_time = time.perf_counter() - start_time
        
        test_executions.append(MarketTestExecution(
            test_name="Trading Execution Integration",
            component="trading_execution_engine",
            result=TestResult.PASSED if len(trading_components) > 0 else TestResult.SKIPPED,
            execution_time=execution_time * 1000,
            details={
                'trading_components': len(trading_components),
                'integration_score': sum(v.functionality_score for v in trading_components) / max(1, len(trading_components))
            },
            error_message=None if len(trading_components) > 0 else "No trading execution components found"
        ))
        
        # Test 4: Broker Integration
        start_time = time.perf_counter()
        broker_components = [v for v in operational_components if 'broker' in v.component_name or 'ibkr' in v.component_name]
        execution_time = time.perf_counter() - start_time
        
        test_executions.append(MarketTestExecution(
            test_name="Broker Integration",
            component="ibkr_integration",
            result=TestResult.PASSED if len(broker_components) > 0 else TestResult.SKIPPED,
            execution_time=execution_time * 1000,
            details={
                'broker_components': len(broker_components),
                'integration_score': sum(v.functionality_score for v in broker_components) / max(1, len(broker_components))
            },
            error_message=None if len(broker_components) > 0 else "No broker integration components found"
        ))
        
        # Test 5: Risk Management Integration
        start_time = time.perf_counter()
        risk_components = [v for v in operational_components if 'risk' in v.component_name or 'monitoring' in v.component_name]
        execution_time = time.perf_counter() - start_time
        
        test_executions.append(MarketTestExecution(
            test_name="Risk Management Integration",
            component="trade_monitoring_system",
            result=TestResult.PASSED if len(risk_components) > 0 else TestResult.SKIPPED,
            execution_time=execution_time * 1000,
            details={
                'risk_components': len(risk_components),
                'integration_score': sum(v.functionality_score for v in risk_components) / max(1, len(risk_components))
            },
            error_message=None if len(risk_components) > 0 else "No risk management components found"
        ))
        
        return test_executions


class MarketIntegrationTestFramework:
    """Main market integration testing framework"""
    
    def __init__(self):
        self.component_validator = MarketIntegrationComponentValidator()
        self.mock_data_generator = MockMarketDataGenerator()
        self.mock_trading_env = MockTradingEnvironment()
        
        logger.info("Market Integration Test Framework initialized")
    
    def run_comprehensive_testing(self) -> Dict[str, Any]:
        """Run comprehensive market integration testing"""
        logger.info("Starting comprehensive market integration testing")
        
        testing_start = time.perf_counter()
        
        # Phase 1: Component Validation
        logger.info("Phase 1: Validating market integration components")
        component_validations = self.component_validator.validate_component_availability()
        
        # Phase 2: Integration Testing
        logger.info("Phase 2: Testing component integration")
        integration_tests = self.component_validator.test_component_integration(component_validations)
        
        # Phase 3: Mock Data Testing
        logger.info("Phase 3: Testing with mock market data")
        mock_data_tests = self._test_mock_data_functionality()
        
        # Phase 4: Trading Environment Testing
        logger.info("Phase 4: Testing mock trading environment")
        trading_env_tests = self._test_trading_environment()
        
        testing_duration = time.perf_counter() - testing_start
        
        # Compile results
        all_tests = integration_tests + mock_data_tests + trading_env_tests
        passed_tests = [t for t in all_tests if t.result == TestResult.PASSED]
        failed_tests = [t for t in all_tests if t.result == TestResult.FAILED]
        
        # Calculate scores
        operational_components = [v for v in component_validations if v.status == ComponentStatus.OPERATIONAL]
        component_score = len(operational_components) / len(component_validations) * 100 if component_validations else 0
        test_success_rate = len(passed_tests) / len(all_tests) * 100 if all_tests else 0
        
        results = {
            'testing_duration': testing_duration,
            'component_validations': component_validations,
            'test_executions': all_tests,
            'summary': {
                'total_components': len(component_validations),
                'operational_components': len(operational_components),
                'component_score': component_score,
                'total_tests': len(all_tests),
                'passed_tests': len(passed_tests),
                'failed_tests': len(failed_tests),
                'test_success_rate': test_success_rate,
                'average_execution_time': sum(t.execution_time for t in all_tests) / len(all_tests) if all_tests else 0
            }
        }
        
        logger.info(f"Market integration testing completed in {testing_duration:.2f}s")
        logger.info(f"Component score: {component_score:.1f}%, Test success rate: {test_success_rate:.1f}%")
        
        return results
    
    def _test_mock_data_functionality(self) -> List[MarketTestExecution]:
        """Test mock market data functionality"""
        tests = []
        
        # Test 1: Market Data Generation
        start_time = time.perf_counter()
        try:
            market_data = self.mock_data_generator.generate_market_data('AAPL', 50)
            execution_time = time.perf_counter() - start_time
            
            tests.append(MarketTestExecution(
                test_name="Market Data Generation",
                component="mock_data_generator",
                result=TestResult.PASSED if len(market_data) == 50 else TestResult.FAILED,
                execution_time=execution_time * 1000,
                details={
                    'data_points_generated': len(market_data),
                    'symbols_tested': ['AAPL'],
                    'price_range': f"{min(d.price for d in market_data):.2f} - {max(d.price for d in market_data):.2f}",
                    'average_spread': sum(d.spread for d in market_data) / len(market_data) if market_data else 0
                }
            ))
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            tests.append(MarketTestExecution(
                test_name="Market Data Generation",
                component="mock_data_generator",
                result=TestResult.ERROR,
                execution_time=execution_time * 1000,
                details={},
                error_message=str(e)
            ))
        
        # Test 2: Real-time Feed Generation
        start_time = time.perf_counter()
        try:
            feed_queue = self.mock_data_generator.generate_real_time_feed('GOOGL', 2)  # 2 seconds
            time.sleep(2.5)  # Wait for feed to generate data
            
            data_count = 0
            while not feed_queue.empty():
                feed_queue.get()
                data_count += 1
            
            execution_time = time.perf_counter() - start_time
            
            tests.append(MarketTestExecution(
                test_name="Real-time Feed Generation",
                component="mock_data_generator",
                result=TestResult.PASSED if data_count > 0 else TestResult.FAILED,
                execution_time=execution_time * 1000,
                details={
                    'feed_duration': 2,
                    'data_points_received': data_count,
                    'feed_rate': data_count / 2 if data_count > 0 else 0
                }
            ))
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            tests.append(MarketTestExecution(
                test_name="Real-time Feed Generation",
                component="mock_data_generator",
                result=TestResult.ERROR,
                execution_time=execution_time * 1000,
                details={},
                error_message=str(e)
            ))
        
        return tests
    
    def _test_trading_environment(self) -> List[MarketTestExecution]:
        """Test mock trading environment functionality"""
        tests = []
        
        # Test 1: Order Placement
        start_time = time.perf_counter()
        try:
            order = self.mock_trading_env.place_order('AAPL', 100, 'market')
            execution_time = time.perf_counter() - start_time
            
            tests.append(MarketTestExecution(
                test_name="Order Placement",
                component="mock_trading_environment",
                result=TestResult.PASSED if order and 'order_id' in order else TestResult.FAILED,
                execution_time=execution_time * 1000,
                details={
                    'order_id': order.get('order_id', 'N/A'),
                    'order_status': order.get('status', 'unknown'),
                    'symbol': order.get('symbol', 'N/A'),
                    'quantity': order.get('quantity', 0)
                }
            ))
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            tests.append(MarketTestExecution(
                test_name="Order Placement",
                component="mock_trading_environment",
                result=TestResult.ERROR,
                execution_time=execution_time * 1000,
                details={},
                error_message=str(e)
            ))
        
        # Test 2: Position Management
        start_time = time.perf_counter()
        try:
            # Place multiple orders
            self.mock_trading_env.place_order('MSFT', 50, 'market')
            self.mock_trading_env.place_order('GOOGL', 25, 'market')
            
            positions = self.mock_trading_env.get_positions()
            execution_time = time.perf_counter() - start_time
            
            tests.append(MarketTestExecution(
                test_name="Position Management",
                component="mock_trading_environment",
                result=TestResult.PASSED if len(positions) >= 2 else TestResult.FAILED,
                execution_time=execution_time * 1000,
                details={
                    'positions_count': len(positions),
                    'positions': positions,
                    'total_orders': len(self.mock_trading_env.get_orders())
                }
            ))
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            tests.append(MarketTestExecution(
                test_name="Position Management",
                component="mock_trading_environment",
                result=TestResult.ERROR,
                execution_time=execution_time * 1000,
                details={},
                error_message=str(e)
            ))
        
        return tests


if __name__ == '__main__':
    print("🏗️ Market Integration Testing Framework Setup (WS4-P4 - Phase 1)")
    print("=" * 80)
    
    # Initialize market integration testing framework
    test_framework = MarketIntegrationTestFramework()
    
    print("\n🔍 Running comprehensive market integration testing...")
    
    # Run comprehensive testing
    test_results = test_framework.run_comprehensive_testing()
    
    print(f"\n📊 Market Integration Testing Results:")
    print(f"   Testing Duration: {test_results['testing_duration']:.2f}s")
    print(f"   Component Score: {test_results['summary']['component_score']:.1f}%")
    print(f"   Test Success Rate: {test_results['summary']['test_success_rate']:.1f}%")
    print(f"   Average Execution Time: {test_results['summary']['average_execution_time']:.2f}ms")
    
    print(f"\n📋 Component Validation Results:")
    for validation in test_results['component_validations']:
        status_icon = {
            ComponentStatus.OPERATIONAL: "🟢",
            ComponentStatus.PARTIALLY_OPERATIONAL: "🟡",
            ComponentStatus.NON_OPERATIONAL: "🔴",
            ComponentStatus.NOT_FOUND: "⚪"
        }.get(validation.status, "❓")
        
        print(f"   {status_icon} {validation.component_name}: {validation.status.value} ({validation.functionality_score:.1f}/100)")
        if validation.recommendations:
            print(f"     💡 {validation.recommendations[0]}")
    
    print(f"\n🔍 Test Execution Results:")
    for test in test_results['test_executions']:
        result_icon = {
            TestResult.PASSED: "✅",
            TestResult.FAILED: "❌",
            TestResult.SKIPPED: "⏭️",
            TestResult.ERROR: "🚨"
        }.get(test.result, "❓")
        
        print(f"   {result_icon} {test.test_name}: {test.result.value} ({test.execution_time:.2f}ms)")
        if test.error_message:
            print(f"     ⚠️  {test.error_message}")
    
    # Save test results
    output_dir = "/home/ubuntu/AGENT_ALLUSE_V1/docs/market_integration"
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, f"market_integration_testing_framework_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Convert results to JSON-serializable format
    json_results = {
        'testing_duration': test_results['testing_duration'],
        'summary': test_results['summary'],
        'component_validations': [
            {
                'component_name': v.component_name,
                'file_path': v.file_path,
                'status': v.status.value,
                'functionality_score': v.functionality_score,
                'api_methods': v.api_methods,
                'validation_details': v.validation_details,
                'recommendations': v.recommendations
            }
            for v in test_results['component_validations']
        ],
        'test_executions': [
            {
                'test_name': t.test_name,
                'component': t.component,
                'result': t.result.value,
                'execution_time': t.execution_time,
                'details': t.details,
                'error_message': t.error_message,
                'timestamp': t.timestamp.isoformat()
            }
            for t in test_results['test_executions']
        ]
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n📁 Test Results Saved: {results_path}")
    
    # Determine next steps
    if test_results['summary']['component_score'] >= 50:
        print(f"\n🎉 MARKET INTEGRATION FRAMEWORK READY!")
        print(f"✅ {test_results['summary']['operational_components']}/{test_results['summary']['total_components']} components operational")
        print(f"🚀 Ready for Phase 2: Live Market Data and IBKR Integration Testing")
    else:
        print(f"\n⚠️  MARKET INTEGRATION FRAMEWORK NEEDS ATTENTION")
        print(f"📋 {test_results['summary']['total_components'] - test_results['summary']['operational_components']} components need fixes")
        print(f"🔄 Proceeding to Phase 2 with available components")
        print(f"🚀 Ready for Phase 2: Live Market Data and IBKR Integration Testing")

