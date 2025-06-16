"""
ALL-USE Testing Utilities

This module provides common testing utilities, mock data generators,
and helper functions for comprehensive testing across all components.
"""

import random
import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock
import numpy as np


class MockDataGenerator:
    """Generate realistic mock data for testing purposes."""
    
    @staticmethod
    def generate_market_data(symbol: str = "SPY", days: int = 30) -> Dict[str, Any]:
        """Generate realistic market data for testing."""
        base_price = 400.0
        dates = []
        prices = []
        volumes = []
        
        current_date = datetime.datetime.now() - datetime.timedelta(days=days)
        current_price = base_price
        
        for i in range(days):
            # Generate realistic price movement
            daily_return = random.gauss(0.001, 0.02)  # 0.1% mean, 2% volatility
            current_price *= (1 + daily_return)
            
            dates.append(current_date.strftime('%Y-%m-%d'))
            prices.append(round(current_price, 2))
            volumes.append(random.randint(50000000, 200000000))
            
            current_date += datetime.timedelta(days=1)
        
        return {
            'symbol': symbol,
            'dates': dates,
            'prices': prices,
            'volumes': volumes,
            'current_price': current_price,
            'daily_returns': [0.0] + [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
        }
    
    @staticmethod
    def generate_portfolio_data(num_positions: int = 5) -> Dict[str, Any]:
        """Generate realistic portfolio data for testing."""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
        selected_symbols = random.sample(symbols, min(num_positions, len(symbols)))
        
        positions = {}
        total_value = 0
        
        for symbol in selected_symbols:
            quantity = random.randint(10, 100)
            price = random.uniform(100, 500)
            value = quantity * price
            
            positions[symbol] = {
                'quantity': quantity,
                'price': price,
                'value': value,
                'sector': MockDataGenerator._get_sector(symbol),
                'volatility': random.uniform(0.15, 0.45),
                'beta': random.uniform(0.8, 1.5)
            }
            total_value += value
        
        return {
            'positions': positions,
            'total_value': total_value,
            'cash': random.uniform(1000, 10000),
            'account_type': random.choice(['GEN_ACC', 'REV_ACC', 'COM_ACC'])
        }
    
    @staticmethod
    def generate_conversation_history(length: int = 10) -> List[Dict[str, Any]]:
        """Generate realistic conversation history for testing."""
        conversation = []
        intents = ['greeting', 'account_inquiry', 'setup_accounts', 'market_analysis', 'risk_assessment']
        
        for i in range(length):
            message = {
                'timestamp': datetime.datetime.now() - datetime.timedelta(minutes=length-i),
                'user_message': f"Test user message {i+1}",
                'agent_response': f"Test agent response {i+1}",
                'intent': random.choice(intents),
                'entities': {'amount': random.randint(1000, 100000)} if random.random() > 0.5 else {},
                'context_score': random.uniform(0.7, 1.0)
            }
            conversation.append(message)
        
        return conversation
    
    @staticmethod
    def generate_performance_data(days: int = 30) -> Dict[str, Any]:
        """Generate realistic performance data for testing."""
        returns = []
        portfolio_values = []
        base_value = 100000
        
        for i in range(days):
            daily_return = random.gauss(0.0008, 0.015)  # Slightly positive mean return
            returns.append(daily_return)
            
            if i == 0:
                portfolio_values.append(base_value)
            else:
                new_value = portfolio_values[-1] * (1 + daily_return)
                portfolio_values.append(new_value)
        
        return {
            'returns': returns,
            'portfolio_values': portfolio_values,
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_drawdown': MockDataGenerator._calculate_max_drawdown(portfolio_values),
            'total_return': (portfolio_values[-1] / portfolio_values[0] - 1) * 100,
            'volatility': np.std(returns) * np.sqrt(252) * 100
        }
    
    @staticmethod
    def _get_sector(symbol: str) -> str:
        """Get sector for a symbol."""
        sector_map = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'TSLA': 'Automotive',
            'NVDA': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'META': 'Technology',
            'NFLX': 'Entertainment'
        }
        return sector_map.get(symbol, 'Technology')
    
    @staticmethod
    def _calculate_max_drawdown(values: List[float]) -> float:
        """Calculate maximum drawdown from a series of values."""
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd * 100


class MockServices:
    """Mock external services for testing."""
    
    @staticmethod
    def create_mock_market_data_service():
        """Create a mock market data service."""
        mock_service = Mock()
        mock_service.get_current_price.return_value = 450.25
        mock_service.get_historical_data.return_value = MockDataGenerator.generate_market_data()
        mock_service.get_option_chain.return_value = {
            'calls': [{'strike': 450, 'delta': 0.5, 'premium': 15.25}],
            'puts': [{'strike': 450, 'delta': -0.5, 'premium': 14.75}]
        }
        return mock_service
    
    @staticmethod
    def create_mock_brokerage_service():
        """Create a mock brokerage service."""
        mock_service = Mock()
        mock_service.place_order.return_value = {'order_id': 'TEST123', 'status': 'filled'}
        mock_service.get_positions.return_value = MockDataGenerator.generate_portfolio_data()
        mock_service.get_account_balance.return_value = 100000.0
        return mock_service
    
    @staticmethod
    def create_mock_database():
        """Create a mock database for testing."""
        mock_db = Mock()
        mock_db.save.return_value = True
        mock_db.load.return_value = {}
        mock_db.query.return_value = []
        return mock_db


class TestAssertions:
    """Custom assertion helpers for ALL-USE testing."""
    
    @staticmethod
    def assert_response_time(func, max_time_ms: float = 100):
        """Assert that a function executes within the specified time."""
        import time
        start_time = time.time()
        result = func()
        execution_time = (time.time() - start_time) * 1000
        
        assert execution_time <= max_time_ms, f"Function took {execution_time:.2f}ms, expected <={max_time_ms}ms"
        return result
    
    @staticmethod
    def assert_memory_usage(func, max_memory_mb: float = 100):
        """Assert that a function uses memory within the specified limit."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func()
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        assert memory_used <= max_memory_mb, f"Function used {memory_used:.2f}MB, expected <={max_memory_mb}MB"
        return result
    
    @staticmethod
    def assert_valid_portfolio_metrics(metrics: Dict[str, Any]):
        """Assert that portfolio metrics are valid."""
        required_keys = ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
        for key in required_keys:
            assert key in metrics, f"Missing required metric: {key}"
        
        # Validate ranges
        assert -100 <= metrics['total_return'] <= 1000, f"Invalid total return: {metrics['total_return']}"
        assert -5 <= metrics['sharpe_ratio'] <= 10, f"Invalid Sharpe ratio: {metrics['sharpe_ratio']}"
        assert 0 <= metrics['max_drawdown'] <= 100, f"Invalid max drawdown: {metrics['max_drawdown']}"
        assert 0 <= metrics['volatility'] <= 200, f"Invalid volatility: {metrics['volatility']}"
    
    @staticmethod
    def assert_valid_risk_metrics(risk_data: Dict[str, Any]):
        """Assert that risk metrics are valid."""
        required_keys = ['var_95', 'var_99', 'cvar_95', 'risk_score']
        for key in required_keys:
            assert key in risk_data, f"Missing required risk metric: {key}"
        
        # Validate ranges
        assert 0 <= risk_data['var_95'] <= 100, f"Invalid VaR 95%: {risk_data['var_95']}"
        assert 0 <= risk_data['var_99'] <= 100, f"Invalid VaR 99%: {risk_data['var_99']}"
        assert 0 <= risk_data['cvar_95'] <= 100, f"Invalid CVaR 95%: {risk_data['cvar_95']}"
        assert 0 <= risk_data['risk_score'] <= 100, f"Invalid risk score: {risk_data['risk_score']}"


class TestFixtures:
    """Common test fixtures and setup utilities."""
    
    @staticmethod
    def setup_test_environment():
        """Set up a clean test environment."""
        # Clear any existing state
        # Set up logging for tests
        import logging
        logging.basicConfig(level=logging.WARNING)  # Reduce noise in tests
        
        return {
            'market_data': MockDataGenerator.generate_market_data(),
            'portfolio_data': MockDataGenerator.generate_portfolio_data(),
            'conversation_history': MockDataGenerator.generate_conversation_history(),
            'performance_data': MockDataGenerator.generate_performance_data()
        }
    
    @staticmethod
    def create_test_agent():
        """Create a test agent instance with mocked dependencies."""
        from src.agent_core.enhanced_agent import EnhancedALLUSEAgent
        
        # Create agent with test configuration
        agent = EnhancedALLUSEAgent()
        
        return agent
    
    @staticmethod
    def create_test_portfolio():
        """Create a test portfolio with known characteristics."""
        portfolio_data = MockDataGenerator.generate_portfolio_data(5)
        
        # Ensure consistent test data
        portfolio_data['total_value'] = 100000.0
        portfolio_data['cash'] = 5000.0
        portfolio_data['account_type'] = 'GEN_ACC'
        
        return portfolio_data


# Configuration for pytest
def pytest_configure():
    """Configure pytest with custom markers and settings."""
    import pytest
    
    # Add custom markers
    pytest.mark.unit = pytest.mark.unit
    pytest.mark.integration = pytest.mark.integration
    pytest.mark.performance = pytest.mark.performance
    pytest.mark.slow = pytest.mark.slow


if __name__ == "__main__":
    # Test the mock data generators
    print("Testing Mock Data Generators...")
    
    market_data = MockDataGenerator.generate_market_data()
    print(f"Generated market data for {market_data['symbol']}: {len(market_data['prices'])} days")
    
    portfolio_data = MockDataGenerator.generate_portfolio_data()
    print(f"Generated portfolio with {len(portfolio_data['positions'])} positions, total value: ${portfolio_data['total_value']:,.2f}")
    
    performance_data = MockDataGenerator.generate_performance_data()
    print(f"Generated performance data: {performance_data['total_return']:.2f}% return, {performance_data['sharpe_ratio']:.3f} Sharpe")
    
    print("Mock data generators working correctly!")

