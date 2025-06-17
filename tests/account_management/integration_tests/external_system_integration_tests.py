#!/usr/bin/env python3
"""
WS3-P4: Comprehensive Testing and Validation
External System Integration Tests

This module implements integration tests for account management with external systems,
validating seamless interaction with other components of the ALL-USE platform.

Author: Manus AI
Date: June 17, 2025
Version: 1.0
"""

import sys
import os
import json
import time
from datetime import datetime, timedelta
import uuid
import requests
from unittest.mock import MagicMock, patch

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import account management test framework
from tests.account_management.account_management_test_framework import (
    AccountManagementTestFramework, TestCategory
)

# Import account management components
from src.account_management.models.account_models import (
    Account, GenerationAccount, RevenueAccount, CompoundingAccount,
    AccountStatus, AccountType, create_account
)
from src.account_management.database.account_database import AccountDatabase
from src.account_management.api.account_operations_api import AccountOperationsAPI
from src.account_management.security.security_framework import SecurityFramework
from src.account_management.integration.integration_layer import IntegrationLayer

class ExternalSystemIntegrationTests:
    """
    Integration tests for account management with external systems.
    
    This class implements tests for:
    - Strategy Engine Integration
    - Market Integration
    - Notification System Integration
    - Reporting System Integration
    - External API Integration
    """
    
    def __init__(self):
        self.framework = AccountManagementTestFramework()
        
        # Initialize components
        self.db = AccountDatabase(":memory:")
        self.security = SecurityFramework(self.db)
        self.api = AccountOperationsAPI(self.db, self.security)
        self.integration = IntegrationLayer(self.db, self.api)
        
        # Create test user
        self.test_user_id = "test_user_123"
        self.security.create_user(self.test_user_id, "Test User", "test@example.com", "password123")
        
        # Generate auth token for API calls
        self.auth_token = self.security.generate_auth_token(self.test_user_id)
        
        # Create test account
        account_data = {
            "name": "Integration Test Account",
            "account_type": AccountType.GENERATION,
            "initial_balance": 100000.0,
            "owner_id": self.test_user_id
        }
        
        create_result = self.api.create_account(self.auth_token, account_data)
        self.test_account_id = create_result["account_id"]
    
    def test_strategy_engine_integration(self):
        """Test integration with the Strategy Engine"""
        try:
            integration_tests = []
            
            # Mock Strategy Engine components
            with patch('src.strategy_engine.core.strategy_template_engine.StrategyTemplateEngine') as mock_template_engine, \
                 patch('src.strategy_engine.core.basic_execution_engine.BasicExecutionEngine') as mock_execution_engine:
                
                # Configure mocks
                mock_template_engine.return_value.get_strategy_templates.return_value = [
                    {"id": "template1", "name": "Conservative Strategy", "risk_level": "low"},
                    {"id": "template2", "name": "Balanced Strategy", "risk_level": "medium"},
                    {"id": "template3", "name": "Aggressive Strategy", "risk_level": "high"}
                ]
                
                mock_template_engine.return_value.get_strategy_template.return_value = {
                    "id": "template1",
                    "name": "Conservative Strategy",
                    "risk_level": "low",
                    "allocation": {
                        "stocks": 30,
                        "bonds": 60,
                        "cash": 10
                    }
                }
                
                mock_execution_engine.return_value.execute_strategy.return_value = {
                    "success": True,
                    "execution_id": "exec123",
                    "status": "completed",
                    "results": {
                        "positions_opened": 5,
                        "total_invested": 50000.0
                    }
                }
                
                # Test 1: Get strategy templates
                templates = self.integration.get_strategy_templates()
                integration_tests.append(templates is not None)
                integration_tests.append(len(templates) == 3)
                
                # Test 2: Get specific strategy template
                template = self.integration.get_strategy_template("template1")
                integration_tests.append(template is not None)
                integration_tests.append(template["name"] == "Conservative Strategy")
                
                # Test 3: Apply strategy to account
                strategy_data = {
                    "template_id": "template1",
                    "account_id": self.test_account_id,
                    "allocation_amount": 50000.0
                }
                
                result = self.integration.apply_strategy_to_account(strategy_data)
                integration_tests.append(result["success"] is True)
                integration_tests.append("execution_id" in result)
                
                # Test 4: Get strategy execution status
                status = self.integration.get_strategy_execution_status(result["execution_id"])
                integration_tests.append(status is not None)
                integration_tests.append(status["status"] == "completed")
                
                # Test 5: Get account strategies
                account_strategies = self.integration.get_account_strategies(self.test_account_id)
                integration_tests.append(account_strategies is not None)
                integration_tests.append(len(account_strategies) > 0)
            
            # Calculate success
            success = all(integration_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "strategy_engine_integration", "value": sum(integration_tests), "target": len(integration_tests), "threshold": len(integration_tests), "passed": success}
                ],
                "integration_results": integration_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_market_integration(self):
        """Test integration with the Market Integration system"""
        try:
            integration_tests = []
            
            # Mock Market Integration components
            with patch('src.market_integration.optimization.trading_system_optimizer.TradingSystemOptimizer') as mock_optimizer, \
                 patch('src.market_integration.analytics.real_time_market_analytics.RealTimeMarketAnalytics') as mock_analytics:
                
                # Configure mocks
                mock_optimizer.return_value.execute_trade.return_value = {
                    "success": True,
                    "trade_id": "trade123",
                    "execution_price": 150.25,
                    "quantity": 100,
                    "total_cost": 15025.0
                }
                
                mock_optimizer.return_value.get_market_data.return_value = {
                    "symbol": "AAPL",
                    "price": 150.25,
                    "change": 2.5,
                    "volume": 1000000
                }
                
                mock_analytics.return_value.get_market_conditions.return_value = {
                    "overall": "bullish",
                    "volatility": "medium",
                    "trend": "upward",
                    "sentiment": "positive"
                }
                
                # Test 1: Execute trade
                trade_data = {
                    "account_id": self.test_account_id,
                    "symbol": "AAPL",
                    "quantity": 100,
                    "order_type": "market",
                    "side": "buy"
                }
                
                result = self.integration.execute_trade(trade_data)
                integration_tests.append(result["success"] is True)
                integration_tests.append("trade_id" in result)
                
                # Test 2: Get market data
                market_data = self.integration.get_market_data("AAPL")
                integration_tests.append(market_data is not None)
                integration_tests.append(market_data["symbol"] == "AAPL")
                
                # Test 3: Get market conditions
                conditions = self.integration.get_market_conditions()
                integration_tests.append(conditions is not None)
                integration_tests.append("overall" in conditions)
                
                # Test 4: Get account positions
                positions = self.integration.get_account_positions(self.test_account_id)
                integration_tests.append(positions is not None)
                
                # Test 5: Update account with market data
                update_result = self.integration.update_account_with_market_data(self.test_account_id)
                integration_tests.append(update_result["success"] is True)
            
            # Calculate success
            success = all(integration_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "market_integration", "value": sum(integration_tests), "target": len(integration_tests), "threshold": len(integration_tests), "passed": success}
                ],
                "integration_results": integration_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_notification_system_integration(self):
        """Test integration with the Notification System"""
        try:
            integration_tests = []
            
            # Mock Notification System
            with patch('src.notification_system.notification_manager.NotificationManager') as mock_notification:
                
                # Configure mock
                mock_notification.return_value.send_notification.return_value = {
                    "success": True,
                    "notification_id": "notif123",
                    "delivery_status": "sent"
                }
                
                mock_notification.return_value.get_notification_preferences.return_value = {
                    "email": True,
                    "sms": False,
                    "push": True,
                    "frequency": "immediate"
                }
                
                # Test 1: Send account notification
                notification_data = {
                    "account_id": self.test_account_id,
                    "type": "account_update",
                    "message": "Your account has been updated",
                    "importance": "medium"
                }
                
                result = self.integration.send_account_notification(notification_data)
                integration_tests.append(result["success"] is True)
                integration_tests.append("notification_id" in result)
                
                # Test 2: Get notification preferences
                preferences = self.integration.get_notification_preferences(self.test_user_id)
                integration_tests.append(preferences is not None)
                integration_tests.append("email" in preferences)
                
                # Test 3: Update notification preferences
                update_data = {
                    "email": True,
                    "sms": True,
                    "push": True,
                    "frequency": "daily"
                }
                
                update_result = self.integration.update_notification_preferences(self.test_user_id, update_data)
                integration_tests.append(update_result["success"] is True)
                
                # Test 4: Get notification history
                history = self.integration.get_notification_history(self.test_user_id)
                integration_tests.append(history is not None)
                
                # Test 5: Send system notification
                system_notification = {
                    "type": "system_alert",
                    "message": "System maintenance scheduled",
                    "importance": "high"
                }
                
                system_result = self.integration.send_system_notification(system_notification)
                integration_tests.append(system_result["success"] is True)
            
            # Calculate success
            success = all(integration_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "notification_system_integration", "value": sum(integration_tests), "target": len(integration_tests), "threshold": len(integration_tests), "passed": success}
                ],
                "integration_results": integration_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_reporting_system_integration(self):
        """Test integration with the Reporting System"""
        try:
            integration_tests = []
            
            # Mock Reporting System
            with patch('src.reporting_system.report_generator.ReportGenerator') as mock_reporting:
                
                # Configure mock
                mock_reporting.return_value.generate_account_report.return_value = {
                    "success": True,
                    "report_id": "report123",
                    "report_url": "https://example.com/reports/report123.pdf"
                }
                
                mock_reporting.return_value.get_report_templates.return_value = [
                    {"id": "template1", "name": "Monthly Statement"},
                    {"id": "template2", "name": "Performance Summary"},
                    {"id": "template3", "name": "Tax Report"}
                ]
                
                # Test 1: Generate account report
                report_data = {
                    "account_id": self.test_account_id,
                    "report_type": "monthly_statement",
                    "period": "2025-05",
                    "format": "pdf"
                }
                
                result = self.integration.generate_account_report(report_data)
                integration_tests.append(result["success"] is True)
                integration_tests.append("report_id" in result)
                integration_tests.append("report_url" in result)
                
                # Test 2: Get report templates
                templates = self.integration.get_report_templates()
                integration_tests.append(templates is not None)
                integration_tests.append(len(templates) == 3)
                
                # Test 3: Schedule recurring report
                schedule_data = {
                    "account_id": self.test_account_id,
                    "report_type": "performance_summary",
                    "frequency": "monthly",
                    "delivery_method": "email"
                }
                
                schedule_result = self.integration.schedule_recurring_report(schedule_data)
                integration_tests.append(schedule_result["success"] is True)
                
                # Test 4: Get report history
                history = self.integration.get_report_history(self.test_account_id)
                integration_tests.append(history is not None)
                
                # Test 5: Generate consolidated report
                consolidated_data = {
                    "user_id": self.test_user_id,
                    "report_type": "consolidated_statement",
                    "period": "2025-Q2",
                    "format": "pdf"
                }
                
                consolidated_result = self.integration.generate_consolidated_report(consolidated_data)
                integration_tests.append(consolidated_result["success"] is True)
            
            # Calculate success
            success = all(integration_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "reporting_system_integration", "value": sum(integration_tests), "target": len(integration_tests), "threshold": len(integration_tests), "passed": success}
                ],
                "integration_results": integration_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_external_api_integration(self):
        """Test integration with external APIs"""
        try:
            integration_tests = []
            
            # Mock external API requests
            with patch('requests.get') as mock_get, \
                 patch('requests.post') as mock_post:
                
                # Configure mocks
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = {
                    "status": "success",
                    "data": {
                        "symbol": "AAPL",
                        "price": 150.25,
                        "change": 2.5,
                        "volume": 1000000
                    }
                }
                
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {
                    "status": "success",
                    "data": {
                        "transaction_id": "tx123",
                        "status": "completed"
                    }
                }
                
                # Test 1: External market data API
                market_data = self.integration.get_external_market_data("AAPL")
                integration_tests.append(market_data is not None)
                integration_tests.append("symbol" in market_data)
                integration_tests.append(market_data["symbol"] == "AAPL")
                
                # Test 2: External payment processor API
                payment_data = {
                    "account_id": self.test_account_id,
                    "amount": 5000.0,
                    "payment_method": "bank_transfer",
                    "description": "Deposit"
                }
                
                payment_result = self.integration.process_external_payment(payment_data)
                integration_tests.append(payment_result["success"] is True)
                integration_tests.append("transaction_id" in payment_result)
                
                # Test 3: External authentication API
                auth_data = {
                    "user_id": self.test_user_id,
                    "external_token": "ext_token_123"
                }
                
                auth_result = self.integration.validate_external_authentication(auth_data)
                integration_tests.append(auth_result["success"] is True)
                
                # Test 4: External reporting API
                report_data = {
                    "account_id": self.test_account_id,
                    "report_type": "regulatory_report",
                    "period": "2025-Q2"
                }
                
                report_result = self.integration.generate_external_report(report_data)
                integration_tests.append(report_result["success"] is True)
                
                # Test 5: External data synchronization
                sync_result = self.integration.synchronize_external_data(self.test_account_id)
                integration_tests.append(sync_result["success"] is True)
                integration_tests.append("last_sync_time" in sync_result)
            
            # Calculate success
            success = all(integration_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "external_api_integration", "value": sum(integration_tests), "target": len(integration_tests), "threshold": len(integration_tests), "passed": success}
                ],
                "integration_results": integration_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_all_tests(self):
        """Run all external system integration tests"""
        test_funcs = {
            "strategy_engine_integration": self.test_strategy_engine_integration,
            "market_integration": self.test_market_integration,
            "notification_system_integration": self.test_notification_system_integration,
            "reporting_system_integration": self.test_reporting_system_integration,
            "external_api_integration": self.test_external_api_integration
        }
        
        results = self.framework.run_test_suite("external_integration", test_funcs, "integration")
        return results


# Run tests if executed directly
if __name__ == "__main__":
    tests = ExternalSystemIntegrationTests()
    results = tests.run_all_tests()
    
    # Generate and print report
    framework = tests.framework
    report = framework.generate_test_report("text")
    print(report["text_report"])
    
    # Export results
    framework.export_test_results("external_system_integration_test_results.json")
    
    # Clean up
    framework.cleanup()

