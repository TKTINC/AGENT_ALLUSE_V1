#!/usr/bin/env python3
"""
WS3-P4: Comprehensive Testing and Validation
Component Integration Tests

This module implements integration tests for account management components,
validating seamless interaction between different modules.

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
from src.account_management.analytics.account_analytics_engine import AccountAnalyticsEngine

class ComponentIntegrationTests:
    """
    Integration tests for account management components.
    
    This class implements tests for:
    - Data Model-Database Integration
    - API-Business Logic Integration
    - Security-API Integration
    - Configuration-Operation Integration
    - Analytics-Database Integration
    """
    
    def __init__(self):
        self.framework = AccountManagementTestFramework()
        
        # Initialize components
        self.db = AccountDatabase(":memory:")
        self.security = SecurityFramework(self.db)
        self.api = AccountOperationsAPI(self.db, self.security)
        self.analytics = AccountAnalyticsEngine(self.db)
        
        # Create test user
        self.test_user_id = "test_user_123"
        self.security.create_user(self.test_user_id, "Test User", "test@example.com", "password123")
        
        # Generate auth token for API calls
        self.auth_token = self.security.generate_auth_token(self.test_user_id)
    
    def test_data_model_database(self):
        """Test integration between data models and database layer"""
        try:
            integration_tests = []
            
            # Test 1: Create account model and save to database
            gen_account = create_account(
                account_type=AccountType.GENERATION,
                name="Integration Test Account",
                initial_balance=100000.0,
                owner_id=self.test_user_id
            )
            
            self.db.save_account(gen_account)
            
            # Test 2: Retrieve account from database and verify model integrity
            retrieved_account = self.db.get_account_by_id(gen_account.account_id)
            
            integration_tests.append(retrieved_account is not None)
            integration_tests.append(isinstance(retrieved_account, GenerationAccount))
            integration_tests.append(retrieved_account.account_id == gen_account.account_id)
            integration_tests.append(retrieved_account.name == gen_account.name)
            integration_tests.append(retrieved_account.initial_balance == gen_account.initial_balance)
            integration_tests.append(retrieved_account.target_delta_range == gen_account.target_delta_range)
            
            # Test 3: Update model and persist changes
            gen_account.name = "Updated Integration Account"
            gen_account.current_balance = 110000.0
            
            self.db.update_account(gen_account)
            
            updated_account = self.db.get_account_by_id(gen_account.account_id)
            integration_tests.append(updated_account.name == "Updated Integration Account")
            integration_tests.append(updated_account.current_balance == 110000.0)
            
            # Test 4: Test account relationships
            child_account = create_account(
                account_type=AccountType.COMPOUNDING,
                name="Child Integration Account",
                initial_balance=50000.0,
                owner_id=self.test_user_id,
                parent_account_id=gen_account.account_id
            )
            
            self.db.save_account(child_account)
            
            # Verify relationship persistence
            child_from_db = self.db.get_account_by_id(child_account.account_id)
            integration_tests.append(child_from_db.parent_account_id == gen_account.account_id)
            
            # Test 5: Test account hierarchy retrieval
            children = self.db.get_child_accounts(gen_account.account_id)
            integration_tests.append(len(children) == 1)
            integration_tests.append(children[0].account_id == child_account.account_id)
            
            # Calculate success
            success = all(integration_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "data_model_database_integration", "value": sum(integration_tests), "target": len(integration_tests), "threshold": len(integration_tests), "passed": success}
                ],
                "integration_results": integration_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_api_business_logic(self):
        """Test integration between API layer and business logic"""
        try:
            integration_tests = []
            
            # Test 1: Create account through API
            account_data = {
                "name": "API Integration Account",
                "account_type": AccountType.GENERATION,
                "initial_balance": 100000.0,
                "owner_id": self.test_user_id
            }
            
            result = self.api.create_account(self.auth_token, account_data)
            integration_tests.append(result["success"] is True)
            
            account_id = result["account_id"]
            
            # Test 2: Retrieve account through API
            get_result = self.api.get_account(self.auth_token, account_id)
            integration_tests.append(get_result["success"] is True)
            integration_tests.append(get_result["account"]["name"] == account_data["name"])
            
            # Test 3: Update account through API
            update_data = {
                "name": "Updated API Account",
                "current_balance": 110000.0
            }
            
            update_result = self.api.update_account(self.auth_token, account_id, update_data)
            integration_tests.append(update_result["success"] is True)
            
            # Verify update through direct database access
            updated_account = self.db.get_account_by_id(account_id)
            integration_tests.append(updated_account.name == update_data["name"])
            integration_tests.append(updated_account.current_balance == update_data["current_balance"])
            
            # Test 4: Test business logic validation through API
            # Try to create account with negative balance
            invalid_data = {
                "name": "Invalid API Account",
                "account_type": AccountType.REVENUE,
                "initial_balance": -10000.0,
                "owner_id": self.test_user_id
            }
            
            invalid_result = self.api.create_account(self.auth_token, invalid_data)
            integration_tests.append(invalid_result["success"] is False)
            
            # Test 5: Test account operations through API
            # Add transaction
            transaction_data = {
                "amount": 5000.0,
                "transaction_type": "deposit",
                "description": "Test deposit"
            }
            
            transaction_result = self.api.add_transaction(self.auth_token, account_id, transaction_data)
            integration_tests.append(transaction_result["success"] is True)
            
            # Verify balance update
            account_after_transaction = self.db.get_account_by_id(account_id)
            integration_tests.append(account_after_transaction.current_balance == 115000.0)
            
            # Calculate success
            success = all(integration_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "api_business_logic_integration", "value": sum(integration_tests), "target": len(integration_tests), "threshold": len(integration_tests), "passed": success}
                ],
                "integration_results": integration_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_security_api(self):
        """Test integration between security framework and API layer"""
        try:
            integration_tests = []
            
            # Test 1: Valid authentication
            account_data = {
                "name": "Security Test Account",
                "account_type": AccountType.GENERATION,
                "initial_balance": 100000.0,
                "owner_id": self.test_user_id
            }
            
            result = self.api.create_account(self.auth_token, account_data)
            integration_tests.append(result["success"] is True)
            
            # Test 2: Invalid authentication
            invalid_token = "invalid_token_123"
            invalid_result = self.api.create_account(invalid_token, account_data)
            integration_tests.append(invalid_result["success"] is False)
            integration_tests.append("authentication" in invalid_result["error"].lower())
            
            # Test 3: Authorization checks
            # Create another user
            other_user_id = "other_user_456"
            self.security.create_user(other_user_id, "Other User", "other@example.com", "password456")
            other_token = self.security.generate_auth_token(other_user_id)
            
            # Try to access first user's account
            account_id = result["account_id"]
            unauthorized_result = self.api.get_account(other_token, account_id)
            integration_tests.append(unauthorized_result["success"] is False)
            integration_tests.append("authorization" in unauthorized_result["error"].lower())
            
            # Test 4: Role-based access control
            # Grant admin role to first user
            self.security.assign_role(self.test_user_id, "admin")
            
            # Create admin-only API endpoint result
            admin_result = self.api.get_all_accounts(self.auth_token)
            integration_tests.append(admin_result["success"] is True)
            
            # Try with non-admin user
            non_admin_result = self.api.get_all_accounts(other_token)
            integration_tests.append(non_admin_result["success"] is False)
            
            # Test 5: Audit logging
            # Perform action that should be logged
            self.api.update_account_status(self.auth_token, account_id, AccountStatus.SUSPENDED)
            
            # Check audit log
            audit_logs = self.security.get_audit_logs(account_id)
            integration_tests.append(len(audit_logs) > 0)
            
            # Verify log content
            latest_log = audit_logs[0]
            integration_tests.append(latest_log["user_id"] == self.test_user_id)
            integration_tests.append("status" in latest_log["action"].lower())
            integration_tests.append(latest_log["target_id"] == account_id)
            
            # Calculate success
            success = all(integration_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "security_api_integration", "value": sum(integration_tests), "target": len(integration_tests), "threshold": len(integration_tests), "passed": success}
                ],
                "integration_results": integration_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_analytics_database(self):
        """Test integration between analytics engine and database"""
        try:
            integration_tests = []
            
            # Create test account with transactions
            account = create_account(
                account_type=AccountType.GENERATION,
                name="Analytics Test Account",
                initial_balance=100000.0,
                owner_id=self.test_user_id
            )
            
            self.db.save_account(account)
            
            # Add transactions
            transactions = [
                {"amount": 5000.0, "transaction_type": "profit", "description": "Week 1 profit"},
                {"amount": 4500.0, "transaction_type": "profit", "description": "Week 2 profit"},
                {"amount": -2000.0, "transaction_type": "loss", "description": "Week 3 loss"},
                {"amount": 6000.0, "transaction_type": "profit", "description": "Week 4 profit"}
            ]
            
            for tx in transactions:
                self.db.add_transaction(
                    account.account_id,
                    tx["amount"],
                    tx["transaction_type"],
                    tx["description"]
                )
                
                # Update account balance
                account.current_balance += tx["amount"]
                self.db.update_account(account)
            
            # Test 1: Performance analysis
            performance = self.analytics.analyze_account_performance(account.account_id)
            integration_tests.append(performance is not None)
            integration_tests.append("total_return" in performance)
            integration_tests.append("win_rate" in performance)
            
            # Verify performance calculations
            total_profit = sum(tx["amount"] for tx in transactions)
            integration_tests.append(abs(performance["total_return"] - total_profit) < 0.01)
            
            win_count = sum(1 for tx in transactions if tx["amount"] > 0)
            expected_win_rate = win_count / len(transactions)
            integration_tests.append(abs(performance["win_rate"] - expected_win_rate) < 0.01)
            
            # Test 2: Trend detection
            trends = self.analytics.detect_account_trends(account.account_id)
            integration_tests.append(trends is not None)
            integration_tests.append("trend_direction" in trends)
            integration_tests.append("momentum" in trends)
            
            # Test 3: Risk assessment
            risk = self.analytics.assess_account_risk(account.account_id)
            integration_tests.append(risk is not None)
            integration_tests.append("risk_score" in risk)
            integration_tests.append("max_drawdown" in risk)
            
            # Test 4: Analytics dashboard
            dashboard = self.analytics.get_analytics_dashboard(account.account_id)
            integration_tests.append(dashboard is not None)
            integration_tests.append("performance" in dashboard)
            integration_tests.append("risk" in dashboard)
            integration_tests.append("trends" in dashboard)
            
            # Test 5: Analytics data persistence
            # Store analytics results
            self.analytics.store_analytics_results(account.account_id, performance, risk, trends)
            
            # Retrieve stored results
            stored_results = self.analytics.get_stored_analytics(account.account_id)
            integration_tests.append(stored_results is not None)
            integration_tests.append("performance" in stored_results)
            integration_tests.append("risk" in stored_results)
            
            # Calculate success
            success = all(integration_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "analytics_database_integration", "value": sum(integration_tests), "target": len(integration_tests), "threshold": len(integration_tests), "passed": success}
                ],
                "integration_results": integration_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_configuration_operation(self):
        """Test integration between configuration system and operations"""
        try:
            integration_tests = []
            
            # Import configuration system
            from src.account_management.config.account_configuration_system import AccountConfigurationSystem
            
            # Initialize configuration system
            config_system = AccountConfigurationSystem(self.db)
            
            # Create test account
            account = create_account(
                account_type=AccountType.GENERATION,
                name="Configuration Test Account",
                initial_balance=100000.0,
                owner_id=self.test_user_id
            )
            
            self.db.save_account(account)
            
            # Test 1: Set configuration and verify persistence
            config_system.set_account_config(account.account_id, "target_delta_range", (45, 55))
            
            stored_config = config_system.get_account_config(account.account_id, "target_delta_range")
            integration_tests.append(stored_config == (45, 55))
            
            # Test 2: Configuration affects operations
            # Update account with new configuration
            account = self.db.get_account_by_id(account.account_id)
            account.apply_configuration(config_system)
            
            # Verify configuration applied
            integration_tests.append(account.target_delta_range == (45, 55))
            
            # Test 3: Default configuration inheritance
            # Set system-wide default
            config_system.set_default_config(AccountType.GENERATION, "cash_buffer_percent", 7.5)
            
            # Create new account without explicit configuration
            new_account = create_account(
                account_type=AccountType.GENERATION,
                name="Default Config Account",
                initial_balance=100000.0,
                owner_id=self.test_user_id
            )
            
            self.db.save_account(new_account)
            
            # Apply default configuration
            new_account.apply_configuration(config_system)
            
            # Verify default applied
            integration_tests.append(new_account.cash_buffer_percent == 7.5)
            
            # Test 4: Configuration versioning
            # Update configuration multiple times
            config_system.set_account_config(account.account_id, "cash_buffer_percent", 6.0)
            config_system.set_account_config(account.account_id, "cash_buffer_percent", 8.0)
            
            # Get configuration history
            config_history = config_system.get_config_history(account.account_id, "cash_buffer_percent")
            integration_tests.append(len(config_history) >= 2)
            integration_tests.append(config_history[0]["value"] == 8.0)  # Latest value
            
            # Test 5: Configuration validation
            # Try to set invalid configuration
            try:
                config_system.set_account_config(account.account_id, "target_delta_range", (60, 70))
                integration_tests.append(False)  # Should not succeed (out of valid range)
            except ValueError:
                integration_tests.append(True)  # Expected exception
            
            # Calculate success
            success = all(integration_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "configuration_operation_integration", "value": sum(integration_tests), "target": len(integration_tests), "threshold": len(integration_tests), "passed": success}
                ],
                "integration_results": integration_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_all_tests(self):
        """Run all component integration tests"""
        test_funcs = {
            "data_model_database": self.test_data_model_database,
            "api_business_logic": self.test_api_business_logic,
            "security_api": self.test_security_api,
            "analytics_database": self.test_analytics_database,
            "configuration_operation": self.test_configuration_operation
        }
        
        results = self.framework.run_test_suite("internal_integration", test_funcs, "integration")
        return results


# Run tests if executed directly
if __name__ == "__main__":
    tests = ComponentIntegrationTests()
    results = tests.run_all_tests()
    
    # Generate and print report
    framework = tests.framework
    report = framework.generate_test_report("text")
    print(report["text_report"])
    
    # Export results
    framework.export_test_results("component_integration_test_results.json")
    
    # Clean up
    framework.cleanup()

