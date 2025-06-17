#!/usr/bin/env python3
"""
WS3-P4: Comprehensive Testing and Validation
Performance Benchmark Tests

This module implements performance tests for the account management system,
measuring throughput, latency, and scalability under various load conditions.

Author: Manus AI
Date: June 17, 2025
Version: 1.0
"""

import sys
import os
import json
import time
import random
import statistics
from datetime import datetime, timedelta
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple

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

class PerformanceBenchmarkTests:
    """
    Performance benchmark tests for the account management system.
    
    This class implements tests for:
    - Account Creation Performance
    - Transaction Processing Performance
    - Query Performance
    - API Response Time
    - Database Operation Performance
    """
    
    def __init__(self):
        self.framework = AccountManagementTestFramework()
        
        # Initialize components
        self.db = AccountDatabase(":memory:")
        self.security = SecurityFramework(self.db)
        self.api = AccountOperationsAPI(self.db, self.security)
        
        # Create test user
        self.test_user_id = "test_user_123"
        self.security.create_user(self.test_user_id, "Test User", "test@example.com", "password123")
        
        # Generate auth token for API calls
        self.auth_token = self.security.generate_auth_token(self.test_user_id)
        
        # Performance targets
        self.performance_targets = {
            "account_creation": 50.0,  # accounts/sec
            "transaction_processing": 100.0,  # transactions/sec
            "query_response": 0.01,  # seconds
            "api_response": 0.05,  # seconds
            "bulk_operations": 25.0  # operations/sec
        }
    
    def test_account_creation_perf(self):
        """Test account creation performance"""
        try:
            # Number of accounts to create
            num_accounts = 100
            
            # Prepare account data
            account_data_list = []
            for i in range(num_accounts):
                account_type = random.choice([
                    AccountType.GENERATION,
                    AccountType.REVENUE,
                    AccountType.COMPOUNDING
                ])
                
                account_data = {
                    "name": f"Performance Test Account {i}",
                    "account_type": account_type,
                    "initial_balance": 100000.0 + (i * 1000),
                    "owner_id": self.test_user_id
                }
                
                account_data_list.append(account_data)
            
            # Measure sequential creation performance
            sequential_start_time = time.time()
            
            for account_data in account_data_list[:20]:  # Test with subset for sequential
                account = create_account(
                    account_type=account_data["account_type"],
                    name=account_data["name"],
                    initial_balance=account_data["initial_balance"],
                    owner_id=account_data["owner_id"]
                )
                self.db.save_account(account)
            
            sequential_time = time.time() - sequential_start_time
            sequential_rate = 20 / sequential_time if sequential_time > 0 else 0
            
            # Measure bulk creation performance
            bulk_accounts = []
            for account_data in account_data_list:
                account = create_account(
                    account_type=account_data["account_type"],
                    name=account_data["name"],
                    initial_balance=account_data["initial_balance"],
                    owner_id=account_data["owner_id"]
                )
                bulk_accounts.append(account)
            
            bulk_start_time = time.time()
            self.db.save_accounts(bulk_accounts)
            bulk_time = time.time() - bulk_start_time
            bulk_rate = num_accounts / bulk_time if bulk_time > 0 else 0
            
            # Measure parallel creation performance using API
            def create_account_api(account_data):
                return self.api.create_account(self.auth_token, account_data)
            
            parallel_start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(create_account_api, account_data_list[:50]))
            
            parallel_time = time.time() - parallel_start_time
            parallel_rate = 50 / parallel_time if parallel_time > 0 else 0
            
            # Calculate success based on performance targets
            target_rate = self.performance_targets["account_creation"]
            
            sequential_success = sequential_rate >= target_rate * 0.2  # Sequential is slower
            bulk_success = bulk_rate >= target_rate
            parallel_success = parallel_rate >= target_rate * 0.5  # Parallel has API overhead
            
            overall_success = bulk_success  # Primary success metric is bulk performance
            
            return {
                "success": overall_success,
                "metrics": [
                    {"name": "sequential_creation_rate", "value": sequential_rate, "target": target_rate * 0.2, "threshold": target_rate * 0.1, "passed": sequential_success},
                    {"name": "bulk_creation_rate", "value": bulk_rate, "target": target_rate, "threshold": target_rate * 0.8, "passed": bulk_success},
                    {"name": "parallel_creation_rate", "value": parallel_rate, "target": target_rate * 0.5, "threshold": target_rate * 0.4, "passed": parallel_success}
                ],
                "performance_results": {
                    "sequential_time": sequential_time,
                    "sequential_rate": sequential_rate,
                    "bulk_time": bulk_time,
                    "bulk_rate": bulk_rate,
                    "parallel_time": parallel_time,
                    "parallel_rate": parallel_rate
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_transaction_processing_perf(self):
        """Test transaction processing performance"""
        try:
            # Create test account
            account = create_account(
                account_type=AccountType.GENERATION,
                name="Transaction Performance Test Account",
                initial_balance=1000000.0,
                owner_id=self.test_user_id
            )
            
            self.db.save_account(account)
            account_id = account.account_id
            
            # Number of transactions to process
            num_transactions = 1000
            
            # Prepare transaction data
            transaction_data_list = []
            for i in range(num_transactions):
                amount = random.uniform(100.0, 5000.0)
                transaction_type = random.choice(["deposit", "withdrawal", "profit", "loss"])
                
                if transaction_type in ["withdrawal", "loss"]:
                    amount = -amount
                
                transaction_data = {
                    "amount": amount,
                    "transaction_type": transaction_type,
                    "description": f"Performance test transaction {i}"
                }
                
                transaction_data_list.append(transaction_data)
            
            # Measure sequential transaction processing
            sequential_start_time = time.time()
            
            for transaction_data in transaction_data_list[:50]:  # Test with subset for sequential
                self.db.add_transaction(
                    account_id,
                    transaction_data["amount"],
                    transaction_data["transaction_type"],
                    transaction_data["description"]
                )
            
            sequential_time = time.time() - sequential_start_time
            sequential_rate = 50 / sequential_time if sequential_time > 0 else 0
            
            # Measure bulk transaction processing
            bulk_start_time = time.time()
            
            self.db.begin_transaction()
            for transaction_data in transaction_data_list:
                self.db.add_transaction(
                    account_id,
                    transaction_data["amount"],
                    transaction_data["transaction_type"],
                    transaction_data["description"]
                )
            self.db.commit_transaction()
            
            bulk_time = time.time() - bulk_start_time
            bulk_rate = num_transactions / bulk_time if bulk_time > 0 else 0
            
            # Measure parallel transaction processing using API
            def add_transaction_api(transaction_data):
                return self.api.add_transaction(self.auth_token, account_id, transaction_data)
            
            parallel_start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=20) as executor:
                results = list(executor.map(add_transaction_api, transaction_data_list[:200]))
            
            parallel_time = time.time() - parallel_start_time
            parallel_rate = 200 / parallel_time if parallel_time > 0 else 0
            
            # Calculate success based on performance targets
            target_rate = self.performance_targets["transaction_processing"]
            
            sequential_success = sequential_rate >= target_rate * 0.2  # Sequential is slower
            bulk_success = bulk_rate >= target_rate
            parallel_success = parallel_rate >= target_rate * 0.5  # Parallel has API overhead
            
            overall_success = bulk_success  # Primary success metric is bulk performance
            
            return {
                "success": overall_success,
                "metrics": [
                    {"name": "sequential_transaction_rate", "value": sequential_rate, "target": target_rate * 0.2, "threshold": target_rate * 0.1, "passed": sequential_success},
                    {"name": "bulk_transaction_rate", "value": bulk_rate, "target": target_rate, "threshold": target_rate * 0.8, "passed": bulk_success},
                    {"name": "parallel_transaction_rate", "value": parallel_rate, "target": target_rate * 0.5, "threshold": target_rate * 0.4, "passed": parallel_success}
                ],
                "performance_results": {
                    "sequential_time": sequential_time,
                    "sequential_rate": sequential_rate,
                    "bulk_time": bulk_time,
                    "bulk_rate": bulk_rate,
                    "parallel_time": parallel_time,
                    "parallel_rate": parallel_rate
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_query_performance_perf(self):
        """Test query performance"""
        try:
            # Create a large number of test accounts for query testing
            num_accounts = 1000
            account_ids = []
            
            # Create accounts in bulk
            bulk_accounts = []
            for i in range(num_accounts):
                account_type = random.choice([
                    AccountType.GENERATION,
                    AccountType.REVENUE,
                    AccountType.COMPOUNDING
                ])
                
                owner_id = f"user_{i % 10}"  # 10 different owners
                
                account = create_account(
                    account_type=account_type,
                    name=f"Query Test Account {i}",
                    initial_balance=100000.0 + (i * 1000),
                    owner_id=owner_id
                )
                
                bulk_accounts.append(account)
                account_ids.append(account.account_id)
            
            self.db.save_accounts(bulk_accounts)
            
            # Measure query performance
            query_times = {}
            
            # Test 1: Get account by ID
            start_time = time.time()
            for _ in range(100):
                account_id = random.choice(account_ids)
                account = self.db.get_account_by_id(account_id)
            query_times["get_by_id"] = (time.time() - start_time) / 100
            
            # Test 2: Get accounts by owner
            start_time = time.time()
            for i in range(10):
                owner_id = f"user_{i}"
                accounts = self.db.get_accounts_by_owner(owner_id)
            query_times["get_by_owner"] = (time.time() - start_time) / 10
            
            # Test 3: Get accounts by type
            start_time = time.time()
            for account_type in [AccountType.GENERATION, AccountType.REVENUE, AccountType.COMPOUNDING]:
                accounts = self.db.get_accounts_by_type(account_type)
            query_times["get_by_type"] = (time.time() - start_time) / 3
            
            # Test 4: Get accounts by balance range
            start_time = time.time()
            for i in range(10):
                min_balance = 100000.0 + (i * 100000)
                max_balance = min_balance + 100000.0
                accounts = self.db.get_accounts_by_balance_range(min_balance, max_balance)
            query_times["get_by_balance"] = (time.time() - start_time) / 10
            
            # Test 5: Complex query (multiple conditions)
            start_time = time.time()
            for i in range(5):
                owner_id = f"user_{i}"
                account_type = random.choice([AccountType.GENERATION, AccountType.REVENUE])
                min_balance = 100000.0 + (i * 100000)
                accounts = self.db.get_accounts_by_criteria({
                    "owner_id": owner_id,
                    "account_type": account_type,
                    "min_balance": min_balance
                })
            query_times["complex_query"] = (time.time() - start_time) / 5
            
            # Calculate average query time
            avg_query_time = statistics.mean(query_times.values())
            
            # Calculate success based on performance targets
            target_time = self.performance_targets["query_response"]
            
            query_success = {
                query: time <= target_time for query, time in query_times.items()
            }
            
            overall_success = avg_query_time <= target_time
            
            return {
                "success": overall_success,
                "metrics": [
                    {"name": "avg_query_time", "value": avg_query_time, "target": target_time, "threshold": target_time * 1.5, "passed": overall_success},
                    {"name": "get_by_id_time", "value": query_times["get_by_id"], "target": target_time, "threshold": target_time * 1.5, "passed": query_success["get_by_id"]},
                    {"name": "get_by_owner_time", "value": query_times["get_by_owner"], "target": target_time * 1.2, "threshold": target_time * 1.8, "passed": query_success["get_by_owner"]},
                    {"name": "complex_query_time", "value": query_times["complex_query"], "target": target_time * 1.5, "threshold": target_time * 2.0, "passed": query_success["complex_query"]}
                ],
                "performance_results": {
                    "query_times": query_times,
                    "avg_query_time": avg_query_time
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_api_response_time(self):
        """Test API response time"""
        try:
            # Create test account
            account_data = {
                "name": "API Performance Test Account",
                "account_type": AccountType.GENERATION,
                "initial_balance": 100000.0,
                "owner_id": self.test_user_id
            }
            
            create_result = self.api.create_account(self.auth_token, account_data)
            account_id = create_result["account_id"]
            
            # Measure API response times
            api_times = {}
            
            # Test 1: Get account
            start_time = time.time()
            for _ in range(50):
                result = self.api.get_account(self.auth_token, account_id)
            api_times["get_account"] = (time.time() - start_time) / 50
            
            # Test 2: Update account
            update_data = {
                "name": "Updated API Test Account"
            }
            
            start_time = time.time()
            for i in range(20):
                update_data["name"] = f"Updated API Test Account {i}"
                result = self.api.update_account(self.auth_token, account_id, update_data)
            api_times["update_account"] = (time.time() - start_time) / 20
            
            # Test 3: Add transaction
            transaction_data = {
                "amount": 1000.0,
                "transaction_type": "profit",
                "description": "API performance test transaction"
            }
            
            start_time = time.time()
            for i in range(20):
                transaction_data["description"] = f"API performance test transaction {i}"
                result = self.api.add_transaction(self.auth_token, account_id, transaction_data)
            api_times["add_transaction"] = (time.time() - start_time) / 20
            
            # Test 4: Get transactions
            start_time = time.time()
            for _ in range(20):
                result = self.api.get_account_transactions(self.auth_token, account_id)
            api_times["get_transactions"] = (time.time() - start_time) / 20
            
            # Test 5: Generate analytics
            start_time = time.time()
            for _ in range(10):
                result = self.api.generate_account_analytics(self.auth_token, account_id)
            api_times["generate_analytics"] = (time.time() - start_time) / 10
            
            # Calculate average API response time
            avg_api_time = statistics.mean(api_times.values())
            
            # Calculate success based on performance targets
            target_time = self.performance_targets["api_response"]
            
            api_success = {
                api: time <= target_time for api, time in api_times.items()
            }
            
            # Analytics is more complex, allow higher threshold
            api_success["generate_analytics"] = api_times["generate_analytics"] <= target_time * 2
            
            overall_success = avg_api_time <= target_time
            
            return {
                "success": overall_success,
                "metrics": [
                    {"name": "avg_api_time", "value": avg_api_time, "target": target_time, "threshold": target_time * 1.5, "passed": overall_success},
                    {"name": "get_account_time", "value": api_times["get_account"], "target": target_time, "threshold": target_time * 1.5, "passed": api_success["get_account"]},
                    {"name": "update_account_time", "value": api_times["update_account"], "target": target_time, "threshold": target_time * 1.5, "passed": api_success["update_account"]},
                    {"name": "add_transaction_time", "value": api_times["add_transaction"], "target": target_time, "threshold": target_time * 1.5, "passed": api_success["add_transaction"]},
                    {"name": "generate_analytics_time", "value": api_times["generate_analytics"], "target": target_time * 2, "threshold": target_time * 3, "passed": api_success["generate_analytics"]}
                ],
                "performance_results": {
                    "api_times": api_times,
                    "avg_api_time": avg_api_time
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_database_operation_perf(self):
        """Test database operation performance"""
        try:
            # Create test accounts
            num_accounts = 100
            account_ids = []
            
            # Create accounts in bulk
            bulk_accounts = []
            for i in range(num_accounts):
                account = create_account(
                    account_type=AccountType.GENERATION,
                    name=f"DB Performance Test Account {i}",
                    initial_balance=100000.0,
                    owner_id=self.test_user_id
                )
                
                bulk_accounts.append(account)
                account_ids.append(account.account_id)
            
            self.db.save_accounts(bulk_accounts)
            
            # Measure database operation performance
            db_times = {}
            
            # Test 1: Bulk status update
            start_time = time.time()
            
            self.db.begin_transaction()
            for account_id in account_ids:
                self.db.update_account_status(account_id, AccountStatus.SUSPENDED)
            self.db.commit_transaction()
            
            db_times["bulk_status_update"] = time.time() - start_time
            bulk_status_rate = num_accounts / db_times["bulk_status_update"] if db_times["bulk_status_update"] > 0 else 0
            
            # Test 2: Bulk balance update
            start_time = time.time()
            
            self.db.begin_transaction()
            for i, account_id in enumerate(account_ids):
                self.db.update_account_balance(account_id, 110000.0 + (i * 1000))
            self.db.commit_transaction()
            
            db_times["bulk_balance_update"] = time.time() - start_time
            bulk_balance_rate = num_accounts / db_times["bulk_balance_update"] if db_times["bulk_balance_update"] > 0 else 0
            
            # Test 3: Bulk data export
            start_time = time.time()
            
            accounts_data = []
            for account_id in account_ids:
                account = self.db.get_account_by_id(account_id)
                accounts_data.append(account.to_dict() if hasattr(account, 'to_dict') else str(account))
            
            db_times["bulk_data_export"] = time.time() - start_time
            bulk_export_rate = num_accounts / db_times["bulk_data_export"] if db_times["bulk_data_export"] > 0 else 0
            
            # Test 4: Bulk deletion
            start_time = time.time()
            
            self.db.begin_transaction()
            for account_id in account_ids[:50]:  # Delete half
                self.db.delete_account(account_id)
            self.db.commit_transaction()
            
            db_times["bulk_deletion"] = time.time() - start_time
            bulk_deletion_rate = 50 / db_times["bulk_deletion"] if db_times["bulk_deletion"] > 0 else 0
            
            # Calculate success based on performance targets
            target_rate = self.performance_targets["bulk_operations"]
            
            bulk_success = {
                "status_update": bulk_status_rate >= target_rate,
                "balance_update": bulk_balance_rate >= target_rate,
                "data_export": bulk_export_rate >= target_rate,
                "deletion": bulk_deletion_rate >= target_rate
            }
            
            overall_success = all(bulk_success.values())
            
            return {
                "success": overall_success,
                "metrics": [
                    {"name": "bulk_status_update_rate", "value": bulk_status_rate, "target": target_rate, "threshold": target_rate * 0.8, "passed": bulk_success["status_update"]},
                    {"name": "bulk_balance_update_rate", "value": bulk_balance_rate, "target": target_rate, "threshold": target_rate * 0.8, "passed": bulk_success["balance_update"]},
                    {"name": "bulk_data_export_rate", "value": bulk_export_rate, "target": target_rate, "threshold": target_rate * 0.8, "passed": bulk_success["data_export"]},
                    {"name": "bulk_deletion_rate", "value": bulk_deletion_rate, "target": target_rate, "threshold": target_rate * 0.8, "passed": bulk_success["deletion"]}
                ],
                "performance_results": {
                    "db_times": db_times,
                    "bulk_rates": {
                        "status_update": bulk_status_rate,
                        "balance_update": bulk_balance_rate,
                        "data_export": bulk_export_rate,
                        "deletion": bulk_deletion_rate
                    }
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_all_tests(self):
        """Run all performance benchmark tests"""
        test_funcs = {
            "account_creation_perf": self.test_account_creation_perf,
            "transaction_processing_perf": self.test_transaction_processing_perf,
            "query_performance_perf": self.test_query_performance_perf,
            "api_response_time": self.test_api_response_time,
            "database_operation_perf": self.test_database_operation_perf
        }
        
        results = self.framework.run_test_suite("performance_benchmarks", test_funcs, "performance")
        return results


# Run tests if executed directly
if __name__ == "__main__":
    tests = PerformanceBenchmarkTests()
    results = tests.run_all_tests()
    
    # Generate and print report
    framework = tests.framework
    report = framework.generate_test_report("text")
    print(report["text_report"])
    
    # Export results
    framework.export_test_results("performance_benchmark_test_results.json")
    
    # Clean up
    framework.cleanup()

