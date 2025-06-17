#!/usr/bin/env python3
"""
WS3-P4: Comprehensive Testing and Validation
Load and Stress Tests

This module implements load and stress tests for the account management system,
validating performance and stability under high load and extreme conditions.

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
import multiprocessing
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np

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

class LoadStressTests:
    """
    Load and stress tests for the account management system.
    
    This class implements tests for:
    - Concurrent User Load
    - High Transaction Volume
    - Data Volume Scaling
    - Long-Running Operations
    - System Resource Limits
    """
    
    def __init__(self):
        self.framework = AccountManagementTestFramework()
        
        # Initialize components
        self.db = AccountDatabase(":memory:")
        self.security = SecurityFramework(self.db)
        self.api = AccountOperationsAPI(self.db, self.security)
        
        # Create test users
        self.test_users = []
        self.auth_tokens = {}
        
        for i in range(10):
            user_id = f"test_user_{i}"
            self.security.create_user(user_id, f"Test User {i}", f"test{i}@example.com", f"password{i}")
            self.test_users.append(user_id)
            self.auth_tokens[user_id] = self.security.generate_auth_token(user_id)
        
        # Performance targets
        self.performance_targets = {
            "concurrent_users": 50,  # Number of concurrent users
            "transaction_volume": 1000,  # Transactions per second
            "data_volume": 10000,  # Number of accounts
            "response_time_under_load": 0.1,  # seconds
            "error_rate_under_load": 0.01  # 1% error rate
        }
        
        # Create output directory for charts
        self.output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_concurrent_user_load(self):
        """Test system performance under concurrent user load"""
        try:
            # Number of concurrent users to simulate
            num_users = self.performance_targets["concurrent_users"]
            
            # Operations to test
            operations = [
                "create_account",
                "get_account",
                "update_account",
                "add_transaction",
                "get_transactions"
            ]
            
            # Results storage
            results = {op: {"times": [], "success": 0, "failure": 0} for op in operations}
            
            # Create test accounts for each user
            accounts = {}
            for user_id in self.test_users:
                account_data = {
                    "name": f"Load Test Account for {user_id}",
                    "account_type": AccountType.GENERATION,
                    "initial_balance": 100000.0,
                    "owner_id": user_id
                }
                
                create_result = self.api.create_account(self.auth_tokens[user_id], account_data)
                accounts[user_id] = create_result["account_id"]
            
            # Function to execute random operations
            def execute_random_operations(user_index, num_operations=10):
                user_id = self.test_users[user_index % len(self.test_users)]
                token = self.auth_tokens[user_id]
                account_id = accounts[user_id]
                
                for _ in range(num_operations):
                    operation = random.choice(operations)
                    
                    try:
                        start_time = time.time()
                        
                        if operation == "create_account":
                            account_data = {
                                "name": f"Concurrent Load Test Account {uuid.uuid4()}",
                                "account_type": random.choice([AccountType.GENERATION, AccountType.REVENUE, AccountType.COMPOUNDING]),
                                "initial_balance": random.uniform(10000.0, 200000.0),
                                "owner_id": user_id
                            }
                            result = self.api.create_account(token, account_data)
                            
                        elif operation == "get_account":
                            result = self.api.get_account(token, account_id)
                            
                        elif operation == "update_account":
                            update_data = {
                                "name": f"Updated Account {uuid.uuid4()}"
                            }
                            result = self.api.update_account(token, account_id, update_data)
                            
                        elif operation == "add_transaction":
                            transaction_data = {
                                "amount": random.uniform(100.0, 5000.0),
                                "transaction_type": random.choice(["deposit", "withdrawal", "profit", "loss"]),
                                "description": f"Concurrent load test transaction {uuid.uuid4()}"
                            }
                            result = self.api.add_transaction(token, account_id, transaction_data)
                            
                        elif operation == "get_transactions":
                            result = self.api.get_account_transactions(token, account_id)
                        
                        execution_time = time.time() - start_time
                        
                        with results_lock:
                            results[operation]["times"].append(execution_time)
                            if result["success"]:
                                results[operation]["success"] += 1
                            else:
                                results[operation]["failure"] += 1
                                
                    except Exception as e:
                        with results_lock:
                            results[operation]["failure"] += 1
            
            # Create lock for thread-safe results update
            results_lock = threading.Lock()
            
            # Execute concurrent operations
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(execute_random_operations, i) for i in range(num_users)]
                
                # Wait for all futures to complete
                for future in futures:
                    future.result()
            
            total_time = time.time() - start_time
            
            # Calculate metrics
            total_operations = sum(len(result["times"]) for result in results.values())
            operations_per_second = total_operations / total_time if total_time > 0 else 0
            
            avg_response_times = {op: statistics.mean(result["times"]) if result["times"] else 0 for op, result in results.items()}
            overall_avg_response_time = statistics.mean([time for result in results.values() for time in result["times"]]) if total_operations > 0 else 0
            
            total_success = sum(result["success"] for result in results.values())
            total_failure = sum(result["failure"] for result in results.values())
            error_rate = total_failure / (total_success + total_failure) if (total_success + total_failure) > 0 else 0
            
            # Generate chart
            self._generate_response_time_chart(results, "concurrent_user_load")
            
            # Calculate success based on performance targets
            target_response_time = self.performance_targets["response_time_under_load"]
            target_error_rate = self.performance_targets["error_rate_under_load"]
            
            response_time_success = overall_avg_response_time <= target_response_time
            error_rate_success = error_rate <= target_error_rate
            
            overall_success = response_time_success and error_rate_success
            
            return {
                "success": overall_success,
                "metrics": [
                    {"name": "concurrent_users", "value": num_users, "target": self.performance_targets["concurrent_users"], "threshold": self.performance_targets["concurrent_users"] * 0.8, "passed": num_users >= self.performance_targets["concurrent_users"] * 0.8},
                    {"name": "operations_per_second", "value": operations_per_second, "target": num_users * 10 / 5, "threshold": num_users * 10 / 10, "passed": True},  # Informational metric
                    {"name": "avg_response_time", "value": overall_avg_response_time, "target": target_response_time, "threshold": target_response_time * 1.5, "passed": response_time_success},
                    {"name": "error_rate", "value": error_rate, "target": target_error_rate, "threshold": target_error_rate * 2, "passed": error_rate_success}
                ],
                "load_results": {
                    "total_operations": total_operations,
                    "total_time": total_time,
                    "operations_per_second": operations_per_second,
                    "avg_response_times": avg_response_times,
                    "overall_avg_response_time": overall_avg_response_time,
                    "total_success": total_success,
                    "total_failure": total_failure,
                    "error_rate": error_rate,
                    "chart_path": os.path.join(self.output_dir, "concurrent_user_load_chart.png")
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_high_transaction_volume(self):
        """Test system performance under high transaction volume"""
        try:
            # Create test account
            account_data = {
                "name": "High Volume Test Account",
                "account_type": AccountType.GENERATION,
                "initial_balance": 1000000.0,
                "owner_id": self.test_users[0]
            }
            
            create_result = self.api.create_account(self.auth_tokens[self.test_users[0]], account_data)
            account_id = create_result["account_id"]
            
            # Number of transactions to process
            target_volume = self.performance_targets["transaction_volume"]
            
            # Prepare transaction data
            transaction_data_list = []
            for i in range(target_volume):
                amount = random.uniform(100.0, 5000.0)
                transaction_type = random.choice(["deposit", "withdrawal", "profit", "loss"])
                
                if transaction_type in ["withdrawal", "loss"]:
                    amount = -amount
                
                transaction_data = {
                    "amount": amount,
                    "transaction_type": transaction_type,
                    "description": f"Volume test transaction {i}"
                }
                
                transaction_data_list.append(transaction_data)
            
            # Results storage
            results = {
                "times": [],
                "success": 0,
                "failure": 0
            }
            
            # Function to process transactions in batches
            def process_transaction_batch(batch):
                batch_results = {
                    "times": [],
                    "success": 0,
                    "failure": 0
                }
                
                for transaction_data in batch:
                    try:
                        start_time = time.time()
                        result = self.api.add_transaction(self.auth_tokens[self.test_users[0]], account_id, transaction_data)
                        execution_time = time.time() - start_time
                        
                        batch_results["times"].append(execution_time)
                        if result["success"]:
                            batch_results["success"] += 1
                        else:
                            batch_results["failure"] += 1
                            
                    except Exception:
                        batch_results["failure"] += 1
                
                return batch_results
            
            # Split transactions into batches
            batch_size = 100
            batches = [transaction_data_list[i:i+batch_size] for i in range(0, len(transaction_data_list), batch_size)]
            
            # Process batches in parallel
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                batch_results = list(executor.map(process_transaction_batch, batches))
            
            total_time = time.time() - start_time
            
            # Combine batch results
            for batch_result in batch_results:
                results["times"].extend(batch_result["times"])
                results["success"] += batch_result["success"]
                results["failure"] += batch_result["failure"]
            
            # Calculate metrics
            total_transactions = results["success"] + results["failure"]
            transactions_per_second = total_transactions / total_time if total_time > 0 else 0
            
            avg_transaction_time = statistics.mean(results["times"]) if results["times"] else 0
            error_rate = results["failure"] / total_transactions if total_transactions > 0 else 0
            
            # Generate chart
            self._generate_transaction_volume_chart(results["times"], "high_transaction_volume")
            
            # Calculate success based on performance targets
            volume_success = transactions_per_second >= target_volume
            response_time_success = avg_transaction_time <= self.performance_targets["response_time_under_load"]
            error_rate_success = error_rate <= self.performance_targets["error_rate_under_load"]
            
            overall_success = volume_success and response_time_success and error_rate_success
            
            return {
                "success": overall_success,
                "metrics": [
                    {"name": "transactions_per_second", "value": transactions_per_second, "target": target_volume, "threshold": target_volume * 0.8, "passed": volume_success},
                    {"name": "avg_transaction_time", "value": avg_transaction_time, "target": self.performance_targets["response_time_under_load"], "threshold": self.performance_targets["response_time_under_load"] * 1.5, "passed": response_time_success},
                    {"name": "error_rate", "value": error_rate, "target": self.performance_targets["error_rate_under_load"], "threshold": self.performance_targets["error_rate_under_load"] * 2, "passed": error_rate_success}
                ],
                "volume_results": {
                    "total_transactions": total_transactions,
                    "total_time": total_time,
                    "transactions_per_second": transactions_per_second,
                    "avg_transaction_time": avg_transaction_time,
                    "error_rate": error_rate,
                    "chart_path": os.path.join(self.output_dir, "high_transaction_volume_chart.png")
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_data_volume_scaling(self):
        """Test system performance with large data volumes"""
        try:
            # Number of accounts to create
            target_volume = self.performance_targets["data_volume"]
            
            # Create accounts in batches
            batch_size = 500
            num_batches = (target_volume + batch_size - 1) // batch_size  # Ceiling division
            
            # Results storage
            creation_times = []
            query_times = []
            account_ids = []
            
            # Create accounts in batches
            for batch in range(num_batches):
                batch_accounts = []
                batch_start = time.time()
                
                for i in range(batch_size):
                    if batch * batch_size + i >= target_volume:
                        break
                    
                    account_type = random.choice([
                        AccountType.GENERATION,
                        AccountType.REVENUE,
                        AccountType.COMPOUNDING
                    ])
                    
                    owner_id = random.choice(self.test_users)
                    
                    account = create_account(
                        account_type=account_type,
                        name=f"Volume Test Account {batch * batch_size + i}",
                        initial_balance=random.uniform(10000.0, 200000.0),
                        owner_id=owner_id
                    )
                    
                    batch_accounts.append(account)
                
                # Save batch to database
                self.db.save_accounts(batch_accounts)
                
                batch_time = time.time() - batch_start
                creation_times.append(batch_time / len(batch_accounts) if batch_accounts else 0)
                
                # Store account IDs for querying
                account_ids.extend([account.account_id for account in batch_accounts])
            
            # Test query performance with large data volume
            # Test 1: Get account by ID
            id_query_times = []
            for _ in range(100):
                account_id = random.choice(account_ids)
                start_time = time.time()
                account = self.db.get_account_by_id(account_id)
                id_query_times.append(time.time() - start_time)
            
            # Test 2: Get accounts by owner
            owner_query_times = []
            for _ in range(20):
                owner_id = random.choice(self.test_users)
                start_time = time.time()
                accounts = self.db.get_accounts_by_owner(owner_id)
                owner_query_times.append(time.time() - start_time)
            
            # Test 3: Get accounts by type
            type_query_times = []
            for _ in range(20):
                account_type = random.choice([
                    AccountType.GENERATION,
                    AccountType.REVENUE,
                    AccountType.COMPOUNDING
                ])
                start_time = time.time()
                accounts = self.db.get_accounts_by_type(account_type)
                type_query_times.append(time.time() - start_time)
            
            # Test 4: Get accounts by balance range
            balance_query_times = []
            for _ in range(20):
                min_balance = random.uniform(10000.0, 100000.0)
                max_balance = min_balance + 100000.0
                start_time = time.time()
                accounts = self.db.get_accounts_by_balance_range(min_balance, max_balance)
                balance_query_times.append(time.time() - start_time)
            
            # Combine query times
            query_times = id_query_times + owner_query_times + type_query_times + balance_query_times
            
            # Calculate metrics
            avg_creation_time = statistics.mean(creation_times)
            creation_rate = 1 / avg_creation_time if avg_creation_time > 0 else 0
            
            avg_query_time = statistics.mean(query_times)
            query_rate = 1 / avg_query_time if avg_query_time > 0 else 0
            
            # Generate charts
            self._generate_data_volume_chart(creation_times, query_times, "data_volume_scaling")
            
            # Calculate success based on performance targets
            volume_success = len(account_ids) >= target_volume * 0.8
            query_time_success = avg_query_time <= self.performance_targets["response_time_under_load"] * 2  # Allow higher latency for large data
            
            overall_success = volume_success and query_time_success
            
            return {
                "success": overall_success,
                "metrics": [
                    {"name": "data_volume", "value": len(account_ids), "target": target_volume, "threshold": target_volume * 0.8, "passed": volume_success},
                    {"name": "avg_creation_time", "value": avg_creation_time, "target": 0.01, "threshold": 0.02, "passed": avg_creation_time <= 0.02},
                    {"name": "avg_query_time", "value": avg_query_time, "target": self.performance_targets["response_time_under_load"] * 2, "threshold": self.performance_targets["response_time_under_load"] * 3, "passed": query_time_success}
                ],
                "volume_results": {
                    "total_accounts": len(account_ids),
                    "avg_creation_time": avg_creation_time,
                    "creation_rate": creation_rate,
                    "avg_query_time": avg_query_time,
                    "query_rate": query_rate,
                    "chart_path": os.path.join(self.output_dir, "data_volume_scaling_chart.png")
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_long_running_operations(self):
        """Test system performance with long-running operations"""
        try:
            # Create test account
            account_data = {
                "name": "Long Operation Test Account",
                "account_type": AccountType.GENERATION,
                "initial_balance": 1000000.0,
                "owner_id": self.test_users[0]
            }
            
            create_result = self.api.create_account(self.auth_tokens[self.test_users[0]], account_data)
            account_id = create_result["account_id"]
            
            # Long-running operations to test
            operations = [
                "bulk_transaction_import",
                "account_analytics_generation",
                "historical_data_processing",
                "complex_report_generation",
                "system_wide_reconciliation"
            ]
            
            # Results storage
            results = {op: {"time": 0, "success": False, "resource_usage": {}} for op in operations}
            
            # Test 1: Bulk Transaction Import
            start_time = time.time()
            
            # Generate large number of transactions
            num_transactions = 10000
            transactions = []
            
            for i in range(num_transactions):
                amount = random.uniform(100.0, 5000.0)
                transaction_type = random.choice(["deposit", "withdrawal", "profit", "loss"])
                
                if transaction_type in ["withdrawal", "loss"]:
                    amount = -amount
                
                transaction = {
                    "account_id": account_id,
                    "amount": amount,
                    "transaction_type": transaction_type,
                    "description": f"Bulk import transaction {i}",
                    "timestamp": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
                }
                
                transactions.append(transaction)
            
            # Import transactions in bulk
            import_result = self.api.bulk_import_transactions(self.auth_tokens[self.test_users[0]], {
                "account_id": account_id,
                "transactions": transactions
            })
            
            results["bulk_transaction_import"]["time"] = time.time() - start_time
            results["bulk_transaction_import"]["success"] = import_result["success"]
            results["bulk_transaction_import"]["resource_usage"] = self._get_resource_usage()
            
            # Test 2: Account Analytics Generation
            start_time = time.time()
            
            analytics_result = self.api.generate_comprehensive_analytics(self.auth_tokens[self.test_users[0]], {
                "account_id": account_id,
                "time_period": "all",
                "metrics": ["performance", "risk", "trends", "forecasts", "comparisons"],
                "detail_level": "maximum"
            })
            
            results["account_analytics_generation"]["time"] = time.time() - start_time
            results["account_analytics_generation"]["success"] = analytics_result["success"]
            results["account_analytics_generation"]["resource_usage"] = self._get_resource_usage()
            
            # Test 3: Historical Data Processing
            start_time = time.time()
            
            history_result = self.api.process_historical_data(self.auth_tokens[self.test_users[0]], {
                "account_id": account_id,
                "data_type": "market_data",
                "time_period": "5y",
                "resolution": "daily"
            })
            
            results["historical_data_processing"]["time"] = time.time() - start_time
            results["historical_data_processing"]["success"] = history_result["success"]
            results["historical_data_processing"]["resource_usage"] = self._get_resource_usage()
            
            # Test 4: Complex Report Generation
            start_time = time.time()
            
            report_result = self.api.generate_complex_report(self.auth_tokens[self.test_users[0]], {
                "account_id": account_id,
                "report_type": "comprehensive_performance",
                "format": "pdf",
                "include_charts": True,
                "include_raw_data": True
            })
            
            results["complex_report_generation"]["time"] = time.time() - start_time
            results["complex_report_generation"]["success"] = report_result["success"]
            results["complex_report_generation"]["resource_usage"] = self._get_resource_usage()
            
            # Test 5: System-Wide Reconciliation
            start_time = time.time()
            
            reconciliation_result = self.api.perform_system_reconciliation(self.auth_tokens[self.test_users[0]], {
                "scope": "all_accounts",
                "verification_level": "detailed",
                "fix_inconsistencies": True
            })
            
            results["system_wide_reconciliation"]["time"] = time.time() - start_time
            results["system_wide_reconciliation"]["success"] = reconciliation_result["success"]
            results["system_wide_reconciliation"]["resource_usage"] = self._get_resource_usage()
            
            # Generate chart
            self._generate_long_operations_chart(results, "long_running_operations")
            
            # Calculate metrics
            operation_times = [result["time"] for result in results.values()]
            avg_operation_time = statistics.mean(operation_times)
            max_operation_time = max(operation_times)
            
            success_count = sum(1 for result in results.values() if result["success"])
            success_rate = success_count / len(operations)
            
            # Calculate success based on performance targets
            # For long-running operations, we care more about success than speed
            success_rate_target = 0.9  # 90% success rate
            
            overall_success = success_rate >= success_rate_target
            
            return {
                "success": overall_success,
                "metrics": [
                    {"name": "success_rate", "value": success_rate, "target": success_rate_target, "threshold": success_rate_target * 0.9, "passed": success_rate >= success_rate_target},
                    {"name": "avg_operation_time", "value": avg_operation_time, "target": None, "threshold": None, "passed": True},  # Informational metric
                    {"name": "max_operation_time", "value": max_operation_time, "target": None, "threshold": None, "passed": True}  # Informational metric
                ],
                "operation_results": {
                    "operation_times": {op: result["time"] for op, result in results.items()},
                    "success_rate": success_rate,
                    "avg_operation_time": avg_operation_time,
                    "max_operation_time": max_operation_time,
                    "chart_path": os.path.join(self.output_dir, "long_running_operations_chart.png")
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_system_resource_limits(self):
        """Test system behavior at resource limits"""
        try:
            # Resource limit tests
            tests = [
                "memory_usage",
                "cpu_usage",
                "database_connections",
                "concurrent_operations",
                "large_result_sets"
            ]
            
            # Results storage
            results = {test: {"success": False, "metrics": {}} for test in tests}
            
            # Test 1: Memory Usage
            # Create large data structures to test memory handling
            start_time = time.time()
            
            try:
                # Generate large number of accounts
                num_accounts = 5000
                accounts = []
                
                for i in range(num_accounts):
                    account = create_account(
                        account_type=random.choice([AccountType.GENERATION, AccountType.REVENUE, AccountType.COMPOUNDING]),
                        name=f"Memory Test Account {i}",
                        initial_balance=random.uniform(10000.0, 200000.0),
                        owner_id=random.choice(self.test_users)
                    )
                    accounts.append(account)
                
                # Save accounts in bulk
                self.db.save_accounts(accounts)
                
                # Retrieve all accounts to force memory usage
                all_accounts = self.db.get_all_accounts()
                
                # Check if all accounts were retrieved
                results["memory_usage"]["success"] = len(all_accounts) >= num_accounts
                results["memory_usage"]["metrics"] = {
                    "accounts_created": num_accounts,
                    "accounts_retrieved": len(all_accounts),
                    "execution_time": time.time() - start_time,
                    "memory_usage": self._get_resource_usage().get("memory", 0)
                }
                
            except Exception as e:
                results["memory_usage"]["error"] = str(e)
            
            # Test 2: CPU Usage
            # Perform CPU-intensive operations
            start_time = time.time()
            
            try:
                # Generate complex analytics for multiple accounts
                cpu_test_accounts = []
                
                # Create test accounts
                for i in range(10):
                    account_data = {
                        "name": f"CPU Test Account {i}",
                        "account_type": AccountType.GENERATION,
                        "initial_balance": 100000.0,
                        "owner_id": self.test_users[i % len(self.test_users)]
                    }
                    
                    create_result = self.api.create_account(
                        self.auth_tokens[self.test_users[i % len(self.test_users)]],
                        account_data
                    )
                    
                    cpu_test_accounts.append(create_result["account_id"])
                
                # Add transactions to accounts
                for account_id in cpu_test_accounts:
                    for _ in range(1000):
                        transaction_data = {
                            "amount": random.uniform(100.0, 5000.0),
                            "transaction_type": random.choice(["deposit", "withdrawal", "profit", "loss"]),
                            "description": f"CPU test transaction {uuid.uuid4()}"
                        }
                        
                        self.api.add_transaction(
                            self.auth_tokens[self.test_users[0]],
                            account_id,
                            transaction_data
                        )
                
                # Generate analytics in parallel
                def generate_analytics(account_id):
                    return self.api.generate_account_analytics(
                        self.auth_tokens[self.test_users[0]],
                        account_id
                    )
                
                with ThreadPoolExecutor(max_workers=10) as executor:
                    analytics_results = list(executor.map(generate_analytics, cpu_test_accounts))
                
                # Check if all analytics were generated
                results["cpu_usage"]["success"] = all(result["success"] for result in analytics_results)
                results["cpu_usage"]["metrics"] = {
                    "accounts_processed": len(cpu_test_accounts),
                    "execution_time": time.time() - start_time,
                    "cpu_usage": self._get_resource_usage().get("cpu", 0)
                }
                
            except Exception as e:
                results["cpu_usage"]["error"] = str(e)
            
            # Test 3: Database Connections
            # Test with many concurrent database connections
            start_time = time.time()
            
            try:
                # Create multiple database connections
                num_connections = 50
                connections = []
                
                for i in range(num_connections):
                    db = AccountDatabase(":memory:")
                    connections.append(db)
                
                # Perform operations on each connection
                for i, db in enumerate(connections):
                    account = create_account(
                        account_type=AccountType.GENERATION,
                        name=f"Connection Test Account {i}",
                        initial_balance=100000.0,
                        owner_id=self.test_users[0]
                    )
                    
                    db.save_account(account)
                    retrieved = db.get_account_by_id(account.account_id)
                    
                    if not retrieved or retrieved.account_id != account.account_id:
                        raise ValueError(f"Failed to retrieve account on connection {i}")
                
                # Close connections
                for db in connections:
                    db.close()
                
                results["database_connections"]["success"] = True
                results["database_connections"]["metrics"] = {
                    "connections_created": num_connections,
                    "execution_time": time.time() - start_time
                }
                
            except Exception as e:
                results["database_connections"]["error"] = str(e)
            
            # Test 4: Concurrent Operations
            # Test with many concurrent API operations
            start_time = time.time()
            
            try:
                # Create test account
                account_data = {
                    "name": "Concurrent Operations Test Account",
                    "account_type": AccountType.GENERATION,
                    "initial_balance": 1000000.0,
                    "owner_id": self.test_users[0]
                }
                
                create_result = self.api.create_account(self.auth_tokens[self.test_users[0]], account_data)
                account_id = create_result["account_id"]
                
                # Define operations to perform concurrently
                operations = [
                    lambda: self.api.get_account(self.auth_tokens[self.test_users[0]], account_id),
                    lambda: self.api.update_account(self.auth_tokens[self.test_users[0]], account_id, {"name": f"Updated Account {uuid.uuid4()}"}),
                    lambda: self.api.add_transaction(self.auth_tokens[self.test_users[0]], account_id, {
                        "amount": random.uniform(100.0, 5000.0),
                        "transaction_type": "deposit",
                        "description": f"Concurrent operation test {uuid.uuid4()}"
                    }),
                    lambda: self.api.get_account_transactions(self.auth_tokens[self.test_users[0]], account_id),
                    lambda: self.api.generate_account_analytics(self.auth_tokens[self.test_users[0]], account_id)
                ]
                
                # Execute operations concurrently
                num_operations = 100
                operation_results = []
                
                with ThreadPoolExecutor(max_workers=20) as executor:
                    futures = []
                    
                    for _ in range(num_operations):
                        operation = random.choice(operations)
                        futures.append(executor.submit(operation))
                    
                    for future in futures:
                        try:
                            result = future.result()
                            operation_results.append(result["success"])
                        except Exception:
                            operation_results.append(False)
                
                # Check success rate
                success_count = sum(1 for result in operation_results if result)
                success_rate = success_count / num_operations
                
                results["concurrent_operations"]["success"] = success_rate >= 0.9  # 90% success rate
                results["concurrent_operations"]["metrics"] = {
                    "operations_attempted": num_operations,
                    "operations_succeeded": success_count,
                    "success_rate": success_rate,
                    "execution_time": time.time() - start_time
                }
                
            except Exception as e:
                results["concurrent_operations"]["error"] = str(e)
            
            # Test 5: Large Result Sets
            # Test handling of large result sets
            start_time = time.time()
            
            try:
                # Create large number of accounts for a single user
                num_accounts = 1000
                bulk_accounts = []
                
                for i in range(num_accounts):
                    account = create_account(
                        account_type=random.choice([AccountType.GENERATION, AccountType.REVENUE, AccountType.COMPOUNDING]),
                        name=f"Large Result Test Account {i}",
                        initial_balance=random.uniform(10000.0, 200000.0),
                        owner_id=self.test_users[0]
                    )
                    
                    bulk_accounts.append(account)
                
                # Save accounts in bulk
                self.db.save_accounts(bulk_accounts)
                
                # Retrieve accounts with pagination
                page_size = 100
                total_retrieved = 0
                
                for page in range(10):
                    accounts = self.api.get_user_accounts(
                        self.auth_tokens[self.test_users[0]],
                        {
                            "user_id": self.test_users[0],
                            "page": page,
                            "page_size": page_size
                        }
                    )
                    
                    if accounts["success"]:
                        total_retrieved += len(accounts["accounts"])
                
                results["large_result_sets"]["success"] = total_retrieved >= num_accounts
                results["large_result_sets"]["metrics"] = {
                    "accounts_created": num_accounts,
                    "accounts_retrieved": total_retrieved,
                    "execution_time": time.time() - start_time
                }
                
            except Exception as e:
                results["large_result_sets"]["error"] = str(e)
            
            # Generate chart
            self._generate_resource_limits_chart(results, "system_resource_limits")
            
            # Calculate overall success
            success_count = sum(1 for result in results.values() if result["success"])
            overall_success = success_count >= len(tests) * 0.8  # 80% success rate
            
            return {
                "success": overall_success,
                "metrics": [
                    {"name": "resource_tests_passed", "value": success_count, "target": len(tests), "threshold": len(tests) * 0.8, "passed": overall_success}
                ],
                "resource_results": {
                    "test_results": {test: {"success": result["success"], "metrics": result["metrics"]} for test, result in results.items()},
                    "success_rate": success_count / len(tests),
                    "chart_path": os.path.join(self.output_dir, "system_resource_limits_chart.png")
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_resource_usage(self):
        """Get current resource usage"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            
            return {
                "memory": process.memory_info().rss / (1024 * 1024),  # MB
                "cpu": process.cpu_percent(),
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections())
            }
        except ImportError:
            # psutil not available, return dummy data
            return {
                "memory": 100.0,
                "cpu": 5.0,
                "threads": 10,
                "open_files": 5,
                "connections": 2
            }
    
    def _generate_response_time_chart(self, results, test_name):
        """Generate response time chart for concurrent user load test"""
        plt.figure(figsize=(10, 6))
        
        operations = list(results.keys())
        avg_times = []
        
        for operation in operations:
            if results[operation]["times"]:
                avg_times.append(statistics.mean(results[operation]["times"]))
            else:
                avg_times.append(0)
        
        plt.bar(operations, avg_times)
        plt.xlabel('Operation')
        plt.ylabel('Average Response Time (s)')
        plt.title('Average Response Time by Operation Under Load')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = os.path.join(self.output_dir, f"{test_name}_chart.png")
        plt.savefig(chart_path)
        plt.close()
    
    def _generate_transaction_volume_chart(self, transaction_times, test_name):
        """Generate transaction volume chart"""
        plt.figure(figsize=(10, 6))
        
        # Plot transaction times
        plt.plot(range(len(transaction_times)), transaction_times, 'b-')
        
        # Plot moving average
        window_size = min(100, len(transaction_times))
        if window_size > 0:
            moving_avg = []
            for i in range(len(transaction_times) - window_size + 1):
                moving_avg.append(sum(transaction_times[i:i+window_size]) / window_size)
            
            plt.plot(range(window_size-1, len(transaction_times)), moving_avg, 'r-', linewidth=2)
        
        plt.xlabel('Transaction Number')
        plt.ylabel('Processing Time (s)')
        plt.title('Transaction Processing Time Under High Volume')
        plt.tight_layout()
        
        chart_path = os.path.join(self.output_dir, f"{test_name}_chart.png")
        plt.savefig(chart_path)
        plt.close()
    
    def _generate_data_volume_chart(self, creation_times, query_times, test_name):
        """Generate data volume scaling chart"""
        plt.figure(figsize=(10, 6))
        
        # Plot creation times
        plt.subplot(2, 1, 1)
        plt.plot(range(len(creation_times)), creation_times, 'b-')
        plt.xlabel('Batch Number')
        plt.ylabel('Avg Creation Time (s)')
        plt.title('Account Creation Time with Increasing Data Volume')
        
        # Plot query times
        plt.subplot(2, 1, 2)
        plt.boxplot(query_times)
        plt.ylabel('Query Time (s)')
        plt.title('Query Performance with Large Data Volume')
        
        plt.tight_layout()
        
        chart_path = os.path.join(self.output_dir, f"{test_name}_chart.png")
        plt.savefig(chart_path)
        plt.close()
    
    def _generate_long_operations_chart(self, results, test_name):
        """Generate long-running operations chart"""
        plt.figure(figsize=(10, 6))
        
        operations = list(results.keys())
        times = [results[op]["time"] for op in operations]
        
        plt.bar(operations, times)
        plt.xlabel('Operation')
        plt.ylabel('Execution Time (s)')
        plt.title('Long-Running Operations Execution Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = os.path.join(self.output_dir, f"{test_name}_chart.png")
        plt.savefig(chart_path)
        plt.close()
    
    def _generate_resource_limits_chart(self, results, test_name):
        """Generate resource limits chart"""
        plt.figure(figsize=(10, 6))
        
        tests = list(results.keys())
        success = [1 if results[test]["success"] else 0 for test in tests]
        
        plt.bar(tests, success)
        plt.xlabel('Resource Limit Test')
        plt.ylabel('Success (1) / Failure (0)')
        plt.title('System Behavior at Resource Limits')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = os.path.join(self.output_dir, f"{test_name}_chart.png")
        plt.savefig(chart_path)
        plt.close()
    
    def run_all_tests(self):
        """Run all load and stress tests"""
        test_funcs = {
            "concurrent_user_load": self.test_concurrent_user_load,
            "high_transaction_volume": self.test_high_transaction_volume,
            "data_volume_scaling": self.test_data_volume_scaling,
            "long_running_operations": self.test_long_running_operations,
            "system_resource_limits": self.test_system_resource_limits
        }
        
        results = self.framework.run_test_suite("load_stress_tests", test_funcs, "performance")
        return results


# Run tests if executed directly
if __name__ == "__main__":
    tests = LoadStressTests()
    results = tests.run_all_tests()
    
    # Generate and print report
    framework = tests.framework
    report = framework.generate_test_report("text")
    print(report["text_report"])
    
    # Export results
    framework.export_test_results("load_stress_test_results.json")
    
    # Clean up
    framework.cleanup()

