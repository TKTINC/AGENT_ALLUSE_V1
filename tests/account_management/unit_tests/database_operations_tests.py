#!/usr/bin/env python3
"""
WS3-P4: Comprehensive Testing and Validation
Database Operations Unit Tests

This module implements comprehensive unit tests for the account database operations,
validating CRUD operations, transaction management, and data integrity.

Author: Manus AI
Date: June 17, 2025
Version: 1.0
"""

import unittest
import sys
import os
import json
import sqlite3
from datetime import datetime, timedelta
import uuid

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import account management test framework
from tests.account_management.account_management_test_framework import (
    AccountManagementTestFramework, TestCategory
)

# Import database operations
from src.account_management.database.account_database import AccountDatabase
from src.account_management.models.account_models import (
    Account, GenerationAccount, RevenueAccount, CompoundingAccount,
    AccountStatus, AccountType, create_account
)

class DatabaseOperationsTests:
    """
    Comprehensive unit tests for account database operations.
    
    This class implements tests for:
    - CRUD operations
    - Transaction management
    - Query performance
    - Data integrity
    - Schema validation
    """
    
    def __init__(self):
        self.framework = AccountManagementTestFramework()
        self.db = AccountDatabase(":memory:")  # Use in-memory database for testing
        self.test_accounts = self._create_test_accounts()
    
    def _create_test_accounts(self):
        """Create test accounts for database testing"""
        accounts = {
            "generation": create_account(
                account_type=AccountType.GENERATION,
                name="Test Generation Account",
                initial_balance=100000.0,
                owner_id="user123"
            ),
            "revenue": create_account(
                account_type=AccountType.REVENUE,
                name="Test Revenue Account",
                initial_balance=75000.0,
                owner_id="user123"
            ),
            "compounding": create_account(
                account_type=AccountType.COMPOUNDING,
                name="Test Compounding Account",
                initial_balance=50000.0,
                owner_id="user123"
            )
        }
        return accounts
    
    def test_crud_operations(self):
        """Test Create, Read, Update, Delete operations"""
        try:
            crud_tests = []
            
            # Test account creation in database
            gen_account = self.test_accounts["generation"]
            self.db.save_account(gen_account)
            
            # Test account retrieval
            retrieved_account = self.db.get_account_by_id(gen_account.account_id)
            crud_tests.append(retrieved_account is not None)
            crud_tests.append(retrieved_account.account_id == gen_account.account_id)
            crud_tests.append(retrieved_account.name == gen_account.name)
            crud_tests.append(retrieved_account.initial_balance == gen_account.initial_balance)
            
            # Test account update
            new_name = "Updated Generation Account"
            gen_account.name = new_name
            self.db.update_account(gen_account)
            updated_account = self.db.get_account_by_id(gen_account.account_id)
            crud_tests.append(updated_account.name == new_name)
            
            # Test account deletion
            self.db.delete_account(gen_account.account_id)
            deleted_account = self.db.get_account_by_id(gen_account.account_id)
            crud_tests.append(deleted_account is None)
            
            # Test bulk operations
            # Create multiple accounts
            accounts = [
                self.test_accounts["revenue"],
                self.test_accounts["compounding"]
            ]
            self.db.save_accounts(accounts)
            
            # Test bulk retrieval
            account_ids = [acc.account_id for acc in accounts]
            retrieved_accounts = self.db.get_accounts_by_ids(account_ids)
            crud_tests.append(len(retrieved_accounts) == len(accounts))
            
            # Test filtering
            filtered_accounts = self.db.get_accounts_by_owner("user123")
            crud_tests.append(len(filtered_accounts) == len(accounts))
            
            # Calculate success
            success = all(crud_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "crud_operations", "value": sum(crud_tests), "target": len(crud_tests), "threshold": len(crud_tests), "passed": success}
                ],
                "crud_results": crud_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_transaction_management(self):
        """Test transaction management and concurrency"""
        try:
            transaction_tests = []
            
            # Test successful transaction
            try:
                self.db.begin_transaction()
                
                # Create account in transaction
                gen_account = self.test_accounts["generation"]
                self.db.save_account(gen_account)
                
                # Update account in same transaction
                gen_account.current_balance = 110000.0
                self.db.update_account(gen_account)
                
                self.db.commit_transaction()
                transaction_tests.append(True)
            except Exception:
                self.db.rollback_transaction()
                transaction_tests.append(False)
            
            # Verify transaction results
            retrieved_account = self.db.get_account_by_id(gen_account.account_id)
            transaction_tests.append(retrieved_account is not None)
            transaction_tests.append(retrieved_account.current_balance == 110000.0)
            
            # Test transaction rollback
            try:
                self.db.begin_transaction()
                
                # Update account
                gen_account.current_balance = 120000.0
                self.db.update_account(gen_account)
                
                # Force rollback
                self.db.rollback_transaction()
                transaction_tests.append(True)
            except Exception:
                transaction_tests.append(False)
            
            # Verify rollback (balance should still be 110000.0)
            retrieved_account = self.db.get_account_by_id(gen_account.account_id)
            transaction_tests.append(retrieved_account.current_balance == 110000.0)
            
            # Test nested transaction (should fail)
            try:
                self.db.begin_transaction()
                self.db.begin_transaction()  # Should raise exception
                self.db.rollback_transaction()
                transaction_tests.append(False)  # Should not reach here
            except Exception:
                self.db.rollback_transaction()
                transaction_tests.append(True)  # Expected exception
            
            # Calculate success
            success = all(transaction_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "transaction_management", "value": sum(transaction_tests), "target": len(transaction_tests), "threshold": len(transaction_tests), "passed": success}
                ],
                "transaction_results": transaction_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_query_performance(self):
        """Test query performance and optimization"""
        try:
            # Create a large number of test accounts for performance testing
            num_accounts = 100
            test_accounts = []
            
            for i in range(num_accounts):
                account_type = AccountType.GENERATION if i % 3 == 0 else (
                    AccountType.REVENUE if i % 3 == 1 else AccountType.COMPOUNDING
                )
                
                account = create_account(
                    account_type=account_type,
                    name=f"Performance Test Account {i}",
                    initial_balance=10000.0 + (i * 1000),
                    owner_id=f"user{i % 10}"
                )
                test_accounts.append(account)
            
            # Bulk insert accounts
            start_time = datetime.now()
            self.db.save_accounts(test_accounts)
            bulk_insert_time = (datetime.now() - start_time).total_seconds()
            
            # Test individual queries
            query_times = []
            
            # Test 1: Get account by ID
            start_time = datetime.now()
            account = self.db.get_account_by_id(test_accounts[0].account_id)
            query_times.append((datetime.now() - start_time).total_seconds())
            
            # Test 2: Get accounts by owner
            start_time = datetime.now()
            accounts = self.db.get_accounts_by_owner("user1")
            query_times.append((datetime.now() - start_time).total_seconds())
            
            # Test 3: Get accounts by type
            start_time = datetime.now()
            accounts = self.db.get_accounts_by_type(AccountType.GENERATION)
            query_times.append((datetime.now() - start_time).total_seconds())
            
            # Test 4: Get accounts by balance range
            start_time = datetime.now()
            accounts = self.db.get_accounts_by_balance_range(20000.0, 50000.0)
            query_times.append((datetime.now() - start_time).total_seconds())
            
            # Test 5: Get all accounts
            start_time = datetime.now()
            all_accounts = self.db.get_all_accounts()
            query_times.append((datetime.now() - start_time).total_seconds())
            
            # Calculate average query time
            avg_query_time = sum(query_times) / len(query_times)
            
            # Performance thresholds
            bulk_insert_threshold = 0.5  # 500ms for 100 accounts
            query_time_threshold = 0.05  # 50ms per query
            
            # Validate performance
            performance_tests = []
            performance_tests.append(bulk_insert_time < bulk_insert_threshold)
            performance_tests.append(avg_query_time < query_time_threshold)
            performance_tests.append(len(all_accounts) == num_accounts)
            
            # Calculate success
            success = all(performance_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "bulk_insert_time", "value": bulk_insert_time, "target": bulk_insert_threshold, "threshold": bulk_insert_threshold * 1.5, "passed": bulk_insert_time < bulk_insert_threshold},
                    {"name": "avg_query_time", "value": avg_query_time, "target": query_time_threshold, "threshold": query_time_threshold * 1.5, "passed": avg_query_time < query_time_threshold}
                ],
                "performance_results": {
                    "bulk_insert_time": bulk_insert_time,
                    "query_times": query_times,
                    "avg_query_time": avg_query_time,
                    "account_count": len(all_accounts)
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_data_integrity(self):
        """Test data integrity constraints and validation"""
        try:
            integrity_tests = []
            
            # Test 1: Unique account ID constraint
            gen_account = self.test_accounts["generation"]
            self.db.save_account(gen_account)
            
            # Try to save another account with same ID
            duplicate_account = create_account(
                account_type=AccountType.GENERATION,
                name="Duplicate ID Account",
                initial_balance=100000.0,
                owner_id="user123"
            )
            duplicate_account.account_id = gen_account.account_id
            
            try:
                self.db.save_account(duplicate_account)
                integrity_tests.append(False)  # Should not succeed
            except Exception:
                integrity_tests.append(True)  # Expected exception
            
            # Test 2: Foreign key constraint (parent account)
            try:
                # Create account with non-existent parent
                invalid_account = create_account(
                    account_type=AccountType.GENERATION,
                    name="Invalid Parent Account",
                    initial_balance=100000.0,
                    owner_id="user123",
                    parent_account_id="non_existent_id"
                )
                
                self.db.save_account(invalid_account)
                integrity_tests.append(False)  # Should not succeed
            except Exception:
                integrity_tests.append(True)  # Expected exception
            
            # Test 3: Data type validation
            try:
                # Create valid account
                valid_account = create_account(
                    account_type=AccountType.REVENUE,
                    name="Valid Account",
                    initial_balance=75000.0,
                    owner_id="user123"
                )
                
                # Save valid account
                self.db.save_account(valid_account)
                
                # Try to update with invalid data type
                valid_account.current_balance = "not_a_number"
                
                self.db.update_account(valid_account)
                integrity_tests.append(False)  # Should not succeed
            except (ValueError, TypeError):
                integrity_tests.append(True)  # Expected exception
            
            # Test 4: Required fields validation
            try:
                # Create account with missing required field
                incomplete_data = {
                    "account_id": str(uuid.uuid4()),
                    "name": "Incomplete Account",
                    # Missing account_type
                    "initial_balance": 50000.0,
                    "owner_id": "user123"
                }
                
                incomplete_account = Account(**incomplete_data)
                self.db.save_account(incomplete_account)
                integrity_tests.append(False)  # Should not succeed
            except Exception:
                integrity_tests.append(True)  # Expected exception
            
            # Calculate success
            success = all(integrity_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "data_integrity", "value": sum(integrity_tests), "target": len(integrity_tests), "threshold": len(integrity_tests), "passed": success}
                ],
                "integrity_results": integrity_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_schema_validation(self):
        """Test schema validation and migration"""
        try:
            schema_tests = []
            
            # Test 1: Schema initialization
            try:
                # Create a new database with schema initialization
                test_db = AccountDatabase(":memory:")
                schema_tests.append(True)
            except Exception:
                schema_tests.append(False)
            
            # Test 2: Schema validation
            try:
                # Validate schema structure
                tables = test_db.get_all_tables()
                required_tables = ["accounts", "transactions", "account_relationships"]
                
                for table in required_tables:
                    schema_tests.append(table in tables)
                
                # Validate account table structure
                account_columns = test_db.get_table_columns("accounts")
                required_columns = ["account_id", "name", "account_type", "initial_balance", 
                                   "current_balance", "creation_date", "status", "owner_id"]
                
                for column in required_columns:
                    schema_tests.append(column in account_columns)
                
            except Exception as e:
                schema_tests.append(False)
            
            # Test 3: Schema migration
            try:
                # Simulate schema migration
                test_db.execute_migration_script("""
                    ALTER TABLE accounts ADD COLUMN last_updated TEXT;
                    CREATE TABLE IF NOT EXISTS account_settings (
                        setting_id TEXT PRIMARY KEY,
                        account_id TEXT,
                        setting_name TEXT,
                        setting_value TEXT,
                        FOREIGN KEY (account_id) REFERENCES accounts (account_id)
                    );
                """)
                
                # Verify migration
                updated_tables = test_db.get_all_tables()
                schema_tests.append("account_settings" in updated_tables)
                
                updated_columns = test_db.get_table_columns("accounts")
                schema_tests.append("last_updated" in updated_columns)
                
            except Exception:
                schema_tests.append(False)
            
            # Calculate success
            success = all(schema_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "schema_validation", "value": sum(schema_tests), "target": len(schema_tests), "threshold": len(schema_tests), "passed": success}
                ],
                "schema_results": schema_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_all_tests(self):
        """Run all database operations tests"""
        test_funcs = {
            "crud_operations": self.test_crud_operations,
            "transaction_management": self.test_transaction_management,
            "query_performance": self.test_query_performance,
            "data_integrity": self.test_data_integrity,
            "schema_validation": self.test_schema_validation
        }
        
        results = self.framework.run_test_suite("database_operations", test_funcs, "database")
        return results


# Run tests if executed directly
if __name__ == "__main__":
    tests = DatabaseOperationsTests()
    results = tests.run_all_tests()
    
    # Generate and print report
    framework = tests.framework
    report = framework.generate_test_report("text")
    print(report["text_report"])
    
    # Export results
    framework.export_test_results("database_operations_test_results.json")
    
    # Clean up
    framework.cleanup()

