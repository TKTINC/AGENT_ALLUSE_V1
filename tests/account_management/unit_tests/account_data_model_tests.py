#!/usr/bin/env python3
"""
WS3-P4: Comprehensive Testing and Validation
Account Data Model Unit Tests

This module implements comprehensive unit tests for the account data models,
validating all account types, properties, and behaviors.

Author: Manus AI
Date: June 17, 2025
Version: 1.0
"""

import unittest
import sys
import os
import json
from datetime import datetime, timedelta
import uuid

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import account management test framework
from tests.account_management.account_management_test_framework import (
    AccountManagementTestFramework, TestCategory
)

# Import account models
from src.account_management.models.account_models import (
    Account, GenerationAccount, RevenueAccount, CompoundingAccount,
    AccountStatus, AccountType, create_account
)

class AccountDataModelTests:
    """
    Comprehensive unit tests for account data models.
    
    This class implements tests for:
    - Account creation and validation
    - Account type-specific behaviors
    - Account relationships
    - Account state management
    """
    
    def __init__(self):
        self.framework = AccountManagementTestFramework()
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self):
        """Generate test data for account model testing"""
        return {
            "generation_account": {
                "account_id": str(uuid.uuid4()),
                "name": "Test Generation Account",
                "account_type": AccountType.GENERATION,
                "initial_balance": 100000.0,
                "current_balance": 105000.0,
                "creation_date": datetime.now().isoformat(),
                "status": AccountStatus.ACTIVE,
                "owner_id": "user123",
                "target_delta_range": (40, 50),
                "cash_buffer_percent": 5.0,
                "parent_account_id": None
            },
            "revenue_account": {
                "account_id": str(uuid.uuid4()),
                "name": "Test Revenue Account",
                "account_type": AccountType.REVENUE,
                "initial_balance": 75000.0,
                "current_balance": 78000.0,
                "creation_date": datetime.now().isoformat(),
                "status": AccountStatus.ACTIVE,
                "owner_id": "user123",
                "target_delta_range": (30, 40),
                "reinvestment_schedule": "quarterly",
                "parent_account_id": None
            },
            "compounding_account": {
                "account_id": str(uuid.uuid4()),
                "name": "Test Compounding Account",
                "account_type": AccountType.COMPOUNDING,
                "initial_balance": 50000.0,
                "current_balance": 51500.0,
                "creation_date": datetime.now().isoformat(),
                "status": AccountStatus.ACTIVE,
                "owner_id": "user123",
                "target_delta_range": (20, 30),
                "withdrawal_policy": "no_withdrawal",
                "parent_account_id": None
            }
        }
    
    def test_account_creation(self):
        """Test account creation for all account types"""
        try:
            # Test Generation Account creation
            gen_data = self.test_data["generation_account"]
            gen_account = create_account(
                account_type=AccountType.GENERATION,
                name=gen_data["name"],
                initial_balance=gen_data["initial_balance"],
                owner_id=gen_data["owner_id"]
            )
            
            # Test Revenue Account creation
            rev_data = self.test_data["revenue_account"]
            rev_account = create_account(
                account_type=AccountType.REVENUE,
                name=rev_data["name"],
                initial_balance=rev_data["initial_balance"],
                owner_id=rev_data["owner_id"]
            )
            
            # Test Compounding Account creation
            comp_data = self.test_data["compounding_account"]
            comp_account = create_account(
                account_type=AccountType.COMPOUNDING,
                name=comp_data["name"],
                initial_balance=comp_data["initial_balance"],
                owner_id=comp_data["owner_id"]
            )
            
            # Validate account types
            type_validation = (
                isinstance(gen_account, GenerationAccount) and
                isinstance(rev_account, RevenueAccount) and
                isinstance(comp_account, CompoundingAccount)
            )
            
            # Validate account properties
            property_validation = (
                gen_account.name == gen_data["name"] and
                gen_account.initial_balance == gen_data["initial_balance"] and
                gen_account.owner_id == gen_data["owner_id"] and
                rev_account.name == rev_data["name"] and
                rev_account.initial_balance == rev_data["initial_balance"] and
                rev_account.owner_id == rev_data["owner_id"] and
                comp_account.name == comp_data["name"] and
                comp_account.initial_balance == comp_data["initial_balance"] and
                comp_account.owner_id == comp_data["owner_id"]
            )
            
            # Validate default values
            default_validation = (
                gen_account.status == AccountStatus.ACTIVE and
                rev_account.status == AccountStatus.ACTIVE and
                comp_account.status == AccountStatus.ACTIVE and
                gen_account.current_balance == gen_data["initial_balance"] and
                rev_account.current_balance == rev_data["initial_balance"] and
                comp_account.current_balance == comp_data["initial_balance"]
            )
            
            success = type_validation and property_validation and default_validation
            
            return {
                "success": success,
                "metrics": [
                    {"name": "type_validation", "value": 1 if type_validation else 0, "target": 1, "threshold": 1, "passed": type_validation},
                    {"name": "property_validation", "value": 1 if property_validation else 0, "target": 1, "threshold": 1, "passed": property_validation},
                    {"name": "default_validation", "value": 1 if default_validation else 0, "target": 1, "threshold": 1, "passed": default_validation}
                ],
                "accounts": {
                    "generation": gen_account.to_dict() if hasattr(gen_account, 'to_dict') else str(gen_account),
                    "revenue": rev_account.to_dict() if hasattr(rev_account, 'to_dict') else str(rev_account),
                    "compounding": comp_account.to_dict() if hasattr(comp_account, 'to_dict') else str(comp_account)
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_account_validation(self):
        """Test account validation rules"""
        try:
            validation_tests = []
            
            # Test 1: Invalid initial balance (negative)
            try:
                invalid_account = create_account(
                    account_type=AccountType.GENERATION,
                    name="Invalid Balance Account",
                    initial_balance=-1000.0,
                    owner_id="user123"
                )
                validation_tests.append(False)  # Should not reach here
            except ValueError:
                validation_tests.append(True)  # Expected exception
            
            # Test 2: Invalid name (empty)
            try:
                invalid_account = create_account(
                    account_type=AccountType.GENERATION,
                    name="",
                    initial_balance=10000.0,
                    owner_id="user123"
                )
                validation_tests.append(False)  # Should not reach here
            except ValueError:
                validation_tests.append(True)  # Expected exception
            
            # Test 3: Invalid owner_id (empty)
            try:
                invalid_account = create_account(
                    account_type=AccountType.GENERATION,
                    name="Valid Name",
                    initial_balance=10000.0,
                    owner_id=""
                )
                validation_tests.append(False)  # Should not reach here
            except ValueError:
                validation_tests.append(True)  # Expected exception
            
            # Test 4: Invalid account type
            try:
                invalid_account = create_account(
                    account_type="INVALID_TYPE",
                    name="Valid Name",
                    initial_balance=10000.0,
                    owner_id="user123"
                )
                validation_tests.append(False)  # Should not reach here
            except (ValueError, TypeError):
                validation_tests.append(True)  # Expected exception
            
            # Test 5: Valid account creation
            try:
                valid_account = create_account(
                    account_type=AccountType.GENERATION,
                    name="Valid Account",
                    initial_balance=10000.0,
                    owner_id="user123"
                )
                validation_tests.append(True)  # Expected success
            except Exception:
                validation_tests.append(False)  # Should not reach here
            
            success = all(validation_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "validation_tests_passed", "value": sum(validation_tests), "target": len(validation_tests), "threshold": len(validation_tests), "passed": success}
                ],
                "validation_results": validation_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_account_type_behavior(self):
        """Test account type-specific behaviors"""
        try:
            # Create accounts of each type
            gen_account = create_account(
                account_type=AccountType.GENERATION,
                name="Generation Test",
                initial_balance=100000.0,
                owner_id="user123"
            )
            
            rev_account = create_account(
                account_type=AccountType.REVENUE,
                name="Revenue Test",
                initial_balance=75000.0,
                owner_id="user123"
            )
            
            comp_account = create_account(
                account_type=AccountType.COMPOUNDING,
                name="Compounding Test",
                initial_balance=50000.0,
                owner_id="user123"
            )
            
            # Test Generation Account specific behaviors
            gen_tests = []
            
            # Test delta range
            gen_tests.append(gen_account.target_delta_range == (40, 50))
            
            # Test cash buffer
            gen_tests.append(gen_account.cash_buffer_percent == 5.0)
            
            # Test forking threshold calculation
            gen_tests.append(gen_account.calculate_forking_threshold() == 50000.0)
            
            # Test Revenue Account specific behaviors
            rev_tests = []
            
            # Test delta range
            rev_tests.append(rev_account.target_delta_range == (30, 40))
            
            # Test reinvestment schedule
            rev_tests.append(rev_account.reinvestment_schedule == "quarterly")
            
            # Test reinvestment calculation
            rev_tests.append(rev_account.calculate_reinvestment_amount(10000.0) == 7500.0)
            
            # Test Compounding Account specific behaviors
            comp_tests = []
            
            # Test delta range
            comp_tests.append(comp_account.target_delta_range == (20, 30))
            
            # Test withdrawal policy
            comp_tests.append(comp_account.withdrawal_policy == "no_withdrawal")
            
            # Test withdrawal validation
            try:
                comp_account.validate_withdrawal(1000.0)
                comp_tests.append(False)  # Should not allow withdrawals
            except ValueError:
                comp_tests.append(True)  # Expected exception
            
            # Calculate overall success
            gen_success = all(gen_tests)
            rev_success = all(rev_tests)
            comp_success = all(comp_tests)
            overall_success = gen_success and rev_success and comp_success
            
            return {
                "success": overall_success,
                "metrics": [
                    {"name": "generation_behavior", "value": sum(gen_tests), "target": len(gen_tests), "threshold": len(gen_tests), "passed": gen_success},
                    {"name": "revenue_behavior", "value": sum(rev_tests), "target": len(rev_tests), "threshold": len(rev_tests), "passed": rev_success},
                    {"name": "compounding_behavior", "value": sum(comp_tests), "target": len(comp_tests), "threshold": len(comp_tests), "passed": comp_success}
                ],
                "behavior_results": {
                    "generation": gen_tests,
                    "revenue": rev_tests,
                    "compounding": comp_tests
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_account_relationships(self):
        """Test account relationships (parent-child, forking)"""
        try:
            # Create parent account
            parent_account = create_account(
                account_type=AccountType.GENERATION,
                name="Parent Account",
                initial_balance=100000.0,
                owner_id="user123"
            )
            
            # Create child accounts
            child_gen_account = create_account(
                account_type=AccountType.GENERATION,
                name="Child Generation Account",
                initial_balance=25000.0,
                owner_id="user123",
                parent_account_id=parent_account.account_id
            )
            
            child_comp_account = create_account(
                account_type=AccountType.COMPOUNDING,
                name="Child Compounding Account",
                initial_balance=25000.0,
                owner_id="user123",
                parent_account_id=parent_account.account_id
            )
            
            # Test relationship establishment
            relationship_tests = []
            
            # Test child accounts have correct parent ID
            relationship_tests.append(child_gen_account.parent_account_id == parent_account.account_id)
            relationship_tests.append(child_comp_account.parent_account_id == parent_account.account_id)
            
            # Test forking relationship
            # Simulate forking by creating a forked account structure
            forked_structure = parent_account.create_forked_structure(50000.0)
            
            # Validate forked structure
            fork_tests = []
            fork_tests.append(isinstance(forked_structure, dict))
            fork_tests.append("generation" in forked_structure)
            fork_tests.append("compounding" in forked_structure)
            fork_tests.append(forked_structure["generation"].initial_balance == 25000.0)
            fork_tests.append(forked_structure["compounding"].initial_balance == 25000.0)
            fork_tests.append(forked_structure["generation"].parent_account_id == parent_account.account_id)
            
            # Calculate overall success
            relationship_success = all(relationship_tests)
            fork_success = all(fork_tests)
            overall_success = relationship_success and fork_success
            
            return {
                "success": overall_success,
                "metrics": [
                    {"name": "relationship_tests", "value": sum(relationship_tests), "target": len(relationship_tests), "threshold": len(relationship_tests), "passed": relationship_success},
                    {"name": "fork_tests", "value": sum(fork_tests), "target": len(fork_tests), "threshold": len(fork_tests), "passed": fork_success}
                ],
                "relationship_results": {
                    "basic_relationships": relationship_tests,
                    "forking_relationships": fork_tests
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_account_state_management(self):
        """Test account state transitions and management"""
        try:
            # Create test account
            test_account = create_account(
                account_type=AccountType.GENERATION,
                name="State Test Account",
                initial_balance=100000.0,
                owner_id="user123"
            )
            
            # Test initial state
            state_tests = []
            state_tests.append(test_account.status == AccountStatus.ACTIVE)
            
            # Test suspend operation
            test_account.suspend("Testing suspension")
            state_tests.append(test_account.status == AccountStatus.SUSPENDED)
            
            # Test reactivate operation
            test_account.reactivate()
            state_tests.append(test_account.status == AccountStatus.ACTIVE)
            
            # Test close operation
            test_account.close("Testing closure")
            state_tests.append(test_account.status == AccountStatus.CLOSED)
            
            # Test operations on closed account
            try:
                test_account.update_balance(5000.0)
                state_tests.append(False)  # Should not allow operations on closed account
            except ValueError:
                state_tests.append(True)  # Expected exception
            
            # Test invalid state transition
            try:
                # Cannot reactivate directly from closed
                test_account.reactivate()
                state_tests.append(False)  # Should not allow invalid transition
            except ValueError:
                state_tests.append(True)  # Expected exception
            
            # Calculate overall success
            success = all(state_tests)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "state_transitions", "value": sum(state_tests), "target": len(state_tests), "threshold": len(state_tests), "passed": success}
                ],
                "state_results": state_tests
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_all_tests(self):
        """Run all account data model tests"""
        test_funcs = {
            "account_creation": self.test_account_creation,
            "account_validation": self.test_account_validation,
            "account_type_behavior": self.test_account_type_behavior,
            "account_relationships": self.test_account_relationships,
            "account_state_management": self.test_account_state_management
        }
        
        results = self.framework.run_test_suite("account_data_model", test_funcs, "account_model")
        return results


# Run tests if executed directly
if __name__ == "__main__":
    tests = AccountDataModelTests()
    results = tests.run_all_tests()
    
    # Generate and print report
    framework = tests.framework
    report = framework.generate_test_report("text")
    print(report["text_report"])
    
    # Export results
    framework.export_test_results("account_data_model_test_results.json")
    
    # Clean up
    framework.cleanup()

