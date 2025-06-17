#!/usr/bin/env python3
"""
WS3-P4: Comprehensive Testing and Validation
End-to-End Workflow Tests

This module implements system-level tests for complete account management workflows,
validating end-to-end functionality across all components.

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
from src.account_management.forking.forking_protocol import ForkingProtocol
from src.account_management.merging.merging_protocol import MergingProtocol
from src.account_management.reinvestment.reinvestment_framework import ReinvestmentFramework

class EndToEndWorkflowTests:
    """
    System-level tests for complete account management workflows.
    
    This class implements tests for:
    - Account Lifecycle
    - Forking Workflow
    - Merging Workflow
    - Reinvestment Workflow
    - Administrative Workflow
    """
    
    def __init__(self):
        self.framework = AccountManagementTestFramework()
        
        # Initialize components
        self.db = AccountDatabase(":memory:")
        self.security = SecurityFramework(self.db)
        self.api = AccountOperationsAPI(self.db, self.security)
        self.analytics = AccountAnalyticsEngine(self.db)
        self.forking = ForkingProtocol(self.db, self.api)
        self.merging = MergingProtocol(self.db, self.api)
        self.reinvestment = ReinvestmentFramework(self.db, self.api)
        
        # Create test user
        self.test_user_id = "test_user_123"
        self.security.create_user(self.test_user_id, "Test User", "test@example.com", "password123")
        
        # Generate auth token for API calls
        self.auth_token = self.security.generate_auth_token(self.test_user_id)
    
    def test_account_lifecycle(self):
        """Test complete account lifecycle from creation to closure"""
        try:
            workflow_steps = []
            
            # Step 1: Create account
            account_data = {
                "name": "Lifecycle Test Account",
                "account_type": AccountType.GENERATION,
                "initial_balance": 100000.0,
                "owner_id": self.test_user_id
            }
            
            create_result = self.api.create_account(self.auth_token, account_data)
            workflow_steps.append(create_result["success"] is True)
            
            account_id = create_result["account_id"]
            
            # Step 2: Configure account
            config_data = {
                "target_delta_range": (42, 48),
                "cash_buffer_percent": 6.0
            }
            
            config_result = self.api.configure_account(self.auth_token, account_id, config_data)
            workflow_steps.append(config_result["success"] is True)
            
            # Step 3: Add transactions
            transactions = [
                {"amount": 5000.0, "transaction_type": "profit", "description": "Week 1 profit"},
                {"amount": 4500.0, "transaction_type": "profit", "description": "Week 2 profit"},
                {"amount": -2000.0, "transaction_type": "loss", "description": "Week 3 loss"},
                {"amount": 6000.0, "transaction_type": "profit", "description": "Week 4 profit"}
            ]
            
            for tx in transactions:
                tx_result = self.api.add_transaction(self.auth_token, account_id, tx)
                workflow_steps.append(tx_result["success"] is True)
            
            # Step 4: Generate analytics
            analytics_result = self.api.generate_account_analytics(self.auth_token, account_id)
            workflow_steps.append(analytics_result["success"] is True)
            workflow_steps.append("performance" in analytics_result)
            workflow_steps.append("risk" in analytics_result)
            
            # Step 5: Suspend account
            suspend_result = self.api.update_account_status(self.auth_token, account_id, AccountStatus.SUSPENDED)
            workflow_steps.append(suspend_result["success"] is True)
            
            # Verify account status
            account_result = self.api.get_account(self.auth_token, account_id)
            workflow_steps.append(account_result["account"]["status"] == AccountStatus.SUSPENDED)
            
            # Step 6: Reactivate account
            reactivate_result = self.api.update_account_status(self.auth_token, account_id, AccountStatus.ACTIVE)
            workflow_steps.append(reactivate_result["success"] is True)
            
            # Verify account status
            account_result = self.api.get_account(self.auth_token, account_id)
            workflow_steps.append(account_result["account"]["status"] == AccountStatus.ACTIVE)
            
            # Step 7: Close account
            close_result = self.api.update_account_status(self.auth_token, account_id, AccountStatus.CLOSED)
            workflow_steps.append(close_result["success"] is True)
            
            # Verify account status
            account_result = self.api.get_account(self.auth_token, account_id)
            workflow_steps.append(account_result["account"]["status"] == AccountStatus.CLOSED)
            
            # Calculate success
            success = all(workflow_steps)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "account_lifecycle_workflow", "value": sum(workflow_steps), "target": len(workflow_steps), "threshold": len(workflow_steps), "passed": success}
                ],
                "workflow_results": workflow_steps
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_forking_workflow(self):
        """Test complete forking workflow"""
        try:
            workflow_steps = []
            
            # Step 1: Create parent account structure
            # Generation Account
            gen_data = {
                "name": "Parent Generation Account",
                "account_type": AccountType.GENERATION,
                "initial_balance": 100000.0,
                "owner_id": self.test_user_id
            }
            
            gen_result = self.api.create_account(self.auth_token, gen_data)
            workflow_steps.append(gen_result["success"] is True)
            gen_account_id = gen_result["account_id"]
            
            # Revenue Account
            rev_data = {
                "name": "Parent Revenue Account",
                "account_type": AccountType.REVENUE,
                "initial_balance": 75000.0,
                "owner_id": self.test_user_id
            }
            
            rev_result = self.api.create_account(self.auth_token, rev_data)
            workflow_steps.append(rev_result["success"] is True)
            rev_account_id = rev_result["account_id"]
            
            # Compounding Account
            comp_data = {
                "name": "Parent Compounding Account",
                "account_type": AccountType.COMPOUNDING,
                "initial_balance": 50000.0,
                "owner_id": self.test_user_id
            }
            
            comp_result = self.api.create_account(self.auth_token, comp_data)
            workflow_steps.append(comp_result["success"] is True)
            comp_account_id = comp_result["account_id"]
            
            # Step 2: Add profits to Generation Account to reach forking threshold
            # Add $50,000 profit to reach forking threshold
            profit_tx = {
                "amount": 50000.0,
                "transaction_type": "profit",
                "description": "Accumulated profit for forking"
            }
            
            tx_result = self.api.add_transaction(self.auth_token, gen_account_id, profit_tx)
            workflow_steps.append(tx_result["success"] is True)
            
            # Step 3: Check forking eligibility
            eligibility_result = self.forking.check_forking_eligibility(gen_account_id)
            workflow_steps.append(eligibility_result["eligible"] is True)
            workflow_steps.append(eligibility_result["surplus"] >= 50000.0)
            
            # Step 4: Execute forking
            forking_result = self.forking.execute_forking(gen_account_id)
            workflow_steps.append(forking_result["success"] is True)
            workflow_steps.append("forked_accounts" in forking_result)
            
            forked_gen_id = forking_result["forked_accounts"]["generation"]
            forked_comp_id = forking_result["forked_accounts"]["compounding"]
            
            # Step 5: Verify forked accounts
            # Verify forked Generation Account
            forked_gen = self.api.get_account(self.auth_token, forked_gen_id)
            workflow_steps.append(forked_gen["success"] is True)
            workflow_steps.append(forked_gen["account"]["initial_balance"] == 25000.0)
            workflow_steps.append(forked_gen["account"]["parent_account_id"] == gen_account_id)
            
            # Verify forked Compounding Account
            forked_comp = self.api.get_account(self.auth_token, forked_comp_id)
            workflow_steps.append(forked_comp["success"] is True)
            workflow_steps.append(forked_comp["account"]["initial_balance"] == 25000.0)
            workflow_steps.append(forked_comp["account"]["parent_account_id"] == gen_account_id)
            
            # Step 6: Verify parent account after forking
            parent_after = self.api.get_account(self.auth_token, gen_account_id)
            workflow_steps.append(parent_after["success"] is True)
            workflow_steps.append(parent_after["account"]["current_balance"] == 125000.0)  # Original + $25K (half of surplus)
            
            # Step 7: Verify forking records
            forking_records = self.forking.get_forking_history(gen_account_id)
            workflow_steps.append(len(forking_records) > 0)
            workflow_steps.append(forking_records[0]["parent_account_id"] == gen_account_id)
            workflow_steps.append(forking_records[0]["forked_amount"] == 50000.0)
            
            # Calculate success
            success = all(workflow_steps)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "forking_workflow", "value": sum(workflow_steps), "target": len(workflow_steps), "threshold": len(workflow_steps), "passed": success}
                ],
                "workflow_results": workflow_steps
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_merging_workflow(self):
        """Test complete merging workflow"""
        try:
            workflow_steps = []
            
            # Step 1: Create parent account structure
            # Compounding Account (target for merging)
            comp_data = {
                "name": "Parent Compounding Account",
                "account_type": AccountType.COMPOUNDING,
                "initial_balance": 400000.0,
                "owner_id": self.test_user_id
            }
            
            comp_result = self.api.create_account(self.auth_token, comp_data)
            workflow_steps.append(comp_result["success"] is True)
            comp_account_id = comp_result["account_id"]
            
            # Step 2: Create forked account structure
            # Generation Account
            gen_data = {
                "name": "Forked Generation Account",
                "account_type": AccountType.GENERATION,
                "initial_balance": 200000.0,
                "owner_id": self.test_user_id,
                "parent_account_id": comp_account_id
            }
            
            gen_result = self.api.create_account(self.auth_token, gen_data)
            workflow_steps.append(gen_result["success"] is True)
            gen_account_id = gen_result["account_id"]
            
            # Revenue Account
            rev_data = {
                "name": "Forked Revenue Account",
                "account_type": AccountType.REVENUE,
                "initial_balance": 150000.0,
                "owner_id": self.test_user_id,
                "parent_account_id": comp_account_id
            }
            
            rev_result = self.api.create_account(self.auth_token, rev_data)
            workflow_steps.append(rev_result["success"] is True)
            rev_account_id = rev_result["account_id"]
            
            # Compounding Account
            forked_comp_data = {
                "name": "Forked Compounding Account",
                "account_type": AccountType.COMPOUNDING,
                "initial_balance": 150000.0,
                "owner_id": self.test_user_id,
                "parent_account_id": comp_account_id
            }
            
            forked_comp_result = self.api.create_account(self.auth_token, forked_comp_data)
            workflow_steps.append(forked_comp_result["success"] is True)
            forked_comp_account_id = forked_comp_result["account_id"]
            
            # Step 3: Check merging eligibility
            # Create a forked structure record
            forked_structure = {
                "parent_account_id": comp_account_id,
                "generation_account_id": gen_account_id,
                "revenue_account_id": rev_account_id,
                "compounding_account_id": forked_comp_account_id,
                "creation_date": datetime.now().isoformat(),
                "total_value": 500000.0
            }
            
            self.forking.record_forked_structure(forked_structure)
            
            # Check eligibility
            eligibility_result = self.merging.check_merging_eligibility(forked_structure["generation_account_id"])
            workflow_steps.append(eligibility_result["eligible"] is True)
            workflow_steps.append(eligibility_result["total_value"] >= 500000.0)
            
            # Step 4: Execute merging
            merging_result = self.merging.execute_merging(forked_structure["generation_account_id"])
            workflow_steps.append(merging_result["success"] is True)
            workflow_steps.append(merging_result["target_account_id"] == comp_account_id)
            workflow_steps.append(merging_result["merged_amount"] > 0)
            
            # Step 5: Verify target account after merging
            target_after = self.api.get_account(self.auth_token, comp_account_id)
            workflow_steps.append(target_after["success"] is True)
            workflow_steps.append(target_after["account"]["current_balance"] >= 900000.0)  # Original + merged accounts
            
            # Step 6: Verify merged accounts status
            # Generation Account should be closed
            gen_after = self.api.get_account(self.auth_token, gen_account_id)
            workflow_steps.append(gen_after["account"]["status"] == AccountStatus.CLOSED)
            
            # Revenue Account should be closed
            rev_after = self.api.get_account(self.auth_token, rev_account_id)
            workflow_steps.append(rev_after["account"]["status"] == AccountStatus.CLOSED)
            
            # Compounding Account should be closed
            comp_after = self.api.get_account(self.auth_token, forked_comp_account_id)
            workflow_steps.append(comp_after["account"]["status"] == AccountStatus.CLOSED)
            
            # Step 7: Verify merging records
            merging_records = self.merging.get_merging_history(comp_account_id)
            workflow_steps.append(len(merging_records) > 0)
            workflow_steps.append(merging_records[0]["target_account_id"] == comp_account_id)
            workflow_steps.append(merging_records[0]["merged_amount"] > 0)
            
            # Calculate success
            success = all(workflow_steps)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "merging_workflow", "value": sum(workflow_steps), "target": len(workflow_steps), "threshold": len(workflow_steps), "passed": success}
                ],
                "workflow_results": workflow_steps
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_reinvestment_workflow(self):
        """Test complete reinvestment workflow"""
        try:
            workflow_steps = []
            
            # Step 1: Create Revenue Account
            rev_data = {
                "name": "Reinvestment Test Account",
                "account_type": AccountType.REVENUE,
                "initial_balance": 100000.0,
                "owner_id": self.test_user_id
            }
            
            rev_result = self.api.create_account(self.auth_token, rev_data)
            workflow_steps.append(rev_result["success"] is True)
            rev_account_id = rev_result["account_id"]
            
            # Step 2: Configure reinvestment settings
            config_data = {
                "reinvestment_schedule": "quarterly",
                "reinvestment_allocation": {
                    "contracts": 75,
                    "leaps": 25
                }
            }
            
            config_result = self.api.configure_account(self.auth_token, rev_account_id, config_data)
            workflow_steps.append(config_result["success"] is True)
            
            # Step 3: Add profits
            profit_tx = {
                "amount": 10000.0,
                "transaction_type": "profit",
                "description": "Quarterly profit"
            }
            
            tx_result = self.api.add_transaction(self.auth_token, rev_account_id, profit_tx)
            workflow_steps.append(tx_result["success"] is True)
            
            # Step 4: Check reinvestment eligibility
            eligibility_result = self.reinvestment.check_reinvestment_eligibility(rev_account_id)
            workflow_steps.append(eligibility_result["eligible"] is True)
            workflow_steps.append(eligibility_result["available_amount"] >= 10000.0)
            
            # Step 5: Calculate reinvestment amount
            calculation_result = self.reinvestment.calculate_reinvestment_amount(rev_account_id)
            workflow_steps.append(calculation_result["success"] is True)
            workflow_steps.append(calculation_result["total_amount"] >= 7500.0)  # 75% of profits
            workflow_steps.append("allocations" in calculation_result)
            workflow_steps.append(calculation_result["allocations"]["contracts"] >= 5625.0)  # 75% of reinvestment
            workflow_steps.append(calculation_result["allocations"]["leaps"] >= 1875.0)  # 25% of reinvestment
            
            # Step 6: Execute reinvestment
            reinvestment_result = self.reinvestment.execute_reinvestment(rev_account_id)
            workflow_steps.append(reinvestment_result["success"] is True)
            workflow_steps.append(reinvestment_result["reinvested_amount"] >= 7500.0)
            
            # Step 7: Verify account after reinvestment
            account_after = self.api.get_account(self.auth_token, rev_account_id)
            workflow_steps.append(account_after["success"] is True)
            
            # Balance should reflect reinvestment (original + profit - reinvested)
            expected_balance = 100000.0 + 10000.0 - 7500.0
            workflow_steps.append(abs(account_after["account"]["current_balance"] - expected_balance) < 0.01)
            
            # Step 8: Verify reinvestment records
            reinvestment_records = self.reinvestment.get_reinvestment_history(rev_account_id)
            workflow_steps.append(len(reinvestment_records) > 0)
            workflow_steps.append(reinvestment_records[0]["account_id"] == rev_account_id)
            workflow_steps.append(reinvestment_records[0]["reinvested_amount"] >= 7500.0)
            
            # Calculate success
            success = all(workflow_steps)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "reinvestment_workflow", "value": sum(workflow_steps), "target": len(workflow_steps), "threshold": len(workflow_steps), "passed": success}
                ],
                "workflow_results": workflow_steps
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_administrative_workflow(self):
        """Test administrative workflow"""
        try:
            workflow_steps = []
            
            # Step 1: Create admin user
            admin_user_id = "admin_user_456"
            self.security.create_user(admin_user_id, "Admin User", "admin@example.com", "adminpass123")
            self.security.assign_role(admin_user_id, "admin")
            admin_token = self.security.generate_auth_token(admin_user_id)
            
            workflow_steps.append(admin_token is not None)
            
            # Step 2: Create multiple test accounts
            accounts = []
            for i in range(5):
                account_data = {
                    "name": f"Admin Test Account {i}",
                    "account_type": AccountType.GENERATION,
                    "initial_balance": 100000.0 + (i * 10000),
                    "owner_id": self.test_user_id
                }
                
                result = self.api.create_account(self.auth_token, account_data)
                workflow_steps.append(result["success"] is True)
                accounts.append(result["account_id"])
            
            # Step 3: Perform bulk status update
            bulk_status_data = {
                "account_ids": accounts,
                "status": AccountStatus.SUSPENDED,
                "reason": "Administrative action"
            }
            
            bulk_result = self.api.bulk_update_account_status(admin_token, bulk_status_data)
            workflow_steps.append(bulk_result["success"] is True)
            workflow_steps.append(bulk_result["updated_count"] == len(accounts))
            
            # Step 4: Verify accounts after bulk update
            for account_id in accounts:
                account = self.api.get_account(admin_token, account_id)
                workflow_steps.append(account["account"]["status"] == AccountStatus.SUSPENDED)
            
            # Step 5: Generate administrative report
            report_result = self.api.generate_administrative_report(admin_token, {
                "report_type": "account_status",
                "filters": {
                    "status": AccountStatus.SUSPENDED,
                    "owner_id": self.test_user_id
                }
            })
            
            workflow_steps.append(report_result["success"] is True)
            workflow_steps.append(len(report_result["accounts"]) >= len(accounts))
            
            # Step 6: Perform bulk configuration update
            bulk_config_data = {
                "account_ids": accounts,
                "configuration": {
                    "cash_buffer_percent": 7.5
                }
            }
            
            config_result = self.api.bulk_update_account_configuration(admin_token, bulk_config_data)
            workflow_steps.append(config_result["success"] is True)
            workflow_steps.append(config_result["updated_count"] == len(accounts))
            
            # Step 7: Verify audit trail
            audit_logs = self.security.get_admin_audit_logs(admin_user_id)
            workflow_steps.append(len(audit_logs) >= 2)  # At least bulk status and config updates
            
            # Calculate success
            success = all(workflow_steps)
            
            return {
                "success": success,
                "metrics": [
                    {"name": "administrative_workflow", "value": sum(workflow_steps), "target": len(workflow_steps), "threshold": len(workflow_steps), "passed": success}
                ],
                "workflow_results": workflow_steps
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_all_tests(self):
        """Run all end-to-end workflow tests"""
        test_funcs = {
            "account_lifecycle": self.test_account_lifecycle,
            "forking_workflow": self.test_forking_workflow,
            "merging_workflow": self.test_merging_workflow,
            "reinvestment_workflow": self.test_reinvestment_workflow,
            "administrative_workflow": self.test_administrative_workflow
        }
        
        results = self.framework.run_test_suite("end_to_end_workflows", test_funcs, "system")
        return results


# Run tests if executed directly
if __name__ == "__main__":
    tests = EndToEndWorkflowTests()
    results = tests.run_all_tests()
    
    # Generate and print report
    framework = tests.framework
    report = framework.generate_test_report("text")
    print(report["text_report"])
    
    # Export results
    framework.export_test_results("end_to_end_workflow_test_results.json")
    
    # Clean up
    framework.cleanup()

