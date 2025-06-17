#!/usr/bin/env python3
"""
ALL-USE Account Management System - System Testing Framework

This module provides a comprehensive framework for system testing of the
ALL-USE Account Management System, enabling thorough validation of system
functionality, performance, security, and reliability.

The system testing framework implements:
- Test case management
- Test execution automation
- Test result collection and analysis
- Test reporting
- Test data management

Author: Manus AI
Date: June 17, 2025
Version: 1.0
"""

import logging
import time
import json
import threading
import datetime
import uuid
import os
import csv
import statistics
from enum import Enum
from typing import Dict, List, Any, Callable, Optional, Union, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Enumeration of test status values."""
    PENDING = "pending"      # Test has not been executed yet
    RUNNING = "running"      # Test is currently running
    PASSED = "passed"        # Test executed successfully and passed
    FAILED = "failed"        # Test executed but failed
    ERROR = "error"          # Test encountered an error during execution
    SKIPPED = "skipped"      # Test was skipped
    BLOCKED = "blocked"      # Test was blocked by a dependency


class TestSeverity(Enum):
    """Enumeration of test severity levels."""
    CRITICAL = "critical"    # Critical functionality
    HIGH = "high"            # Important functionality
    MEDIUM = "medium"        # Standard functionality
    LOW = "low"              # Minor functionality


class TestCategory(Enum):
    """Enumeration of test categories."""
    FUNCTIONAL = "functional"        # Functional tests
    PERFORMANCE = "performance"      # Performance tests
    SECURITY = "security"            # Security tests
    RELIABILITY = "reliability"      # Reliability tests
    USABILITY = "usability"          # Usability tests
    INTEGRATION = "integration"      # Integration tests


class TestCase:
    """
    Represents a test case in the system testing framework.
    
    A test case defines a specific test to be executed, including
    its setup, execution, and validation steps.
    """
    
    def __init__(
        self,
        test_id: str,
        name: str,
        description: str,
        category: TestCategory,
        severity: TestSeverity,
        test_function: Callable,
        setup_function: Callable = None,
        teardown_function: Callable = None,
        dependencies: List[str] = None,
        tags: List[str] = None,
        timeout_seconds: int = 60
    ):
        """
        Initialize a test case.
        
        Args:
            test_id: Unique identifier for the test case
            name: Human-readable name for the test case
            description: Description of what the test case validates
            category: Category of the test case
            severity: Severity level of the test case
            test_function: Function that executes the test
            setup_function: Function to set up the test environment (optional)
            teardown_function: Function to clean up after the test (optional)
            dependencies: List of test IDs that must pass before this test can run
            tags: List of tags for categorizing and filtering tests
            timeout_seconds: Maximum time the test is allowed to run
        """
        self.test_id = test_id
        self.name = name
        self.description = description
        self.category = category
        self.severity = severity
        self.test_function = test_function
        self.setup_function = setup_function
        self.teardown_function = teardown_function
        self.dependencies = dependencies or []
        self.tags = tags or []
        self.timeout_seconds = timeout_seconds
        self.status = TestStatus.PENDING
        self.result = None
        self.error_message = None
        self.execution_time = None
        self.start_time = None
        self.end_time = None
        self.metrics = {}
    
    def execute(self, test_context: Dict[str, Any] = None) -> bool:
        """
        Execute the test case.
        
        Args:
            test_context: Context data for the test execution
            
        Returns:
            True if the test passed, False otherwise
        """
        if not test_context:
            test_context = {}
        
        self.status = TestStatus.RUNNING
        self.start_time = datetime.datetime.now()
        
        try:
            # Run setup if provided
            if self.setup_function:
                self.setup_function(test_context)
            
            # Run the test with timeout
            start_execution = time.time()
            self.result = self._run_with_timeout(test_context)
            end_execution = time.time()
            
            # Calculate execution time
            self.execution_time = end_execution - start_execution
            
            # Update status based on result
            self.status = TestStatus.PASSED if self.result else TestStatus.FAILED
            
            return self.result
        except Exception as e:
            self.status = TestStatus.ERROR
            self.error_message = str(e)
            logger.error(f"Error executing test {self.test_id}: {str(e)}")
            return False
        finally:
            # Run teardown if provided, even if test failed
            if self.teardown_function:
                try:
                    self.teardown_function(test_context)
                except Exception as e:
                    logger.error(f"Error in teardown for test {self.test_id}: {str(e)}")
            
            self.end_time = datetime.datetime.now()
    
    def _run_with_timeout(self, test_context: Dict[str, Any]) -> bool:
        """
        Run the test function with a timeout.
        
        Args:
            test_context: Context data for the test execution
            
        Returns:
            The result of the test function
            
        Raises:
            TimeoutError: If the test exceeds its timeout
        """
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = self.test_function(test_context)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(self.timeout_seconds)
        
        if thread.is_alive():
            # Test timed out
            raise TimeoutError(f"Test {self.test_id} timed out after {self.timeout_seconds} seconds")
        
        if exception[0]:
            # Re-raise any exception from the test
            raise exception[0]
        
        return result[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the test case to a dictionary."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity.value,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "timeout_seconds": self.timeout_seconds,
            "status": self.status.value,
            "result": self.result,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metrics": self.metrics
        }


class TestSuite:
    """
    Represents a collection of related test cases.
    
    A test suite groups related test cases and provides methods
    for executing them as a batch.
    """
    
    def __init__(
        self,
        suite_id: str,
        name: str,
        description: str,
        test_cases: List[TestCase] = None
    ):
        """
        Initialize a test suite.
        
        Args:
            suite_id: Unique identifier for the test suite
            name: Human-readable name for the test suite
            description: Description of what the test suite validates
            test_cases: List of test cases in the suite
        """
        self.suite_id = suite_id
        self.name = name
        self.description = description
        self.test_cases = test_cases or []
        self.status = TestStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.execution_time = None
    
    def add_test_case(self, test_case: TestCase):
        """
        Add a test case to the suite.
        
        Args:
            test_case: The test case to add
        """
        self.test_cases.append(test_case)
    
    def execute(self, test_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute all test cases in the suite.
        
        Args:
            test_context: Context data for the test execution
            
        Returns:
            Dictionary with execution results
        """
        if not test_context:
            test_context = {}
        
        self.status = TestStatus.RUNNING
        self.start_time = datetime.datetime.now()
        
        # Track results
        results = {
            "total": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "error": 0,
            "skipped": 0,
            "blocked": 0,
            "test_results": {}
        }
        
        # Build dependency map
        dependency_map = {}
        for test_case in self.test_cases:
            for dep_id in test_case.dependencies:
                if dep_id not in dependency_map:
                    dependency_map[dep_id] = []
                dependency_map[dep_id].append(test_case.test_id)
        
        # Track failed tests to block dependent tests
        failed_tests = set()
        
        # Execute tests in order, respecting dependencies
        for test_case in self.test_cases:
            # Check if this test depends on any failed tests
            blocked = False
            for dep_id in test_case.dependencies:
                if dep_id in failed_tests:
                    test_case.status = TestStatus.BLOCKED
                    test_case.error_message = f"Blocked by failed dependency: {dep_id}"
                    blocked = True
                    break
            
            if blocked:
                results["blocked"] += 1
                results["test_results"][test_case.test_id] = {
                    "status": TestStatus.BLOCKED.value,
                    "error_message": test_case.error_message
                }
                continue
            
            # Execute the test
            logger.info(f"Executing test: {test_case.name} ({test_case.test_id})")
            success = test_case.execute(test_context)
            
            # Record result
            if test_case.status == TestStatus.PASSED:
                results["passed"] += 1
            elif test_case.status == TestStatus.FAILED:
                results["failed"] += 1
                failed_tests.add(test_case.test_id)
            elif test_case.status == TestStatus.ERROR:
                results["error"] += 1
                failed_tests.add(test_case.test_id)
            elif test_case.status == TestStatus.SKIPPED:
                results["skipped"] += 1
            
            results["test_results"][test_case.test_id] = {
                "status": test_case.status.value,
                "result": test_case.result,
                "error_message": test_case.error_message,
                "execution_time": test_case.execution_time
            }
        
        self.end_time = datetime.datetime.now()
        self.execution_time = (self.end_time - self.start_time).total_seconds()
        
        # Determine overall suite status
        if results["failed"] > 0 or results["error"] > 0:
            self.status = TestStatus.FAILED
        elif results["passed"] == results["total"]:
            self.status = TestStatus.PASSED
        else:
            self.status = TestStatus.SKIPPED
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the test suite to a dictionary."""
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "description": self.description,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_time": self.execution_time
        }


class TestPlan:
    """
    Represents a test plan containing multiple test suites.
    
    A test plan organizes test suites and provides methods for
    executing them according to a defined strategy.
    """
    
    def __init__(
        self,
        plan_id: str,
        name: str,
        description: str,
        test_suites: List[TestSuite] = None
    ):
        """
        Initialize a test plan.
        
        Args:
            plan_id: Unique identifier for the test plan
            name: Human-readable name for the test plan
            description: Description of what the test plan validates
            test_suites: List of test suites in the plan
        """
        self.plan_id = plan_id
        self.name = name
        self.description = description
        self.test_suites = test_suites or []
        self.status = TestStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.execution_time = None
        self.results = {}
    
    def add_test_suite(self, test_suite: TestSuite):
        """
        Add a test suite to the plan.
        
        Args:
            test_suite: The test suite to add
        """
        self.test_suites.append(test_suite)
    
    def execute(self, test_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute all test suites in the plan.
        
        Args:
            test_context: Context data for the test execution
            
        Returns:
            Dictionary with execution results
        """
        if not test_context:
            test_context = {}
        
        self.status = TestStatus.RUNNING
        self.start_time = datetime.datetime.now()
        
        # Track results
        results = {
            "total_suites": len(self.test_suites),
            "passed_suites": 0,
            "failed_suites": 0,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "error_tests": 0,
            "skipped_tests": 0,
            "blocked_tests": 0,
            "suite_results": {}
        }
        
        # Execute each test suite
        for test_suite in self.test_suites:
            logger.info(f"Executing test suite: {test_suite.name} ({test_suite.suite_id})")
            suite_results = test_suite.execute(test_context)
            
            # Record suite results
            if test_suite.status == TestStatus.PASSED:
                results["passed_suites"] += 1
            elif test_suite.status == TestStatus.FAILED:
                results["failed_suites"] += 1
            
            # Aggregate test results
            results["total_tests"] += suite_results["total"]
            results["passed_tests"] += suite_results["passed"]
            results["failed_tests"] += suite_results["failed"]
            results["error_tests"] += suite_results["error"]
            results["skipped_tests"] += suite_results["skipped"]
            results["blocked_tests"] += suite_results["blocked"]
            
            results["suite_results"][test_suite.suite_id] = {
                "status": test_suite.status.value,
                "execution_time": test_suite.execution_time,
                "test_results": suite_results["test_results"]
            }
        
        self.end_time = datetime.datetime.now()
        self.execution_time = (self.end_time - self.start_time).total_seconds()
        
        # Determine overall plan status
        if results["failed_suites"] > 0:
            self.status = TestStatus.FAILED
        elif results["passed_suites"] == results["total_suites"]:
            self.status = TestStatus.PASSED
        else:
            self.status = TestStatus.SKIPPED
        
        self.results = results
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the test plan to a dictionary."""
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "description": self.description,
            "test_suites": [ts.to_dict() for ts in self.test_suites],
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_time": self.execution_time,
            "results": self.results
        }


class TestDataManager:
    """
    Manages test data for system testing.
    
    This class provides methods for generating, managing, and
    cleaning up test data used in system tests.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the test data manager.
        
        Args:
            data_dir: Directory for storing test data files
        """
        self.data_dir = data_dir or os.path.join(os.getcwd(), "test_data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_sets = {}
        logger.info(f"Test Data Manager initialized with data directory: {self.data_dir}")
    
    def create_data_set(self, name: str, data: Dict[str, Any]) -> str:
        """
        Create a new test data set.
        
        Args:
            name: Name of the data set
            data: Data to include in the set
            
        Returns:
            ID of the created data set
        """
        data_set_id = f"{name}_{str(uuid.uuid4())[:8]}"
        self.data_sets[data_set_id] = data
        
        # Save to file
        file_path = os.path.join(self.data_dir, f"{data_set_id}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Created test data set: {data_set_id}")
        return data_set_id
    
    def get_data_set(self, data_set_id: str) -> Dict[str, Any]:
        """
        Get a test data set by ID.
        
        Args:
            data_set_id: ID of the data set to retrieve
            
        Returns:
            The requested data set
            
        Raises:
            ValueError: If the data set is not found
        """
        if data_set_id in self.data_sets:
            return self.data_sets[data_set_id]
        
        # Try to load from file
        file_path = os.path.join(self.data_dir, f"{data_set_id}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.data_sets[data_set_id] = data
                return data
        
        raise ValueError(f"Data set not found: {data_set_id}")
    
    def update_data_set(self, data_set_id: str, data: Dict[str, Any]) -> bool:
        """
        Update an existing test data set.
        
        Args:
            data_set_id: ID of the data set to update
            data: New data for the set
            
        Returns:
            True if the update was successful, False otherwise
        """
        if data_set_id not in self.data_sets:
            file_path = os.path.join(self.data_dir, f"{data_set_id}.json")
            if not os.path.exists(file_path):
                return False
        
        self.data_sets[data_set_id] = data
        
        # Save to file
        file_path = os.path.join(self.data_dir, f"{data_set_id}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Updated test data set: {data_set_id}")
        return True
    
    def delete_data_set(self, data_set_id: str) -> bool:
        """
        Delete a test data set.
        
        Args:
            data_set_id: ID of the data set to delete
            
        Returns:
            True if the deletion was successful, False otherwise
        """
        if data_set_id in self.data_sets:
            del self.data_sets[data_set_id]
        
        # Delete file if it exists
        file_path = os.path.join(self.data_dir, f"{data_set_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted test data set: {data_set_id}")
            return True
        
        return False
    
    def list_data_sets(self) -> List[str]:
        """
        List all available test data sets.
        
        Returns:
            List of data set IDs
        """
        # Combine in-memory data sets with those on disk
        data_set_ids = set(self.data_sets.keys())
        
        # Add data sets from files
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".json"):
                data_set_id = file_name[:-5]  # Remove .json extension
                data_set_ids.add(data_set_id)
        
        return list(data_set_ids)
    
    def generate_account_data(self, count: int = 10) -> str:
        """
        Generate test account data.
        
        Args:
            count: Number of test accounts to generate
            
        Returns:
            ID of the created data set
        """
        accounts = []
        for i in range(count):
            account_id = f"ACC{100000 + i}"
            account = {
                "account_id": account_id,
                "account_type": "STANDARD",
                "status": "ACTIVE",
                "balance": round(1000 + i * 100 + random.random() * 1000, 2),
                "created_at": (datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 365))).isoformat(),
                "owner_id": f"USER{200000 + i}",
                "metadata": {
                    "risk_score": random.randint(1, 100),
                    "tier": random.choice(["BASIC", "PREMIUM", "PLATINUM"]),
                    "tags": random.sample(["retail", "business", "vip", "new", "legacy"], k=random.randint(1, 3))
                }
            }
            accounts.append(account)
        
        return self.create_data_set("account_data", {"accounts": accounts})
    
    def generate_transaction_data(self, account_ids: List[str], count_per_account: int = 20) -> str:
        """
        Generate test transaction data.
        
        Args:
            account_ids: List of account IDs to generate transactions for
            count_per_account: Number of transactions per account
            
        Returns:
            ID of the created data set
        """
        transactions = []
        transaction_types = ["DEPOSIT", "WITHDRAWAL", "TRANSFER", "PAYMENT", "FEE"]
        
        for account_id in account_ids:
            for i in range(count_per_account):
                transaction_id = f"TXN{300000 + len(transactions)}"
                transaction_type = random.choice(transaction_types)
                amount = round(10 + random.random() * 990, 2)
                
                if transaction_type in ["WITHDRAWAL", "TRANSFER", "PAYMENT", "FEE"]:
                    amount = -amount
                
                transaction = {
                    "transaction_id": transaction_id,
                    "account_id": account_id,
                    "type": transaction_type,
                    "amount": amount,
                    "balance_after": round(1000 + random.random() * 9000, 2),
                    "timestamp": (datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 90))).isoformat(),
                    "status": "COMPLETED",
                    "metadata": {
                        "category": random.choice(["INCOME", "EXPENSE", "TRANSFER", "FEE"]),
                        "description": f"Test transaction {i+1} for {account_id}"
                    }
                }
                transactions.append(transaction)
        
        return self.create_data_set("transaction_data", {"transactions": transactions})
    
    def cleanup(self):
        """Clean up all test data."""
        self.data_sets.clear()
        
        # Delete all data files
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".json"):
                os.remove(os.path.join(self.data_dir, file_name))
        
        logger.info("Cleaned up all test data")


class TestReporter:
    """
    Generates reports from test execution results.
    
    This class provides methods for generating various types of
    reports from test execution results.
    """
    
    def __init__(self, report_dir: str = None):
        """
        Initialize the test reporter.
        
        Args:
            report_dir: Directory for storing test reports
        """
        self.report_dir = report_dir or os.path.join(os.getcwd(), "test_reports")
        os.makedirs(self.report_dir, exist_ok=True)
        logger.info(f"Test Reporter initialized with report directory: {self.report_dir}")
    
    def generate_summary_report(self, test_plan: TestPlan) -> str:
        """
        Generate a summary report for a test plan execution.
        
        Args:
            test_plan: The executed test plan
            
        Returns:
            Path to the generated report file
        """
        report_id = f"summary_{test_plan.plan_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_path = os.path.join(self.report_dir, f"{report_id}.json")
        
        # Create the report data
        report = {
            "report_id": report_id,
            "test_plan": {
                "plan_id": test_plan.plan_id,
                "name": test_plan.name,
                "description": test_plan.description
            },
            "execution_summary": {
                "status": test_plan.status.value,
                "start_time": test_plan.start_time.isoformat() if test_plan.start_time else None,
                "end_time": test_plan.end_time.isoformat() if test_plan.end_time else None,
                "execution_time": test_plan.execution_time,
                "total_suites": test_plan.results.get("total_suites", 0),
                "passed_suites": test_plan.results.get("passed_suites", 0),
                "failed_suites": test_plan.results.get("failed_suites", 0),
                "total_tests": test_plan.results.get("total_tests", 0),
                "passed_tests": test_plan.results.get("passed_tests", 0),
                "failed_tests": test_plan.results.get("failed_tests", 0),
                "error_tests": test_plan.results.get("error_tests", 0),
                "skipped_tests": test_plan.results.get("skipped_tests", 0),
                "blocked_tests": test_plan.results.get("blocked_tests", 0)
            },
            "suite_summary": []
        }
        
        # Add suite summaries
        for test_suite in test_plan.test_suites:
            suite_summary = {
                "suite_id": test_suite.suite_id,
                "name": test_suite.name,
                "status": test_suite.status.value,
                "execution_time": test_suite.execution_time,
                "total_tests": len(test_suite.test_cases),
                "passed_tests": sum(1 for tc in test_suite.test_cases if tc.status == TestStatus.PASSED),
                "failed_tests": sum(1 for tc in test_suite.test_cases if tc.status == TestStatus.FAILED),
                "error_tests": sum(1 for tc in test_suite.test_cases if tc.status == TestStatus.ERROR),
                "skipped_tests": sum(1 for tc in test_suite.test_cases if tc.status == TestStatus.SKIPPED),
                "blocked_tests": sum(1 for tc in test_suite.test_cases if tc.status == TestStatus.BLOCKED)
            }
            report["suite_summary"].append(suite_summary)
        
        # Save the report
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated summary report: {file_path}")
        return file_path
    
    def generate_detailed_report(self, test_plan: TestPlan) -> str:
        """
        Generate a detailed report for a test plan execution.
        
        Args:
            test_plan: The executed test plan
            
        Returns:
            Path to the generated report file
        """
        report_id = f"detailed_{test_plan.plan_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_path = os.path.join(self.report_dir, f"{report_id}.json")
        
        # Create the report data (full test plan dictionary)
        report = test_plan.to_dict()
        report["report_id"] = report_id
        report["generated_at"] = datetime.datetime.now().isoformat()
        
        # Save the report
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated detailed report: {file_path}")
        return file_path
    
    def generate_csv_report(self, test_plan: TestPlan) -> str:
        """
        Generate a CSV report for a test plan execution.
        
        Args:
            test_plan: The executed test plan
            
        Returns:
            Path to the generated report file
        """
        report_id = f"csv_{test_plan.plan_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_path = os.path.join(self.report_dir, f"{report_id}.csv")
        
        # Create CSV data
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "Suite ID", "Suite Name", "Test ID", "Test Name", "Category",
                "Severity", "Status", "Execution Time (s)", "Error Message"
            ])
            
            # Write test results
            for test_suite in test_plan.test_suites:
                for test_case in test_suite.test_cases:
                    writer.writerow([
                        test_suite.suite_id,
                        test_suite.name,
                        test_case.test_id,
                        test_case.name,
                        test_case.category.value,
                        test_case.severity.value,
                        test_case.status.value,
                        test_case.execution_time,
                        test_case.error_message or ""
                    ])
        
        logger.info(f"Generated CSV report: {file_path}")
        return file_path
    
    def generate_html_report(self, test_plan: TestPlan) -> str:
        """
        Generate an HTML report for a test plan execution.
        
        Args:
            test_plan: The executed test plan
            
        Returns:
            Path to the generated report file
        """
        report_id = f"html_{test_plan.plan_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_path = os.path.join(self.report_dir, f"{report_id}.html")
        
        # Calculate summary statistics
        total_tests = test_plan.results.get("total_tests", 0)
        passed_tests = test_plan.results.get("passed_tests", 0)
        failed_tests = test_plan.results.get("failed_tests", 0)
        error_tests = test_plan.results.get("error_tests", 0)
        skipped_tests = test_plan.results.get("skipped_tests", 0)
        blocked_tests = test_plan.results.get("blocked_tests", 0)
        
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Report: {test_plan.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .error {{ color: orange; }}
                .skipped {{ color: gray; }}
                .blocked {{ color: purple; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .progress-bar {{ height: 20px; background-color: #e0e0e0; border-radius: 10px; margin-bottom: 10px; }}
                .progress {{ height: 100%; border-radius: 10px; }}
                .progress-passed {{ background-color: green; }}
                .progress-failed {{ background-color: red; }}
                .progress-error {{ background-color: orange; }}
                .progress-skipped {{ background-color: gray; }}
                .progress-blocked {{ background-color: purple; }}
            </style>
        </head>
        <body>
            <h1>Test Report: {test_plan.name}</h1>
            <p>{test_plan.description}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Status:</strong> <span class="{test_plan.status.value.lower()}">{test_plan.status.value}</span></p>
                <p><strong>Execution Time:</strong> {test_plan.execution_time:.2f} seconds</p>
                <p><strong>Start Time:</strong> {test_plan.start_time.strftime('%Y-%m-%d %H:%M:%S') if test_plan.start_time else 'N/A'}</p>
                <p><strong>End Time:</strong> {test_plan.end_time.strftime('%Y-%m-%d %H:%M:%S') if test_plan.end_time else 'N/A'}</p>
                
                <h3>Test Results</h3>
                <div class="progress-bar">
                    <div class="progress progress-passed" style="width: {pass_rate}%"></div>
                </div>
                <p><strong>Pass Rate:</strong> {pass_rate:.1f}% ({passed_tests}/{total_tests})</p>
                <p>
                    <span class="passed">Passed: {passed_tests}</span> | 
                    <span class="failed">Failed: {failed_tests}</span> | 
                    <span class="error">Error: {error_tests}</span> | 
                    <span class="skipped">Skipped: {skipped_tests}</span> | 
                    <span class="blocked">Blocked: {blocked_tests}</span>
                </p>
            </div>
            
            <h2>Test Suites</h2>
        """
        
        # Add test suite results
        for test_suite in test_plan.test_suites:
            suite_pass_count = sum(1 for tc in test_suite.test_cases if tc.status == TestStatus.PASSED)
            suite_total_count = len(test_suite.test_cases)
            suite_pass_rate = (suite_pass_count / suite_total_count * 100) if suite_total_count > 0 else 0
            
            html_content += f"""
            <h3>{test_suite.name} ({test_suite.suite_id})</h3>
            <p>{test_suite.description}</p>
            <p><strong>Status:</strong> <span class="{test_suite.status.value.lower()}">{test_suite.status.value}</span></p>
            <p><strong>Pass Rate:</strong> {suite_pass_rate:.1f}% ({suite_pass_count}/{suite_total_count})</p>
            
            <table>
                <tr>
                    <th>Test ID</th>
                    <th>Name</th>
                    <th>Category</th>
                    <th>Severity</th>
                    <th>Status</th>
                    <th>Execution Time (s)</th>
                    <th>Error Message</th>
                </tr>
            """
            
            # Add test case results
            for test_case in test_suite.test_cases:
                html_content += f"""
                <tr>
                    <td>{test_case.test_id}</td>
                    <td>{test_case.name}</td>
                    <td>{test_case.category.value}</td>
                    <td>{test_case.severity.value}</td>
                    <td class="{test_case.status.value.lower()}">{test_case.status.value}</td>
                    <td>{test_case.execution_time:.2f if test_case.execution_time else 'N/A'}</td>
                    <td>{test_case.error_message or ''}</td>
                </tr>
                """
            
            html_content += "</table>"
        
        html_content += """
        </body>
        </html>
        """
        
        # Save the HTML report
        with open(file_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {file_path}")
        return file_path


class SystemTestingFramework:
    """
    Main class for the system testing framework.
    
    This class coordinates the various components of the system testing
    framework and provides a facade for test execution and reporting.
    """
    
    def __init__(self, data_dir: str = None, report_dir: str = None):
        """
        Initialize the system testing framework.
        
        Args:
            data_dir: Directory for storing test data
            report_dir: Directory for storing test reports
        """
        self.data_manager = TestDataManager(data_dir)
        self.reporter = TestReporter(report_dir)
        self.test_plans = {}
        logger.info("System Testing Framework initialized")
    
    def create_test_case(
        self,
        name: str,
        description: str,
        category: TestCategory,
        severity: TestSeverity,
        test_function: Callable,
        setup_function: Callable = None,
        teardown_function: Callable = None,
        dependencies: List[str] = None,
        tags: List[str] = None,
        timeout_seconds: int = 60
    ) -> TestCase:
        """
        Create a new test case.
        
        Args:
            name: Human-readable name for the test case
            description: Description of what the test case validates
            category: Category of the test case
            severity: Severity level of the test case
            test_function: Function that executes the test
            setup_function: Function to set up the test environment (optional)
            teardown_function: Function to clean up after the test (optional)
            dependencies: List of test IDs that must pass before this test can run
            tags: List of tags for categorizing and filtering tests
            timeout_seconds: Maximum time the test is allowed to run
            
        Returns:
            The created test case
        """
        test_id = f"test_{name.lower().replace(' ', '_')}_{str(uuid.uuid4())[:8]}"
        
        return TestCase(
            test_id=test_id,
            name=name,
            description=description,
            category=category,
            severity=severity,
            test_function=test_function,
            setup_function=setup_function,
            teardown_function=teardown_function,
            dependencies=dependencies,
            tags=tags,
            timeout_seconds=timeout_seconds
        )
    
    def create_test_suite(
        self,
        name: str,
        description: str,
        test_cases: List[TestCase] = None
    ) -> TestSuite:
        """
        Create a new test suite.
        
        Args:
            name: Human-readable name for the test suite
            description: Description of what the test suite validates
            test_cases: List of test cases to include in the suite
            
        Returns:
            The created test suite
        """
        suite_id = f"suite_{name.lower().replace(' ', '_')}_{str(uuid.uuid4())[:8]}"
        
        return TestSuite(
            suite_id=suite_id,
            name=name,
            description=description,
            test_cases=test_cases
        )
    
    def create_test_plan(
        self,
        name: str,
        description: str,
        test_suites: List[TestSuite] = None
    ) -> TestPlan:
        """
        Create a new test plan.
        
        Args:
            name: Human-readable name for the test plan
            description: Description of what the test plan validates
            test_suites: List of test suites to include in the plan
            
        Returns:
            The created test plan
        """
        plan_id = f"plan_{name.lower().replace(' ', '_')}_{str(uuid.uuid4())[:8]}"
        
        test_plan = TestPlan(
            plan_id=plan_id,
            name=name,
            description=description,
            test_suites=test_suites
        )
        
        self.test_plans[plan_id] = test_plan
        return test_plan
    
    def execute_test_plan(self, plan_id: str, test_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a test plan.
        
        Args:
            plan_id: ID of the test plan to execute
            test_context: Context data for the test execution
            
        Returns:
            Dictionary with execution results
            
        Raises:
            ValueError: If the test plan is not found
        """
        if plan_id not in self.test_plans:
            raise ValueError(f"Test plan not found: {plan_id}")
        
        test_plan = self.test_plans[plan_id]
        logger.info(f"Executing test plan: {test_plan.name} ({plan_id})")
        
        # Execute the test plan
        results = test_plan.execute(test_context)
        
        # Generate reports
        self.reporter.generate_summary_report(test_plan)
        self.reporter.generate_detailed_report(test_plan)
        self.reporter.generate_csv_report(test_plan)
        self.reporter.generate_html_report(test_plan)
        
        return results
    
    def get_test_plan(self, plan_id: str) -> TestPlan:
        """
        Get a test plan by ID.
        
        Args:
            plan_id: ID of the test plan to retrieve
            
        Returns:
            The requested test plan
            
        Raises:
            ValueError: If the test plan is not found
        """
        if plan_id not in self.test_plans:
            raise ValueError(f"Test plan not found: {plan_id}")
        
        return self.test_plans[plan_id]
    
    def list_test_plans(self) -> List[Dict[str, Any]]:
        """
        List all test plans.
        
        Returns:
            List of dictionaries with test plan information
        """
        return [
            {
                "plan_id": plan.plan_id,
                "name": plan.name,
                "description": plan.description,
                "status": plan.status.value,
                "suite_count": len(plan.test_suites)
            }
            for plan in self.test_plans.values()
        ]
    
    def cleanup(self):
        """Clean up all test data and reports."""
        self.data_manager.cleanup()
        self.test_plans.clear()
        logger.info("System Testing Framework cleaned up")


# Singleton instance of the system testing framework
_system_testing_framework = None

def get_system_testing_framework() -> SystemTestingFramework:
    """
    Get the singleton instance of the system testing framework.
    
    Returns:
        The system testing framework instance
    """
    global _system_testing_framework
    if _system_testing_framework is None:
        _system_testing_framework = SystemTestingFramework()
    return _system_testing_framework


# Example usage
if __name__ == "__main__":
    import random
    
    # Create the system testing framework
    framework = get_system_testing_framework()
    
    # Define a sample test function
    def sample_test_function(context):
        # Simulate a test that usually passes
        success = random.random() > 0.2
        if success:
            return True
        else:
            return False
    
    # Create test cases
    test_cases = []
    for i in range(5):
        test_case = framework.create_test_case(
            name=f"Sample Test {i+1}",
            description=f"A sample test case {i+1}",
            category=TestCategory.FUNCTIONAL,
            severity=TestSeverity.MEDIUM,
            test_function=sample_test_function
        )
        test_cases.append(test_case)
    
    # Create a test suite
    test_suite = framework.create_test_suite(
        name="Sample Test Suite",
        description="A sample test suite",
        test_cases=test_cases
    )
    
    # Create a test plan
    test_plan = framework.create_test_plan(
        name="Sample Test Plan",
        description="A sample test plan",
        test_suites=[test_suite]
    )
    
    # Execute the test plan
    results = framework.execute_test_plan(test_plan.plan_id)
    
    # Print results
    print(f"Test plan execution results: {json.dumps(results, indent=2)}")
    
    # Clean up
    framework.cleanup()

