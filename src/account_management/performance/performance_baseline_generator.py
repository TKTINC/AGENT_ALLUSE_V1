#!/usr/bin/env python3
"""
ALL-USE Account Management System - Performance Baseline Generator

This module generates comprehensive performance baselines for the ALL-USE
Account Management System, establishing reference points for optimization
efforts and ongoing performance monitoring.

The baseline generator creates controlled test scenarios, executes them
with varying parameters, and collects detailed performance metrics to
establish a performance baseline.

Author: Manus AI
Date: June 17, 2025
"""

import os
import sys
import time
import json
import random
import logging
import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import account management modules
from models.account_models import AccountType, AccountStatus
from database.account_database import AccountDatabase
from api.account_operations_api import AccountOperationsAPI
from performance.performance_analyzer import PerformanceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("performance_baseline")

class PerformanceBaselineGenerator:
    """Class for generating performance baselines for the account management system."""
    
    def __init__(self, output_dir="./performance_baselines"):
        """Initialize the performance baseline generator.
        
        Args:
            output_dir (str): Directory for storing baseline reports
        """
        self.output_dir = output_dir
        self.analyzer = PerformanceAnalyzer(output_dir=output_dir)
        self.api = None
        self.db = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Performance baseline generator initialized. Reports will be saved to {output_dir}")
    
    def initialize_system(self):
        """Initialize the account management system components."""
        logger.info("Initializing account management system components")
        
        # Initialize database
        self.db = AccountDatabase()
        
        # Initialize API
        self.api = AccountOperationsAPI(self.db)
        
        logger.info("System components initialized")
    
    def generate_baseline(self, scenario_name, scenario_config):
        """Generate a performance baseline for a specific scenario.
        
        Args:
            scenario_name (str): Name of the scenario
            scenario_config (dict): Configuration for the scenario
            
        Returns:
            str: Path to the generated baseline report
        """
        logger.info(f"Generating baseline for scenario: {scenario_name}")
        
        # Initialize system if not already initialized
        if self.api is None or self.db is None:
            self.initialize_system()
        
        # Start performance monitoring
        self.analyzer.start_monitoring()
        
        try:
            # Execute the appropriate scenario
            if scenario_name == "account_creation":
                self._run_account_creation_scenario(scenario_config)
            elif scenario_name == "account_retrieval":
                self._run_account_retrieval_scenario(scenario_config)
            elif scenario_name == "transaction_processing":
                self._run_transaction_processing_scenario(scenario_config)
            elif scenario_name == "account_forking":
                self._run_account_forking_scenario(scenario_config)
            elif scenario_name == "account_merging":
                self._run_account_merging_scenario(scenario_config)
            elif scenario_name == "analytics_generation":
                self._run_analytics_generation_scenario(scenario_config)
            elif scenario_name == "mixed_workload":
                self._run_mixed_workload_scenario(scenario_config)
            else:
                logger.error(f"Unknown scenario: {scenario_name}")
                return None
        finally:
            # Stop performance monitoring
            self.analyzer.stop_monitoring()
        
        # Generate baseline report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"baseline_{scenario_name}"
        report_path = self.analyzer.generate_report(report_name)
        
        # Save scenario configuration with the report
        config_path = os.path.join(self.output_dir, f"{report_name}_{timestamp}_config.json")
        with open(config_path, 'w') as f:
            json.dump(scenario_config, f, indent=2)
        
        logger.info(f"Baseline generated: {report_path}")
        return report_path
    
    def _run_account_creation_scenario(self, config):
        """Run the account creation scenario.
        
        Args:
            config (dict): Scenario configuration
        """
        logger.info("Running account creation scenario")
        
        # Extract configuration parameters
        count = config.get("count", 100)
        concurrency = config.get("concurrency", 10)
        
        # Create accounts with specified concurrency
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            
            for i in range(count):
                account_data = {
                    "account_name": f"Test Account {i}",
                    "account_type": random.choice(["STANDARD", "PREMIUM", "ENTERPRISE"]),
                    "initial_balance": random.uniform(1000, 10000),
                    "owner_id": f"owner_{i % 10}",
                    "status": "ACTIVE"
                }
                
                # Submit account creation task
                futures.append(executor.submit(
                    self._create_account_with_metrics, account_data
                ))
            
            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in account creation: {e}")
        
        logger.info(f"Created {count} accounts")
    
    def _create_account_with_metrics(self, account_data):
        """Create an account and measure performance.
        
        Args:
            account_data (dict): Account data
            
        Returns:
            str: Account ID
        """
        metric = self.analyzer.get_metric("account_creation")
        self.analyzer.record_operation("account_creation")
        
        metric.start()
        try:
            account_id = self.api.create_account(
                account_data["account_name"],
                account_data["account_type"],
                account_data["initial_balance"],
                account_data["owner_id"],
                account_data["status"]
            )
            return account_id
        except Exception as e:
            self.analyzer.record_error("account_creation", type(e).__name__)
            raise
        finally:
            metric.stop()
    
    def _run_account_retrieval_scenario(self, config):
        """Run the account retrieval scenario.
        
        Args:
            config (dict): Scenario configuration
        """
        logger.info("Running account retrieval scenario")
        
        # Extract configuration parameters
        count = config.get("count", 100)
        concurrency = config.get("concurrency", 10)
        
        # Create test accounts if needed
        account_ids = self._ensure_test_accounts(count)
        
        # Retrieve accounts with specified concurrency
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            
            for i in range(count):
                # Get a random account ID from the list
                account_id = random.choice(account_ids)
                
                # Submit account retrieval task
                futures.append(executor.submit(
                    self._retrieve_account_with_metrics, account_id
                ))
            
            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in account retrieval: {e}")
        
        logger.info(f"Retrieved {count} accounts")
    
    def _retrieve_account_with_metrics(self, account_id):
        """Retrieve an account and measure performance.
        
        Args:
            account_id (str): Account ID
            
        Returns:
            Account: Retrieved account
        """
        metric = self.analyzer.get_metric("account_retrieval")
        self.analyzer.record_operation("account_retrieval")
        
        metric.start()
        try:
            account = self.api.get_account(account_id)
            return account
        except Exception as e:
            self.analyzer.record_error("account_retrieval", type(e).__name__)
            raise
        finally:
            metric.stop()
    
    def _run_transaction_processing_scenario(self, config):
        """Run the transaction processing scenario.
        
        Args:
            config (dict): Scenario configuration
        """
        logger.info("Running transaction processing scenario")
        
        # Extract configuration parameters
        count = config.get("count", 100)
        concurrency = config.get("concurrency", 10)
        
        # Create test accounts if needed
        account_ids = self._ensure_test_accounts(10)  # Need fewer accounts for transactions
        
        # Process transactions with specified concurrency
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            
            for i in range(count):
                # Get random source and destination accounts
                source_id = random.choice(account_ids)
                dest_id = random.choice([id for id in account_ids if id != source_id])
                
                transaction_data = {
                    "source_account_id": source_id,
                    "destination_account_id": dest_id,
                    "amount": random.uniform(10, 100),
                    "transaction_type": random.choice(["TRANSFER", "PAYMENT", "DEPOSIT"]),
                    "description": f"Test Transaction {i}"
                }
                
                # Submit transaction processing task
                futures.append(executor.submit(
                    self._process_transaction_with_metrics, transaction_data
                ))
            
            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in transaction processing: {e}")
        
        logger.info(f"Processed {count} transactions")
    
    def _process_transaction_with_metrics(self, transaction_data):
        """Process a transaction and measure performance.
        
        Args:
            transaction_data (dict): Transaction data
            
        Returns:
            str: Transaction ID
        """
        metric = self.analyzer.get_metric("transaction_processing")
        self.analyzer.record_operation("transaction_processing")
        
        metric.start()
        try:
            transaction_id = self.api.create_transaction(
                transaction_data["source_account_id"],
                transaction_data["destination_account_id"],
                transaction_data["amount"],
                transaction_data["transaction_type"],
                transaction_data["description"]
            )
            return transaction_id
        except Exception as e:
            self.analyzer.record_error("transaction_processing", type(e).__name__)
            raise
        finally:
            metric.stop()
    
    def _run_account_forking_scenario(self, config):
        """Run the account forking scenario.
        
        Args:
            config (dict): Scenario configuration
        """
        logger.info("Running account forking scenario")
        
        # Extract configuration parameters
        count = config.get("count", 50)
        concurrency = config.get("concurrency", 5)
        
        # Create test accounts if needed
        account_ids = self._ensure_test_accounts(count)
        
        # Fork accounts with specified concurrency
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            
            for i in range(count):
                # Get a random account ID from the list
                account_id = random.choice(account_ids)
                
                fork_data = {
                    "source_account_id": account_id,
                    "fork_name": f"Forked Account {i}",
                    "fork_percentage": random.uniform(0.1, 0.5),
                    "fork_reason": f"Test Fork {i}"
                }
                
                # Submit account forking task
                futures.append(executor.submit(
                    self._fork_account_with_metrics, fork_data
                ))
            
            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in account forking: {e}")
        
        logger.info(f"Forked {count} accounts")
    
    def _fork_account_with_metrics(self, fork_data):
        """Fork an account and measure performance.
        
        Args:
            fork_data (dict): Fork data
            
        Returns:
            str: Forked account ID
        """
        metric = self.analyzer.get_metric("account_forking")
        self.analyzer.record_operation("account_forking")
        
        metric.start()
        try:
            forked_account_id = self.api.fork_account(
                fork_data["source_account_id"],
                fork_data["fork_name"],
                fork_data["fork_percentage"],
                fork_data["fork_reason"]
            )
            return forked_account_id
        except Exception as e:
            self.analyzer.record_error("account_forking", type(e).__name__)
            raise
        finally:
            metric.stop()
    
    def _run_account_merging_scenario(self, config):
        """Run the account merging scenario.
        
        Args:
            config (dict): Scenario configuration
        """
        logger.info("Running account merging scenario")
        
        # Extract configuration parameters
        count = config.get("count", 25)
        concurrency = config.get("concurrency", 5)
        
        # Create test accounts and fork them to create merge candidates
        source_account_ids = self._ensure_test_accounts(count)
        
        # Create forked accounts for merging
        forked_account_pairs = []
        for i in range(count):
            source_id = source_account_ids[i % len(source_account_ids)]
            
            # Fork the account
            fork_data = {
                "source_account_id": source_id,
                "fork_name": f"Merge Candidate {i}",
                "fork_percentage": random.uniform(0.1, 0.3),
                "fork_reason": f"Test Merge Candidate {i}"
            }
            
            try:
                forked_id = self._fork_account_with_metrics(fork_data)
                forked_account_pairs.append((source_id, forked_id))
            except Exception as e:
                logger.error(f"Error creating fork for merge: {e}")
        
        # Merge accounts with specified concurrency
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            
            for i, (source_id, forked_id) in enumerate(forked_account_pairs):
                merge_data = {
                    "source_account_id": source_id,
                    "target_account_id": forked_id,
                    "merge_reason": f"Test Merge {i}"
                }
                
                # Submit account merging task
                futures.append(executor.submit(
                    self._merge_accounts_with_metrics, merge_data
                ))
            
            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in account merging: {e}")
        
        logger.info(f"Merged {len(forked_account_pairs)} account pairs")
    
    def _merge_accounts_with_metrics(self, merge_data):
        """Merge accounts and measure performance.
        
        Args:
            merge_data (dict): Merge data
            
        Returns:
            bool: Success indicator
        """
        metric = self.analyzer.get_metric("account_merging")
        self.analyzer.record_operation("account_merging")
        
        metric.start()
        try:
            success = self.api.merge_accounts(
                merge_data["source_account_id"],
                merge_data["target_account_id"],
                merge_data["merge_reason"]
            )
            return success
        except Exception as e:
            self.analyzer.record_error("account_merging", type(e).__name__)
            raise
        finally:
            metric.stop()
    
    def _run_analytics_generation_scenario(self, config):
        """Run the analytics generation scenario.
        
        Args:
            config (dict): Scenario configuration
        """
        logger.info("Running analytics generation scenario")
        
        # Extract configuration parameters
        count = config.get("count", 50)
        concurrency = config.get("concurrency", 5)
        
        # Create test accounts if needed
        account_ids = self._ensure_test_accounts(count)
        
        # Add some transactions to the accounts
        self._add_test_transactions(account_ids, 10)
        
        # Generate analytics with specified concurrency
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            
            for i in range(count):
                # Get a random account ID from the list
                account_id = random.choice(account_ids)
                
                # Submit analytics generation task
                futures.append(executor.submit(
                    self._generate_analytics_with_metrics, account_id
                ))
            
            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in analytics generation: {e}")
        
        logger.info(f"Generated analytics for {count} accounts")
    
    def _generate_analytics_with_metrics(self, account_id):
        """Generate analytics for an account and measure performance.
        
        Args:
            account_id (str): Account ID
            
        Returns:
            dict: Analytics data
        """
        metric = self.analyzer.get_metric("analytics_generation")
        self.analyzer.record_operation("analytics_generation")
        
        metric.start()
        try:
            # Import analytics module only when needed
            from analytics.account_analytics_engine import AccountAnalyticsEngine
            analytics_engine = AccountAnalyticsEngine(self.db)
            
            analytics_data = analytics_engine.generate_account_analytics(account_id)
            return analytics_data
        except Exception as e:
            self.analyzer.record_error("analytics_generation", type(e).__name__)
            raise
        finally:
            metric.stop()
    
    def _run_mixed_workload_scenario(self, config):
        """Run a mixed workload scenario with various operations.
        
        Args:
            config (dict): Scenario configuration
        """
        logger.info("Running mixed workload scenario")
        
        # Extract configuration parameters
        total_operations = config.get("total_operations", 200)
        concurrency = config.get("concurrency", 20)
        
        # Define operation mix percentages
        operation_mix = {
            "account_creation": 0.15,
            "account_retrieval": 0.40,
            "transaction_processing": 0.30,
            "account_forking": 0.05,
            "account_merging": 0.02,
            "analytics_generation": 0.08
        }
        
        # Create initial test accounts
        initial_accounts = 50
        account_ids = self._ensure_test_accounts(initial_accounts)
        
        # Add some initial transactions
        self._add_test_transactions(account_ids, 5)
        
        # Create operation list based on mix percentages
        operations = []
        for op_type, percentage in operation_mix.items():
            op_count = int(total_operations * percentage)
            operations.extend([op_type] * op_count)
        
        # Add any remaining operations to account retrieval (most common operation)
        while len(operations) < total_operations:
            operations.append("account_retrieval")
        
        # Shuffle operations for randomness
        random.shuffle(operations)
        
        # Execute mixed workload with specified concurrency
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            
            for i, operation in enumerate(operations):
                # Submit appropriate task based on operation type
                if operation == "account_creation":
                    account_data = {
                        "account_name": f"Mixed Workload Account {i}",
                        "account_type": random.choice(["STANDARD", "PREMIUM", "ENTERPRISE"]),
                        "initial_balance": random.uniform(1000, 10000),
                        "owner_id": f"owner_{i % 10}",
                        "status": "ACTIVE"
                    }
                    futures.append(executor.submit(
                        self._create_account_with_metrics, account_data
                    ))
                
                elif operation == "account_retrieval":
                    account_id = random.choice(account_ids)
                    futures.append(executor.submit(
                        self._retrieve_account_with_metrics, account_id
                    ))
                
                elif operation == "transaction_processing":
                    if len(account_ids) >= 2:
                        source_id = random.choice(account_ids)
                        dest_id = random.choice([id for id in account_ids if id != source_id])
                        
                        transaction_data = {
                            "source_account_id": source_id,
                            "destination_account_id": dest_id,
                            "amount": random.uniform(10, 100),
                            "transaction_type": random.choice(["TRANSFER", "PAYMENT", "DEPOSIT"]),
                            "description": f"Mixed Workload Transaction {i}"
                        }
                        
                        futures.append(executor.submit(
                            self._process_transaction_with_metrics, transaction_data
                        ))
                
                elif operation == "account_forking":
                    account_id = random.choice(account_ids)
                    
                    fork_data = {
                        "source_account_id": account_id,
                        "fork_name": f"Mixed Workload Fork {i}",
                        "fork_percentage": random.uniform(0.1, 0.5),
                        "fork_reason": f"Mixed Workload Test Fork {i}"
                    }
                    
                    future = executor.submit(
                        self._fork_account_with_metrics, fork_data
                    )
                    futures.append(future)
                    
                    # Add the forked account ID to our list when it's available
                    def add_forked_id(future):
                        try:
                            forked_id = future.result()
                            if forked_id:
                                account_ids.append(forked_id)
                        except Exception:
                            pass
                    
                    future.add_done_callback(add_forked_id)
                
                elif operation == "account_merging":
                    if len(account_ids) >= 2:
                        source_id = random.choice(account_ids)
                        target_id = random.choice([id for id in account_ids if id != source_id])
                        
                        merge_data = {
                            "source_account_id": source_id,
                            "target_account_id": target_id,
                            "merge_reason": f"Mixed Workload Test Merge {i}"
                        }
                        
                        futures.append(executor.submit(
                            self._merge_accounts_with_metrics, merge_data
                        ))
                
                elif operation == "analytics_generation":
                    account_id = random.choice(account_ids)
                    futures.append(executor.submit(
                        self._generate_analytics_with_metrics, account_id
                    ))
            
            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in mixed workload operation: {e}")
        
        logger.info(f"Completed mixed workload with {total_operations} operations")
    
    def _ensure_test_accounts(self, count):
        """Ensure that the specified number of test accounts exist.
        
        Args:
            count (int): Number of accounts needed
            
        Returns:
            list: List of account IDs
        """
        # Check if we already have enough accounts
        existing_accounts = self.api.list_accounts(limit=count)
        
        if len(existing_accounts) >= count:
            return [account.account_id for account in existing_accounts[:count]]
        
        # Create additional accounts as needed
        account_ids = [account.account_id for account in existing_accounts]
        accounts_to_create = count - len(account_ids)
        
        logger.info(f"Creating {accounts_to_create} additional test accounts")
        
        for i in range(accounts_to_create):
            account_data = {
                "account_name": f"Baseline Test Account {i}",
                "account_type": random.choice(["STANDARD", "PREMIUM", "ENTERPRISE"]),
                "initial_balance": random.uniform(1000, 10000),
                "owner_id": f"owner_{i % 10}",
                "status": "ACTIVE"
            }
            
            try:
                account_id = self.api.create_account(
                    account_data["account_name"],
                    account_data["account_type"],
                    account_data["initial_balance"],
                    account_data["owner_id"],
                    account_data["status"]
                )
                account_ids.append(account_id)
            except Exception as e:
                logger.error(f"Error creating test account: {e}")
        
        return account_ids
    
    def _add_test_transactions(self, account_ids, transactions_per_account):
        """Add test transactions to accounts.
        
        Args:
            account_ids (list): List of account IDs
            transactions_per_account (int): Number of transactions per account
            
        Returns:
            int: Number of transactions created
        """
        if len(account_ids) < 2:
            logger.warning("Need at least 2 accounts to create transactions")
            return 0
        
        logger.info(f"Adding {transactions_per_account} test transactions per account")
        
        transaction_count = 0
        
        for source_id in account_ids:
            for i in range(transactions_per_account):
                # Get a random destination account different from source
                dest_id = random.choice([id for id in account_ids if id != source_id])
                
                transaction_data = {
                    "source_account_id": source_id,
                    "destination_account_id": dest_id,
                    "amount": random.uniform(10, 100),
                    "transaction_type": random.choice(["TRANSFER", "PAYMENT", "DEPOSIT"]),
                    "description": f"Baseline Test Transaction {i}"
                }
                
                try:
                    self.api.create_transaction(
                        transaction_data["source_account_id"],
                        transaction_data["destination_account_id"],
                        transaction_data["amount"],
                        transaction_data["transaction_type"],
                        transaction_data["description"]
                    )
                    transaction_count += 1
                except Exception as e:
                    logger.error(f"Error creating test transaction: {e}")
        
        return transaction_count
    
    def run_all_baseline_scenarios(self):
        """Run all baseline scenarios with default configurations.
        
        Returns:
            dict: Dictionary of scenario names and report paths
        """
        logger.info("Running all baseline scenarios")
        
        # Define default configurations for each scenario
        scenarios = {
            "account_creation": {
                "count": 100,
                "concurrency": 10
            },
            "account_retrieval": {
                "count": 200,
                "concurrency": 20
            },
            "transaction_processing": {
                "count": 300,
                "concurrency": 30
            },
            "account_forking": {
                "count": 50,
                "concurrency": 5
            },
            "account_merging": {
                "count": 25,
                "concurrency": 5
            },
            "analytics_generation": {
                "count": 50,
                "concurrency": 5
            },
            "mixed_workload": {
                "total_operations": 500,
                "concurrency": 50
            }
        }
        
        # Run each scenario and collect report paths
        reports = {}
        
        for scenario_name, config in scenarios.items():
            try:
                report_path = self.generate_baseline(scenario_name, config)
                reports[scenario_name] = report_path
            except Exception as e:
                logger.error(f"Error running {scenario_name} scenario: {e}")
                reports[scenario_name] = None
        
        return reports

def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="ALL-USE Account Management System - Performance Baseline Generator")
    parser.add_argument("--scenario", choices=["account_creation", "account_retrieval", 
                                              "transaction_processing", "account_forking",
                                              "account_merging", "analytics_generation",
                                              "mixed_workload", "all"],
                       default="all", help="Scenario to run (default: all)")
    parser.add_argument("--count", type=int, help="Number of operations to perform")
    parser.add_argument("--concurrency", type=int, help="Concurrency level")
    parser.add_argument("--output-dir", default="./performance_baselines", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Create baseline generator
    generator = PerformanceBaselineGenerator(output_dir=args.output_dir)
    
    if args.scenario == "all":
        # Run all scenarios
        reports = generator.run_all_baseline_scenarios()
        
        print("\nBaseline Reports:")
        for scenario, report_path in reports.items():
            status = "Success" if report_path else "Failed"
            print(f"  {scenario}: {status}")
            if report_path:
                print(f"    Report: {report_path}")
    else:
        # Run specific scenario
        config = {}
        if args.count:
            config["count"] = args.count
        if args.concurrency:
            config["concurrency"] = args.concurrency
        
        report_path = generator.generate_baseline(args.scenario, config)
        
        if report_path:
            print(f"\nBaseline report generated: {report_path}")
        else:
            print("\nFailed to generate baseline report")

if __name__ == "__main__":
    main()

