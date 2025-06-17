#!/usr/bin/env python3
"""
ALL-USE Account Management System - Account Monitoring System

This module implements a specialized monitoring system for the ALL-USE
Account Management System, providing real-time visibility into account
operations, performance metrics, and health indicators.

The system integrates with the core monitoring framework to provide
comprehensive monitoring capabilities specific to account management.

Author: Manus AI
Date: June 17, 2025
"""

import os
import sys
import time
import logging
import json
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import account management modules
from monitoring.monitoring_framework import MonitoringFramework, MetricType, AlertSeverity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("account_monitoring")

class AccountMonitoringSystem:
    """Specialized monitoring system for the ALL-USE Account Management System."""
    
    def __init__(self, monitoring_framework=None, storage_dir="./account_monitoring_data"):
        """Initialize the account monitoring system.
        
        Args:
            monitoring_framework (MonitoringFramework, optional): Core monitoring framework
            storage_dir (str): Directory for storing monitoring data
        """
        # Use provided monitoring framework or create a new one
        self.monitoring = monitoring_framework or MonitoringFramework(storage_dir)
        self.storage_dir = storage_dir
        self.lock = threading.RLock()
        
        # Create storage directory
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize account metrics
        self._initialize_account_metrics()
        
        # Initialize alert rules
        self._initialize_alert_rules()
        
        logger.info("Account monitoring system initialized")
    
    def _initialize_account_metrics(self):
        """Initialize account-specific metrics."""
        # Account operation metrics
        self.monitoring.create_metric(
            "account.operations.create",
            MetricType.COUNTER,
            "Number of account creation operations"
        )
        self.monitoring.create_metric(
            "account.operations.update",
            MetricType.COUNTER,
            "Number of account update operations"
        )
        self.monitoring.create_metric(
            "account.operations.delete",
            MetricType.COUNTER,
            "Number of account deletion operations"
        )
        self.monitoring.create_metric(
            "account.operations.query",
            MetricType.COUNTER,
            "Number of account query operations"
        )
        
        # Account operation performance metrics
        self.monitoring.create_timer_metric(
            "account.performance.create",
            "Account creation time"
        )
        self.monitoring.create_timer_metric(
            "account.performance.update",
            "Account update time"
        )
        self.monitoring.create_timer_metric(
            "account.performance.delete",
            "Account deletion time"
        )
        self.monitoring.create_timer_metric(
            "account.performance.query",
            "Account query time"
        )
        
        # Account statistics metrics
        self.monitoring.create_metric(
            "account.stats.total",
            MetricType.GAUGE,
            "Total number of accounts"
        )
        self.monitoring.create_metric(
            "account.stats.active",
            MetricType.GAUGE,
            "Number of active accounts"
        )
        self.monitoring.create_metric(
            "account.stats.inactive",
            MetricType.GAUGE,
            "Number of inactive accounts"
        )
        self.monitoring.create_metric(
            "account.stats.suspended",
            MetricType.GAUGE,
            "Number of suspended accounts"
        )
        
        # Transaction metrics
        self.monitoring.create_metric(
            "account.transactions.count",
            MetricType.COUNTER,
            "Number of transactions"
        )
        self.monitoring.create_metric(
            "account.transactions.volume",
            MetricType.COUNTER,
            "Transaction volume"
        )
        self.monitoring.create_timer_metric(
            "account.transactions.processing_time",
            "Transaction processing time"
        )
        
        # Error metrics
        self.monitoring.create_metric(
            "account.errors.count",
            MetricType.COUNTER,
            "Number of errors"
        )
        self.monitoring.create_metric(
            "account.errors.validation",
            MetricType.COUNTER,
            "Number of validation errors"
        )
        self.monitoring.create_metric(
            "account.errors.database",
            MetricType.COUNTER,
            "Number of database errors"
        )
        self.monitoring.create_metric(
            "account.errors.authorization",
            MetricType.COUNTER,
            "Number of authorization errors"
        )
        
        # Forking and merging metrics
        self.monitoring.create_metric(
            "account.fork.count",
            MetricType.COUNTER,
            "Number of account fork operations"
        )
        self.monitoring.create_timer_metric(
            "account.fork.time",
            "Account fork operation time"
        )
        self.monitoring.create_metric(
            "account.merge.count",
            MetricType.COUNTER,
            "Number of account merge operations"
        )
        self.monitoring.create_timer_metric(
            "account.merge.time",
            "Account merge operation time"
        )
        
        # Reinvestment metrics
        self.monitoring.create_metric(
            "account.reinvest.count",
            MetricType.COUNTER,
            "Number of reinvestment operations"
        )
        self.monitoring.create_metric(
            "account.reinvest.amount",
            MetricType.COUNTER,
            "Reinvestment amount"
        )
        self.monitoring.create_timer_metric(
            "account.reinvest.time",
            "Reinvestment operation time"
        )
        
        # Analytics metrics
        self.monitoring.create_timer_metric(
            "account.analytics.generation_time",
            "Analytics generation time"
        )
        self.monitoring.create_metric(
            "account.analytics.cache_hit_rate",
            MetricType.GAUGE,
            "Analytics cache hit rate",
            "%"
        )
        
        # Security metrics
        self.monitoring.create_metric(
            "account.security.login_attempts",
            MetricType.COUNTER,
            "Number of login attempts"
        )
        self.monitoring.create_metric(
            "account.security.failed_logins",
            MetricType.COUNTER,
            "Number of failed login attempts"
        )
        self.monitoring.create_metric(
            "account.security.permission_denials",
            MetricType.COUNTER,
            "Number of permission denials"
        )
    
    def _initialize_alert_rules(self):
        """Initialize account-specific alert rules."""
        # High error rate alert
        self.monitoring.add_alert_rule(
            name="high_error_rate",
            metric_name="account.errors.count",
            condition=">",
            threshold=100,
            severity=AlertSeverity.WARNING,
            message_template="High account error rate: {value} errors (threshold: {threshold})"
        )
        
        # Slow account creation alert
        self.monitoring.add_alert_rule(
            name="slow_account_creation",
            metric_name="account.performance.create",
            condition=">",
            threshold=2.0,  # 2 seconds
            severity=AlertSeverity.WARNING,
            message_template="Slow account creation: {value:.2f}s (threshold: {threshold}s)"
        )
        
        # Slow transaction processing alert
        self.monitoring.add_alert_rule(
            name="slow_transaction_processing",
            metric_name="account.transactions.processing_time",
            condition=">",
            threshold=1.0,  # 1 second
            severity=AlertSeverity.WARNING,
            message_template="Slow transaction processing: {value:.2f}s (threshold: {threshold}s)"
        )
        
        # High failed login rate alert
        self.monitoring.add_alert_rule(
            name="high_failed_login_rate",
            metric_name="account.security.failed_logins",
            condition=">",
            threshold=10,
            severity=AlertSeverity.ERROR,
            message_template="High failed login rate: {value} attempts (threshold: {threshold})"
        )
        
        # Low analytics cache hit rate alert
        self.monitoring.add_alert_rule(
            name="low_analytics_cache_hit_rate",
            metric_name="account.analytics.cache_hit_rate",
            condition="<",
            threshold=50,  # 50%
            severity=AlertSeverity.WARNING,
            message_template="Low analytics cache hit rate: {value:.2f}% (threshold: {threshold}%)"
        )
    
    def start_monitoring(self, interval=60):
        """Start the monitoring system.
        
        Args:
            interval (int): Monitoring interval in seconds
            
        Returns:
            bool: True if started
        """
        return self.monitoring.start_monitoring(interval)
    
    def stop_monitoring(self):
        """Stop the monitoring system.
        
        Returns:
            bool: True if stopped
        """
        return self.monitoring.stop_monitoring()
    
    def record_account_operation(self, operation_type, duration=None):
        """Record an account operation.
        
        Args:
            operation_type (str): Operation type ('create', 'update', 'delete', 'query')
            duration (float, optional): Operation duration in seconds
            
        Returns:
            bool: True if successful
        """
        # Validate operation type
        valid_types = ['create', 'update', 'delete', 'query']
        if operation_type not in valid_types:
            logger.warning(f"Invalid operation type: {operation_type}")
            return False
        
        # Record operation count
        self.monitoring.record_metric(f"account.operations.{operation_type}", 1)
        
        # Record operation duration if provided
        if duration is not None:
            self.monitoring.record_metric(f"account.performance.{operation_type}", duration)
        
        return True
    
    def start_operation_timer(self, operation_type):
        """Start a timer for an account operation.
        
        Args:
            operation_type (str): Operation type ('create', 'update', 'delete', 'query')
            
        Returns:
            bool: True if successful
        """
        # Validate operation type
        valid_types = ['create', 'update', 'delete', 'query']
        if operation_type not in valid_types:
            logger.warning(f"Invalid operation type: {operation_type}")
            return False
        
        # Start timer
        return self.monitoring.start_timer(f"account.performance.{operation_type}")
    
    def stop_operation_timer(self, operation_type):
        """Stop a timer for an account operation and record the elapsed time.
        
        Args:
            operation_type (str): Operation type ('create', 'update', 'delete', 'query')
            
        Returns:
            float: Elapsed time in seconds or None if failed
        """
        # Validate operation type
        valid_types = ['create', 'update', 'delete', 'query']
        if operation_type not in valid_types:
            logger.warning(f"Invalid operation type: {operation_type}")
            return None
        
        # Record operation count
        self.monitoring.record_metric(f"account.operations.{operation_type}", 1)
        
        # Stop timer
        return self.monitoring.stop_timer(f"account.performance.{operation_type}")
    
    def update_account_statistics(self, total, active, inactive, suspended):
        """Update account statistics.
        
        Args:
            total (int): Total number of accounts
            active (int): Number of active accounts
            inactive (int): Number of inactive accounts
            suspended (int): Number of suspended accounts
            
        Returns:
            bool: True if successful
        """
        self.monitoring.record_metric("account.stats.total", total)
        self.monitoring.record_metric("account.stats.active", active)
        self.monitoring.record_metric("account.stats.inactive", inactive)
        self.monitoring.record_metric("account.stats.suspended", suspended)
        return True
    
    def record_transaction(self, count=1, volume=0, processing_time=None):
        """Record transaction metrics.
        
        Args:
            count (int): Number of transactions
            volume (float): Transaction volume
            processing_time (float, optional): Processing time in seconds
            
        Returns:
            bool: True if successful
        """
        self.monitoring.record_metric("account.transactions.count", count)
        self.monitoring.record_metric("account.transactions.volume", volume)
        
        if processing_time is not None:
            self.monitoring.record_metric("account.transactions.processing_time", processing_time)
        
        return True
    
    def start_transaction_timer(self):
        """Start a timer for transaction processing.
        
        Returns:
            bool: True if successful
        """
        return self.monitoring.start_timer("account.transactions.processing_time")
    
    def stop_transaction_timer(self, count=1, volume=0):
        """Stop a timer for transaction processing and record the elapsed time.
        
        Args:
            count (int): Number of transactions
            volume (float): Transaction volume
            
        Returns:
            float: Elapsed time in seconds or None if failed
        """
        # Record transaction count and volume
        self.monitoring.record_metric("account.transactions.count", count)
        self.monitoring.record_metric("account.transactions.volume", volume)
        
        # Stop timer
        return self.monitoring.stop_timer("account.transactions.processing_time")
    
    def record_error(self, error_type=None):
        """Record an error.
        
        Args:
            error_type (str, optional): Error type ('validation', 'database', 'authorization')
            
        Returns:
            bool: True if successful
        """
        # Record total error count
        self.monitoring.record_metric("account.errors.count", 1)
        
        # Record specific error type if provided
        if error_type in ['validation', 'database', 'authorization']:
            self.monitoring.record_metric(f"account.errors.{error_type}", 1)
        
        return True
    
    def record_fork_operation(self, count=1, duration=None):
        """Record a fork operation.
        
        Args:
            count (int): Number of fork operations
            duration (float, optional): Operation duration in seconds
            
        Returns:
            bool: True if successful
        """
        self.monitoring.record_metric("account.fork.count", count)
        
        if duration is not None:
            self.monitoring.record_metric("account.fork.time", duration)
        
        return True
    
    def start_fork_timer(self):
        """Start a timer for a fork operation.
        
        Returns:
            bool: True if successful
        """
        return self.monitoring.start_timer("account.fork.time")
    
    def stop_fork_timer(self, count=1):
        """Stop a timer for a fork operation and record the elapsed time.
        
        Args:
            count (int): Number of fork operations
            
        Returns:
            float: Elapsed time in seconds or None if failed
        """
        # Record fork count
        self.monitoring.record_metric("account.fork.count", count)
        
        # Stop timer
        return self.monitoring.stop_timer("account.fork.time")
    
    def record_merge_operation(self, count=1, duration=None):
        """Record a merge operation.
        
        Args:
            count (int): Number of merge operations
            duration (float, optional): Operation duration in seconds
            
        Returns:
            bool: True if successful
        """
        self.monitoring.record_metric("account.merge.count", count)
        
        if duration is not None:
            self.monitoring.record_metric("account.merge.time", duration)
        
        return True
    
    def start_merge_timer(self):
        """Start a timer for a merge operation.
        
        Returns:
            bool: True if successful
        """
        return self.monitoring.start_timer("account.merge.time")
    
    def stop_merge_timer(self, count=1):
        """Stop a timer for a merge operation and record the elapsed time.
        
        Args:
            count (int): Number of merge operations
            
        Returns:
            float: Elapsed time in seconds or None if failed
        """
        # Record merge count
        self.monitoring.record_metric("account.merge.count", count)
        
        # Stop timer
        return self.monitoring.stop_timer("account.merge.time")
    
    def record_reinvestment(self, count=1, amount=0, duration=None):
        """Record a reinvestment operation.
        
        Args:
            count (int): Number of reinvestment operations
            amount (float): Reinvestment amount
            duration (float, optional): Operation duration in seconds
            
        Returns:
            bool: True if successful
        """
        self.monitoring.record_metric("account.reinvest.count", count)
        self.monitoring.record_metric("account.reinvest.amount", amount)
        
        if duration is not None:
            self.monitoring.record_metric("account.reinvest.time", duration)
        
        return True
    
    def start_reinvestment_timer(self):
        """Start a timer for a reinvestment operation.
        
        Returns:
            bool: True if successful
        """
        return self.monitoring.start_timer("account.reinvest.time")
    
    def stop_reinvestment_timer(self, count=1, amount=0):
        """Stop a timer for a reinvestment operation and record the elapsed time.
        
        Args:
            count (int): Number of reinvestment operations
            amount (float): Reinvestment amount
            
        Returns:
            float: Elapsed time in seconds or None if failed
        """
        # Record reinvestment count and amount
        self.monitoring.record_metric("account.reinvest.count", count)
        self.monitoring.record_metric("account.reinvest.amount", amount)
        
        # Stop timer
        return self.monitoring.stop_timer("account.reinvest.time")
    
    def record_analytics_generation(self, duration=None, cache_hit_rate=None):
        """Record analytics generation metrics.
        
        Args:
            duration (float, optional): Generation time in seconds
            cache_hit_rate (float, optional): Cache hit rate percentage
            
        Returns:
            bool: True if successful
        """
        if duration is not None:
            self.monitoring.record_metric("account.analytics.generation_time", duration)
        
        if cache_hit_rate is not None:
            self.monitoring.record_metric("account.analytics.cache_hit_rate", cache_hit_rate)
        
        return True
    
    def start_analytics_timer(self):
        """Start a timer for analytics generation.
        
        Returns:
            bool: True if successful
        """
        return self.monitoring.start_timer("account.analytics.generation_time")
    
    def stop_analytics_timer(self, cache_hit_rate=None):
        """Stop a timer for analytics generation and record the elapsed time.
        
        Args:
            cache_hit_rate (float, optional): Cache hit rate percentage
            
        Returns:
            float: Elapsed time in seconds or None if failed
        """
        # Record cache hit rate if provided
        if cache_hit_rate is not None:
            self.monitoring.record_metric("account.analytics.cache_hit_rate", cache_hit_rate)
        
        # Stop timer
        return self.monitoring.stop_timer("account.analytics.generation_time")
    
    def record_security_event(self, event_type, count=1):
        """Record a security event.
        
        Args:
            event_type (str): Event type ('login_attempts', 'failed_logins', 'permission_denials')
            count (int): Number of events
            
        Returns:
            bool: True if successful
        """
        # Validate event type
        valid_types = ['login_attempts', 'failed_logins', 'permission_denials']
        if event_type not in valid_types:
            logger.warning(f"Invalid security event type: {event_type}")
            return False
        
        # Record event
        self.monitoring.record_metric(f"account.security.{event_type}", count)
        return True
    
    def generate_account_monitoring_report(self, start_time=None, end_time=None, output_file=None):
        """Generate an account monitoring report.
        
        Args:
            start_time (datetime, optional): Report start time
            end_time (datetime, optional): Report end time
            output_file (str, optional): Output file path
            
        Returns:
            str: Report file path
        """
        # Default time range is last 24 hours
        if not start_time:
            start_time = datetime.now() - timedelta(days=1)
        if not end_time:
            end_time = datetime.now()
        
        # Default output file
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.storage_dir, f"account_monitoring_report_{timestamp}.json")
        
        # Get account-specific metrics
        account_metrics = {}
        
        with self.lock:
            for name, metric in self.monitoring.metrics.items():
                if name.startswith("account."):
                    values = metric.get_values(start_time, end_time)
                    if values:
                        account_metrics[name] = {
                            "type": metric.metric_type.value,
                            "description": metric.description,
                            "unit": metric.unit,
                            "values": [v.to_dict() for v in values],
                            "statistics": metric.get_statistics()
                        }
        
        # Get account-specific alerts
        alerts = self.monitoring.get_alerts(start_time=start_time, end_time=end_time)
        account_alerts = [
            alert.to_dict() for alert in alerts
            if alert.source and alert.source.startswith("account.")
        ]
        
        # Create report
        report = {
            "generated_at": datetime.now().isoformat(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "metrics": account_metrics,
            "alerts": account_alerts,
            "summary": self._generate_summary(account_metrics)
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated account monitoring report: {output_file}")
        return output_file
    
    def _generate_summary(self, metrics):
        """Generate a summary of account metrics.
        
        Args:
            metrics (dict): Account metrics
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            "operations": {
                "total": 0,
                "create": 0,
                "update": 0,
                "delete": 0,
                "query": 0
            },
            "performance": {
                "create_avg": None,
                "update_avg": None,
                "delete_avg": None,
                "query_avg": None
            },
            "transactions": {
                "count": 0,
                "volume": 0,
                "avg_processing_time": None
            },
            "errors": {
                "total": 0,
                "validation": 0,
                "database": 0,
                "authorization": 0
            },
            "fork_merge": {
                "fork_count": 0,
                "merge_count": 0,
                "avg_fork_time": None,
                "avg_merge_time": None
            },
            "reinvestment": {
                "count": 0,
                "amount": 0,
                "avg_time": None
            },
            "analytics": {
                "avg_generation_time": None,
                "avg_cache_hit_rate": None
            },
            "security": {
                "login_attempts": 0,
                "failed_logins": 0,
                "permission_denials": 0
            }
        }
        
        # Extract operation counts
        for op_type in ["create", "update", "delete", "query"]:
            metric_name = f"account.operations.{op_type}"
            if metric_name in metrics:
                count = sum(v["value"] for v in metrics[metric_name]["values"])
                summary["operations"][op_type] = count
                summary["operations"]["total"] += count
        
        # Extract performance averages
        for op_type in ["create", "update", "delete", "query"]:
            metric_name = f"account.performance.{op_type}"
            if metric_name in metrics and metrics[metric_name]["statistics"]["count"] > 0:
                summary["performance"][f"{op_type}_avg"] = metrics[metric_name]["statistics"]["mean"]
        
        # Extract transaction metrics
        if "account.transactions.count" in metrics:
            summary["transactions"]["count"] = sum(v["value"] for v in metrics["account.transactions.count"]["values"])
        
        if "account.transactions.volume" in metrics:
            summary["transactions"]["volume"] = sum(v["value"] for v in metrics["account.transactions.volume"]["values"])
        
        if "account.transactions.processing_time" in metrics and metrics["account.transactions.processing_time"]["statistics"]["count"] > 0:
            summary["transactions"]["avg_processing_time"] = metrics["account.transactions.processing_time"]["statistics"]["mean"]
        
        # Extract error counts
        if "account.errors.count" in metrics:
            summary["errors"]["total"] = sum(v["value"] for v in metrics["account.errors.count"]["values"])
        
        for error_type in ["validation", "database", "authorization"]:
            metric_name = f"account.errors.{error_type}"
            if metric_name in metrics:
                summary["errors"][error_type] = sum(v["value"] for v in metrics[metric_name]["values"])
        
        # Extract fork/merge metrics
        if "account.fork.count" in metrics:
            summary["fork_merge"]["fork_count"] = sum(v["value"] for v in metrics["account.fork.count"]["values"])
        
        if "account.merge.count" in metrics:
            summary["fork_merge"]["merge_count"] = sum(v["value"] for v in metrics["account.merge.count"]["values"])
        
        if "account.fork.time" in metrics and metrics["account.fork.time"]["statistics"]["count"] > 0:
            summary["fork_merge"]["avg_fork_time"] = metrics["account.fork.time"]["statistics"]["mean"]
        
        if "account.merge.time" in metrics and metrics["account.merge.time"]["statistics"]["count"] > 0:
            summary["fork_merge"]["avg_merge_time"] = metrics["account.merge.time"]["statistics"]["mean"]
        
        # Extract reinvestment metrics
        if "account.reinvest.count" in metrics:
            summary["reinvestment"]["count"] = sum(v["value"] for v in metrics["account.reinvest.count"]["values"])
        
        if "account.reinvest.amount" in metrics:
            summary["reinvestment"]["amount"] = sum(v["value"] for v in metrics["account.reinvest.amount"]["values"])
        
        if "account.reinvest.time" in metrics and metrics["account.reinvest.time"]["statistics"]["count"] > 0:
            summary["reinvestment"]["avg_time"] = metrics["account.reinvest.time"]["statistics"]["mean"]
        
        # Extract analytics metrics
        if "account.analytics.generation_time" in metrics and metrics["account.analytics.generation_time"]["statistics"]["count"] > 0:
            summary["analytics"]["avg_generation_time"] = metrics["account.analytics.generation_time"]["statistics"]["mean"]
        
        if "account.analytics.cache_hit_rate" in metrics and metrics["account.analytics.cache_hit_rate"]["statistics"]["count"] > 0:
            summary["analytics"]["avg_cache_hit_rate"] = metrics["account.analytics.cache_hit_rate"]["statistics"]["mean"]
        
        # Extract security metrics
        for event_type in ["login_attempts", "failed_logins", "permission_denials"]:
            metric_name = f"account.security.{event_type}"
            if metric_name in metrics:
                summary["security"][event_type] = sum(v["value"] for v in metrics[metric_name]["values"])
        
        return summary
    
    def generate_account_visualizations(self, start_time=None, end_time=None, output_dir=None):
        """Generate visualizations for account metrics.
        
        Args:
            start_time (datetime, optional): Start time
            end_time (datetime, optional): End time
            output_dir (str, optional): Output directory
            
        Returns:
            list: List of visualization file paths
        """
        # Default time range is last 24 hours
        if not start_time:
            start_time = datetime.now() - timedelta(days=1)
        if not end_time:
            end_time = datetime.now()
        
        # Default output directory
        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.storage_dir, f"account_visualizations_{timestamp}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get account-specific metrics
        account_metrics = []
        
        with self.lock:
            for name, metric in self.monitoring.metrics.items():
                if name.startswith("account."):
                    account_metrics.append(name)
        
        # Generate visualizations
        return self.monitoring.generate_visualizations(
            metrics=account_metrics,
            start_time=start_time,
            end_time=end_time,
            output_dir=output_dir
        )

# Global account monitoring instance
_account_monitoring_instance = None

def get_account_monitoring_instance():
    """Get the global account monitoring instance.
    
    Returns:
        AccountMonitoringSystem: Global account monitoring instance
    """
    global _account_monitoring_instance
    if _account_monitoring_instance is None:
        _account_monitoring_instance = AccountMonitoringSystem()
    return _account_monitoring_instance

def set_account_monitoring_instance(instance):
    """Set the global account monitoring instance.
    
    Args:
        instance (AccountMonitoringSystem): Account monitoring instance
    """
    global _account_monitoring_instance
    _account_monitoring_instance = instance

def main():
    """Main function for standalone execution."""
    print("ALL-USE Account Management System - Account Monitoring System")
    print("==========================================================")
    print("\nThis module provides account monitoring capabilities and should be imported by other modules.")
    print("For standalone testing, this script will perform a basic self-test.")
    
    # Create account monitoring system
    monitoring_dir = os.path.join(os.getcwd(), "account_monitoring_data")
    system = AccountMonitoringSystem(storage_dir=monitoring_dir)
    
    # Run self-test
    print("\nRunning account monitoring system self-test...")
    
    # Start monitoring
    system.start_monitoring(interval=5)
    
    # Test account operations
    print("\nTesting account operations:")
    
    # Create operation
    system.start_operation_timer("create")
    time.sleep(0.2)
    elapsed = system.stop_operation_timer("create")
    print(f"  Account creation: {elapsed:.2f}s")
    
    # Update operation
    system.start_operation_timer("update")
    time.sleep(0.1)
    elapsed = system.stop_operation_timer("update")
    print(f"  Account update: {elapsed:.2f}s")
    
    # Query operation
    system.record_account_operation("query", 0.05)
    print("  Account query: 0.05s")
    
    # Update account statistics
    system.update_account_statistics(total=1000, active=800, inactive=150, suspended=50)
    print("  Updated account statistics")
    
    # Test transactions
    print("\nTesting transactions:")
    
    # Record transactions
    system.start_transaction_timer()
    time.sleep(0.3)
    elapsed = system.stop_transaction_timer(count=5, volume=1000)
    print(f"  Processed 5 transactions with volume 1000: {elapsed:.2f}s")
    
    # Test errors
    print("\nTesting errors:")
    
    # Record errors
    system.record_error("validation")
    system.record_error("database")
    system.record_error()
    print("  Recorded 3 errors")
    
    # Test fork/merge operations
    print("\nTesting fork/merge operations:")
    
    # Fork operation
    system.start_fork_timer()
    time.sleep(0.4)
    elapsed = system.stop_fork_timer(count=2)
    print(f"  Fork operation (2 accounts): {elapsed:.2f}s")
    
    # Merge operation
    system.record_merge_operation(count=1, duration=0.3)
    print("  Merge operation: 0.3s")
    
    # Test reinvestment
    print("\nTesting reinvestment:")
    
    # Reinvestment operation
    system.start_reinvestment_timer()
    time.sleep(0.2)
    elapsed = system.stop_reinvestment_timer(count=1, amount=500)
    print(f"  Reinvestment operation (amount 500): {elapsed:.2f}s")
    
    # Test analytics
    print("\nTesting analytics:")
    
    # Analytics generation
    system.start_analytics_timer()
    time.sleep(0.5)
    elapsed = system.stop_analytics_timer(cache_hit_rate=75)
    print(f"  Analytics generation (75% cache hit rate): {elapsed:.2f}s")
    
    # Test security events
    print("\nTesting security events:")
    
    # Security events
    system.record_security_event("login_attempts", 10)
    system.record_security_event("failed_logins", 2)
    print("  Recorded security events")
    
    # Wait for monitoring to collect some data
    print("\nWaiting for monitoring to collect data...")
    time.sleep(10)
    
    # Generate report
    print("\nGenerating account monitoring report...")
    report_path = system.generate_account_monitoring_report()
    print(f"  Generated report: {report_path}")
    
    # Generate visualizations
    print("\nGenerating account visualizations...")
    viz_files = system.generate_account_visualizations()
    print(f"  Generated {len(viz_files)} visualization files")
    
    # Stop monitoring
    system.stop_monitoring()
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()

