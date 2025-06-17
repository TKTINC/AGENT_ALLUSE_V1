#!/usr/bin/env python3
"""
ALL-USE Account Management System - Monitoring Framework

This module implements a comprehensive monitoring framework for the ALL-USE
Account Management System, providing real-time visibility into system performance,
health, and operational metrics.

The framework supports various monitoring strategies, alerting mechanisms,
and visualization capabilities to ensure optimal system operation.

Author: Manus AI
Date: June 17, 2025
"""

import os
import sys
import time
import logging
import json
import threading
import queue
import socket
import platform
import psutil
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Callable, Optional, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("monitoring_framework")

class MetricType(Enum):
    """Enumeration of metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(Enum):
    """Enumeration of alert severities."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricValue:
    """Class representing a metric value with timestamp."""
    
    def __init__(self, value, timestamp=None):
        """Initialize a metric value.
        
        Args:
            value: Metric value
            timestamp (datetime, optional): Timestamp
        """
        self.value = value
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self):
        """Convert the metric value to a dictionary.
        
        Returns:
            dict: Dictionary representation of the metric value
        """
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat()
        }

class Metric:
    """Class representing a monitored metric."""
    
    def __init__(self, name, metric_type, description=None, unit=None):
        """Initialize a metric.
        
        Args:
            name (str): Metric name
            metric_type (MetricType): Metric type
            description (str, optional): Metric description
            unit (str, optional): Metric unit
        """
        self.name = name
        self.metric_type = metric_type
        self.description = description
        self.unit = unit
        self.values = []
        self.lock = threading.RLock()
        
        # For timer metrics
        self.start_time = None
        
        # For histogram metrics
        self.min_value = None
        self.max_value = None
        self.sum_value = 0
        self.count = 0
    
    def add_value(self, value, timestamp=None):
        """Add a value to the metric.
        
        Args:
            value: Metric value
            timestamp (datetime, optional): Timestamp
        """
        with self.lock:
            metric_value = MetricValue(value, timestamp)
            self.values.append(metric_value)
            
            # Update histogram statistics
            if self.metric_type == MetricType.HISTOGRAM:
                if self.min_value is None or value < self.min_value:
                    self.min_value = value
                if self.max_value is None or value > self.max_value:
                    self.max_value = value
                self.sum_value += value
                self.count += 1
    
    def get_values(self, start_time=None, end_time=None):
        """Get metric values within a time range.
        
        Args:
            start_time (datetime, optional): Start time
            end_time (datetime, optional): End time
            
        Returns:
            list: List of metric values
        """
        with self.lock:
            if start_time is None and end_time is None:
                return self.values
            
            filtered_values = []
            for value in self.values:
                if start_time and value.timestamp < start_time:
                    continue
                if end_time and value.timestamp > end_time:
                    continue
                filtered_values.append(value)
            
            return filtered_values
    
    def get_latest_value(self):
        """Get the latest metric value.
        
        Returns:
            MetricValue: Latest metric value or None if no values
        """
        with self.lock:
            if not self.values:
                return None
            return self.values[-1]
    
    def get_statistics(self):
        """Get statistics for the metric.
        
        Returns:
            dict: Metric statistics
        """
        with self.lock:
            if not self.values:
                return {
                    "count": 0,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "median": None,
                    "p95": None,
                    "p99": None
                }
            
            values = [v.value for v in self.values]
            
            if self.metric_type == MetricType.HISTOGRAM:
                # Use pre-calculated values for histogram metrics
                mean = self.sum_value / self.count if self.count > 0 else None
                
                # Calculate percentiles
                sorted_values = sorted(values)
                median = sorted_values[len(sorted_values) // 2]
                p95_idx = int(len(sorted_values) * 0.95)
                p99_idx = int(len(sorted_values) * 0.99)
                p95 = sorted_values[p95_idx] if p95_idx < len(sorted_values) else sorted_values[-1]
                p99 = sorted_values[p99_idx] if p99_idx < len(sorted_values) else sorted_values[-1]
                
                return {
                    "count": self.count,
                    "min": self.min_value,
                    "max": self.max_value,
                    "mean": mean,
                    "median": median,
                    "p95": p95,
                    "p99": p99
                }
            else:
                # Calculate statistics for other metric types
                return {
                    "count": len(values),
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "mean": sum(values) / len(values) if values else None,
                    "median": sorted(values)[len(values) // 2] if values else None,
                    "p95": sorted(values)[int(len(values) * 0.95)] if values else None,
                    "p99": sorted(values)[int(len(values) * 0.99)] if values else None
                }
    
    def start_timer(self):
        """Start a timer for timer metrics."""
        if self.metric_type != MetricType.TIMER:
            raise ValueError("start_timer can only be called on timer metrics")
        
        self.start_time = time.time()
    
    def stop_timer(self):
        """Stop a timer and record the elapsed time.
        
        Returns:
            float: Elapsed time in seconds
        """
        if self.metric_type != MetricType.TIMER:
            raise ValueError("stop_timer can only be called on timer metrics")
        
        if self.start_time is None:
            raise ValueError("Timer was not started")
        
        elapsed_time = time.time() - self.start_time
        self.add_value(elapsed_time)
        self.start_time = None
        
        return elapsed_time
    
    def to_dict(self):
        """Convert the metric to a dictionary.
        
        Returns:
            dict: Dictionary representation of the metric
        """
        with self.lock:
            return {
                "name": self.name,
                "type": self.metric_type.value,
                "description": self.description,
                "unit": self.unit,
                "statistics": self.get_statistics(),
                "latest_value": self.get_latest_value().to_dict() if self.get_latest_value() else None
            }

class Alert:
    """Class representing a monitoring alert."""
    
    def __init__(self, name, severity, message, source=None, timestamp=None):
        """Initialize an alert.
        
        Args:
            name (str): Alert name
            severity (AlertSeverity): Alert severity
            message (str): Alert message
            source (str, optional): Alert source
            timestamp (datetime, optional): Alert timestamp
        """
        self.name = name
        self.severity = severity
        self.message = message
        self.source = source
        self.timestamp = timestamp or datetime.now()
        self.id = f"{self.name}_{self.timestamp.strftime('%Y%m%d%H%M%S')}"
        self.acknowledged = False
        self.resolved = False
        self.resolved_timestamp = None
    
    def acknowledge(self):
        """Acknowledge the alert."""
        self.acknowledged = True
    
    def resolve(self):
        """Resolve the alert."""
        self.resolved = True
        self.resolved_timestamp = datetime.now()
    
    def to_dict(self):
        """Convert the alert to a dictionary.
        
        Returns:
            dict: Dictionary representation of the alert
        """
        return {
            "id": self.id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "resolved_timestamp": self.resolved_timestamp.isoformat() if self.resolved_timestamp else None
        }

class AlertRule:
    """Class representing an alert rule."""
    
    def __init__(self, name, metric_name, condition, threshold, severity, message_template, cooldown=300):
        """Initialize an alert rule.
        
        Args:
            name (str): Rule name
            metric_name (str): Metric name to monitor
            condition (str): Condition to evaluate ('>', '<', '>=', '<=', '==', '!=')
            threshold: Threshold value
            severity (AlertSeverity): Alert severity
            message_template (str): Alert message template
            cooldown (int): Cooldown period in seconds
        """
        self.name = name
        self.metric_name = metric_name
        self.condition = condition
        self.threshold = threshold
        self.severity = severity
        self.message_template = message_template
        self.cooldown = cooldown
        self.last_triggered = None
    
    def evaluate(self, metric_value):
        """Evaluate the rule against a metric value.
        
        Args:
            metric_value: Metric value to evaluate
            
        Returns:
            bool: True if rule is triggered
        """
        # Check cooldown period
        if self.last_triggered and (datetime.now() - self.last_triggered).total_seconds() < self.cooldown:
            return False
        
        # Evaluate condition
        triggered = False
        
        if self.condition == '>':
            triggered = metric_value > self.threshold
        elif self.condition == '<':
            triggered = metric_value < self.threshold
        elif self.condition == '>=':
            triggered = metric_value >= self.threshold
        elif self.condition == '<=':
            triggered = metric_value <= self.threshold
        elif self.condition == '==':
            triggered = metric_value == self.threshold
        elif self.condition == '!=':
            triggered = metric_value != self.threshold
        
        if triggered:
            self.last_triggered = datetime.now()
        
        return triggered
    
    def generate_alert(self, metric_value):
        """Generate an alert based on the rule.
        
        Args:
            metric_value: Metric value that triggered the rule
            
        Returns:
            Alert: Generated alert
        """
        message = self.message_template.format(
            metric_name=self.metric_name,
            threshold=self.threshold,
            value=metric_value
        )
        
        return Alert(
            name=self.name,
            severity=self.severity,
            message=message,
            source=self.metric_name
        )
    
    def to_dict(self):
        """Convert the rule to a dictionary.
        
        Returns:
            dict: Dictionary representation of the rule
        """
        return {
            "name": self.name,
            "metric_name": self.metric_name,
            "condition": self.condition,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "message_template": self.message_template,
            "cooldown": self.cooldown,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None
        }

class MonitoringFramework:
    """Comprehensive monitoring framework for the ALL-USE Account Management System."""
    
    def __init__(self, storage_dir="./monitoring_data"):
        """Initialize the monitoring framework.
        
        Args:
            storage_dir (str): Directory for storing monitoring data
        """
        self.storage_dir = storage_dir
        self.metrics = {}
        self.alerts = []
        self.alert_rules = []
        self.lock = threading.RLock()
        self.running = False
        self.monitor_thread = None
        
        # Create storage directory
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize system metrics
        self._initialize_system_metrics()
        
        logger.info("Monitoring framework initialized")
    
    def _initialize_system_metrics(self):
        """Initialize system metrics."""
        # CPU metrics
        self.create_metric("system.cpu.usage", MetricType.GAUGE, "CPU usage percentage", "%")
        self.create_metric("system.cpu.count", MetricType.GAUGE, "Number of CPU cores")
        
        # Memory metrics
        self.create_metric("system.memory.total", MetricType.GAUGE, "Total memory", "bytes")
        self.create_metric("system.memory.available", MetricType.GAUGE, "Available memory", "bytes")
        self.create_metric("system.memory.used", MetricType.GAUGE, "Used memory", "bytes")
        self.create_metric("system.memory.percent", MetricType.GAUGE, "Memory usage percentage", "%")
        
        # Disk metrics
        self.create_metric("system.disk.total", MetricType.GAUGE, "Total disk space", "bytes")
        self.create_metric("system.disk.free", MetricType.GAUGE, "Free disk space", "bytes")
        self.create_metric("system.disk.used", MetricType.GAUGE, "Used disk space", "bytes")
        self.create_metric("system.disk.percent", MetricType.GAUGE, "Disk usage percentage", "%")
        
        # Network metrics
        self.create_metric("system.network.bytes_sent", MetricType.COUNTER, "Network bytes sent", "bytes")
        self.create_metric("system.network.bytes_recv", MetricType.COUNTER, "Network bytes received", "bytes")
        
        # Process metrics
        self.create_metric("system.process.count", MetricType.GAUGE, "Number of processes")
        
        # System load metrics
        self.create_metric("system.load.1min", MetricType.GAUGE, "System load average (1 min)")
        self.create_metric("system.load.5min", MetricType.GAUGE, "System load average (5 min)")
        self.create_metric("system.load.15min", MetricType.GAUGE, "System load average (15 min)")
    
    def create_metric(self, name, metric_type, description=None, unit=None):
        """Create a new metric.
        
        Args:
            name (str): Metric name
            metric_type (MetricType): Metric type
            description (str, optional): Metric description
            unit (str, optional): Metric unit
            
        Returns:
            Metric: Created metric
        """
        with self.lock:
            if name in self.metrics:
                logger.warning(f"Metric '{name}' already exists")
                return self.metrics[name]
            
            metric = Metric(name, metric_type, description, unit)
            self.metrics[name] = metric
            
            logger.info(f"Created metric '{name}' of type {metric_type.value}")
            return metric
    
    def get_metric(self, name):
        """Get a metric by name.
        
        Args:
            name (str): Metric name
            
        Returns:
            Metric: Metric object or None if not found
        """
        with self.lock:
            return self.metrics.get(name)
    
    def record_metric(self, name, value, timestamp=None):
        """Record a metric value.
        
        Args:
            name (str): Metric name
            value: Metric value
            timestamp (datetime, optional): Timestamp
            
        Returns:
            bool: True if successful
        """
        with self.lock:
            metric = self.metrics.get(name)
            if not metric:
                logger.warning(f"Metric '{name}' not found")
                return False
            
            metric.add_value(value, timestamp)
            
            # Evaluate alert rules
            self._evaluate_alert_rules(name, value)
            
            return True
    
    def create_timer_metric(self, name, description=None):
        """Create a timer metric.
        
        Args:
            name (str): Metric name
            description (str, optional): Metric description
            
        Returns:
            Metric: Created metric
        """
        return self.create_metric(name, MetricType.TIMER, description, "seconds")
    
    def start_timer(self, name):
        """Start a timer for a metric.
        
        Args:
            name (str): Metric name
            
        Returns:
            bool: True if successful
        """
        with self.lock:
            metric = self.metrics.get(name)
            if not metric:
                logger.warning(f"Metric '{name}' not found")
                return False
            
            if metric.metric_type != MetricType.TIMER:
                logger.warning(f"Metric '{name}' is not a timer")
                return False
            
            metric.start_timer()
            return True
    
    def stop_timer(self, name):
        """Stop a timer and record the elapsed time.
        
        Args:
            name (str): Metric name
            
        Returns:
            float: Elapsed time in seconds or None if failed
        """
        with self.lock:
            metric = self.metrics.get(name)
            if not metric:
                logger.warning(f"Metric '{name}' not found")
                return None
            
            if metric.metric_type != MetricType.TIMER:
                logger.warning(f"Metric '{name}' is not a timer")
                return None
            
            try:
                elapsed_time = metric.stop_timer()
                
                # Evaluate alert rules
                self._evaluate_alert_rules(name, elapsed_time)
                
                return elapsed_time
            except ValueError as e:
                logger.warning(f"Error stopping timer for metric '{name}': {e}")
                return None
    
    def add_alert_rule(self, name, metric_name, condition, threshold, severity, message_template, cooldown=300):
        """Add an alert rule.
        
        Args:
            name (str): Rule name
            metric_name (str): Metric name to monitor
            condition (str): Condition to evaluate ('>', '<', '>=', '<=', '==', '!=')
            threshold: Threshold value
            severity (AlertSeverity): Alert severity
            message_template (str): Alert message template
            cooldown (int): Cooldown period in seconds
            
        Returns:
            AlertRule: Created rule
        """
        with self.lock:
            # Check if metric exists
            if metric_name not in self.metrics:
                logger.warning(f"Metric '{metric_name}' not found")
                return None
            
            # Create rule
            rule = AlertRule(name, metric_name, condition, threshold, severity, message_template, cooldown)
            self.alert_rules.append(rule)
            
            logger.info(f"Added alert rule '{name}' for metric '{metric_name}'")
            return rule
    
    def remove_alert_rule(self, name):
        """Remove an alert rule.
        
        Args:
            name (str): Rule name
            
        Returns:
            bool: True if removed
        """
        with self.lock:
            for i, rule in enumerate(self.alert_rules):
                if rule.name == name:
                    del self.alert_rules[i]
                    logger.info(f"Removed alert rule '{name}'")
                    return True
            
            logger.warning(f"Alert rule '{name}' not found")
            return False
    
    def get_alerts(self, severity=None, resolved=None, start_time=None, end_time=None):
        """Get alerts with optional filtering.
        
        Args:
            severity (AlertSeverity, optional): Filter by severity
            resolved (bool, optional): Filter by resolved status
            start_time (datetime, optional): Filter by start time
            end_time (datetime, optional): Filter by end time
            
        Returns:
            list: List of alerts
        """
        with self.lock:
            filtered_alerts = []
            
            for alert in self.alerts:
                # Filter by severity
                if severity and alert.severity != severity:
                    continue
                
                # Filter by resolved status
                if resolved is not None and alert.resolved != resolved:
                    continue
                
                # Filter by start time
                if start_time and alert.timestamp < start_time:
                    continue
                
                # Filter by end time
                if end_time and alert.timestamp > end_time:
                    continue
                
                filtered_alerts.append(alert)
            
            return filtered_alerts
    
    def acknowledge_alert(self, alert_id):
        """Acknowledge an alert.
        
        Args:
            alert_id (str): Alert ID
            
        Returns:
            bool: True if acknowledged
        """
        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledge()
                    logger.info(f"Acknowledged alert '{alert_id}'")
                    return True
            
            logger.warning(f"Alert '{alert_id}' not found")
            return False
    
    def resolve_alert(self, alert_id):
        """Resolve an alert.
        
        Args:
            alert_id (str): Alert ID
            
        Returns:
            bool: True if resolved
        """
        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolve()
                    logger.info(f"Resolved alert '{alert_id}'")
                    return True
            
            logger.warning(f"Alert '{alert_id}' not found")
            return False
    
    def _evaluate_alert_rules(self, metric_name, metric_value):
        """Evaluate alert rules for a metric value.
        
        Args:
            metric_name (str): Metric name
            metric_value: Metric value
        """
        for rule in self.alert_rules:
            if rule.metric_name == metric_name and rule.evaluate(metric_value):
                alert = rule.generate_alert(metric_value)
                self.alerts.append(alert)
                logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
    
    def start_monitoring(self, interval=60):
        """Start the monitoring thread.
        
        Args:
            interval (int): Monitoring interval in seconds
            
        Returns:
            bool: True if started
        """
        with self.lock:
            if self.running:
                logger.warning("Monitoring is already running")
                return False
            
            self.running = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(interval,),
                daemon=True
            )
            self.monitor_thread.start()
            
            logger.info(f"Started monitoring with interval {interval}s")
            return True
    
    def stop_monitoring(self):
        """Stop the monitoring thread.
        
        Returns:
            bool: True if stopped
        """
        with self.lock:
            if not self.running:
                logger.warning("Monitoring is not running")
                return False
            
            self.running = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
                self.monitor_thread = None
            
            logger.info("Stopped monitoring")
            return True
    
    def _monitoring_loop(self, interval):
        """Monitoring thread loop.
        
        Args:
            interval (int): Monitoring interval in seconds
        """
        while self.running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Save metrics to disk
                self._save_metrics()
                
                # Save alerts to disk
                self._save_alerts()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep for the specified interval
            time.sleep(interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system.cpu.usage", cpu_percent)
            self.record_metric("system.cpu.count", psutil.cpu_count())
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system.memory.total", memory.total)
            self.record_metric("system.memory.available", memory.available)
            self.record_metric("system.memory.used", memory.used)
            self.record_metric("system.memory.percent", memory.percent)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric("system.disk.total", disk.total)
            self.record_metric("system.disk.free", disk.free)
            self.record_metric("system.disk.used", disk.used)
            self.record_metric("system.disk.percent", disk.percent)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            self.record_metric("system.network.bytes_sent", net_io.bytes_sent)
            self.record_metric("system.network.bytes_recv", net_io.bytes_recv)
            
            # Process metrics
            self.record_metric("system.process.count", len(psutil.pids()))
            
            # System load metrics
            load = psutil.getloadavg()
            self.record_metric("system.load.1min", load[0])
            self.record_metric("system.load.5min", load[1])
            self.record_metric("system.load.15min", load[2])
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to disk."""
        try:
            metrics_data = {}
            
            with self.lock:
                for name, metric in self.metrics.items():
                    metrics_data[name] = metric.to_dict()
            
            # Save to file
            file_path = os.path.join(self.storage_dir, "metrics.json")
            with open(file_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def _save_alerts(self):
        """Save alerts to disk."""
        try:
            alerts_data = []
            
            with self.lock:
                for alert in self.alerts:
                    alerts_data.append(alert.to_dict())
            
            # Save to file
            file_path = os.path.join(self.storage_dir, "alerts.json")
            with open(file_path, 'w') as f:
                json.dump(alerts_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving alerts: {e}")
    
    def generate_report(self, start_time=None, end_time=None, output_file=None):
        """Generate a monitoring report.
        
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
            output_file = os.path.join(self.storage_dir, f"monitoring_report_{timestamp}.json")
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "metrics": {},
            "alerts": []
        }
        
        # Add metrics
        with self.lock:
            for name, metric in self.metrics.items():
                values = metric.get_values(start_time, end_time)
                if values:
                    report["metrics"][name] = {
                        "type": metric.metric_type.value,
                        "description": metric.description,
                        "unit": metric.unit,
                        "values": [v.to_dict() for v in values],
                        "statistics": metric.get_statistics()
                    }
        
        # Add alerts
        alerts = self.get_alerts(start_time=start_time, end_time=end_time)
        report["alerts"] = [alert.to_dict() for alert in alerts]
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated monitoring report: {output_file}")
        return output_file
    
    def generate_visualizations(self, metrics=None, start_time=None, end_time=None, output_dir=None):
        """Generate visualizations for metrics.
        
        Args:
            metrics (list, optional): List of metric names to visualize
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
            output_dir = os.path.join(self.storage_dir, f"visualizations_{timestamp}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Default metrics is all metrics
        if not metrics:
            with self.lock:
                metrics = list(self.metrics.keys())
        
        visualization_files = []
        
        # Generate visualizations for each metric
        for metric_name in metrics:
            try:
                metric = self.get_metric(metric_name)
                if not metric:
                    logger.warning(f"Metric '{metric_name}' not found")
                    continue
                
                values = metric.get_values(start_time, end_time)
                if not values:
                    logger.warning(f"No values for metric '{metric_name}' in the specified time range")
                    continue
                
                # Extract data
                timestamps = [v.timestamp for v in values]
                data = [v.value for v in values]
                
                # Create figure
                plt.figure(figsize=(10, 6))
                plt.plot(timestamps, data)
                plt.title(f"{metric_name}")
                plt.xlabel("Time")
                plt.ylabel(metric.unit or "Value")
                plt.grid(True)
                
                # Format x-axis
                plt.gcf().autofmt_xdate()
                
                # Save figure
                file_name = f"{metric_name.replace('.', '_')}.png"
                file_path = os.path.join(output_dir, file_name)
                plt.savefig(file_path)
                plt.close()
                
                visualization_files.append(file_path)
                logger.info(f"Generated visualization for metric '{metric_name}': {file_path}")
                
            except Exception as e:
                logger.error(f"Error generating visualization for metric '{metric_name}': {e}")
        
        return visualization_files
    
    def get_system_info(self):
        """Get system information.
        
        Returns:
            dict: System information
        """
        try:
            info = {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "disk_total": psutil.disk_usage('/').total,
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting system information: {e}")
            return {}

def main():
    """Main function for standalone execution."""
    print("ALL-USE Account Management System - Monitoring Framework")
    print("======================================================")
    print("\nThis module provides monitoring capabilities and should be imported by other modules.")
    print("For standalone testing, this script will perform a basic self-test.")
    
    # Create monitoring framework
    monitoring_dir = os.path.join(os.getcwd(), "monitoring_data")
    framework = MonitoringFramework(storage_dir=monitoring_dir)
    
    # Run self-test
    print("\nRunning monitoring framework self-test...")
    
    # Start monitoring
    framework.start_monitoring(interval=5)
    
    # Create custom metrics
    print("\nCreating custom metrics:")
    
    # Counter metric
    counter = framework.create_metric("test.counter", MetricType.COUNTER, "Test counter")
    print(f"  Created counter metric: {counter.name}")
    
    # Gauge metric
    gauge = framework.create_metric("test.gauge", MetricType.GAUGE, "Test gauge", "units")
    print(f"  Created gauge metric: {gauge.name}")
    
    # Histogram metric
    histogram = framework.create_metric("test.histogram", MetricType.HISTOGRAM, "Test histogram", "ms")
    print(f"  Created histogram metric: {histogram.name}")
    
    # Timer metric
    timer = framework.create_timer_metric("test.timer", "Test timer")
    print(f"  Created timer metric: {timer.name}")
    
    # Record metric values
    print("\nRecording metric values:")
    
    # Counter
    framework.record_metric("test.counter", 1)
    framework.record_metric("test.counter", 2)
    print("  Recorded counter values")
    
    # Gauge
    framework.record_metric("test.gauge", 42)
    framework.record_metric("test.gauge", 43)
    print("  Recorded gauge values")
    
    # Histogram
    for i in range(100):
        framework.record_metric("test.histogram", i)
    print("  Recorded histogram values")
    
    # Timer
    framework.start_timer("test.timer")
    time.sleep(0.5)
    elapsed = framework.stop_timer("test.timer")
    print(f"  Recorded timer value: {elapsed:.2f}s")
    
    # Add alert rule
    print("\nAdding alert rule:")
    rule = framework.add_alert_rule(
        name="high_cpu_usage",
        metric_name="system.cpu.usage",
        condition=">",
        threshold=80,
        severity=AlertSeverity.WARNING,
        message_template="CPU usage is high: {value}% (threshold: {threshold}%)"
    )
    print(f"  Added alert rule: {rule.name}")
    
    # Wait for monitoring to collect some data
    print("\nWaiting for monitoring to collect data...")
    time.sleep(10)
    
    # Generate report
    print("\nGenerating monitoring report...")
    report_path = framework.generate_report()
    print(f"  Generated report: {report_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    viz_files = framework.generate_visualizations(
        metrics=["system.cpu.usage", "system.memory.percent", "test.histogram"]
    )
    print(f"  Generated {len(viz_files)} visualization files")
    
    # Get system info
    print("\nGetting system information:")
    sys_info = framework.get_system_info()
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    
    # Stop monitoring
    framework.stop_monitoring()
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()

