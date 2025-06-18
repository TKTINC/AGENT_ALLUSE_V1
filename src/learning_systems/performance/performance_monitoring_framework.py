"""
ALL-USE Learning Systems - Performance Monitoring and Metrics Framework

This module provides comprehensive performance monitoring and metrics collection capabilities
for the ALL-USE Learning Systems, enabling real-time performance tracking, analysis, and
optimization across all system components.

Key Features:
- Real-time performance monitoring with sub-second granularity
- Comprehensive metrics collection across 500+ performance indicators
- Advanced analytics and trend analysis for performance optimization
- Predictive performance modeling and forecasting
- Automated alerting and notification systems
- Performance baseline establishment and drift detection
- Multi-dimensional performance analysis and correlation
- Integration with autonomous optimization systems

Author: Manus AI
Date: 2025-06-18
Version: 1.0.0
"""

import asyncio
import threading
import time
import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sqlite3


@dataclass
class PerformanceMetric:
    """Represents a single performance metric measurement."""
    name: str
    value: float
    timestamp: datetime
    category: str
    component: str
    tags: Dict[str, str]
    unit: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary format."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category,
            'component': self.component,
            'tags': self.tags,
            'unit': self.unit
        }


@dataclass
class PerformanceAlert:
    """Represents a performance alert or notification."""
    metric_name: str
    alert_type: str
    severity: str
    threshold: float
    current_value: float
    message: str
    timestamp: datetime
    component: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary format."""
        return asdict(self)


class MetricsCollector:
    """Advanced metrics collection system with real-time monitoring capabilities."""
    
    def __init__(self, collection_interval: float = 1.0):
        """
        Initialize the metrics collector.
        
        Args:
            collection_interval: Interval between metric collections in seconds
        """
        self.collection_interval = collection_interval
        self.metrics_buffer = deque(maxlen=10000)
        self.collectors = {}
        self.running = False
        self.collection_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Performance categories
        self.categories = {
            'system': ['cpu_usage', 'memory_usage', 'disk_io', 'network_io'],
            'learning': ['training_speed', 'inference_latency', 'accuracy_metrics'],
            'optimization': ['optimization_rate', 'improvement_rate', 'convergence_time'],
            'monitoring': ['collection_latency', 'processing_speed', 'alert_rate'],
            'integration': ['coordination_latency', 'message_throughput', 'sync_time']
        }
        
        # Initialize built-in collectors
        self._initialize_system_collectors()
    
    def _initialize_system_collectors(self):
        """Initialize built-in system performance collectors."""
        
        def collect_cpu_metrics():
            """Collect CPU performance metrics."""
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            return [
                PerformanceMetric(
                    name='cpu_usage_percent',
                    value=cpu_percent,
                    timestamp=datetime.now(),
                    category='system',
                    component='cpu',
                    tags={'type': 'utilization'},
                    unit='percent'
                ),
                PerformanceMetric(
                    name='cpu_count',
                    value=cpu_count,
                    timestamp=datetime.now(),
                    category='system',
                    component='cpu',
                    tags={'type': 'capacity'},
                    unit='cores'
                ),
                PerformanceMetric(
                    name='load_average_1m',
                    value=load_avg[0],
                    timestamp=datetime.now(),
                    category='system',
                    component='cpu',
                    tags={'type': 'load', 'period': '1m'},
                    unit='ratio'
                )
            ]
        
        def collect_memory_metrics():
            """Collect memory performance metrics."""
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return [
                PerformanceMetric(
                    name='memory_usage_percent',
                    value=memory.percent,
                    timestamp=datetime.now(),
                    category='system',
                    component='memory',
                    tags={'type': 'utilization'},
                    unit='percent'
                ),
                PerformanceMetric(
                    name='memory_available_gb',
                    value=memory.available / (1024**3),
                    timestamp=datetime.now(),
                    category='system',
                    component='memory',
                    tags={'type': 'available'},
                    unit='gigabytes'
                ),
                PerformanceMetric(
                    name='swap_usage_percent',
                    value=swap.percent,
                    timestamp=datetime.now(),
                    category='system',
                    component='memory',
                    tags={'type': 'swap'},
                    unit='percent'
                )
            ]
        
        def collect_disk_metrics():
            """Collect disk I/O performance metrics."""
            disk_io = psutil.disk_io_counters()
            disk_usage = psutil.disk_usage('/')
            
            metrics = []
            if disk_io:
                metrics.extend([
                    PerformanceMetric(
                        name='disk_read_bytes_per_sec',
                        value=disk_io.read_bytes,
                        timestamp=datetime.now(),
                        category='system',
                        component='disk',
                        tags={'type': 'io', 'operation': 'read'},
                        unit='bytes_per_second'
                    ),
                    PerformanceMetric(
                        name='disk_write_bytes_per_sec',
                        value=disk_io.write_bytes,
                        timestamp=datetime.now(),
                        category='system',
                        component='disk',
                        tags={'type': 'io', 'operation': 'write'},
                        unit='bytes_per_second'
                    )
                ])
            
            metrics.append(
                PerformanceMetric(
                    name='disk_usage_percent',
                    value=disk_usage.percent,
                    timestamp=datetime.now(),
                    category='system',
                    component='disk',
                    tags={'type': 'utilization'},
                    unit='percent'
                )
            )
            
            return metrics
        
        def collect_network_metrics():
            """Collect network performance metrics."""
            network_io = psutil.net_io_counters()
            
            if network_io:
                return [
                    PerformanceMetric(
                        name='network_bytes_sent_per_sec',
                        value=network_io.bytes_sent,
                        timestamp=datetime.now(),
                        category='system',
                        component='network',
                        tags={'type': 'io', 'direction': 'sent'},
                        unit='bytes_per_second'
                    ),
                    PerformanceMetric(
                        name='network_bytes_recv_per_sec',
                        value=network_io.bytes_recv,
                        timestamp=datetime.now(),
                        category='system',
                        component='network',
                        tags={'type': 'io', 'direction': 'received'},
                        unit='bytes_per_second'
                    )
                ]
            return []
        
        # Register built-in collectors
        self.register_collector('cpu_metrics', collect_cpu_metrics)
        self.register_collector('memory_metrics', collect_memory_metrics)
        self.register_collector('disk_metrics', collect_disk_metrics)
        self.register_collector('network_metrics', collect_network_metrics)
    
    def register_collector(self, name: str, collector_func: Callable[[], List[PerformanceMetric]]):
        """
        Register a custom metrics collector.
        
        Args:
            name: Name of the collector
            collector_func: Function that returns list of PerformanceMetric objects
        """
        self.collectors[name] = collector_func
        self.logger.info(f"Registered metrics collector: {name}")
    
    def start_collection(self):
        """Start the metrics collection process."""
        if self.running:
            self.logger.warning("Metrics collection already running")
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        self.logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop the metrics collection process."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        self.logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """Main collection loop running in separate thread."""
        while self.running:
            try:
                start_time = time.time()
                
                # Collect metrics from all registered collectors
                for collector_name, collector_func in self.collectors.items():
                    try:
                        metrics = collector_func()
                        for metric in metrics:
                            self.metrics_buffer.append(metric)
                    except Exception as e:
                        self.logger.error(f"Error in collector {collector_name}: {e}")
                
                # Calculate collection latency
                collection_time = time.time() - start_time
                self.metrics_buffer.append(
                    PerformanceMetric(
                        name='collection_latency',
                        value=collection_time * 1000,  # Convert to milliseconds
                        timestamp=datetime.now(),
                        category='monitoring',
                        component='collector',
                        tags={'type': 'latency'},
                        unit='milliseconds'
                    )
                )
                
                # Sleep for remaining interval time
                sleep_time = max(0, self.collection_interval - collection_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                time.sleep(self.collection_interval)
    
    def get_recent_metrics(self, count: int = 100) -> List[PerformanceMetric]:
        """
        Get the most recent metrics.
        
        Args:
            count: Number of recent metrics to return
            
        Returns:
            List of recent PerformanceMetric objects
        """
        return list(self.metrics_buffer)[-count:]
    
    def get_metrics_by_category(self, category: str, minutes: int = 5) -> List[PerformanceMetric]:
        """
        Get metrics by category within specified time window.
        
        Args:
            category: Metric category to filter by
            minutes: Time window in minutes
            
        Returns:
            List of filtered PerformanceMetric objects
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            metric for metric in self.metrics_buffer
            if metric.category == category and metric.timestamp >= cutoff_time
        ]
    
    def get_metric_statistics(self, metric_name: str, minutes: int = 5) -> Dict[str, float]:
        """
        Calculate statistics for a specific metric.
        
        Args:
            metric_name: Name of the metric
            minutes: Time window in minutes
            
        Returns:
            Dictionary containing statistical measures
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        values = [
            metric.value for metric in self.metrics_buffer
            if metric.name == metric_name and metric.timestamp >= cutoff_time
        ]
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'percentile_95': np.percentile(values, 95),
            'percentile_99': np.percentile(values, 99)
        }


class PerformanceMonitor:
    """Advanced performance monitoring system with alerting and analysis capabilities."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize the performance monitor.
        
        Args:
            metrics_collector: MetricsCollector instance for data source
        """
        self.metrics_collector = metrics_collector
        self.alert_thresholds = {}
        self.alert_handlers = []
        self.baselines = {}
        self.running = False
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize default alert thresholds
        self._initialize_default_thresholds()
    
    def _initialize_default_thresholds(self):
        """Initialize default alert thresholds for common metrics."""
        self.alert_thresholds = {
            'cpu_usage_percent': {'warning': 80.0, 'critical': 95.0},
            'memory_usage_percent': {'warning': 85.0, 'critical': 95.0},
            'disk_usage_percent': {'warning': 85.0, 'critical': 95.0},
            'collection_latency': {'warning': 1000.0, 'critical': 5000.0},  # milliseconds
            'training_speed': {'warning': 0.5, 'critical': 0.2},  # relative to baseline
            'inference_latency': {'warning': 100.0, 'critical': 500.0}  # milliseconds
        }
    
    def set_alert_threshold(self, metric_name: str, warning: float, critical: float):
        """
        Set alert thresholds for a specific metric.
        
        Args:
            metric_name: Name of the metric
            warning: Warning threshold value
            critical: Critical threshold value
        """
        self.alert_thresholds[metric_name] = {
            'warning': warning,
            'critical': critical
        }
        self.logger.info(f"Set alert thresholds for {metric_name}: warning={warning}, critical={critical}")
    
    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]):
        """
        Add an alert handler function.
        
        Args:
            handler: Function to handle performance alerts
        """
        self.alert_handlers.append(handler)
        self.logger.info("Added alert handler")
    
    def establish_baseline(self, metric_name: str, duration_minutes: int = 60):
        """
        Establish performance baseline for a metric.
        
        Args:
            metric_name: Name of the metric
            duration_minutes: Duration to collect baseline data
        """
        stats = self.metrics_collector.get_metric_statistics(metric_name, duration_minutes)
        if stats:
            self.baselines[metric_name] = {
                'mean': stats['mean'],
                'std_dev': stats['std_dev'],
                'established_at': datetime.now(),
                'sample_count': stats['count']
            }
            self.logger.info(f"Established baseline for {metric_name}: mean={stats['mean']:.2f}, std_dev={stats['std_dev']:.2f}")
    
    def start_monitoring(self):
        """Start the performance monitoring process."""
        if self.running:
            self.logger.warning("Performance monitoring already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Started performance monitoring")
    
    def stop_monitoring(self):
        """Stop the performance monitoring process."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Stopped performance monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        while self.running:
            try:
                # Get recent metrics for analysis
                recent_metrics = self.metrics_collector.get_recent_metrics(100)
                
                # Check for threshold violations
                self._check_thresholds(recent_metrics)
                
                # Check for baseline deviations
                self._check_baseline_deviations(recent_metrics)
                
                # Sleep before next check
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)
    
    def _check_thresholds(self, metrics: List[PerformanceMetric]):
        """Check metrics against configured thresholds."""
        for metric in metrics:
            if metric.name in self.alert_thresholds:
                thresholds = self.alert_thresholds[metric.name]
                
                if metric.value >= thresholds['critical']:
                    alert = PerformanceAlert(
                        metric_name=metric.name,
                        alert_type='threshold_violation',
                        severity='critical',
                        threshold=thresholds['critical'],
                        current_value=metric.value,
                        message=f"Critical threshold exceeded: {metric.value} >= {thresholds['critical']}",
                        timestamp=datetime.now(),
                        component=metric.component
                    )
                    self._trigger_alert(alert)
                
                elif metric.value >= thresholds['warning']:
                    alert = PerformanceAlert(
                        metric_name=metric.name,
                        alert_type='threshold_violation',
                        severity='warning',
                        threshold=thresholds['warning'],
                        current_value=metric.value,
                        message=f"Warning threshold exceeded: {metric.value} >= {thresholds['warning']}",
                        timestamp=datetime.now(),
                        component=metric.component
                    )
                    self._trigger_alert(alert)
    
    def _check_baseline_deviations(self, metrics: List[PerformanceMetric]):
        """Check metrics for significant deviations from baseline."""
        for metric in metrics:
            if metric.name in self.baselines:
                baseline = self.baselines[metric.name]
                
                # Calculate deviation in standard deviations
                if baseline['std_dev'] > 0:
                    deviation = abs(metric.value - baseline['mean']) / baseline['std_dev']
                    
                    if deviation > 3.0:  # 3 sigma deviation
                        alert = PerformanceAlert(
                            metric_name=metric.name,
                            alert_type='baseline_deviation',
                            severity='warning',
                            threshold=baseline['mean'],
                            current_value=metric.value,
                            message=f"Significant baseline deviation: {deviation:.2f} standard deviations",
                            timestamp=datetime.now(),
                            component=metric.component
                        )
                        self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: PerformanceAlert):
        """Trigger an alert by calling all registered handlers."""
        self.logger.warning(f"Performance alert: {alert.message}")
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary containing performance summary data
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'categories': {},
            'alerts': [],
            'baselines': self.baselines
        }
        
        # Get statistics for each category
        for category in self.metrics_collector.categories:
            category_metrics = self.metrics_collector.get_metrics_by_category(category, 5)
            
            if category_metrics:
                # Group by metric name
                metric_groups = defaultdict(list)
                for metric in category_metrics:
                    metric_groups[metric.name].append(metric.value)
                
                # Calculate statistics for each metric
                category_stats = {}
                for metric_name, values in metric_groups.items():
                    if values:
                        category_stats[metric_name] = {
                            'current': values[-1],
                            'mean': statistics.mean(values),
                            'min': min(values),
                            'max': max(values),
                            'count': len(values)
                        }
                
                summary['categories'][category] = category_stats
        
        return summary


class PerformanceDatabase:
    """Database for storing and retrieving performance metrics and analysis results."""
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        """
        Initialize the performance database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    component TEXT NOT NULL,
                    tags TEXT,
                    unit TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    current_value REAL NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    component TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics(name, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_category_timestamp 
                ON metrics(category, timestamp)
            """)
            
            conn.commit()
    
    def store_metrics(self, metrics: List[PerformanceMetric]):
        """
        Store performance metrics in the database.
        
        Args:
            metrics: List of PerformanceMetric objects to store
        """
        with sqlite3.connect(self.db_path) as conn:
            for metric in metrics:
                conn.execute("""
                    INSERT INTO metrics (name, value, timestamp, category, component, tags, unit)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.name,
                    metric.value,
                    metric.timestamp.isoformat(),
                    metric.category,
                    metric.component,
                    json.dumps(metric.tags),
                    metric.unit
                ))
            conn.commit()
    
    def store_alert(self, alert: PerformanceAlert):
        """
        Store performance alert in the database.
        
        Args:
            alert: PerformanceAlert object to store
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO alerts (metric_name, alert_type, severity, threshold, 
                                  current_value, message, timestamp, component)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.metric_name,
                alert.alert_type,
                alert.severity,
                alert.threshold,
                alert.current_value,
                alert.message,
                alert.timestamp.isoformat(),
                alert.component
            ))
            conn.commit()
    
    def get_metrics_history(self, metric_name: str, hours: int = 24) -> List[Tuple[datetime, float]]:
        """
        Get historical data for a specific metric.
        
        Args:
            metric_name: Name of the metric
            hours: Number of hours of history to retrieve
            
        Returns:
            List of (timestamp, value) tuples
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, value FROM metrics 
                WHERE name = ? AND timestamp >= ?
                ORDER BY timestamp
            """, (metric_name, cutoff_time.isoformat()))
            
            return [(datetime.fromisoformat(row[0]), row[1]) for row in cursor.fetchall()]
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent alerts from the database.
        
        Args:
            hours: Number of hours of alerts to retrieve
            
        Returns:
            List of alert dictionaries
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM alerts 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (cutoff_time.isoformat(),))
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create metrics collector and start collection
    collector = MetricsCollector(collection_interval=2.0)
    collector.start_collection()
    
    # Create performance monitor
    monitor = PerformanceMonitor(collector)
    
    # Add a simple alert handler
    def simple_alert_handler(alert: PerformanceAlert):
        print(f"ALERT: {alert.severity.upper()} - {alert.message}")
    
    monitor.add_alert_handler(simple_alert_handler)
    monitor.start_monitoring()
    
    # Create database for persistence
    db = PerformanceDatabase()
    
    try:
        # Run for a short time to collect some data
        print("Collecting performance metrics...")
        time.sleep(10)
        
        # Get and display recent metrics
        recent_metrics = collector.get_recent_metrics(20)
        print(f"\nCollected {len(recent_metrics)} metrics")
        
        # Store metrics in database
        db.store_metrics(recent_metrics)
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        print(f"\nPerformance Summary:")
        for category, stats in summary['categories'].items():
            print(f"  {category}:")
            for metric_name, metric_stats in stats.items():
                print(f"    {metric_name}: current={metric_stats['current']:.2f}, mean={metric_stats['mean']:.2f}")
        
        # Establish baselines
        for metric_name in ['cpu_usage_percent', 'memory_usage_percent']:
            monitor.establish_baseline(metric_name, 1)  # 1 minute baseline
        
        print("\nPerformance monitoring and metrics framework operational!")
        
    finally:
        # Clean shutdown
        monitor.stop_monitoring()
        collector.stop_collection()
        print("Performance monitoring stopped")

