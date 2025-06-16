"""
ALL-USE Monitoring Infrastructure

This module provides comprehensive monitoring and real-time analytics for all WS1 components,
including performance metrics, system health monitoring, and alerting capabilities.
"""

import time
import threading
import queue
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import asyncio
import weakref
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger('all_use_monitoring')


@dataclass
class MetricPoint:
    """Represents a single metric data point."""
    timestamp: datetime
    name: str
    value: Union[int, float]
    tags: Dict[str, str]
    unit: str = ""


@dataclass
class Alert:
    """Represents a monitoring alert."""
    id: str
    name: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    source: str
    tags: Dict[str, str]
    acknowledged: bool = False


class MetricsCollector:
    """
    Comprehensive metrics collection system for ALL-USE components.
    
    Provides:
    - Real-time performance metrics collection
    - System health monitoring
    - Custom metric tracking
    - Metric aggregation and storage
    """
    
    def __init__(self, max_points: int = 10000):
        """Initialize the metrics collector."""
        self.max_points = max_points
        self.metrics = defaultdict(lambda: deque(maxlen=max_points))
        self.metric_configs = {}
        self.collection_enabled = True
        
        # Background collection thread
        self.collection_thread = None
        self.collection_interval = 1.0  # seconds
        self.stop_collection = threading.Event()
        
        # Metric aggregations
        self.aggregations = defaultdict(dict)
        
        logger.info("Metrics collector initialized")
    
    def start_collection(self):
        """Start background metrics collection."""
        if self.collection_thread and self.collection_thread.is_alive():
            return
        
        self.stop_collection.clear()
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info("Background metrics collection started")
    
    def stop_collection_service(self):
        """Stop background metrics collection."""
        if self.collection_thread and self.collection_thread.is_alive():
            self.stop_collection.set()
            self.collection_thread.join(timeout=5.0)
        
        logger.info("Background metrics collection stopped")
    
    def record_metric(self, name: str, value: Union[int, float], tags: Dict[str, str] = None, unit: str = ""):
        """Record a metric data point."""
        if not self.collection_enabled:
            return
        
        tags = tags or {}
        point = MetricPoint(
            timestamp=datetime.now(),
            name=name,
            value=value,
            tags=tags,
            unit=unit
        )
        
        self.metrics[name].append(point)
        self._update_aggregations(name, value)
    
    def record_timing(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record a timing metric."""
        self.record_metric(name, duration_ms, tags, "ms")
    
    def record_counter(self, name: str, increment: int = 1, tags: Dict[str, str] = None):
        """Record a counter metric."""
        # Get current counter value
        current_value = self.get_latest_value(name, default=0)
        new_value = current_value + increment
        self.record_metric(name, new_value, tags, "count")
    
    def record_gauge(self, name: str, value: Union[int, float], tags: Dict[str, str] = None, unit: str = ""):
        """Record a gauge metric."""
        self.record_metric(name, value, tags, unit)
    
    def get_metrics(self, name: str, since: datetime = None, limit: int = None) -> List[MetricPoint]:
        """Get metric data points."""
        if name not in self.metrics:
            return []
        
        points = list(self.metrics[name])
        
        # Filter by time
        if since:
            points = [p for p in points if p.timestamp >= since]
        
        # Apply limit
        if limit:
            points = points[-limit:]
        
        return points
    
    def get_latest_value(self, name: str, default: Union[int, float] = None) -> Union[int, float]:
        """Get the latest value for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return default
        
        return self.metrics[name][-1].value
    
    def get_metric_summary(self, name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get summary statistics for a metric over a time window."""
        since = datetime.now() - timedelta(minutes=window_minutes)
        points = self.get_metrics(name, since=since)
        
        if not points:
            return {'count': 0, 'min': None, 'max': None, 'avg': None, 'latest': None}
        
        values = [p.value for p in points]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1],
            'unit': points[-1].unit if points else "",
            'window_minutes': window_minutes
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary for all metrics."""
        return {
            name: self.get_metric_summary(name)
            for name in self.metrics.keys()
        }
    
    def _collection_loop(self):
        """Background collection loop for system metrics."""
        while not self.stop_collection.wait(self.collection_interval):
            try:
                self._collect_system_metrics()
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        process = psutil.Process()
        
        # CPU metrics
        self.record_gauge("system.cpu.percent", psutil.cpu_percent(), unit="%")
        self.record_gauge("process.cpu.percent", process.cpu_percent(), unit="%")
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.record_gauge("system.memory.percent", memory.percent, unit="%")
        self.record_gauge("system.memory.available_mb", memory.available / 1024 / 1024, unit="MB")
        
        process_memory = process.memory_info()
        self.record_gauge("process.memory.rss_mb", process_memory.rss / 1024 / 1024, unit="MB")
        self.record_gauge("process.memory.vms_mb", process_memory.vms / 1024 / 1024, unit="MB")
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.record_gauge("system.disk.percent", disk.percent, unit="%")
        self.record_gauge("system.disk.free_gb", disk.free / 1024 / 1024 / 1024, unit="GB")
        
        # Network metrics (if available)
        try:
            network = psutil.net_io_counters()
            self.record_gauge("system.network.bytes_sent", network.bytes_sent, unit="bytes")
            self.record_gauge("system.network.bytes_recv", network.bytes_recv, unit="bytes")
        except:
            pass  # Network metrics not available
        
        # Process metrics
        self.record_gauge("process.threads", process.num_threads(), unit="count")
        self.record_gauge("process.fds", process.num_fds(), unit="count")
    
    def _update_aggregations(self, name: str, value: Union[int, float]):
        """Update metric aggregations."""
        if name not in self.aggregations:
            self.aggregations[name] = {
                'count': 0,
                'sum': 0,
                'min': float('inf'),
                'max': float('-inf'),
                'last_reset': datetime.now()
            }
        
        agg = self.aggregations[name]
        agg['count'] += 1
        agg['sum'] += value
        agg['min'] = min(agg['min'], value)
        agg['max'] = max(agg['max'], value)


class HealthMonitor:
    """
    System health monitoring for ALL-USE components.
    
    Provides:
    - Component health checks
    - Service availability monitoring
    - Health status aggregation
    - Health-based alerting
    """
    
    def __init__(self):
        """Initialize the health monitor."""
        self.health_checks = {}
        self.health_status = {}
        self.health_history = defaultdict(lambda: deque(maxlen=100))
        
        # Health check execution
        self.check_interval = 30.0  # seconds
        self.check_thread = None
        self.stop_checks = threading.Event()
        
        logger.info("Health monitor initialized")
    
    def register_health_check(self, name: str, check_func: Callable[[], bool], 
                            critical: bool = False, timeout: float = 5.0):
        """Register a health check function."""
        self.health_checks[name] = {
            'func': check_func,
            'critical': critical,
            'timeout': timeout,
            'last_check': None,
            'last_result': None,
            'consecutive_failures': 0
        }
        
        logger.info(f"Health check '{name}' registered (critical: {critical})")
    
    def start_monitoring(self):
        """Start background health monitoring."""
        if self.check_thread and self.check_thread.is_alive():
            return
        
        self.stop_checks.clear()
        self.check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.check_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop background health monitoring."""
        if self.check_thread and self.check_thread.is_alive():
            self.stop_checks.set()
            self.check_thread.join(timeout=5.0)
        
        logger.info("Health monitoring stopped")
    
    def run_health_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.health_checks:
            return {'status': 'unknown', 'error': f"Health check '{name}' not found"}
        
        check_info = self.health_checks[name]
        start_time = time.time()
        
        try:
            # Run health check with timeout
            result = self._run_with_timeout(check_info['func'], check_info['timeout'])
            duration = time.time() - start_time
            
            status = 'healthy' if result else 'unhealthy'
            
            # Update check info
            check_info['last_check'] = datetime.now()
            check_info['last_result'] = result
            
            if result:
                check_info['consecutive_failures'] = 0
            else:
                check_info['consecutive_failures'] += 1
            
            # Record health status
            health_record = {
                'timestamp': datetime.now(),
                'status': status,
                'duration_ms': duration * 1000,
                'consecutive_failures': check_info['consecutive_failures']
            }
            
            self.health_history[name].append(health_record)
            self.health_status[name] = health_record
            
            return {
                'status': status,
                'duration_ms': duration * 1000,
                'consecutive_failures': check_info['consecutive_failures'],
                'critical': check_info['critical']
            }
            
        except Exception as e:
            duration = time.time() - start_time
            check_info['consecutive_failures'] += 1
            
            health_record = {
                'timestamp': datetime.now(),
                'status': 'error',
                'error': str(e),
                'duration_ms': duration * 1000,
                'consecutive_failures': check_info['consecutive_failures']
            }
            
            self.health_history[name].append(health_record)
            self.health_status[name] = health_record
            
            return {
                'status': 'error',
                'error': str(e),
                'duration_ms': duration * 1000,
                'consecutive_failures': check_info['consecutive_failures'],
                'critical': check_info['critical']
            }
    
    def run_all_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_status = 'healthy'
        critical_failures = 0
        total_failures = 0
        
        for name in self.health_checks.keys():
            result = self.run_health_check(name)
            results[name] = result
            
            if result['status'] != 'healthy':
                total_failures += 1
                if result['critical']:
                    critical_failures += 1
                    overall_status = 'critical'
                elif overall_status == 'healthy':
                    overall_status = 'degraded'
        
        return {
            'overall_status': overall_status,
            'critical_failures': critical_failures,
            'total_failures': total_failures,
            'total_checks': len(self.health_checks),
            'checks': results,
            'timestamp': datetime.now()
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all components."""
        summary = {
            'overall_status': 'unknown',
            'components': {},
            'last_check': None
        }
        
        if not self.health_status:
            return summary
        
        # Determine overall status
        critical_unhealthy = False
        any_unhealthy = False
        latest_check = None
        
        for name, status in self.health_status.items():
            check_info = self.health_checks[name]
            
            if status['status'] != 'healthy':
                any_unhealthy = True
                if check_info['critical']:
                    critical_unhealthy = True
            
            if latest_check is None or status['timestamp'] > latest_check:
                latest_check = status['timestamp']
            
            summary['components'][name] = {
                'status': status['status'],
                'critical': check_info['critical'],
                'consecutive_failures': status.get('consecutive_failures', 0),
                'last_check': status['timestamp']
            }
        
        if critical_unhealthy:
            summary['overall_status'] = 'critical'
        elif any_unhealthy:
            summary['overall_status'] = 'degraded'
        else:
            summary['overall_status'] = 'healthy'
        
        summary['last_check'] = latest_check
        
        return summary
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self.stop_checks.wait(self.check_interval):
            try:
                self.run_all_health_checks()
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
    
    def _run_with_timeout(self, func: Callable, timeout: float):
        """Run function with timeout."""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Timeout occurred
            raise TimeoutError(f"Health check timed out after {timeout} seconds")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]


class AlertManager:
    """
    Alert management and notification system for ALL-USE components.
    
    Provides:
    - Alert generation and management
    - Alert routing and notification
    - Alert escalation and acknowledgment
    - Alert history and analytics
    """
    
    def __init__(self):
        """Initialize the alert manager."""
        self.alerts = {}  # Active alerts
        self.alert_history = deque(maxlen=1000)
        self.alert_rules = {}
        self.notification_handlers = {}
        
        # Alert processing
        self.alert_queue = queue.Queue()
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
        logger.info("Alert manager initialized")
    
    def start_processing(self):
        """Start background alert processing."""
        if self.processing_thread and self.processing_thread.is_alive():
            return
        
        self.stop_processing.clear()
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("Alert processing started")
    
    def stop_processing_service(self):
        """Stop background alert processing."""
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_processing.set()
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Alert processing stopped")
    
    def create_alert(self, name: str, severity: str, message: str, source: str, 
                    tags: Dict[str, str] = None) -> Alert:
        """Create a new alert."""
        alert_id = f"{source}:{name}:{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            name=name,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            source=source,
            tags=tags or {}
        )
        
        self.alert_queue.put(alert)
        return alert
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert."""
        if alert_id in self.alerts:
            alert = self.alerts.pop(alert_id)
            self.alert_history.append({
                'alert': alert,
                'resolved_at': datetime.now(),
                'resolved_by': resolved_by
            })
            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True
        return False
    
    def register_notification_handler(self, name: str, handler: Callable[[Alert], None]):
        """Register a notification handler."""
        self.notification_handlers[name] = handler
        logger.info(f"Notification handler '{name}' registered")
    
    def add_alert_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool], 
                      severity: str, message_template: str):
        """Add an alert rule."""
        self.alert_rules[name] = {
            'condition': condition,
            'severity': severity,
            'message_template': message_template,
            'last_triggered': None,
            'trigger_count': 0
        }
        logger.info(f"Alert rule '{name}' added")
    
    def evaluate_rules(self, metrics: Dict[str, Any]):
        """Evaluate alert rules against current metrics."""
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule['condition'](metrics):
                    # Rule triggered
                    message = rule['message_template'].format(**metrics)
                    
                    alert = self.create_alert(
                        name=rule_name,
                        severity=rule['severity'],
                        message=message,
                        source='alert_rules',
                        tags={'rule': rule_name}
                    )
                    
                    rule['last_triggered'] = datetime.now()
                    rule['trigger_count'] += 1
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule '{rule_name}': {e}")
    
    def get_active_alerts(self, severity: str = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        active_alerts = list(self.alerts.values())
        
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity] += 1
        
        acknowledged_count = sum(1 for alert in active_alerts if alert.acknowledged)
        
        return {
            'total_active': len(active_alerts),
            'acknowledged': acknowledged_count,
            'unacknowledged': len(active_alerts) - acknowledged_count,
            'by_severity': dict(severity_counts),
            'total_resolved': len(self.alert_history),
            'oldest_active': min((a.timestamp for a in active_alerts), default=None),
            'newest_active': max((a.timestamp for a in active_alerts), default=None)
        }
    
    def _processing_loop(self):
        """Background alert processing loop."""
        while not self.stop_processing.is_set():
            try:
                # Get alert from queue with timeout
                alert = self.alert_queue.get(timeout=1.0)
                
                # Add to active alerts
                self.alerts[alert.id] = alert
                
                # Send notifications
                self._send_notifications(alert)
                
                logger.info(f"Alert processed: {alert.name} ({alert.severity})")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        for handler_name, handler in self.notification_handlers.items():
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in notification handler '{handler_name}': {e}")


# Global monitoring instances
metrics_collector = MetricsCollector()
health_monitor = HealthMonitor()
alert_manager = AlertManager()


# Convenience decorators and functions
def monitor_performance(metric_name: str = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        name = metric_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                duration = (time.perf_counter() - start_time) * 1000
                metrics_collector.record_timing(f"{name}.duration", duration)
                metrics_collector.record_counter(f"{name}.calls")
                return result
                
            except Exception as e:
                duration = (time.perf_counter() - start_time) * 1000
                metrics_collector.record_timing(f"{name}.duration", duration)
                metrics_collector.record_counter(f"{name}.errors")
                raise
        
        return wrapper
    return decorator


def health_check(name: str, critical: bool = False, timeout: float = 5.0):
    """Decorator to register a function as a health check."""
    def decorator(func):
        health_monitor.register_health_check(name, func, critical, timeout)
        return func
    return decorator


if __name__ == "__main__":
    # Test the monitoring system
    import functools
    
    @monitor_performance("test_function")
    def test_function(duration: float = 0.01):
        """Test function for monitoring."""
        time.sleep(duration)
        return "success"
    
    @health_check("test_service", critical=True)
    def test_health_check():
        """Test health check."""
        return True  # Always healthy for test
    
    # Start monitoring
    metrics_collector.start_collection()
    health_monitor.start_monitoring()
    alert_manager.start_processing()
    
    # Test performance monitoring
    print("Testing performance monitoring...")
    for i in range(5):
        result = test_function(0.01)
        print(f"Function call {i+1}: {result}")
    
    # Test health monitoring
    print("\nTesting health monitoring...")
    health_result = health_monitor.run_health_check("test_service")
    print(f"Health check result: {health_result}")
    
    # Test alerting
    print("\nTesting alerting...")
    alert = alert_manager.create_alert(
        name="test_alert",
        severity="medium",
        message="This is a test alert",
        source="test_system"
    )
    print(f"Created alert: {alert.id}")
    
    # Wait a moment for background processing
    time.sleep(2)
    
    # Get monitoring reports
    print("\nMetrics Summary:")
    metrics_summary = metrics_collector.get_all_metrics_summary()
    for name, summary in metrics_summary.items():
        if summary['count'] > 0:
            print(f"  {name}: {summary}")
    
    print("\nHealth Summary:")
    health_summary = health_monitor.get_health_summary()
    print(f"  Overall status: {health_summary['overall_status']}")
    
    print("\nAlert Summary:")
    alert_summary = alert_manager.get_alert_summary()
    print(f"  Active alerts: {alert_summary['total_active']}")
    
    # Stop monitoring
    metrics_collector.stop_collection_service()
    health_monitor.stop_monitoring()
    alert_manager.stop_processing_service()
    
    print("\nMonitoring system test completed successfully!")

