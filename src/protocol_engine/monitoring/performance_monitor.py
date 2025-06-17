#!/usr/bin/env python3
"""
ALL-USE Agent - Protocol Engine Performance Monitor
P5 of WS2: Performance Optimization and Monitoring - Phase 4

This module provides comprehensive performance monitoring and metrics collection
for the Protocol Engine, implementing real-time tracking, alerting, and
dashboard backend capabilities for ongoing performance optimization.

Monitoring Components:
1. Performance Monitor - Real-time performance tracking and measurement
2. Metrics Collector - Comprehensive metrics aggregation and storage
3. Alerting System - Automated performance alerts and notifications
4. Dashboard Backend - Performance visualization and reporting API
5. Health Checker - System health monitoring and diagnostics
6. Monitoring Coordinator - Centralized monitoring management
"""

import time
import threading
import queue
import json
import sqlite3
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = None
    unit: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class Alert:
    """Performance alert data structure"""
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class HealthStatus:
    """System health status"""
    component: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_queue = queue.Queue()
        self.active_timers = {}
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        
        self.monitoring_thread = None
        self.monitoring_active = False
        
        logger.info(f"Performance Monitor initialized (interval: {monitoring_interval}s)")
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Process any queued metrics
                self._process_metrics_queue()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        now = datetime.now()
        
        # Memory metrics
        process = psutil.Process()
        memory_info = process.memory_info()
        
        self.record_gauge("system.memory.rss", memory_info.rss / 1024 / 1024, "MB")
        self.record_gauge("system.memory.vms", memory_info.vms / 1024 / 1024, "MB")
        
        # CPU metrics
        cpu_percent = process.cpu_percent()
        self.record_gauge("system.cpu.percent", cpu_percent, "%")
        
        # Garbage collection metrics
        gc_stats = gc.get_stats()
        for i, stats in enumerate(gc_stats):
            self.record_gauge(f"system.gc.generation_{i}.collections", stats['collections'], "count")
            self.record_gauge(f"system.gc.generation_{i}.collected", stats['collected'], "count")
        
        # Thread metrics
        thread_count = threading.active_count()
        self.record_gauge("system.threads.active", thread_count, "count")
    
    def _process_metrics_queue(self):
        """Process metrics from the queue"""
        processed_count = 0
        
        while not self.metrics_queue.empty() and processed_count < 100:
            try:
                metric = self.metrics_queue.get_nowait()
                self._store_metric(metric)
                processed_count += 1
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing metric: {e}")
    
    def _store_metric(self, metric: PerformanceMetric):
        """Store metric in appropriate data structure"""
        if metric.metric_type == MetricType.COUNTER:
            self.counters[metric.name] += metric.value
        elif metric.metric_type == MetricType.GAUGE:
            self.gauges[metric.name] = metric.value
        elif metric.metric_type == MetricType.HISTOGRAM:
            self.histograms[metric.name].append(metric.value)
            # Keep only last 1000 values
            if len(self.histograms[metric.name]) > 1000:
                self.histograms[metric.name] = self.histograms[metric.name][-1000:]
    
    def record_counter(self, name: str, value: Union[int, float] = 1, tags: Dict[str, str] = None):
        """Record a counter metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.metrics_queue.put(metric)
    
    def record_gauge(self, name: str, value: Union[int, float], unit: str = "", tags: Dict[str, str] = None):
        """Record a gauge metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit
        )
        self.metrics_queue.put(metric)
    
    def record_histogram(self, name: str, value: Union[int, float], tags: Dict[str, str] = None):
        """Record a histogram metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.metrics_queue.put(metric)
    
    def start_timer(self, name: str, tags: Dict[str, str] = None) -> str:
        """Start a timer and return timer ID"""
        timer_id = f"{name}_{int(time.time() * 1000000)}"
        self.active_timers[timer_id] = {
            'name': name,
            'start_time': time.perf_counter(),
            'tags': tags or {}
        }
        return timer_id
    
    def stop_timer(self, timer_id: str) -> float:
        """Stop a timer and record the duration"""
        if timer_id not in self.active_timers:
            logger.warning(f"Timer {timer_id} not found")
            return 0.0
        
        timer_info = self.active_timers.pop(timer_id)
        duration = time.perf_counter() - timer_info['start_time']
        
        # Record as histogram
        self.record_histogram(f"{timer_info['name']}.duration", duration * 1000, timer_info['tags'])
        
        return duration
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values"""
        # Process any pending metrics
        self._process_metrics_queue()
        
        metrics = {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms': {}
        }
        
        # Calculate histogram statistics
        for name, values in self.histograms.items():
            if values:
                metrics['histograms'][name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                    'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
                }
        
        return metrics


class MetricsCollector:
    """Comprehensive metrics collection and storage"""
    
    def __init__(self, db_path: str = "/tmp/protocol_engine_metrics.db"):
        self.db_path = db_path
        self.metrics_buffer = deque(maxlen=10000)
        self.collection_stats = defaultdict(int)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Metrics Collector initialized (db: {db_path})")
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tags TEXT,
                    unit TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics(name, timestamp)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    current_value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TEXT
                )
            """)
    
    def collect_metric(self, metric: PerformanceMetric):
        """Collect a performance metric"""
        self.metrics_buffer.append(metric)
        self.collection_stats['total_collected'] += 1
        self.collection_stats[f'type_{metric.metric_type.value}'] += 1
        
        # Flush buffer if it's getting full
        if len(self.metrics_buffer) >= 1000:
            self.flush_metrics()
    
    def flush_metrics(self):
        """Flush metrics buffer to database"""
        if not self.metrics_buffer:
            return
        
        metrics_to_store = list(self.metrics_buffer)
        self.metrics_buffer.clear()
        
        with sqlite3.connect(self.db_path) as conn:
            for metric in metrics_to_store:
                conn.execute("""
                    INSERT INTO metrics (name, value, metric_type, timestamp, tags, unit)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric.name,
                    metric.value,
                    metric.metric_type.value,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.tags) if metric.tags else None,
                    metric.unit
                ))
        
        self.collection_stats['flushed_to_db'] += len(metrics_to_store)
        logger.debug(f"Flushed {len(metrics_to_store)} metrics to database")
    
    def query_metrics(self, name: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Query metrics from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM metrics 
                WHERE name = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """, (name, start_time.isoformat(), end_time.isoformat()))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_metric_summary(self, name: str, hours: int = 24) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = self.query_metrics(name, start_time, end_time)
        
        if not metrics:
            return {}
        
        values = [m['value'] for m in metrics]
        
        return {
            'name': name,
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'latest': values[-1],
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get metrics collection statistics"""
        return dict(self.collection_stats)


class AlertingSystem:
    """Automated performance alerting system"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_callbacks = []
        
        logger.info("Alerting System initialized")
    
    def add_alert_rule(self, name: str, metric_name: str, threshold: float, 
                      comparison: str = "greater", level: AlertLevel = AlertLevel.WARNING,
                      cooldown_minutes: int = 5):
        """Add an alert rule"""
        self.alert_rules[name] = {
            'metric_name': metric_name,
            'threshold': threshold,
            'comparison': comparison,  # greater, less, equal
            'level': level,
            'cooldown_minutes': cooldown_minutes,
            'last_triggered': None
        }
        
        logger.info(f"Added alert rule '{name}' for metric '{metric_name}'")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def check_alerts(self, current_metrics: Dict[str, Any]):
        """Check all alert rules against current metrics"""
        now = datetime.now()
        
        for rule_name, rule in self.alert_rules.items():
            metric_name = rule['metric_name']
            
            # Get current metric value
            current_value = self._get_metric_value(current_metrics, metric_name)
            if current_value is None:
                continue
            
            # Check if alert should trigger
            should_alert = self._evaluate_alert_condition(current_value, rule)
            
            if should_alert:
                # Check cooldown
                if rule['last_triggered']:
                    time_since_last = now - rule['last_triggered']
                    if time_since_last.total_seconds() < rule['cooldown_minutes'] * 60:
                        continue
                
                # Trigger alert
                alert = Alert(
                    id=f"{rule_name}_{int(now.timestamp())}",
                    level=rule['level'],
                    message=f"Alert: {metric_name} is {current_value} (threshold: {rule['threshold']})",
                    metric_name=metric_name,
                    threshold=rule['threshold'],
                    current_value=current_value,
                    timestamp=now
                )
                
                self._trigger_alert(alert)
                rule['last_triggered'] = now
    
    def _get_metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value from metrics dictionary"""
        # Check gauges first
        if metric_name in metrics.get('gauges', {}):
            return metrics['gauges'][metric_name]
        
        # Check counters
        if metric_name in metrics.get('counters', {}):
            return metrics['counters'][metric_name]
        
        # Check histograms (use mean)
        if metric_name in metrics.get('histograms', {}):
            return metrics['histograms'][metric_name].get('mean')
        
        return None
    
    def _evaluate_alert_condition(self, current_value: float, rule: Dict[str, Any]) -> bool:
        """Evaluate if alert condition is met"""
        threshold = rule['threshold']
        comparison = rule['comparison']
        
        if comparison == "greater":
            return current_value > threshold
        elif comparison == "less":
            return current_value < threshold
        elif comparison == "equal":
            return abs(current_value - threshold) < 0.001
        
        return False
    
    def _trigger_alert(self, alert: Alert):
        """Trigger an alert"""
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Store alert in database
        with sqlite3.connect(self.metrics_collector.db_path) as conn:
            conn.execute("""
                INSERT INTO alerts (id, level, message, metric_name, threshold, current_value, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id, alert.level.value, alert.message, alert.metric_name,
                alert.threshold, alert.current_value, alert.timestamp.isoformat()
            ))
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"ALERT TRIGGERED: {alert.message}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            alert.resolved = True
            alert.resolved_at = datetime.now()
            
            # Update database
            with sqlite3.connect(self.metrics_collector.db_path) as conn:
                conn.execute("""
                    UPDATE alerts SET resolved = 1, resolved_at = ?
                    WHERE id = ?
                """, (alert.resolved_at.isoformat(), alert_id))
            
            logger.info(f"Alert resolved: {alert_id}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]


class HealthChecker:
    """System health monitoring and diagnostics"""
    
    def __init__(self):
        self.health_checks = {}
        self.health_status = {}
        self.check_interval = 30  # seconds
        
        # Register default health checks
        self._register_default_checks()
        
        logger.info("Health Checker initialized")
    
    def _register_default_checks(self):
        """Register default health checks"""
        self.register_health_check("memory_usage", self._check_memory_usage)
        self.register_health_check("cpu_usage", self._check_cpu_usage)
        self.register_health_check("thread_count", self._check_thread_count)
        self.register_health_check("gc_performance", self._check_gc_performance)
    
    def register_health_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Register a health check function"""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def run_health_checks(self) -> Dict[str, HealthStatus]:
        """Run all health checks"""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                check_result = check_func()
                
                status = HealthStatus(
                    component=name,
                    status=check_result.get('status', 'unknown'),
                    last_check=datetime.now(),
                    details=check_result.get('details', {})
                )
                
                results[name] = status
                self.health_status[name] = status
                
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                results[name] = HealthStatus(
                    component=name,
                    status='unhealthy',
                    last_check=datetime.now(),
                    details={'error': str(e)}
                )
        
        return results
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage health"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb < 100:
            status = "healthy"
        elif memory_mb < 200:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            'status': status,
            'details': {
                'memory_mb': memory_mb,
                'threshold_healthy': 100,
                'threshold_degraded': 200
            }
        }
    
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage health"""
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=1)
        
        if cpu_percent < 50:
            status = "healthy"
        elif cpu_percent < 80:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            'status': status,
            'details': {
                'cpu_percent': cpu_percent,
                'threshold_healthy': 50,
                'threshold_degraded': 80
            }
        }
    
    def _check_thread_count(self) -> Dict[str, Any]:
        """Check thread count health"""
        thread_count = threading.active_count()
        
        if thread_count < 20:
            status = "healthy"
        elif thread_count < 50:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            'status': status,
            'details': {
                'thread_count': thread_count,
                'threshold_healthy': 20,
                'threshold_degraded': 50
            }
        }
    
    def _check_gc_performance(self) -> Dict[str, Any]:
        """Check garbage collection performance"""
        gc_stats = gc.get_stats()
        total_collections = sum(stats['collections'] for stats in gc_stats)
        
        if total_collections < 100:
            status = "healthy"
        elif total_collections < 500:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            'status': status,
            'details': {
                'total_collections': total_collections,
                'gc_stats': gc_stats,
                'threshold_healthy': 100,
                'threshold_degraded': 500
            }
        }
    
    def get_overall_health(self) -> str:
        """Get overall system health status"""
        if not self.health_status:
            return "unknown"
        
        statuses = [status.status for status in self.health_status.values()]
        
        if any(status == "unhealthy" for status in statuses):
            return "unhealthy"
        elif any(status == "degraded" for status in statuses):
            return "degraded"
        else:
            return "healthy"


class MonitoringCoordinator:
    """Centralized monitoring management and coordination"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor(monitoring_interval=1.0)
        self.metrics_collector = MetricsCollector()
        self.alerting_system = AlertingSystem(self.metrics_collector)
        self.health_checker = HealthChecker()
        
        self.monitoring_active = False
        self.coordination_thread = None
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        logger.info("Monitoring Coordinator initialized")
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        # Memory usage alerts
        self.alerting_system.add_alert_rule(
            "high_memory_usage",
            "system.memory.rss",
            150.0,  # 150 MB
            "greater",
            AlertLevel.WARNING
        )
        
        self.alerting_system.add_alert_rule(
            "critical_memory_usage",
            "system.memory.rss",
            200.0,  # 200 MB
            "greater",
            AlertLevel.CRITICAL
        )
        
        # CPU usage alerts
        self.alerting_system.add_alert_rule(
            "high_cpu_usage",
            "system.cpu.percent",
            70.0,  # 70%
            "greater",
            AlertLevel.WARNING
        )
        
        # Add alert callback for logging
        self.alerting_system.add_alert_callback(self._log_alert)
    
    def _log_alert(self, alert: Alert):
        """Log alert callback"""
        logger.warning(f"PERFORMANCE ALERT [{alert.level.value.upper()}]: {alert.message}")
    
    def start_monitoring(self):
        """Start comprehensive monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        # Start coordination thread
        self.coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self.coordination_thread.start()
        
        logger.info("Comprehensive monitoring started")
    
    def stop_monitoring(self):
        """Stop comprehensive monitoring"""
        self.monitoring_active = False
        
        # Stop performance monitoring
        self.performance_monitor.stop_monitoring()
        
        # Stop coordination thread
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5.0)
        
        # Flush any remaining metrics
        self.metrics_collector.flush_metrics()
        
        logger.info("Comprehensive monitoring stopped")
    
    def _coordination_loop(self):
        """Main coordination loop"""
        while self.monitoring_active:
            try:
                # Get current metrics
                current_metrics = self.performance_monitor.get_current_metrics()
                
                # Collect metrics for storage
                for metric_type, metrics in current_metrics.items():
                    for name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            metric = PerformanceMetric(
                                name=name,
                                value=value,
                                metric_type=MetricType.GAUGE,
                                timestamp=datetime.now()
                            )
                            self.metrics_collector.collect_metric(metric)
                
                # Check alerts
                self.alerting_system.check_alerts(current_metrics)
                
                # Run health checks (every 30 seconds)
                if int(time.time()) % 30 == 0:
                    self.health_checker.run_health_checks()
                
                time.sleep(5.0)  # Coordination interval
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                time.sleep(5.0)
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring data for dashboard"""
        current_metrics = self.performance_monitor.get_current_metrics()
        active_alerts = self.alerting_system.get_active_alerts()
        health_status = self.health_checker.run_health_checks()
        collection_stats = self.metrics_collector.get_collection_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics,
            'active_alerts': [asdict(alert) for alert in active_alerts],
            'health_status': {name: asdict(status) for name, status in health_status.items()},
            'overall_health': self.health_checker.get_overall_health(),
            'collection_stats': collection_stats,
            'monitoring_active': self.monitoring_active
        }


# Global monitoring coordinator instance
_global_monitoring_coordinator = None


def get_monitoring_coordinator() -> MonitoringCoordinator:
    """Get the global monitoring coordinator instance"""
    global _global_monitoring_coordinator
    if _global_monitoring_coordinator is None:
        _global_monitoring_coordinator = MonitoringCoordinator()
        logger.info("Global monitoring coordinator initialized")
    
    return _global_monitoring_coordinator


# Convenience functions for easy integration
def monitor_function_performance(func_name: str):
    """Decorator to monitor function performance"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            coordinator = get_monitoring_coordinator()
            
            # Start timer
            timer_id = coordinator.performance_monitor.start_timer(f"function.{func_name}")
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Record success
                coordinator.performance_monitor.record_counter(f"function.{func_name}.success")
                
                return result
                
            except Exception as e:
                # Record error
                coordinator.performance_monitor.record_counter(f"function.{func_name}.error")
                raise
                
            finally:
                # Stop timer
                coordinator.performance_monitor.stop_timer(timer_id)
        
        return wrapper
    return decorator


if __name__ == '__main__':
    print("📊 Testing Monitoring and Metrics Collection Framework (P5 of WS2 - Phase 4)")
    print("=" * 85)
    
    # Initialize monitoring coordinator
    coordinator = get_monitoring_coordinator()
    
    print("\n🚀 Starting Comprehensive Monitoring:")
    coordinator.start_monitoring()
    
    # Test function monitoring
    @monitor_function_performance("test_calculation")
    def test_calculation(x, y):
        time.sleep(0.01)  # Simulate work
        return x * y + x ** 2
    
    print("\n📈 Testing Function Performance Monitoring:")
    for i in range(10):
        result = test_calculation(i, i + 1)
        coordinator.performance_monitor.record_gauge("test.result", result)
    
    # Wait for some monitoring data
    print("\n⏱️  Collecting monitoring data...")
    time.sleep(3)
    
    # Get dashboard data
    dashboard_data = coordinator.get_monitoring_dashboard_data()
    
    print("\n📊 Monitoring Dashboard Data:")
    print(f"   Overall Health: {dashboard_data['overall_health']}")
    print(f"   Active Alerts: {len(dashboard_data['active_alerts'])}")
    print(f"   Health Components: {len(dashboard_data['health_status'])}")
    print(f"   Metrics Collected: {dashboard_data['collection_stats'].get('total_collected', 0)}")
    
    # Show current metrics
    current_metrics = dashboard_data['current_metrics']
    print(f"\n📈 Current Metrics Summary:")
    print(f"   Gauges: {len(current_metrics.get('gauges', {}))}")
    print(f"   Counters: {len(current_metrics.get('counters', {}))}")
    print(f"   Histograms: {len(current_metrics.get('histograms', {}))}")
    
    # Show some specific metrics
    if 'gauges' in current_metrics:
        memory_usage = current_metrics['gauges'].get('system.memory.rss', 0)
        cpu_usage = current_metrics['gauges'].get('system.cpu.percent', 0)
        print(f"   Memory Usage: {memory_usage:.2f} MB")
        print(f"   CPU Usage: {cpu_usage:.2f}%")
    
    # Test alerting by triggering a high memory alert
    print("\n🚨 Testing Alerting System:")
    coordinator.performance_monitor.record_gauge("system.memory.rss", 180.0)  # Above warning threshold
    time.sleep(2)
    
    # Check for alerts
    dashboard_data = coordinator.get_monitoring_dashboard_data()
    active_alerts = dashboard_data['active_alerts']
    if active_alerts:
        print(f"   Alert triggered: {active_alerts[0]['message']}")
    else:
        print("   No alerts triggered")
    
    print("\n🏥 Health Check Results:")
    health_status = dashboard_data['health_status']
    for component, status in health_status.items():
        print(f"   {component}: {status['status']}")
    
    print("\n🛑 Stopping Monitoring:")
    coordinator.stop_monitoring()
    
    print("\n✅ P5 of WS2 - Phase 4: Monitoring and Metrics Collection Framework COMPLETE")
    print("🚀 Ready for Phase 5: Performance Analytics and Real-time Tracking")

