"""
WS5-P5: Performance Monitoring Framework
Advanced performance monitoring and analytics for autonomous learning systems.

This module provides comprehensive performance monitoring capabilities including:
- Real-time metrics collection and analysis
- Multi-dimensional performance tracking
- Anomaly detection and alerting
- Performance baseline management
- Historical trend analysis
"""

import time
import threading
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
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Represents a single performance metric measurement."""
    name: str
    value: float
    timestamp: datetime
    category: str
    unit: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary format."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category,
            'unit': self.unit,
            'metadata': self.metadata or {}
        }

@dataclass
class PerformanceBaseline:
    """Represents performance baseline for a metric."""
    metric_name: str
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_95: float
    sample_count: int
    last_updated: datetime
    
class MetricsCollector:
    """Collects various system and application performance metrics."""
    
    def __init__(self, collection_interval: float = 0.1):
        """
        Initialize metrics collector.
        
        Args:
            collection_interval: Interval between metric collections in seconds
        """
        self.collection_interval = collection_interval
        self.is_collecting = False
        self.metrics_buffer = deque(maxlen=10000)  # Buffer for recent metrics
        self.custom_collectors = {}
        self.collection_thread = None
        
    def register_custom_collector(self, name: str, collector_func: Callable[[], Dict[str, float]]):
        """Register a custom metrics collector function."""
        self.custom_collectors[name] = collector_func
        logger.info(f"Registered custom collector: {name}")
    
    def collect_system_metrics(self) -> List[PerformanceMetric]:
        """Collect system-level performance metrics."""
        timestamp = datetime.now()
        metrics = []
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            metrics.extend([
                PerformanceMetric("cpu_usage_percent", cpu_percent, timestamp, "system", "percent"),
                PerformanceMetric("cpu_count", cpu_count, timestamp, "system", "count"),
                PerformanceMetric("cpu_frequency_mhz", cpu_freq.current if cpu_freq else 0, timestamp, "system", "mhz")
            ])
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics.extend([
                PerformanceMetric("memory_usage_percent", memory.percent, timestamp, "system", "percent"),
                PerformanceMetric("memory_available_gb", memory.available / (1024**3), timestamp, "system", "gb"),
                PerformanceMetric("memory_used_gb", memory.used / (1024**3), timestamp, "system", "gb"),
                PerformanceMetric("swap_usage_percent", swap.percent, timestamp, "system", "percent")
            ])
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            metrics.extend([
                PerformanceMetric("disk_usage_percent", (disk_usage.used / disk_usage.total) * 100, timestamp, "system", "percent"),
                PerformanceMetric("disk_free_gb", disk_usage.free / (1024**3), timestamp, "system", "gb"),
                PerformanceMetric("disk_read_bytes_per_sec", disk_io.read_bytes if disk_io else 0, timestamp, "system", "bytes/sec"),
                PerformanceMetric("disk_write_bytes_per_sec", disk_io.write_bytes if disk_io else 0, timestamp, "system", "bytes/sec")
            ])
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            metrics.extend([
                PerformanceMetric("network_bytes_sent_per_sec", network_io.bytes_sent if network_io else 0, timestamp, "system", "bytes/sec"),
                PerformanceMetric("network_bytes_recv_per_sec", network_io.bytes_recv if network_io else 0, timestamp, "system", "bytes/sec"),
                PerformanceMetric("network_packets_sent_per_sec", network_io.packets_sent if network_io else 0, timestamp, "system", "packets/sec"),
                PerformanceMetric("network_packets_recv_per_sec", network_io.packets_recv if network_io else 0, timestamp, "system", "packets/sec")
            ])
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
        return metrics
    
    def collect_application_metrics(self) -> List[PerformanceMetric]:
        """Collect application-specific performance metrics."""
        timestamp = datetime.now()
        metrics = []
        
        try:
            # Process-specific metrics
            process = psutil.Process()
            
            metrics.extend([
                PerformanceMetric("process_cpu_percent", process.cpu_percent(), timestamp, "application", "percent"),
                PerformanceMetric("process_memory_mb", process.memory_info().rss / (1024**2), timestamp, "application", "mb"),
                PerformanceMetric("process_threads", process.num_threads(), timestamp, "application", "count"),
                PerformanceMetric("process_open_files", len(process.open_files()), timestamp, "application", "count")
            ])
            
            # Custom application metrics
            for collector_name, collector_func in self.custom_collectors.items():
                try:
                    custom_metrics = collector_func()
                    for metric_name, value in custom_metrics.items():
                        metrics.append(
                            PerformanceMetric(
                                f"{collector_name}_{metric_name}", 
                                value, 
                                timestamp, 
                                "application", 
                                "custom"
                            )
                        )
                except Exception as e:
                    logger.error(f"Error in custom collector {collector_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            
        return metrics
    
    def collect_all_metrics(self) -> List[PerformanceMetric]:
        """Collect all available metrics."""
        all_metrics = []
        all_metrics.extend(self.collect_system_metrics())
        all_metrics.extend(self.collect_application_metrics())
        return all_metrics
    
    def start_collection(self):
        """Start continuous metrics collection."""
        if self.is_collecting:
            logger.warning("Metrics collection already running")
            return
            
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """Main collection loop running in separate thread."""
        while self.is_collecting:
            try:
                metrics = self.collect_all_metrics()
                for metric in metrics:
                    self.metrics_buffer.append(metric)
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(1)  # Prevent tight error loop
    
    def get_recent_metrics(self, count: int = 100) -> List[PerformanceMetric]:
        """Get the most recent metrics."""
        return list(self.metrics_buffer)[-count:]

class PerformanceAnalyzer:
    """Analyzes performance metrics for trends, anomalies, and insights."""
    
    def __init__(self, anomaly_threshold: float = 2.0):
        """
        Initialize performance analyzer.
        
        Args:
            anomaly_threshold: Standard deviations from mean to consider anomalous
        """
        self.anomaly_threshold = anomaly_threshold
        self.baselines = {}
        self.analysis_history = deque(maxlen=1000)
        
    def calculate_baseline(self, metrics: List[PerformanceMetric], metric_name: str) -> Optional[PerformanceBaseline]:
        """Calculate performance baseline for a specific metric."""
        metric_values = [m.value for m in metrics if m.name == metric_name]
        
        if len(metric_values) < 10:  # Need minimum samples for baseline
            return None
            
        try:
            mean_val = statistics.mean(metric_values)
            std_dev = statistics.stdev(metric_values) if len(metric_values) > 1 else 0
            min_val = min(metric_values)
            max_val = max(metric_values)
            percentile_95 = np.percentile(metric_values, 95)
            
            baseline = PerformanceBaseline(
                metric_name=metric_name,
                mean=mean_val,
                std_dev=std_dev,
                min_value=min_val,
                max_value=max_val,
                percentile_95=percentile_95,
                sample_count=len(metric_values),
                last_updated=datetime.now()
            )
            
            self.baselines[metric_name] = baseline
            return baseline
            
        except Exception as e:
            logger.error(f"Error calculating baseline for {metric_name}: {e}")
            return None
    
    def detect_anomalies(self, metrics: List[PerformanceMetric]) -> List[Dict[str, Any]]:
        """Detect anomalous performance metrics."""
        anomalies = []
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.name].append(metric)
        
        for metric_name, metric_list in metrics_by_name.items():
            if metric_name not in self.baselines:
                continue
                
            baseline = self.baselines[metric_name]
            
            for metric in metric_list:
                # Check if metric is anomalous
                if baseline.std_dev > 0:
                    z_score = abs(metric.value - baseline.mean) / baseline.std_dev
                    
                    if z_score > self.anomaly_threshold:
                        anomalies.append({
                            'metric': metric,
                            'baseline': baseline,
                            'z_score': z_score,
                            'severity': 'high' if z_score > 3.0 else 'medium',
                            'detected_at': datetime.now()
                        })
        
        return anomalies
    
    def analyze_trends(self, metrics: List[PerformanceMetric], window_size: int = 100) -> Dict[str, Dict[str, Any]]:
        """Analyze performance trends over time."""
        trends = {}
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.name].append(metric)
        
        for metric_name, metric_list in metrics_by_name.items():
            if len(metric_list) < window_size:
                continue
                
            # Sort by timestamp
            metric_list.sort(key=lambda x: x.timestamp)
            recent_metrics = metric_list[-window_size:]
            
            values = [m.value for m in recent_metrics]
            timestamps = [m.timestamp for m in recent_metrics]
            
            # Calculate trend
            if len(values) > 1:
                # Simple linear trend calculation
                x = np.arange(len(values))
                coeffs = np.polyfit(x, values, 1)
                trend_slope = coeffs[0]
                
                # Calculate trend strength
                correlation = np.corrcoef(x, values)[0, 1] if len(values) > 2 else 0
                
                trends[metric_name] = {
                    'slope': trend_slope,
                    'correlation': correlation,
                    'direction': 'increasing' if trend_slope > 0 else 'decreasing' if trend_slope < 0 else 'stable',
                    'strength': abs(correlation),
                    'recent_mean': statistics.mean(values),
                    'recent_std': statistics.stdev(values) if len(values) > 1 else 0,
                    'sample_count': len(values),
                    'time_range': {
                        'start': timestamps[0].isoformat(),
                        'end': timestamps[-1].isoformat()
                    }
                }
        
        return trends
    
    def generate_performance_summary(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Generate comprehensive performance summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_metrics': len(metrics),
            'metric_categories': {},
            'top_metrics': {},
            'anomalies_detected': 0,
            'trends_analyzed': 0
        }
        
        # Categorize metrics
        categories = defaultdict(int)
        for metric in metrics:
            categories[metric.category] += 1
        summary['metric_categories'] = dict(categories)
        
        # Find top metrics by value
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.name].append(metric.value)
        
        for metric_name, values in metrics_by_name.items():
            if values:
                summary['top_metrics'][metric_name] = {
                    'current': values[-1],
                    'average': statistics.mean(values),
                    'max': max(values),
                    'min': min(values)
                }
        
        # Detect anomalies
        anomalies = self.detect_anomalies(metrics)
        summary['anomalies_detected'] = len(anomalies)
        
        # Analyze trends
        trends = self.analyze_trends(metrics)
        summary['trends_analyzed'] = len(trends)
        
        return summary

class PerformanceReporter:
    """Generates performance reports and visualizations."""
    
    def __init__(self, output_dir: str = "performance_reports"):
        """Initialize performance reporter."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        
    def generate_text_report(self, metrics: List[PerformanceMetric], analyzer: PerformanceAnalyzer) -> str:
        """Generate text-based performance report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PERFORMANCE MONITORING REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Metrics: {len(metrics)}")
        report_lines.append("")
        
        # Summary statistics
        summary = analyzer.generate_performance_summary(metrics)
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 40)
        for category, count in summary['metric_categories'].items():
            report_lines.append(f"{category.capitalize()}: {count} metrics")
        report_lines.append("")
        
        # Top metrics
        report_lines.append("TOP PERFORMANCE METRICS")
        report_lines.append("-" * 40)
        for metric_name, stats in list(summary['top_metrics'].items())[:10]:
            report_lines.append(f"{metric_name}:")
            report_lines.append(f"  Current: {stats['current']:.2f}")
            report_lines.append(f"  Average: {stats['average']:.2f}")
            report_lines.append(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
        report_lines.append("")
        
        # Anomalies
        anomalies = analyzer.detect_anomalies(metrics)
        if anomalies:
            report_lines.append("ANOMALIES DETECTED")
            report_lines.append("-" * 40)
            for anomaly in anomalies[:5]:  # Top 5 anomalies
                metric = anomaly['metric']
                report_lines.append(f"{metric.name}: {metric.value:.2f} (Z-score: {anomaly['z_score']:.2f})")
        else:
            report_lines.append("No anomalies detected")
        report_lines.append("")
        
        # Trends
        trends = analyzer.analyze_trends(metrics)
        if trends:
            report_lines.append("PERFORMANCE TRENDS")
            report_lines.append("-" * 40)
            for metric_name, trend in list(trends.items())[:5]:  # Top 5 trends
                report_lines.append(f"{metric_name}: {trend['direction']} (strength: {trend['strength']:.2f})")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        return "\n".join(report_lines)
    
    def save_report(self, report_content: str, filename: str = None) -> str:
        """Save report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Performance report saved to: {filepath}")
        return filepath
    
    def generate_json_report(self, metrics: List[PerformanceMetric], analyzer: PerformanceAnalyzer) -> Dict[str, Any]:
        """Generate JSON-formatted performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': analyzer.generate_performance_summary(metrics),
            'metrics': [metric.to_dict() for metric in metrics[-100:]],  # Last 100 metrics
            'anomalies': [],
            'trends': analyzer.analyze_trends(metrics),
            'baselines': {}
        }
        
        # Add anomalies
        anomalies = analyzer.detect_anomalies(metrics)
        for anomaly in anomalies:
            report['anomalies'].append({
                'metric_name': anomaly['metric'].name,
                'value': anomaly['metric'].value,
                'z_score': anomaly['z_score'],
                'severity': anomaly['severity'],
                'timestamp': anomaly['metric'].timestamp.isoformat()
            })
        
        # Add baselines
        for metric_name, baseline in analyzer.baselines.items():
            report['baselines'][metric_name] = {
                'mean': baseline.mean,
                'std_dev': baseline.std_dev,
                'min_value': baseline.min_value,
                'max_value': baseline.max_value,
                'percentile_95': baseline.percentile_95,
                'sample_count': baseline.sample_count,
                'last_updated': baseline.last_updated.isoformat()
            }
        
        return report

class BaselineManager:
    """Manages performance baselines and their updates."""
    
    def __init__(self, db_path: str = "performance_baselines.db"):
        """Initialize baseline manager with SQLite database."""
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for baseline storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS baselines (
                    metric_name TEXT PRIMARY KEY,
                    mean REAL,
                    std_dev REAL,
                    min_value REAL,
                    max_value REAL,
                    percentile_95 REAL,
                    sample_count INTEGER,
                    last_updated TEXT
                )
            """)
            conn.commit()
    
    def save_baseline(self, baseline: PerformanceBaseline):
        """Save baseline to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO baselines 
                (metric_name, mean, std_dev, min_value, max_value, percentile_95, sample_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                baseline.metric_name,
                baseline.mean,
                baseline.std_dev,
                baseline.min_value,
                baseline.max_value,
                baseline.percentile_95,
                baseline.sample_count,
                baseline.last_updated.isoformat()
            ))
            conn.commit()
    
    def load_baseline(self, metric_name: str) -> Optional[PerformanceBaseline]:
        """Load baseline from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM baselines WHERE metric_name = ?", 
                (metric_name,)
            )
            row = cursor.fetchone()
            
            if row:
                return PerformanceBaseline(
                    metric_name=row[0],
                    mean=row[1],
                    std_dev=row[2],
                    min_value=row[3],
                    max_value=row[4],
                    percentile_95=row[5],
                    sample_count=row[6],
                    last_updated=datetime.fromisoformat(row[7])
                )
            return None
    
    def update_baselines(self, analyzer: PerformanceAnalyzer, metrics: List[PerformanceMetric]):
        """Update all baselines with new metrics."""
        metric_names = set(m.name for m in metrics)
        
        for metric_name in metric_names:
            baseline = analyzer.calculate_baseline(metrics, metric_name)
            if baseline:
                self.save_baseline(baseline)
                logger.info(f"Updated baseline for {metric_name}")

class PerformanceMonitoringFramework:
    """Main framework orchestrating all performance monitoring components."""
    
    def __init__(self, 
                 collection_interval: float = 0.1,
                 analysis_interval: float = 60.0,
                 report_interval: float = 300.0):
        """
        Initialize performance monitoring framework.
        
        Args:
            collection_interval: Metrics collection interval in seconds
            analysis_interval: Analysis update interval in seconds  
            report_interval: Report generation interval in seconds
        """
        self.collection_interval = collection_interval
        self.analysis_interval = analysis_interval
        self.report_interval = report_interval
        
        # Initialize components
        self.collector = MetricsCollector(collection_interval)
        self.analyzer = PerformanceAnalyzer()
        self.reporter = PerformanceReporter()
        self.baseline_manager = BaselineManager()
        
        # Control variables
        self.is_running = False
        self.analysis_thread = None
        self.report_thread = None
        
        # Performance tracking
        self.start_time = None
        self.total_metrics_collected = 0
        self.total_anomalies_detected = 0
        
        logger.info("Performance monitoring framework initialized")
    
    def register_custom_collector(self, name: str, collector_func: Callable[[], Dict[str, float]]):
        """Register custom metrics collector."""
        self.collector.register_custom_collector(name, collector_func)
    
    def start_monitoring(self):
        """Start comprehensive performance monitoring."""
        if self.is_running:
            logger.warning("Performance monitoring already running")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start metrics collection
        self.collector.start_collection()
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        # Start reporting thread
        self.report_thread = threading.Thread(target=self._report_loop, daemon=True)
        self.report_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_running = False
        
        # Stop collection
        self.collector.stop_collection()
        
        # Wait for threads to finish
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        if self.report_thread:
            self.report_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")
    
    def _analysis_loop(self):
        """Analysis loop running in separate thread."""
        while self.is_running:
            try:
                # Get recent metrics
                metrics = self.collector.get_recent_metrics(1000)
                self.total_metrics_collected = len(metrics)
                
                if metrics:
                    # Update baselines
                    self.baseline_manager.update_baselines(self.analyzer, metrics)
                    
                    # Load baselines into analyzer
                    metric_names = set(m.name for m in metrics)
                    for metric_name in metric_names:
                        baseline = self.baseline_manager.load_baseline(metric_name)
                        if baseline:
                            self.analyzer.baselines[metric_name] = baseline
                    
                    # Detect anomalies
                    anomalies = self.analyzer.detect_anomalies(metrics)
                    self.total_anomalies_detected += len(anomalies)
                    
                    # Log significant anomalies
                    for anomaly in anomalies:
                        if anomaly['severity'] == 'high':
                            logger.warning(f"High severity anomaly detected: {anomaly['metric'].name} = {anomaly['metric'].value:.2f} (Z-score: {anomaly['z_score']:.2f})")
                
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                time.sleep(5)  # Prevent tight error loop
    
    def _report_loop(self):
        """Report generation loop running in separate thread."""
        while self.is_running:
            try:
                # Get recent metrics
                metrics = self.collector.get_recent_metrics(1000)
                
                if metrics:
                    # Generate and save text report
                    text_report = self.reporter.generate_text_report(metrics, self.analyzer)
                    self.reporter.save_report(text_report)
                    
                    # Generate and save JSON report
                    json_report = self.reporter.generate_json_report(metrics, self.analyzer)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    json_filename = f"performance_report_{timestamp}.json"
                    json_filepath = os.path.join(self.reporter.output_dir, json_filename)
                    
                    with open(json_filepath, 'w') as f:
                        json.dump(json_report, f, indent=2)
                    
                    logger.info(f"Performance reports generated: {json_filepath}")
                
                time.sleep(self.report_interval)
                
            except Exception as e:
                logger.error(f"Error in report loop: {e}")
                time.sleep(10)  # Prevent tight error loop
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'total_metrics_collected': self.total_metrics_collected,
            'total_anomalies_detected': self.total_anomalies_detected,
            'collection_interval': self.collection_interval,
            'analysis_interval': self.analysis_interval,
            'report_interval': self.report_interval,
            'baselines_count': len(self.analyzer.baselines),
            'recent_metrics_count': len(self.collector.get_recent_metrics())
        }
    
    def generate_immediate_report(self) -> Dict[str, Any]:
        """Generate immediate performance report."""
        metrics = self.collector.get_recent_metrics(1000)
        return self.reporter.generate_json_report(metrics, self.analyzer)
    
    def run_self_test(self) -> Dict[str, Any]:
        """Run comprehensive self-test of monitoring framework."""
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'test_status': 'passed',
            'tests': {},
            'performance_metrics': {}
        }
        
        try:
            # Test metrics collection
            start_time = time.time()
            metrics = self.collector.collect_all_metrics()
            collection_time = time.time() - start_time
            
            test_results['tests']['metrics_collection'] = {
                'status': 'passed' if metrics else 'failed',
                'metrics_count': len(metrics),
                'collection_time_ms': collection_time * 1000
            }
            
            # Test analysis
            start_time = time.time()
            summary = self.analyzer.generate_performance_summary(metrics)
            analysis_time = time.time() - start_time
            
            test_results['tests']['analysis'] = {
                'status': 'passed' if summary else 'failed',
                'analysis_time_ms': analysis_time * 1000
            }
            
            # Test reporting
            start_time = time.time()
            report = self.reporter.generate_json_report(metrics, self.analyzer)
            reporting_time = time.time() - start_time
            
            test_results['tests']['reporting'] = {
                'status': 'passed' if report else 'failed',
                'reporting_time_ms': reporting_time * 1000
            }
            
            # Test baseline management
            start_time = time.time()
            if metrics:
                baseline = self.analyzer.calculate_baseline(metrics, metrics[0].name)
                baseline_time = time.time() - start_time
                
                test_results['tests']['baseline_management'] = {
                    'status': 'passed' if baseline else 'failed',
                    'baseline_time_ms': baseline_time * 1000
                }
            
            # Overall performance metrics
            test_results['performance_metrics'] = {
                'total_test_time_ms': (collection_time + analysis_time + reporting_time) * 1000,
                'metrics_per_second': len(metrics) / max(collection_time, 0.001),
                'memory_usage_mb': psutil.Process().memory_info().rss / (1024**2)
            }
            
        except Exception as e:
            test_results['test_status'] = 'failed'
            test_results['error'] = str(e)
            logger.error(f"Self-test failed: {e}")
        
        return test_results

# Example usage and testing
if __name__ == "__main__":
    # Create monitoring framework
    monitor = PerformanceMonitoringFramework(
        collection_interval=0.1,  # Collect every 100ms
        analysis_interval=30.0,   # Analyze every 30 seconds
        report_interval=120.0     # Report every 2 minutes
    )
    
    # Register custom collector example
    def custom_app_metrics():
        return {
            'active_connections': np.random.randint(10, 100),
            'request_rate': np.random.uniform(50, 200),
            'error_rate': np.random.uniform(0, 5)
        }
    
    monitor.register_custom_collector('application', custom_app_metrics)
    
    # Run self-test
    print("Running performance monitoring self-test...")
    test_results = monitor.run_self_test()
    print(f"Self-test status: {test_results['test_status']}")
    
    # Start monitoring for demonstration
    print("Starting performance monitoring...")
    monitor.start_monitoring()
    
    try:
        # Let it run for a short time
        time.sleep(10)
        
        # Get status
        status = monitor.get_current_status()
        print(f"Monitoring status: {status}")
        
        # Generate immediate report
        report = monitor.generate_immediate_report()
        print(f"Generated report with {len(report['metrics'])} metrics")
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        print("Performance monitoring stopped")

