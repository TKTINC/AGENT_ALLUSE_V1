#!/usr/bin/env python3
"""
ALL-USE Account Management System - Performance Analyzer

This module provides comprehensive performance analysis capabilities for the
ALL-USE Account Management System, enabling detailed profiling, bottleneck
identification, and optimization opportunity assessment.

The analyzer collects performance metrics across various system components,
analyzes execution patterns, and generates detailed reports to guide
optimization efforts.

Author: Manus AI
Date: June 17, 2025
"""

import time
import statistics
import threading
import logging
import json
import os
import sys
import datetime
import psutil
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from collections import defaultdict, deque

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import account management modules
from models.account_models import AccountType, AccountStatus
from database.account_database import AccountDatabase
from api.account_operations_api import AccountOperationsAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("performance_analyzer")

class PerformanceMetric:
    """Class representing a performance metric with statistical analysis capabilities."""
    
    def __init__(self, name, unit="ms", description=""):
        """Initialize a new performance metric.
        
        Args:
            name (str): Name of the metric
            unit (str): Unit of measurement (default: ms)
            description (str): Description of the metric
        """
        self.name = name
        self.unit = unit
        self.description = description
        self.values = []
        self.start_time = None
        self.end_time = None
    
    def record(self, value):
        """Record a value for this metric.
        
        Args:
            value: The value to record
        """
        self.values.append(value)
    
    def start(self):
        """Start timing for this metric."""
        self.start_time = time.time()
    
    def stop(self):
        """Stop timing and record the elapsed time."""
        if self.start_time is not None:
            self.end_time = time.time()
            elapsed = (self.end_time - self.start_time) * 1000  # Convert to ms
            self.record(elapsed)
            self.start_time = None
            return elapsed
        return None
    
    def get_statistics(self):
        """Calculate statistics for this metric.
        
        Returns:
            dict: Dictionary containing statistical measures
        """
        if not self.values:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "p95": None,
                "p99": None,
                "std_dev": None
            }
        
        values = sorted(self.values)
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": values[int(len(values) * 0.95)] if len(values) > 20 else values[-1],
            "p99": values[int(len(values) * 0.99)] if len(values) > 100 else values[-1],
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def reset(self):
        """Reset all recorded values."""
        self.values = []
        self.start_time = None
        self.end_time = None

class PerformanceAnalyzer:
    """Main class for analyzing and profiling system performance."""
    
    def __init__(self, output_dir="./performance_reports"):
        """Initialize the performance analyzer.
        
        Args:
            output_dir (str): Directory for storing performance reports
        """
        self.metrics = {}
        self.output_dir = output_dir
        self.resource_usage = defaultdict(list)
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.start_time = None
        self.end_time = None
        self.resource_monitor_thread = None
        self.resource_monitor_running = False
        self.sampling_interval = 1.0  # seconds
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Performance analyzer initialized. Reports will be saved to {output_dir}")
    
    def create_metric(self, name, unit="ms", description=""):
        """Create a new performance metric.
        
        Args:
            name (str): Name of the metric
            unit (str): Unit of measurement
            description (str): Description of the metric
            
        Returns:
            PerformanceMetric: The created metric
        """
        metric = PerformanceMetric(name, unit, description)
        self.metrics[name] = metric
        return metric
    
    def get_metric(self, name):
        """Get a metric by name, creating it if it doesn't exist.
        
        Args:
            name (str): Name of the metric
            
        Returns:
            PerformanceMetric: The requested metric
        """
        if name not in self.metrics:
            return self.create_metric(name)
        return self.metrics[name]
    
    def start_monitoring(self):
        """Start performance monitoring session."""
        self.start_time = time.time()
        
        # Start resource monitoring in a separate thread
        self.resource_monitor_running = True
        self.resource_monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self.resource_monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring session."""
        self.end_time = time.time()
        
        # Stop resource monitoring thread
        if self.resource_monitor_running:
            self.resource_monitor_running = False
            if self.resource_monitor_thread:
                self.resource_monitor_thread.join(timeout=2.0)
        
        logger.info(f"Performance monitoring stopped. Duration: {self.end_time - self.start_time:.2f} seconds")
    
    def _monitor_resources(self):
        """Monitor system resource usage in a background thread."""
        while self.resource_monitor_running:
            try:
                # Get current resource usage
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                
                # Record resource usage
                timestamp = time.time()
                self.resource_usage["timestamp"].append(timestamp)
                self.resource_usage["cpu_percent"].append(cpu_percent)
                self.resource_usage["memory_percent"].append(memory_info.percent)
                self.resource_usage["memory_used"].append(memory_info.used)
                self.resource_usage["disk_read_bytes"].append(disk_io.read_bytes)
                self.resource_usage["disk_write_bytes"].append(disk_io.write_bytes)
                
                # Sleep for the sampling interval
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.sampling_interval)
    
    def record_operation(self, operation_name):
        """Record an operation execution.
        
        Args:
            operation_name (str): Name of the operation
        """
        self.operation_counts[operation_name] += 1
    
    def record_error(self, operation_name, error_type=None):
        """Record an operation error.
        
        Args:
            operation_name (str): Name of the operation
            error_type (str, optional): Type of error
        """
        error_key = f"{operation_name}:{error_type}" if error_type else operation_name
        self.error_counts[error_key] += 1
    
    def measure_execution_time(self, func=None, metric_name=None):
        """Decorator to measure execution time of a function.
        
        Args:
            func: The function to decorate
            metric_name (str, optional): Name of the metric to use
            
        Returns:
            function: Decorated function
        """
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                # Determine metric name
                name = metric_name or f.__qualname__
                metric = self.get_metric(name)
                
                # Record operation
                self.record_operation(name)
                
                # Measure execution time
                metric.start()
                try:
                    result = f(*args, **kwargs)
                    return result
                except Exception as e:
                    self.record_error(name, type(e).__name__)
                    raise
                finally:
                    metric.stop()
            return wrapper
        
        # Handle both @measure_execution_time and @measure_execution_time(metric_name="...")
        if func is None:
            return decorator
        return decorator(func)
    
    def generate_report(self, report_name="performance_report"):
        """Generate a comprehensive performance report.
        
        Args:
            report_name (str): Base name for the report files
            
        Returns:
            str: Path to the generated report
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{report_name}_{timestamp}.json"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Prepare report data
        report_data = {
            "timestamp": timestamp,
            "duration": self.end_time - self.start_time if self.end_time and self.start_time else None,
            "metrics": {},
            "operations": dict(self.operation_counts),
            "errors": dict(self.error_counts),
            "resource_usage_summary": self._calculate_resource_summary()
        }
        
        # Add metrics data
        for name, metric in self.metrics.items():
            report_data["metrics"][name] = {
                "unit": metric.unit,
                "description": metric.description,
                "statistics": metric.get_statistics()
            }
        
        # Write report to file
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Performance report generated: {report_path}")
        
        # Generate visualizations
        self._generate_visualizations(report_name, timestamp)
        
        return report_path
    
    def _calculate_resource_summary(self):
        """Calculate summary statistics for resource usage.
        
        Returns:
            dict: Summary statistics for each resource metric
        """
        summary = {}
        
        for metric, values in self.resource_usage.items():
            if metric == "timestamp" or not values:
                continue
                
            summary[metric] = {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "p95": sorted(values)[int(len(values) * 0.95)] if values else None
            }
            
        return summary
    
    def _generate_visualizations(self, report_name, timestamp):
        """Generate visualization charts for the performance data.
        
        Args:
            report_name (str): Base name for the report files
            timestamp (str): Timestamp string for the report
        """
        # Create visualizations directory
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate resource usage charts
        if self.resource_usage["timestamp"]:
            self._generate_resource_charts(viz_dir, report_name, timestamp)
        
        # Generate metrics distribution charts
        self._generate_metrics_charts(viz_dir, report_name, timestamp)
        
        # Generate operations summary chart
        self._generate_operations_chart(viz_dir, report_name, timestamp)
        
        logger.info(f"Performance visualizations generated in {viz_dir}")
    
    def _generate_resource_charts(self, viz_dir, report_name, timestamp):
        """Generate resource usage charts.
        
        Args:
            viz_dir (str): Directory for visualizations
            report_name (str): Base name for the report files
            timestamp (str): Timestamp string for the report
        """
        # Convert timestamps to relative seconds
        start_time = self.resource_usage["timestamp"][0]
        time_points = [(t - start_time) for t in self.resource_usage["timestamp"]]
        
        # CPU and Memory Usage Chart
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(time_points, self.resource_usage["cpu_percent"], 'b-', label='CPU Usage')
        plt.xlabel('Time (seconds)')
        plt.ylabel('CPU Usage (%)')
        plt.title('CPU Utilization')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(time_points, self.resource_usage["memory_percent"], 'r-', label='Memory Usage')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (%)')
        plt.title('Memory Utilization')
        plt.grid(True)
        
        plt.tight_layout()
        resource_chart_path = os.path.join(viz_dir, f"{report_name}_resources_{timestamp}.png")
        plt.savefig(resource_chart_path)
        plt.close()
        
        # Disk I/O Chart
        plt.figure(figsize=(12, 6))
        
        # Calculate I/O rates
        read_rates = []
        write_rates = []
        for i in range(1, len(time_points)):
            time_diff = time_points[i] - time_points[i-1]
            if time_diff > 0:
                read_diff = self.resource_usage["disk_read_bytes"][i] - self.resource_usage["disk_read_bytes"][i-1]
                write_diff = self.resource_usage["disk_write_bytes"][i] - self.resource_usage["disk_write_bytes"][i-1]
                read_rates.append(read_diff / time_diff / 1024)  # KB/s
                write_rates.append(write_diff / time_diff / 1024)  # KB/s
            else:
                read_rates.append(0)
                write_rates.append(0)
        
        plt.plot(time_points[1:], read_rates, 'g-', label='Read Rate')
        plt.plot(time_points[1:], write_rates, 'm-', label='Write Rate')
        plt.xlabel('Time (seconds)')
        plt.ylabel('I/O Rate (KB/s)')
        plt.title('Disk I/O Activity')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        io_chart_path = os.path.join(viz_dir, f"{report_name}_disk_io_{timestamp}.png")
        plt.savefig(io_chart_path)
        plt.close()
    
    def _generate_metrics_charts(self, viz_dir, report_name, timestamp):
        """Generate charts for performance metrics.
        
        Args:
            viz_dir (str): Directory for visualizations
            report_name (str): Base name for the report files
            timestamp (str): Timestamp string for the report
        """
        # Filter metrics with sufficient data
        metrics_to_plot = [name for name, metric in self.metrics.items() 
                          if len(metric.values) >= 5]
        
        if not metrics_to_plot:
            return
        
        # Determine chart layout
        n_metrics = len(metrics_to_plot)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        plt.figure(figsize=(5*cols, 4*rows))
        
        for i, metric_name in enumerate(metrics_to_plot, 1):
            metric = self.metrics[metric_name]
            
            plt.subplot(rows, cols, i)
            
            # Create histogram
            plt.hist(metric.values, bins=20, alpha=0.7, color='blue')
            
            # Add vertical lines for key statistics
            stats = metric.get_statistics()
            plt.axvline(stats["mean"], color='r', linestyle='dashed', linewidth=1, label=f'Mean: {stats["mean"]:.2f}')
            plt.axvline(stats["p95"], color='g', linestyle='dashed', linewidth=1, label=f'95th: {stats["p95"]:.2f}')
            
            plt.title(f'{metric_name} Distribution')
            plt.xlabel(f'Value ({metric.unit})')
            plt.ylabel('Frequency')
            plt.legend(fontsize='small')
            
        plt.tight_layout()
        metrics_chart_path = os.path.join(viz_dir, f"{report_name}_metrics_{timestamp}.png")
        plt.savefig(metrics_chart_path)
        plt.close()
    
    def _generate_operations_chart(self, viz_dir, report_name, timestamp):
        """Generate operations summary chart.
        
        Args:
            viz_dir (str): Directory for visualizations
            report_name (str): Base name for the report files
            timestamp (str): Timestamp string for the report
        """
        if not self.operation_counts:
            return
        
        # Sort operations by count
        sorted_ops = sorted(self.operation_counts.items(), key=lambda x: x[1], reverse=True)
        op_names = [op[0] for op in sorted_ops]
        op_counts = [op[1] for op in sorted_ops]
        
        # Limit to top 15 operations if there are many
        if len(op_names) > 15:
            op_names = op_names[:15]
            op_counts = op_counts[:15]
            op_names.append("Others")
            op_counts.append(sum(op[1] for op in sorted_ops[15:]))
        
        plt.figure(figsize=(12, 6))
        
        # Create bar chart
        bars = plt.bar(range(len(op_names)), op_counts, color='skyblue')
        plt.xticks(range(len(op_names)), op_names, rotation=45, ha='right')
        plt.xlabel('Operation')
        plt.ylabel('Count')
        plt.title('Operation Execution Counts')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        ops_chart_path = os.path.join(viz_dir, f"{report_name}_operations_{timestamp}.png")
        plt.savefig(ops_chart_path)
        plt.close()

class DatabasePerformanceAnalyzer:
    """Specialized analyzer for database performance."""
    
    def __init__(self, db_connection, analyzer):
        """Initialize the database performance analyzer.
        
        Args:
            db_connection: Database connection object
            analyzer (PerformanceAnalyzer): Main performance analyzer
        """
        self.db_connection = db_connection
        self.analyzer = analyzer
        self.query_stats = defaultdict(list)
    
    def analyze_query(self, query, params=None, description=None):
        """Decorator to analyze query performance.
        
        Args:
            query (str): SQL query to analyze
            params (tuple, optional): Query parameters
            description (str, optional): Query description
            
        Returns:
            function: Decorated function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create a metric name based on the function and query
                query_hash = hash(query)
                metric_name = f"db_query_{func.__qualname__}_{query_hash}"
                description_text = description or func.__doc__ or func.__name__
                
                metric = self.analyzer.get_metric(metric_name, "ms", description_text)
                
                # Record operation
                self.analyzer.record_operation(metric_name)
                
                # Measure execution time
                metric.start()
                try:
                    result = func(*args, **kwargs)
                    
                    # Record query statistics
                    self.query_stats[metric_name].append({
                        "query": query,
                        "params": params,
                        "execution_time": metric.stop(),
                        "timestamp": time.time()
                    })
                    
                    return result
                except Exception as e:
                    self.analyzer.record_error(metric_name, type(e).__name__)
                    raise
                finally:
                    if metric.start_time is not None:
                        metric.stop()
            return wrapper
        return decorator
    
    def analyze_query_plan(self, query, params=None):
        """Analyze the execution plan for a query.
        
        Args:
            query (str): SQL query to analyze
            params (tuple, optional): Query parameters
            
        Returns:
            dict: Query execution plan analysis
        """
        # This is a placeholder - actual implementation would depend on the database
        # For PostgreSQL, this would use EXPLAIN ANALYZE
        # For other databases, appropriate explain plan commands would be used
        logger.info(f"Query plan analysis requested for: {query}")
        return {"query": query, "params": params, "analysis": "Query plan analysis not implemented"}
    
    def get_slow_queries(self, threshold_ms=100):
        """Get list of slow queries exceeding the threshold.
        
        Args:
            threshold_ms (float): Threshold in milliseconds
            
        Returns:
            list: List of slow query information
        """
        slow_queries = []
        
        for query_name, executions in self.query_stats.items():
            for execution in executions:
                if execution["execution_time"] > threshold_ms:
                    slow_queries.append({
                        "name": query_name,
                        "query": execution["query"],
                        "params": execution["params"],
                        "execution_time": execution["execution_time"],
                        "timestamp": execution["timestamp"]
                    })
        
        # Sort by execution time (slowest first)
        slow_queries.sort(key=lambda q: q["execution_time"], reverse=True)
        return slow_queries
    
    def generate_query_report(self):
        """Generate a report of query performance.
        
        Returns:
            dict: Query performance report
        """
        report = {
            "query_summary": {},
            "slow_queries": self.get_slow_queries(),
            "query_patterns": self._analyze_query_patterns()
        }
        
        # Generate summary statistics for each query
        for query_name, executions in self.query_stats.items():
            execution_times = [e["execution_time"] for e in executions]
            
            if not execution_times:
                continue
                
            report["query_summary"][query_name] = {
                "count": len(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "mean": statistics.mean(execution_times),
                "p95": sorted(execution_times)[int(len(execution_times) * 0.95)] if execution_times else None,
                "query": executions[0]["query"]  # Include the query text
            }
        
        return report
    
    def _analyze_query_patterns(self):
        """Analyze patterns in query execution.
        
        Returns:
            dict: Query pattern analysis
        """
        # This is a simplified implementation
        # A more sophisticated implementation would analyze query structures,
        # identify common patterns, and provide optimization recommendations
        
        patterns = {
            "frequent_queries": [],
            "variable_performance_queries": []
        }
        
        # Identify frequent queries
        query_counts = {name: len(execs) for name, execs in self.query_stats.items()}
        frequent_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for name, count in frequent_queries:
            if name in self.query_stats and self.query_stats[name]:
                patterns["frequent_queries"].append({
                    "name": name,
                    "count": count,
                    "query": self.query_stats[name][0]["query"]
                })
        
        # Identify queries with variable performance
        for name, executions in self.query_stats.items():
            execution_times = [e["execution_time"] for e in executions]
            
            if len(execution_times) < 5:
                continue
                
            mean = statistics.mean(execution_times)
            std_dev = statistics.stdev(execution_times)
            
            # If standard deviation is high relative to mean, performance is variable
            if std_dev > mean * 0.5:
                patterns["variable_performance_queries"].append({
                    "name": name,
                    "count": len(execution_times),
                    "mean": mean,
                    "std_dev": std_dev,
                    "coefficient_of_variation": std_dev / mean,
                    "query": self.query_stats[name][0]["query"]
                })
        
        # Sort by coefficient of variation (highest first)
        patterns["variable_performance_queries"].sort(
            key=lambda x: x["coefficient_of_variation"], 
            reverse=True
        )
        
        return patterns

class ApplicationPerformanceAnalyzer:
    """Specialized analyzer for application performance."""
    
    def __init__(self, analyzer):
        """Initialize the application performance analyzer.
        
        Args:
            analyzer (PerformanceAnalyzer): Main performance analyzer
        """
        self.analyzer = analyzer
        self.method_stats = defaultdict(list)
        self.call_graph = defaultdict(set)
        self.current_call_stack = threading.local()
    
    def analyze_method(self, method_name=None):
        """Decorator to analyze method performance.
        
        Args:
            method_name (str, optional): Custom name for the method
            
        Returns:
            function: Decorated function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Determine method name
                name = method_name or func.__qualname__
                metric = self.analyzer.get_metric(name, "ms", func.__doc__)
                
                # Record operation
                self.analyzer.record_operation(name)
                
                # Update call graph
                try:
                    caller = getattr(self.current_call_stack, "name", None)
                    if caller:
                        self.call_graph[caller].add(name)
                except Exception:
                    pass
                
                # Set current method in call stack
                prev_method = getattr(self.current_call_stack, "name", None)
                self.current_call_stack.name = name
                
                # Measure execution time
                start_time = time.time()
                metric.start()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record method statistics
                    execution_time = metric.stop()
                    self.method_stats[name].append({
                        "execution_time": execution_time,
                        "timestamp": time.time(),
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    })
                    
                    return result
                except Exception as e:
                    self.analyzer.record_error(name, type(e).__name__)
                    raise
                finally:
                    if metric.start_time is not None:
                        metric.stop()
                    
                    # Restore previous method in call stack
                    self.current_call_stack.name = prev_method
            return wrapper
        return decorator
    
    def get_hotspots(self, limit=10):
        """Get the performance hotspots (methods consuming most time).
        
        Args:
            limit (int): Maximum number of hotspots to return
            
        Returns:
            list: List of hotspot information
        """
        hotspots = []
        
        for method_name, executions in self.method_stats.items():
            if not executions:
                continue
                
            total_time = sum(e["execution_time"] for e in executions)
            count = len(executions)
            avg_time = total_time / count
            
            hotspots.append({
                "name": method_name,
                "total_time": total_time,
                "count": count,
                "avg_time": avg_time
            })
        
        # Sort by total time (highest first)
        hotspots.sort(key=lambda h: h["total_time"], reverse=True)
        return hotspots[:limit]
    
    def generate_call_graph_visualization(self, output_dir):
        """Generate a visualization of the call graph.
        
        Args:
            output_dir (str): Directory for output files
            
        Returns:
            str: Path to the generated visualization
        """
        # This is a simplified implementation
        # A more sophisticated implementation would use a proper graph visualization library
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_filename = f"call_graph_{timestamp}.txt"
        graph_path = os.path.join(output_dir, graph_filename)
        
        with open(graph_path, 'w') as f:
            f.write("Call Graph:\n")
            f.write("==========\n\n")
            
            for caller, callees in self.call_graph.items():
                f.write(f"{caller}:\n")
                for callee in callees:
                    f.write(f"  -> {callee}\n")
                f.write("\n")
        
        logger.info(f"Call graph visualization generated: {graph_path}")
        return graph_path
    
    def generate_method_report(self):
        """Generate a report of method performance.
        
        Returns:
            dict: Method performance report
        """
        report = {
            "method_summary": {},
            "hotspots": self.get_hotspots(),
            "call_graph_summary": self._summarize_call_graph()
        }
        
        # Generate summary statistics for each method
        for method_name, executions in self.method_stats.items():
            execution_times = [e["execution_time"] for e in executions]
            
            if not execution_times:
                continue
                
            report["method_summary"][method_name] = {
                "count": len(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "mean": statistics.mean(execution_times),
                "p95": sorted(execution_times)[int(len(execution_times) * 0.95)] if execution_times else None,
                "total_time": sum(execution_times)
            }
        
        return report
    
    def _summarize_call_graph(self):
        """Generate a summary of the call graph.
        
        Returns:
            dict: Call graph summary
        """
        summary = {
            "entry_points": [],
            "leaf_methods": [],
            "central_methods": []
        }
        
        # Find all methods that appear in the call graph
        all_methods = set(self.call_graph.keys())
        for callees in self.call_graph.values():
            all_methods.update(callees)
        
        # Entry points are methods that are not called by any other method
        called_methods = set()
        for callees in self.call_graph.values():
            called_methods.update(callees)
        
        entry_points = all_methods - called_methods
        summary["entry_points"] = list(entry_points)
        
        # Leaf methods are methods that don't call any other method
        leaf_methods = set()
        for method, callees in self.call_graph.items():
            if not callees:
                leaf_methods.add(method)
        
        summary["leaf_methods"] = list(leaf_methods)
        
        # Central methods are those with many incoming and outgoing calls
        method_scores = {}
        for method in all_methods:
            # Count incoming calls
            incoming = sum(1 for caller, callees in self.call_graph.items() if method in callees)
            
            # Count outgoing calls
            outgoing = len(self.call_graph.get(method, set()))
            
            # Score is product of incoming and outgoing
            method_scores[method] = incoming * outgoing
        
        # Get top 5 central methods
        central_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        summary["central_methods"] = [{"name": m[0], "score": m[1]} for m in central_methods]
        
        return summary

class LoadGenerator:
    """Utility for generating load to test system performance."""
    
    def __init__(self, analyzer):
        """Initialize the load generator.
        
        Args:
            analyzer (PerformanceAnalyzer): Performance analyzer to use
        """
        self.analyzer = analyzer
        self.running = False
        self.executor = None
    
    def start_load(self, target_func, args_list, concurrency=10, duration=60):
        """Start generating load.
        
        Args:
            target_func: Function to call
            args_list: List of arguments to pass to the function
            concurrency (int): Number of concurrent threads
            duration (int): Duration in seconds
            
        Returns:
            dict: Load test results
        """
        logger.info(f"Starting load test with concurrency={concurrency}, duration={duration}s")
        
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=concurrency)
        
        start_time = time.time()
        end_time = start_time + duration
        
        # Start analyzer
        self.analyzer.start_monitoring()
        
        # Submit tasks
        futures = []
        task_count = 0
        
        while time.time() < end_time and self.running:
            # Submit tasks up to concurrency level
            while len(futures) < concurrency and time.time() < end_time and self.running:
                # Get arguments for this task (cycling through the list)
                args = args_list[task_count % len(args_list)]
                
                # Submit task
                future = self.executor.submit(target_func, *args)
                futures.append(future)
                task_count += 1
            
            # Wait for at least one task to complete
            done, futures = concurrent.futures.wait(
                futures, 
                timeout=1.0,
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            # Process completed tasks
            for future in done:
                try:
                    future.result()  # Get result to check for exceptions
                except Exception as e:
                    logger.error(f"Task error: {e}")
        
        # Stop analyzer
        self.analyzer.stop_monitoring()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        self.running = False
        
        # Calculate results
        actual_duration = time.time() - start_time
        operations_per_second = task_count / actual_duration
        
        results = {
            "concurrency": concurrency,
            "duration": actual_duration,
            "total_operations": task_count,
            "operations_per_second": operations_per_second,
            "success_rate": 1.0 - sum(self.analyzer.error_counts.values()) / task_count if task_count > 0 else 0
        }
        
        logger.info(f"Load test completed: {results}")
        return results
    
    def stop_load(self):
        """Stop the load generation."""
        logger.info("Stopping load test")
        self.running = False
        
        if self.executor:
            self.executor.shutdown(wait=False)

def main():
    """Main function for standalone execution."""
    print("ALL-USE Account Management System - Performance Analyzer")
    print("======================================================")
    print("\nThis module provides performance analysis capabilities and should be imported by other modules.")
    print("For standalone testing, this script will perform a basic self-test.")
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "performance_reports")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create analyzer
    analyzer = PerformanceAnalyzer(output_dir=output_dir)
    
    # Simple test function
    @analyzer.measure_execution_time
    def test_function(iterations):
        result = 0
        for i in range(iterations):
            result += i
        time.sleep(0.01)  # Simulate some work
        return result
    
    # Run test
    print("\nRunning performance analyzer self-test...")
    analyzer.start_monitoring()
    
    for i in range(10):
        test_function(1000 * (i + 1))
    
    analyzer.stop_monitoring()
    
    # Generate report
    report_path = analyzer.generate_report("self_test")
    print(f"\nTest completed. Performance report generated: {report_path}")

if __name__ == "__main__":
    main()

