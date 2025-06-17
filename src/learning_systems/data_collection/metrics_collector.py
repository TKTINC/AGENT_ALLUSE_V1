"""
ALL-USE Learning Systems - Metrics Collector

This module implements specialized metrics collectors for the ALL-USE Learning Systems,
providing components for collecting various types of metrics from different system components.

The metrics collectors are designed to:
- Collect specific types of metrics with minimal overhead
- Provide standardized metric formats for consistent processing
- Support both pull and push-based collection models
- Integrate with the collection agent framework

Classes:
- SystemMetricsCollector: Collects system-level metrics (CPU, memory, disk, network)
- AccountMetricsCollector: Collects account-related metrics
- ProtocolMetricsCollector: Collects protocol-related metrics
- MarketMetricsCollector: Collects market-related metrics
- PerformanceMetricsCollector: Collects performance-related metrics

Version: 1.0.0
"""

import os
import time
import psutil
import platform
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

from .collection_agent import MetricsCollectionAgent, CollectionConfig, CollectionPriority

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemMetricsCollector:
    """Collects system-level metrics (CPU, memory, disk, network)."""
    
    def __init__(self, collection_interval: float = 5.0):
        """Initialize the system metrics collector.
        
        Args:
            collection_interval: Interval between collections in seconds.
        """
        self.agent = MetricsCollectionAgent(
            agent_id="system-metrics",
            config=CollectionConfig(
                collection_interval=collection_interval,
                priority=CollectionPriority.HIGH
            )
        )
        
        # Register metric collection functions
        self.agent.add_metric("cpu_percent", self._get_cpu_percent)
        self.agent.add_metric("cpu_count", self._get_cpu_count)
        self.agent.add_metric("memory_percent", self._get_memory_percent)
        self.agent.add_metric("memory_used", self._get_memory_used)
        self.agent.add_metric("memory_total", self._get_memory_total)
        self.agent.add_metric("disk_usage", self._get_disk_usage)
        self.agent.add_metric("network_io", self._get_network_io)
        self.agent.add_metric("system_load", self._get_system_load)
        self.agent.add_metric("boot_time", self._get_boot_time)
        self.agent.add_metric("system_info", self._get_system_info)
        
        # Initialize previous network IO counters for rate calculation
        self._prev_net_io = psutil.net_io_counters()
        self._prev_net_io_time = time.time()
        self._net_io_lock = threading.Lock()
    
    def start(self):
        """Start collecting system metrics."""
        self.agent.start()
    
    def stop(self):
        """Stop collecting system metrics."""
        self.agent.stop()
    
    def _get_cpu_percent(self) -> float:
        """Get CPU usage percentage.
        
        Returns:
            CPU usage percentage (0-100).
        """
        return psutil.cpu_percent(interval=0.1)
    
    def _get_cpu_count(self) -> Dict[str, int]:
        """Get CPU count information.
        
        Returns:
            Dictionary with physical and logical CPU counts.
        """
        return {
            "physical": psutil.cpu_count(logical=False) or 0,
            "logical": psutil.cpu_count(logical=True) or 0
        }
    
    def _get_memory_percent(self) -> float:
        """Get memory usage percentage.
        
        Returns:
            Memory usage percentage (0-100).
        """
        return psutil.virtual_memory().percent
    
    def _get_memory_used(self) -> int:
        """Get used memory in bytes.
        
        Returns:
            Used memory in bytes.
        """
        return psutil.virtual_memory().used
    
    def _get_memory_total(self) -> int:
        """Get total memory in bytes.
        
        Returns:
            Total memory in bytes.
        """
        return psutil.virtual_memory().total
    
    def _get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage information.
        
        Returns:
            Dictionary with disk usage information.
        """
        result = {}
        for partition in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                result[partition.mountpoint] = {
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent
                }
            except (PermissionError, FileNotFoundError):
                # Skip partitions that can't be accessed
                pass
        return result
    
    def _get_network_io(self) -> Dict[str, Any]:
        """Get network I/O information.
        
        Returns:
            Dictionary with network I/O information.
        """
        with self._net_io_lock:
            current_net_io = psutil.net_io_counters()
            current_time = time.time()
            
            # Calculate rates
            time_diff = current_time - self._prev_net_io_time
            if time_diff > 0:
                bytes_sent_rate = (current_net_io.bytes_sent - self._prev_net_io.bytes_sent) / time_diff
                bytes_recv_rate = (current_net_io.bytes_recv - self._prev_net_io.bytes_recv) / time_diff
            else:
                bytes_sent_rate = 0
                bytes_recv_rate = 0
            
            # Update previous values
            self._prev_net_io = current_net_io
            self._prev_net_io_time = current_time
            
            return {
                "bytes_sent": current_net_io.bytes_sent,
                "bytes_recv": current_net_io.bytes_recv,
                "packets_sent": current_net_io.packets_sent,
                "packets_recv": current_net_io.packets_recv,
                "bytes_sent_rate": bytes_sent_rate,
                "bytes_recv_rate": bytes_recv_rate
            }
    
    def _get_system_load(self) -> List[float]:
        """Get system load averages.
        
        Returns:
            List of load averages (1, 5, 15 minutes).
        """
        try:
            return list(os.getloadavg())
        except (AttributeError, OSError):
            # Windows doesn't support getloadavg
            return [0.0, 0.0, 0.0]
    
    def _get_boot_time(self) -> float:
        """Get system boot time.
        
        Returns:
            System boot time as a Unix timestamp.
        """
        return psutil.boot_time()
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information.
        
        Returns:
            Dictionary with system information.
        """
        return {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }

class AccountMetricsCollector:
    """Collects account-related metrics."""
    
    def __init__(self, collection_interval: float = 10.0, account_service=None):
        """Initialize the account metrics collector.
        
        Args:
            collection_interval: Interval between collections in seconds.
            account_service: Service for accessing account information.
        """
        self.account_service = account_service
        self.agent = MetricsCollectionAgent(
            agent_id="account-metrics",
            config=CollectionConfig(
                collection_interval=collection_interval,
                priority=CollectionPriority.MEDIUM
            )
        )
        
        # Register metric collection functions
        self.agent.add_metric("account_count", self._get_account_count)
        self.agent.add_metric("account_types", self._get_account_types)
        self.agent.add_metric("transaction_count", self._get_transaction_count)
        self.agent.add_metric("active_accounts", self._get_active_accounts)
        self.agent.add_metric("account_growth", self._get_account_growth)
    
    def start(self):
        """Start collecting account metrics."""
        self.agent.start()
    
    def stop(self):
        """Stop collecting account metrics."""
        self.agent.stop()
    
    def _get_account_count(self) -> int:
        """Get total number of accounts.
        
        Returns:
            Total number of accounts.
        """
        if self.account_service:
            try:
                return self.account_service.get_account_count()
            except Exception as e:
                logger.error(f"Error getting account count: {e}")
        
        # Simulated data if service not available
        return 1000
    
    def _get_account_types(self) -> Dict[str, int]:
        """Get account counts by type.
        
        Returns:
            Dictionary with account counts by type.
        """
        if self.account_service:
            try:
                return self.account_service.get_account_types_count()
            except Exception as e:
                logger.error(f"Error getting account types: {e}")
        
        # Simulated data if service not available
        return {
            "standard": 700,
            "premium": 200,
            "enterprise": 100
        }
    
    def _get_transaction_count(self) -> int:
        """Get total number of transactions.
        
        Returns:
            Total number of transactions.
        """
        if self.account_service:
            try:
                return self.account_service.get_transaction_count()
            except Exception as e:
                logger.error(f"Error getting transaction count: {e}")
        
        # Simulated data if service not available
        return 50000
    
    def _get_active_accounts(self) -> int:
        """Get number of active accounts.
        
        Returns:
            Number of active accounts.
        """
        if self.account_service:
            try:
                return self.account_service.get_active_account_count()
            except Exception as e:
                logger.error(f"Error getting active accounts: {e}")
        
        # Simulated data if service not available
        return 800
    
    def _get_account_growth(self) -> Dict[str, int]:
        """Get account growth metrics.
        
        Returns:
            Dictionary with account growth metrics.
        """
        if self.account_service:
            try:
                return self.account_service.get_account_growth_metrics()
            except Exception as e:
                logger.error(f"Error getting account growth: {e}")
        
        # Simulated data if service not available
        return {
            "daily_new": 10,
            "weekly_new": 70,
            "monthly_new": 300,
            "daily_closed": 2,
            "weekly_closed": 14,
            "monthly_closed": 60
        }

class ProtocolMetricsCollector:
    """Collects protocol-related metrics."""
    
    def __init__(self, collection_interval: float = 15.0, protocol_service=None):
        """Initialize the protocol metrics collector.
        
        Args:
            collection_interval: Interval between collections in seconds.
            protocol_service: Service for accessing protocol information.
        """
        self.protocol_service = protocol_service
        self.agent = MetricsCollectionAgent(
            agent_id="protocol-metrics",
            config=CollectionConfig(
                collection_interval=collection_interval,
                priority=CollectionPriority.MEDIUM
            )
        )
        
        # Register metric collection functions
        self.agent.add_metric("protocol_operations", self._get_protocol_operations)
        self.agent.add_metric("protocol_compliance", self._get_protocol_compliance)
        self.agent.add_metric("protocol_errors", self._get_protocol_errors)
        self.agent.add_metric("protocol_latency", self._get_protocol_latency)
    
    def start(self):
        """Start collecting protocol metrics."""
        self.agent.start()
    
    def stop(self):
        """Stop collecting protocol metrics."""
        self.agent.stop()
    
    def _get_protocol_operations(self) -> Dict[str, int]:
        """Get protocol operation counts.
        
        Returns:
            Dictionary with protocol operation counts.
        """
        if self.protocol_service:
            try:
                return self.protocol_service.get_operation_counts()
            except Exception as e:
                logger.error(f"Error getting protocol operations: {e}")
        
        # Simulated data if service not available
        return {
            "create": 1000,
            "read": 5000,
            "update": 2000,
            "delete": 500
        }
    
    def _get_protocol_compliance(self) -> float:
        """Get protocol compliance rate.
        
        Returns:
            Protocol compliance rate (0-100).
        """
        if self.protocol_service:
            try:
                return self.protocol_service.get_compliance_rate()
            except Exception as e:
                logger.error(f"Error getting protocol compliance: {e}")
        
        # Simulated data if service not available
        return 99.5
    
    def _get_protocol_errors(self) -> Dict[str, int]:
        """Get protocol error counts.
        
        Returns:
            Dictionary with protocol error counts.
        """
        if self.protocol_service:
            try:
                return self.protocol_service.get_error_counts()
            except Exception as e:
                logger.error(f"Error getting protocol errors: {e}")
        
        # Simulated data if service not available
        return {
            "validation": 50,
            "timeout": 20,
            "permission": 10,
            "system": 5
        }
    
    def _get_protocol_latency(self) -> Dict[str, float]:
        """Get protocol operation latencies.
        
        Returns:
            Dictionary with protocol operation latencies.
        """
        if self.protocol_service:
            try:
                return self.protocol_service.get_operation_latencies()
            except Exception as e:
                logger.error(f"Error getting protocol latency: {e}")
        
        # Simulated data if service not available
        return {
            "create": 15.0,
            "read": 5.0,
            "update": 10.0,
            "delete": 8.0
        }

class MarketMetricsCollector:
    """Collects market-related metrics."""
    
    def __init__(self, collection_interval: float = 5.0, market_service=None):
        """Initialize the market metrics collector.
        
        Args:
            collection_interval: Interval between collections in seconds.
            market_service: Service for accessing market information.
        """
        self.market_service = market_service
        self.agent = MetricsCollectionAgent(
            agent_id="market-metrics",
            config=CollectionConfig(
                collection_interval=collection_interval,
                priority=CollectionPriority.HIGH
            )
        )
        
        # Register metric collection functions
        self.agent.add_metric("market_volume", self._get_market_volume)
        self.agent.add_metric("market_transactions", self._get_market_transactions)
        self.agent.add_metric("market_latency", self._get_market_latency)
        self.agent.add_metric("market_errors", self._get_market_errors)
    
    def start(self):
        """Start collecting market metrics."""
        self.agent.start()
    
    def stop(self):
        """Stop collecting market metrics."""
        self.agent.stop()
    
    def _get_market_volume(self) -> Dict[str, float]:
        """Get market volume metrics.
        
        Returns:
            Dictionary with market volume metrics.
        """
        if self.market_service:
            try:
                return self.market_service.get_volume_metrics()
            except Exception as e:
                logger.error(f"Error getting market volume: {e}")
        
        # Simulated data if service not available
        return {
            "total_volume": 1000000.0,
            "buy_volume": 550000.0,
            "sell_volume": 450000.0
        }
    
    def _get_market_transactions(self) -> int:
        """Get number of market transactions.
        
        Returns:
            Number of market transactions.
        """
        if self.market_service:
            try:
                return self.market_service.get_transaction_count()
            except Exception as e:
                logger.error(f"Error getting market transactions: {e}")
        
        # Simulated data if service not available
        return 5000
    
    def _get_market_latency(self) -> Dict[str, float]:
        """Get market operation latencies.
        
        Returns:
            Dictionary with market operation latencies.
        """
        if self.market_service:
            try:
                return self.market_service.get_operation_latencies()
            except Exception as e:
                logger.error(f"Error getting market latency: {e}")
        
        # Simulated data if service not available
        return {
            "order_submission": 5.0,
            "order_execution": 10.0,
            "market_data": 2.0
        }
    
    def _get_market_errors(self) -> Dict[str, int]:
        """Get market error counts.
        
        Returns:
            Dictionary with market error counts.
        """
        if self.market_service:
            try:
                return self.market_service.get_error_counts()
            except Exception as e:
                logger.error(f"Error getting market errors: {e}")
        
        # Simulated data if service not available
        return {
            "order_rejection": 20,
            "connection": 5,
            "timeout": 10,
            "data": 15
        }

class PerformanceMetricsCollector:
    """Collects performance-related metrics."""
    
    def __init__(self, collection_interval: float = 10.0):
        """Initialize the performance metrics collector.
        
        Args:
            collection_interval: Interval between collections in seconds.
        """
        self.agent = MetricsCollectionAgent(
            agent_id="performance-metrics",
            config=CollectionConfig(
                collection_interval=collection_interval,
                priority=CollectionPriority.HIGH
            )
        )
        
        # Register metric collection functions
        self.agent.add_metric("response_times", self._get_response_times)
        self.agent.add_metric("throughput", self._get_throughput)
        self.agent.add_metric("error_rates", self._get_error_rates)
        self.agent.add_metric("resource_utilization", self._get_resource_utilization)
        
        # Initialize performance tracking
        self._response_times = {}
        self._throughput_counts = {}
        self._error_counts = {}
        self._last_collection_time = time.time()
    
    def start(self):
        """Start collecting performance metrics."""
        self.agent.start()
    
    def stop(self):
        """Stop collecting performance metrics."""
        self.agent.stop()
    
    def record_response_time(self, operation: str, response_time: float):
        """Record a response time for an operation.
        
        Args:
            operation: Name of the operation.
            response_time: Response time in milliseconds.
        """
        if operation not in self._response_times:
            self._response_times[operation] = []
        
        self._response_times[operation].append(response_time)
        
        # Keep only the last 100 response times
        if len(self._response_times[operation]) > 100:
            self._response_times[operation] = self._response_times[operation][-100:]
    
    def record_operation(self, operation: str):
        """Record an operation for throughput calculation.
        
        Args:
            operation: Name of the operation.
        """
        if operation not in self._throughput_counts:
            self._throughput_counts[operation] = 0
        
        self._throughput_counts[operation] += 1
    
    def record_error(self, error_type: str):
        """Record an error for error rate calculation.
        
        Args:
            error_type: Type of the error.
        """
        if error_type not in self._error_counts:
            self._error_counts[error_type] = 0
        
        self._error_counts[error_type] += 1
    
    def _get_response_times(self) -> Dict[str, Dict[str, float]]:
        """Get response time metrics.
        
        Returns:
            Dictionary with response time metrics.
        """
        result = {}
        
        for operation, times in self._response_times.items():
            if not times:
                continue
            
            result[operation] = {
                "min": min(times),
                "max": max(times),
                "avg": sum(times) / len(times),
                "p95": sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else max(times),
                "count": len(times)
            }
        
        return result
    
    def _get_throughput(self) -> Dict[str, float]:
        """Get throughput metrics.
        
        Returns:
            Dictionary with throughput metrics.
        """
        current_time = time.time()
        time_diff = current_time - self._last_collection_time
        
        result = {}
        
        if time_diff > 0:
            for operation, count in self._throughput_counts.items():
                result[operation] = count / time_diff
                
            # Reset counts
            self._throughput_counts = {}
            self._last_collection_time = current_time
        
        return result
    
    def _get_error_rates(self) -> Dict[str, int]:
        """Get error count metrics.
        
        Returns:
            Dictionary with error count metrics.
        """
        result = dict(self._error_counts)
        
        # Reset counts
        self._error_counts = {}
        
        return result
    
    def _get_resource_utilization(self) -> Dict[str, float]:
        """Get resource utilization metrics.
        
        Returns:
            Dictionary with resource utilization metrics.
        """
        return {
            "cpu": psutil.cpu_percent(interval=0.1),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage('/').percent
        }

# Example usage
if __name__ == "__main__":
    # Create system metrics collector
    system_collector = SystemMetricsCollector(collection_interval=5.0)
    system_collector.start()
    
    # Create performance metrics collector
    perf_collector = PerformanceMetricsCollector(collection_interval=10.0)
    perf_collector.start()
    
    # Simulate some operations and response times
    for i in range(100):
        perf_collector.record_response_time("api_call", 10.0 + (i % 10))
        perf_collector.record_operation("api_call")
        
        if i % 20 == 0:
            perf_collector.record_error("timeout")
        
        time.sleep(0.1)
    
    # Let collectors run for a while
    time.sleep(30)
    
    # Stop collectors
    system_collector.stop()
    perf_collector.stop()
    
    print("Metrics collection complete")

