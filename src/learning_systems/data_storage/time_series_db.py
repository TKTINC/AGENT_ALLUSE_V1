"""
ALL-USE Learning Systems - Time Series Database

This module implements a time series database for the ALL-USE Learning Systems,
providing efficient storage and retrieval of time-based performance metrics.

The time series database is designed to:
- Store high-volume time-series data efficiently
- Provide fast querying by time ranges
- Support aggregation and downsampling
- Enable efficient data retention policies
- Support multiple metric types and dimensions

Classes:
- TimeSeriesDB: Core time series database implementation
- TimeSeriesMetric: Represents a time series metric
- TimeSeriesQuery: Query builder for time series data
- TimeSeriesAggregator: Aggregates time series data

Version: 1.0.0
"""

import time
import logging
import threading
import sqlite3
import json
import os
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics that can be stored in the time series database."""
    COUNTER = 1    # Monotonically increasing value
    GAUGE = 2      # Value that can go up and down
    HISTOGRAM = 3  # Distribution of values
    SUMMARY = 4    # Summary statistics (min, max, avg, etc.)
    EVENT = 5      # Discrete events

class AggregationType(Enum):
    """Types of aggregations that can be performed on time series data."""
    SUM = 1
    AVG = 2
    MIN = 3
    MAX = 4
    COUNT = 5
    PERCENTILE = 6
    STDDEV = 7
    RATE = 8

@dataclass
class TimeSeriesConfig:
    """Configuration for a time series database."""
    db_path: str = "time_series.db"
    retention_days: int = 30
    auto_vacuum: bool = True
    vacuum_interval_hours: int = 24
    max_points_per_query: int = 10000
    default_aggregation_window: int = 60  # seconds
    enable_compression: bool = True
    enable_caching: bool = True
    cache_size_mb: int = 100
    max_concurrent_queries: int = 10

class TimeSeriesMetric:
    """Represents a time series metric."""
    
    def __init__(self, name: str, metric_type: MetricType, dimensions: Optional[Dict[str, str]] = None):
        """Initialize the time series metric.
        
        Args:
            name: Name of the metric.
            metric_type: Type of the metric.
            dimensions: Additional dimensions for the metric.
        """
        self.name = name
        self.metric_type = metric_type
        self.dimensions = dimensions or {}
        
        # Generate a unique ID for this metric
        dimension_str = "_".join(f"{k}={v}" for k, v in sorted(self.dimensions.items()))
        self.id = f"{name}_{dimension_str}" if dimension_str else name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the metric to a dictionary.
        
        Returns:
            Dictionary representation of the metric.
        """
        return {
            "id": self.id,
            "name": self.name,
            "type": self.metric_type.name,
            "dimensions": self.dimensions,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeSeriesMetric':
        """Create a metric from a dictionary.
        
        Args:
            data: Dictionary representation of the metric.
            
        Returns:
            The created metric.
        """
        return cls(
            name=data["name"],
            metric_type=MetricType[data["type"]],
            dimensions=data.get("dimensions", {})
        )

class TimeSeriesPoint:
    """Represents a single point in a time series."""
    
    def __init__(self, timestamp: float, value: float, metric: TimeSeriesMetric):
        """Initialize the time series point.
        
        Args:
            timestamp: Unix timestamp of the point.
            value: Value of the point.
            metric: The metric this point belongs to.
        """
        self.timestamp = timestamp
        self.value = value
        self.metric = metric
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the point to a dictionary.
        
        Returns:
            Dictionary representation of the point.
        """
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "metric": self.metric.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeSeriesPoint':
        """Create a point from a dictionary.
        
        Args:
            data: Dictionary representation of the point.
            
        Returns:
            The created point.
        """
        return cls(
            timestamp=data["timestamp"],
            value=data["value"],
            metric=TimeSeriesMetric.from_dict(data["metric"])
        )

class TimeSeriesQuery:
    """Query builder for time series data."""
    
    def __init__(self, db: 'TimeSeriesDB'):
        """Initialize the time series query.
        
        Args:
            db: The time series database to query.
        """
        self.db = db
        self.metric_name = None
        self.dimensions = {}
        self.start_time = None
        self.end_time = None
        self.limit = db.config.max_points_per_query
        self.aggregation = None
        self.aggregation_window = db.config.default_aggregation_window
        self.order_by = "timestamp"
        self.order_direction = "ASC"
    
    def for_metric(self, metric_name: str) -> 'TimeSeriesQuery':
        """Set the metric name to query.
        
        Args:
            metric_name: Name of the metric to query.
            
        Returns:
            This query builder.
        """
        self.metric_name = metric_name
        return self
    
    def with_dimensions(self, dimensions: Dict[str, str]) -> 'TimeSeriesQuery':
        """Set the dimensions to filter by.
        
        Args:
            dimensions: Dimensions to filter by.
            
        Returns:
            This query builder.
        """
        self.dimensions = dimensions
        return self
    
    def in_time_range(self, start_time: float, end_time: float) -> 'TimeSeriesQuery':
        """Set the time range to query.
        
        Args:
            start_time: Start time of the range (Unix timestamp).
            end_time: End time of the range (Unix timestamp).
            
        Returns:
            This query builder.
        """
        self.start_time = start_time
        self.end_time = end_time
        return self
    
    def in_last(self, seconds: int) -> 'TimeSeriesQuery':
        """Set the time range to the last N seconds.
        
        Args:
            seconds: Number of seconds to look back.
            
        Returns:
            This query builder.
        """
        self.end_time = time.time()
        self.start_time = self.end_time - seconds
        return self
    
    def limit_to(self, limit: int) -> 'TimeSeriesQuery':
        """Set the maximum number of points to return.
        
        Args:
            limit: Maximum number of points.
            
        Returns:
            This query builder.
        """
        self.limit = min(limit, self.db.config.max_points_per_query)
        return self
    
    def aggregate_by(self, aggregation: AggregationType, window_seconds: int) -> 'TimeSeriesQuery':
        """Set the aggregation to perform.
        
        Args:
            aggregation: Type of aggregation to perform.
            window_seconds: Size of the aggregation window in seconds.
            
        Returns:
            This query builder.
        """
        self.aggregation = aggregation
        self.aggregation_window = window_seconds
        return self
    
    def order_by_time(self, direction: str = "ASC") -> 'TimeSeriesQuery':
        """Set the ordering of results.
        
        Args:
            direction: Direction to order by ("ASC" or "DESC").
            
        Returns:
            This query builder.
        """
        self.order_by = "timestamp"
        self.order_direction = direction
        return self
    
    def execute(self) -> List[TimeSeriesPoint]:
        """Execute the query.
        
        Returns:
            List of time series points matching the query.
        """
        return self.db.query(self)

class TimeSeriesDB:
    """Core time series database implementation."""
    
    def __init__(self, db_id: str, config: Optional[TimeSeriesConfig] = None):
        """Initialize the time series database.
        
        Args:
            db_id: Unique identifier for this database.
            config: Configuration for this database.
        """
        self.db_id = db_id
        self.config = config or TimeSeriesConfig()
        self.logger = logging.getLogger(f"{__name__}.db.{db_id}")
        self.conn = None
        self.metrics: Dict[str, TimeSeriesMetric] = {}
        self.running = False
        self.maintenance_thread = None
        self.query_semaphore = threading.Semaphore(self.config.max_concurrent_queries)
        
        # Initialize the database
        self._init_db()
    
    def _init_db(self):
        """Initialize the database."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.config.db_path)), exist_ok=True)
            
            # Connect to the database
            self.conn = sqlite3.connect(self.config.db_path, check_same_thread=False)
            
            # Enable WAL mode for better concurrency
            self.conn.execute("PRAGMA journal_mode=WAL")
            
            # Set cache size
            self.conn.execute(f"PRAGMA cache_size={self.config.cache_size_mb * 1024}")
            
            # Create tables if they don't exist
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    dimensions TEXT NOT NULL
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    value REAL NOT NULL,
                    FOREIGN KEY (metric_id) REFERENCES metrics (id)
                )
            """)
            
            # Create indexes
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_points_metric_time ON points (metric_id, timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_points_time ON points (timestamp)")
            
            # Load existing metrics
            cursor = self.conn.execute("SELECT id, name, type, dimensions FROM metrics")
            for row in cursor:
                metric_id, name, type_str, dimensions_json = row
                dimensions = json.loads(dimensions_json)
                metric = TimeSeriesMetric(name, MetricType[type_str], dimensions)
                self.metrics[metric_id] = metric
            
            self.logger.info(f"Initialized database with {len(self.metrics)} metrics")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}", exc_info=True)
            raise
    
    def start(self):
        """Start the time series database."""
        if self.running:
            self.logger.warning("Database already running")
            return
        
        self.running = True
        self.logger.info(f"Starting time series database {self.db_id}")
        
        # Start maintenance thread
        self.maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            name=f"db-maintenance-{self.db_id}",
            daemon=True
        )
        self.maintenance_thread.start()
    
    def stop(self):
        """Stop the time series database."""
        if not self.running:
            self.logger.warning("Database not running")
            return
        
        self.logger.info(f"Stopping time series database {self.db_id}")
        self.running = False
        
        # Wait for thread to terminate
        if self.maintenance_thread and self.maintenance_thread.is_alive():
            self.maintenance_thread.join(timeout=5.0)
        
        # Close the database connection
        if self.conn:
            self.conn.close()
            self.conn = None
        
        self.logger.info(f"Time series database {self.db_id} stopped")
    
    def _maintenance_loop(self):
        """Maintenance loop for the database."""
        self.logger.info(f"Maintenance loop started for database {self.db_id}")
        
        last_vacuum_time = time.time()
        
        while self.running:
            try:
                # Apply retention policy
                self._apply_retention_policy()
                
                # Vacuum the database if needed
                current_time = time.time()
                if (self.config.auto_vacuum and 
                    current_time - last_vacuum_time >= self.config.vacuum_interval_hours * 3600):
                    
                    self._vacuum_database()
                    last_vacuum_time = current_time
                
                # Sleep for a while
                time.sleep(3600)  # 1 hour
                
            except Exception as e:
                self.logger.error(f"Error in maintenance loop: {e}", exc_info=True)
                time.sleep(3600)  # Sleep longer on error
    
    def _apply_retention_policy(self):
        """Apply the retention policy to the database."""
        try:
            # Calculate the cutoff time
            cutoff_time = time.time() - (self.config.retention_days * 24 * 3600)
            
            # Delete old data
            cursor = self.conn.execute(
                "DELETE FROM points WHERE timestamp < ?",
                (cutoff_time,)
            )
            deleted_count = cursor.rowcount
            self.conn.commit()
            
            if deleted_count > 0:
                self.logger.info(f"Deleted {deleted_count} points older than {self.config.retention_days} days")
            
        except Exception as e:
            self.logger.error(f"Error applying retention policy: {e}", exc_info=True)
    
    def _vacuum_database(self):
        """Vacuum the database to reclaim space."""
        try:
            self.logger.info("Vacuuming database...")
            self.conn.execute("VACUUM")
            self.logger.info("Database vacuum completed")
        except Exception as e:
            self.logger.error(f"Error vacuuming database: {e}", exc_info=True)
    
    def register_metric(self, metric: TimeSeriesMetric) -> TimeSeriesMetric:
        """Register a metric with the database.
        
        Args:
            metric: The metric to register.
            
        Returns:
            The registered metric.
        """
        try:
            # Check if metric already exists
            if metric.id in self.metrics:
                return self.metrics[metric.id]
            
            # Insert into database
            self.conn.execute(
                "INSERT INTO metrics (id, name, type, dimensions) VALUES (?, ?, ?, ?)",
                (metric.id, metric.name, metric.metric_type.name, json.dumps(metric.dimensions))
            )
            self.conn.commit()
            
            # Add to cache
            self.metrics[metric.id] = metric
            
            self.logger.debug(f"Registered metric {metric.id}")
            return metric
            
        except Exception as e:
            self.logger.error(f"Error registering metric: {e}", exc_info=True)
            raise
    
    def write_point(self, point: TimeSeriesPoint) -> bool:
        """Write a single point to the database.
        
        Args:
            point: The point to write.
            
        Returns:
            True if the point was successfully written, False otherwise.
        """
        try:
            # Register metric if needed
            if point.metric.id not in self.metrics:
                self.register_metric(point.metric)
            
            # Insert point
            self.conn.execute(
                "INSERT INTO points (metric_id, timestamp, value) VALUES (?, ?, ?)",
                (point.metric.id, point.timestamp, point.value)
            )
            self.conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing point: {e}", exc_info=True)
            return False
    
    def write_points(self, points: List[TimeSeriesPoint]) -> bool:
        """Write multiple points to the database.
        
        Args:
            points: The points to write.
            
        Returns:
            True if all points were successfully written, False otherwise.
        """
        if not points:
            return True
        
        try:
            # Register metrics if needed
            for point in points:
                if point.metric.id not in self.metrics:
                    self.register_metric(point.metric)
            
            # Insert points
            self.conn.executemany(
                "INSERT INTO points (metric_id, timestamp, value) VALUES (?, ?, ?)",
                [(p.metric.id, p.timestamp, p.value) for p in points]
            )
            self.conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing points: {e}", exc_info=True)
            return False
    
    def query(self, query: TimeSeriesQuery) -> List[TimeSeriesPoint]:
        """Query the database.
        
        Args:
            query: The query to execute.
            
        Returns:
            List of time series points matching the query.
        """
        # Acquire semaphore to limit concurrent queries
        with self.query_semaphore:
            try:
                # Build SQL query
                sql = "SELECT m.id, m.name, m.type, m.dimensions, p.timestamp, p.value FROM points p JOIN metrics m ON p.metric_id = m.id WHERE 1=1"
                params = []
                
                # Add metric filter
                if query.metric_name:
                    sql += " AND m.name = ?"
                    params.append(query.metric_name)
                
                # Add dimension filters
                if query.dimensions:
                    # This is a simplification; in a real implementation, we would need to parse the JSON
                    # and filter based on the dimensions
                    dimension_str = json.dumps(query.dimensions)
                    sql += " AND m.dimensions = ?"
                    params.append(dimension_str)
                
                # Add time range filter
                if query.start_time is not None:
                    sql += " AND p.timestamp >= ?"
                    params.append(query.start_time)
                
                if query.end_time is not None:
                    sql += " AND p.timestamp <= ?"
                    params.append(query.end_time)
                
                # Add ordering
                sql += f" ORDER BY p.{query.order_by} {query.order_direction}"
                
                # Add limit
                sql += " LIMIT ?"
                params.append(query.limit)
                
                # Execute query
                cursor = self.conn.execute(sql, params)
                
                # Process results
                results = []
                for row in cursor:
                    metric_id, name, type_str, dimensions_json, timestamp, value = row
                    dimensions = json.loads(dimensions_json)
                    metric = TimeSeriesMetric(name, MetricType[type_str], dimensions)
                    point = TimeSeriesPoint(timestamp, value, metric)
                    results.append(point)
                
                # Apply aggregation if requested
                if query.aggregation and results:
                    results = self._aggregate_results(results, query.aggregation, query.aggregation_window)
                
                return results
                
            except Exception as e:
                self.logger.error(f"Error executing query: {e}", exc_info=True)
                return []
    
    def _aggregate_results(self, points: List[TimeSeriesPoint], aggregation: AggregationType, window_seconds: int) -> List[TimeSeriesPoint]:
        """Aggregate query results.
        
        Args:
            points: The points to aggregate.
            aggregation: Type of aggregation to perform.
            window_seconds: Size of the aggregation window in seconds.
            
        Returns:
            List of aggregated time series points.
        """
        if not points:
            return []
        
        # Group points by metric and time window
        grouped_points = {}
        for point in points:
            metric_id = point.metric.id
            window_start = int(point.timestamp / window_seconds) * window_seconds
            key = (metric_id, window_start)
            
            if key not in grouped_points:
                grouped_points[key] = {
                    "metric": point.metric,
                    "window_start": window_start,
                    "values": []
                }
            
            grouped_points[key]["values"].append(point.value)
        
        # Aggregate each group
        aggregated_points = []
        for group_data in grouped_points.values():
            metric = group_data["metric"]
            window_start = group_data["window_start"]
            values = group_data["values"]
            
            # Calculate aggregated value
            if aggregation == AggregationType.SUM:
                value = sum(values)
            elif aggregation == AggregationType.AVG:
                value = sum(values) / len(values)
            elif aggregation == AggregationType.MIN:
                value = min(values)
            elif aggregation == AggregationType.MAX:
                value = max(values)
            elif aggregation == AggregationType.COUNT:
                value = len(values)
            elif aggregation == AggregationType.PERCENTILE:
                # 95th percentile as default
                value = np.percentile(values, 95)
            elif aggregation == AggregationType.STDDEV:
                value = np.std(values)
            elif aggregation == AggregationType.RATE:
                # Rate per second over the window
                value = sum(values) / window_seconds
            else:
                value = sum(values)  # Default to sum
            
            # Create aggregated point
            point = TimeSeriesPoint(window_start, value, metric)
            aggregated_points.append(point)
        
        # Sort by timestamp
        aggregated_points.sort(key=lambda p: p.timestamp)
        
        return aggregated_points
    
    def get_metrics_list(self) -> List[TimeSeriesMetric]:
        """Get a list of all registered metrics.
        
        Returns:
            List of all registered metrics.
        """
        return list(self.metrics.values())
    
    def create_query(self) -> TimeSeriesQuery:
        """Create a new query builder.
        
        Returns:
            A new query builder.
        """
        return TimeSeriesQuery(self)

# Example usage
if __name__ == "__main__":
    # Create a time series database
    db = TimeSeriesDB("example-db")
    
    # Start the database
    db.start()
    
    # Create some metrics
    cpu_metric = TimeSeriesMetric("cpu_usage", MetricType.GAUGE, {"host": "server1"})
    memory_metric = TimeSeriesMetric("memory_usage", MetricType.GAUGE, {"host": "server1"})
    
    # Register metrics
    db.register_metric(cpu_metric)
    db.register_metric(memory_metric)
    
    # Write some points
    current_time = time.time()
    for i in range(100):
        timestamp = current_time - (100 - i) * 60  # One point per minute, going back 100 minutes
        cpu_value = 50 + 20 * np.sin(i / 10)  # Sine wave between 30-70%
        memory_value = 40 + 10 * np.cos(i / 8)  # Cosine wave between 30-50%
        
        cpu_point = TimeSeriesPoint(timestamp, cpu_value, cpu_metric)
        memory_point = TimeSeriesPoint(timestamp, memory_value, memory_metric)
        
        db.write_point(cpu_point)
        db.write_point(memory_point)
    
    # Query the data
    query = db.create_query() \
        .for_metric("cpu_usage") \
        .with_dimensions({"host": "server1"}) \
        .in_last(3600) \
        .aggregate_by(AggregationType.AVG, 300) \
        .limit_to(20)
    
    results = query.execute()
    
    # Print results
    print(f"Query returned {len(results)} points")
    for point in results:
        dt = datetime.fromtimestamp(point.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{dt}: {point.metric.name} = {point.value:.2f}")
    
    # Stop the database
    db.stop()

