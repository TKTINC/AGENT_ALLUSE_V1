"""
ALL-USE Learning Systems - Collection Agent

This module implements the base collection agent framework for the ALL-USE Learning Systems,
providing a lightweight, configurable agent that can be deployed across all system components
to collect performance data with minimal impact on system performance.

The collection agent is designed to be:
- Lightweight: Minimal resource usage and performance impact
- Configurable: Flexible configuration for different collection needs
- Resilient: Handles errors and connectivity issues gracefully
- Extensible: Easy to extend for different types of metrics

Classes:
- CollectionAgent: Base class for all collection agents
- MetricsCollectionAgent: Agent for collecting performance metrics
- EventCollectionAgent: Agent for collecting event data
- LogCollectionAgent: Agent for collecting log data

Version: 1.0.0
"""

import time
import uuid
import logging
import threading
import queue
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CollectionPriority(Enum):
    """Priority levels for data collection."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3

@dataclass
class CollectionConfig:
    """Configuration for a collection agent."""
    collection_interval: float = 1.0  # seconds
    batch_size: int = 100
    max_queue_size: int = 10000
    retry_interval: float = 5.0  # seconds
    max_retries: int = 3
    priority: CollectionPriority = CollectionPriority.MEDIUM
    enabled: bool = True

class CollectionAgent:
    """Base class for all collection agents."""
    
    def __init__(self, agent_id: Optional[str] = None, config: Optional[CollectionConfig] = None):
        """Initialize the collection agent.
        
        Args:
            agent_id: Unique identifier for this agent. If None, a UUID will be generated.
            config: Configuration for this agent. If None, default configuration will be used.
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.config = config or CollectionConfig()
        self.queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.running = False
        self.collection_thread = None
        self.processing_thread = None
        self.logger = logging.getLogger(f"{__name__}.{self.agent_id}")
        self.metrics = {
            "collected_items": 0,
            "processed_items": 0,
            "dropped_items": 0,
            "errors": 0,
            "last_collection_time": 0,
            "last_processing_time": 0,
        }
    
    def start(self):
        """Start the collection agent."""
        if self.running:
            self.logger.warning("Agent already running")
            return
        
        self.running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            name=f"collection-{self.agent_id}",
            daemon=True
        )
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name=f"processing-{self.agent_id}",
            daemon=True
        )
        
        self.logger.info(f"Starting collection agent {self.agent_id}")
        self.collection_thread.start()
        self.processing_thread.start()
    
    def stop(self):
        """Stop the collection agent."""
        if not self.running:
            self.logger.warning("Agent not running")
            return
        
        self.logger.info(f"Stopping collection agent {self.agent_id}")
        self.running = False
        
        # Wait for threads to terminate
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        self.logger.info(f"Collection agent {self.agent_id} stopped")
    
    def _collection_loop(self):
        """Main collection loop."""
        self.logger.info(f"Collection loop started for agent {self.agent_id}")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Skip collection if disabled
                if not self.config.enabled:
                    time.sleep(self.config.collection_interval)
                    continue
                
                # Collect data
                data = self.collect()
                if data:
                    # Add metadata
                    data["agent_id"] = self.agent_id
                    data["timestamp"] = time.time()
                    
                    # Try to add to queue, drop if full
                    try:
                        self.queue.put(data, block=False)
                        self.metrics["collected_items"] += 1
                    except queue.Full:
                        self.logger.warning("Queue full, dropping data")
                        self.metrics["dropped_items"] += 1
                
                # Update metrics
                self.metrics["last_collection_time"] = time.time()
                
                # Sleep until next collection interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.collection_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}", exc_info=True)
                self.metrics["errors"] += 1
                time.sleep(self.config.retry_interval)
    
    def _processing_loop(self):
        """Main processing loop."""
        self.logger.info(f"Processing loop started for agent {self.agent_id}")
        
        batch = []
        last_flush_time = time.time()
        
        while self.running:
            try:
                # Get data from queue with timeout
                try:
                    data = self.queue.get(timeout=0.1)
                    batch.append(data)
                    self.queue.task_done()
                except queue.Empty:
                    pass
                
                # Process batch if full or timeout reached
                current_time = time.time()
                if (len(batch) >= self.config.batch_size or 
                    (batch and current_time - last_flush_time >= self.config.collection_interval)):
                    
                    if batch:
                        self.process_batch(batch)
                        self.metrics["processed_items"] += len(batch)
                        self.metrics["last_processing_time"] = time.time()
                        batch = []
                        last_flush_time = current_time
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}", exc_info=True)
                self.metrics["errors"] += 1
                time.sleep(self.config.retry_interval)
    
    def collect(self) -> Dict[str, Any]:
        """Collect data.
        
        This method should be overridden by subclasses to implement specific
        data collection logic.
        
        Returns:
            A dictionary containing the collected data.
        """
        raise NotImplementedError("Subclasses must implement collect()")
    
    def process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of collected data.
        
        This method should be overridden by subclasses to implement specific
        data processing logic.
        
        Args:
            batch: A list of dictionaries containing the collected data.
        """
        raise NotImplementedError("Subclasses must implement process_batch()")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics.
        
        Returns:
            A dictionary containing agent metrics.
        """
        return {
            "agent_id": self.agent_id,
            "running": self.running,
            "queue_size": self.queue.qsize(),
            "metrics": self.metrics,
            "config": {
                "collection_interval": self.config.collection_interval,
                "batch_size": self.config.batch_size,
                "max_queue_size": self.config.max_queue_size,
                "priority": self.config.priority.name,
                "enabled": self.config.enabled,
            }
        }

class MetricsCollectionAgent(CollectionAgent):
    """Agent for collecting performance metrics."""
    
    def __init__(
        self, 
        agent_id: Optional[str] = None, 
        config: Optional[CollectionConfig] = None,
        metrics_functions: Optional[Dict[str, Callable[[], Any]]] = None,
        output_handler: Optional[Callable[[List[Dict[str, Any]]], None]] = None
    ):
        """Initialize the metrics collection agent.
        
        Args:
            agent_id: Unique identifier for this agent. If None, a UUID will be generated.
            config: Configuration for this agent. If None, default configuration will be used.
            metrics_functions: Dictionary of functions that return metric values.
            output_handler: Function to handle processed batches.
        """
        super().__init__(agent_id, config)
        self.metrics_functions = metrics_functions or {}
        self.output_handler = output_handler
    
    def collect(self) -> Dict[str, Any]:
        """Collect metrics using the registered metrics functions.
        
        Returns:
            A dictionary containing the collected metrics.
        """
        result = {}
        
        for metric_name, metric_func in self.metrics_functions.items():
            try:
                result[metric_name] = metric_func()
            except Exception as e:
                self.logger.error(f"Error collecting metric {metric_name}: {e}")
                result[metric_name] = None
        
        return {"type": "metrics", "data": result}
    
    def process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of collected metrics.
        
        Args:
            batch: A list of dictionaries containing the collected metrics.
        """
        if self.output_handler:
            try:
                self.output_handler(batch)
            except Exception as e:
                self.logger.error(f"Error in output handler: {e}")
        else:
            # Default processing - just log the batch size
            self.logger.debug(f"Processed batch of {len(batch)} metrics")
    
    def add_metric(self, name: str, func: Callable[[], Any]):
        """Add a metric collection function.
        
        Args:
            name: Name of the metric.
            func: Function that returns the metric value.
        """
        self.metrics_functions[name] = func
        self.logger.debug(f"Added metric {name}")
    
    def remove_metric(self, name: str):
        """Remove a metric collection function.
        
        Args:
            name: Name of the metric to remove.
        """
        if name in self.metrics_functions:
            del self.metrics_functions[name]
            self.logger.debug(f"Removed metric {name}")

class EventCollectionAgent(CollectionAgent):
    """Agent for collecting event data."""
    
    def __init__(
        self, 
        agent_id: Optional[str] = None, 
        config: Optional[CollectionConfig] = None,
        output_handler: Optional[Callable[[List[Dict[str, Any]]], None]] = None
    ):
        """Initialize the event collection agent.
        
        Args:
            agent_id: Unique identifier for this agent. If None, a UUID will be generated.
            config: Configuration for this agent. If None, default configuration will be used.
            output_handler: Function to handle processed batches.
        """
        super().__init__(agent_id, config)
        self.output_handler = output_handler
        self.events_queue = queue.Queue()
        
        # Override collection interval for event-based collection
        if config is None:
            self.config.collection_interval = 0.1  # Check for events more frequently
    
    def collect(self) -> Dict[str, Any]:
        """Collect events from the events queue.
        
        Returns:
            A dictionary containing the collected events, or None if no events.
        """
        if self.events_queue.empty():
            return None
        
        events = []
        while not self.events_queue.empty() and len(events) < self.config.batch_size:
            try:
                events.append(self.events_queue.get(block=False))
                self.events_queue.task_done()
            except queue.Empty:
                break
        
        if not events:
            return None
        
        return {"type": "events", "data": events}
    
    def process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of collected events.
        
        Args:
            batch: A list of dictionaries containing the collected events.
        """
        if self.output_handler:
            try:
                self.output_handler(batch)
            except Exception as e:
                self.logger.error(f"Error in output handler: {e}")
        else:
            # Default processing - just log the batch size
            self.logger.debug(f"Processed batch of {len(batch)} events")
    
    def add_event(self, event_type: str, event_data: Dict[str, Any]):
        """Add an event to the collection queue.
        
        Args:
            event_type: Type of the event.
            event_data: Data associated with the event.
        """
        event = {
            "event_type": event_type,
            "timestamp": time.time(),
            "data": event_data
        }
        
        try:
            self.events_queue.put(event, block=False)
        except queue.Full:
            self.logger.warning("Events queue full, dropping event")
            self.metrics["dropped_items"] += 1

class LogCollectionAgent(CollectionAgent):
    """Agent for collecting log data."""
    
    def __init__(
        self, 
        agent_id: Optional[str] = None, 
        config: Optional[CollectionConfig] = None,
        log_file_path: Optional[str] = None,
        log_pattern: Optional[str] = None,
        output_handler: Optional[Callable[[List[Dict[str, Any]]], None]] = None
    ):
        """Initialize the log collection agent.
        
        Args:
            agent_id: Unique identifier for this agent. If None, a UUID will be generated.
            config: Configuration for this agent. If None, default configuration will be used.
            log_file_path: Path to the log file to collect from.
            log_pattern: Regex pattern to match log entries.
            output_handler: Function to handle processed batches.
        """
        super().__init__(agent_id, config)
        self.log_file_path = log_file_path
        self.log_pattern = log_pattern
        self.output_handler = output_handler
        self.last_position = 0
    
    def collect(self) -> Dict[str, Any]:
        """Collect logs from the log file.
        
        Returns:
            A dictionary containing the collected logs, or None if no logs.
        """
        if not self.log_file_path:
            return None
        
        try:
            logs = []
            with open(self.log_file_path, 'r') as f:
                f.seek(self.last_position)
                for line in f:
                    logs.append(line.strip())
                self.last_position = f.tell()
            
            if not logs:
                return None
            
            return {"type": "logs", "data": logs}
        except Exception as e:
            self.logger.error(f"Error collecting logs: {e}")
            return None
    
    def process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of collected logs.
        
        Args:
            batch: A list of dictionaries containing the collected logs.
        """
        if self.output_handler:
            try:
                self.output_handler(batch)
            except Exception as e:
                self.logger.error(f"Error in output handler: {e}")
        else:
            # Default processing - just log the batch size
            self.logger.debug(f"Processed batch of {len(batch)} logs")

# Example usage
if __name__ == "__main__":
    # Example metrics collection
    def get_cpu_usage():
        return 0.5  # Simulated CPU usage
    
    def get_memory_usage():
        return 1024  # Simulated memory usage in MB
    
    # Create and configure a metrics collection agent
    metrics_agent = MetricsCollectionAgent(
        agent_id="system-metrics",
        config=CollectionConfig(collection_interval=5.0),
        metrics_functions={
            "cpu_usage": get_cpu_usage,
            "memory_usage": get_memory_usage
        },
        output_handler=lambda batch: print(f"Processed {len(batch)} metrics")
    )
    
    # Start the agent
    metrics_agent.start()
    
    # Let it run for a while
    time.sleep(30)
    
    # Stop the agent
    metrics_agent.stop()
    
    # Print metrics
    print(metrics_agent.get_metrics())

