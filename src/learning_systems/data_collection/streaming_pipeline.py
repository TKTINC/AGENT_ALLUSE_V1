"""
ALL-USE Learning Systems - Streaming Pipeline

This module implements the streaming pipeline for the ALL-USE Learning Systems,
providing components for real-time data streaming from collection agents to
storage and processing components.

The streaming pipeline is designed to:
- Handle high-volume data streams with minimal latency
- Provide buffering and batching capabilities
- Support multiple data sources and destinations
- Implement backpressure mechanisms
- Ensure data integrity and delivery guarantees

Classes:
- StreamingPipeline: Core streaming pipeline implementation
- DataStream: Represents a stream of data
- StreamProcessor: Processes data in a stream
- StreamSink: Destination for processed data

Version: 1.0.0
"""

import time
import queue
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamingMode(Enum):
    """Streaming modes for the pipeline."""
    REAL_TIME = 1  # Process data as soon as it arrives
    BATCH = 2      # Process data in batches
    HYBRID = 3     # Combination of real-time and batch processing

class DeliveryGuarantee(Enum):
    """Delivery guarantees for the pipeline."""
    AT_MOST_ONCE = 1    # Data may be lost, but never duplicated
    AT_LEAST_ONCE = 2   # Data may be duplicated, but never lost
    EXACTLY_ONCE = 3    # Data is neither lost nor duplicated

@dataclass
class StreamingConfig:
    """Configuration for a streaming pipeline."""
    mode: StreamingMode = StreamingMode.REAL_TIME
    batch_size: int = 100
    batch_timeout: float = 1.0  # seconds
    max_queue_size: int = 10000
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE
    num_workers: int = 1
    retry_interval: float = 5.0  # seconds
    max_retries: int = 3
    buffer_flush_interval: float = 5.0  # seconds

class DataStream:
    """Represents a stream of data."""
    
    def __init__(self, stream_id: str, config: Optional[StreamingConfig] = None):
        """Initialize the data stream.
        
        Args:
            stream_id: Unique identifier for this stream.
            config: Configuration for this stream.
        """
        self.stream_id = stream_id
        self.config = config or StreamingConfig()
        self.queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.processors: List[StreamProcessor] = []
        self.running = False
        self.processing_thread = None
        self.logger = logging.getLogger(f"{__name__}.stream.{stream_id}")
        self.metrics = {
            "received_items": 0,
            "processed_items": 0,
            "dropped_items": 0,
            "errors": 0,
            "last_processing_time": 0,
        }
    
    def start(self):
        """Start processing the data stream."""
        if self.running:
            self.logger.warning("Stream already running")
            return
        
        self.running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name=f"stream-{self.stream_id}",
            daemon=True
        )
        
        self.logger.info(f"Starting data stream {self.stream_id}")
        self.processing_thread.start()
    
    def stop(self):
        """Stop processing the data stream."""
        if not self.running:
            self.logger.warning("Stream not running")
            return
        
        self.logger.info(f"Stopping data stream {self.stream_id}")
        self.running = False
        
        # Wait for thread to terminate
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        self.logger.info(f"Data stream {self.stream_id} stopped")
    
    def add_processor(self, processor: 'StreamProcessor'):
        """Add a processor to the stream.
        
        Args:
            processor: The processor to add.
        """
        self.processors.append(processor)
        self.logger.debug(f"Added processor {processor.processor_id} to stream {self.stream_id}")
    
    def remove_processor(self, processor: 'StreamProcessor'):
        """Remove a processor from the stream.
        
        Args:
            processor: The processor to remove.
        """
        if processor in self.processors:
            self.processors.remove(processor)
            self.logger.debug(f"Removed processor {processor.processor_id} from stream {self.stream_id}")
    
    def push(self, data: Dict[str, Any]) -> bool:
        """Push data into the stream.
        
        Args:
            data: The data to push.
            
        Returns:
            True if the data was successfully pushed, False otherwise.
        """
        try:
            # Add metadata
            if "timestamp" not in data:
                data["timestamp"] = time.time()
            if "stream_id" not in data:
                data["stream_id"] = self.stream_id
            
            # Try to add to queue, drop if full
            if self.config.mode == StreamingMode.REAL_TIME:
                try:
                    self.queue.put(data, block=False)
                    self.metrics["received_items"] += 1
                    return True
                except queue.Full:
                    self.logger.warning("Queue full, dropping data")
                    self.metrics["dropped_items"] += 1
                    return False
            else:
                # For batch mode, block until space is available
                self.queue.put(data, block=True, timeout=1.0)
                self.metrics["received_items"] += 1
                return True
                
        except Exception as e:
            self.logger.error(f"Error pushing data: {e}", exc_info=True)
            self.metrics["errors"] += 1
            return False
    
    def _processing_loop(self):
        """Main processing loop."""
        self.logger.info(f"Processing loop started for stream {self.stream_id}")
        
        batch = []
        last_flush_time = time.time()
        
        while self.running:
            try:
                # Process based on mode
                if self.config.mode == StreamingMode.REAL_TIME:
                    # Get data from queue with timeout
                    try:
                        data = self.queue.get(timeout=0.1)
                        self._process_item(data)
                        self.queue.task_done()
                    except queue.Empty:
                        pass
                    
                elif self.config.mode == StreamingMode.BATCH:
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
                        (batch and current_time - last_flush_time >= self.config.batch_timeout)):
                        
                        if batch:
                            self._process_batch(batch)
                            batch = []
                            last_flush_time = current_time
                
                elif self.config.mode == StreamingMode.HYBRID:
                    # Get data from queue with timeout
                    try:
                        data = self.queue.get(timeout=0.1)
                        
                        # Process item immediately
                        self._process_item(data)
                        
                        # Also add to batch for batch processing
                        batch.append(data)
                        
                        self.queue.task_done()
                    except queue.Empty:
                        pass
                    
                    # Process batch if full or timeout reached
                    current_time = time.time()
                    if (len(batch) >= self.config.batch_size or 
                        (batch and current_time - last_flush_time >= self.config.batch_timeout)):
                        
                        if batch:
                            self._process_batch(batch)
                            batch = []
                            last_flush_time = current_time
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}", exc_info=True)
                self.metrics["errors"] += 1
                time.sleep(self.config.retry_interval)
    
    def _process_item(self, data: Dict[str, Any]):
        """Process a single data item.
        
        Args:
            data: The data item to process.
        """
        for processor in self.processors:
            try:
                data = processor.process(data)
                if data is None:
                    # Processor filtered out the data
                    break
            except Exception as e:
                self.logger.error(f"Error in processor {processor.processor_id}: {e}", exc_info=True)
                self.metrics["errors"] += 1
                
                # Handle based on delivery guarantee
                if self.config.delivery_guarantee == DeliveryGuarantee.AT_LEAST_ONCE:
                    # Retry later
                    self.push(data)
                
                break
        
        self.metrics["processed_items"] += 1
        self.metrics["last_processing_time"] = time.time()
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of data items.
        
        Args:
            batch: The batch of data items to process.
        """
        for processor in self.processors:
            try:
                batch = processor.process_batch(batch)
                if not batch:
                    # Processor filtered out all data
                    break
            except Exception as e:
                self.logger.error(f"Error in processor {processor.processor_id}: {e}", exc_info=True)
                self.metrics["errors"] += 1
                
                # Handle based on delivery guarantee
                if self.config.delivery_guarantee == DeliveryGuarantee.AT_LEAST_ONCE:
                    # Retry later
                    for data in batch:
                        self.push(data)
                
                break
        
        self.metrics["processed_items"] += len(batch)
        self.metrics["last_processing_time"] = time.time()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get stream metrics.
        
        Returns:
            A dictionary containing stream metrics.
        """
        return {
            "stream_id": self.stream_id,
            "running": self.running,
            "queue_size": self.queue.qsize(),
            "metrics": self.metrics,
            "config": {
                "mode": self.config.mode.name,
                "batch_size": self.config.batch_size,
                "batch_timeout": self.config.batch_timeout,
                "max_queue_size": self.config.max_queue_size,
                "delivery_guarantee": self.config.delivery_guarantee.name,
                "num_workers": self.config.num_workers,
            }
        }

class StreamProcessor:
    """Processes data in a stream."""
    
    def __init__(self, processor_id: str):
        """Initialize the stream processor.
        
        Args:
            processor_id: Unique identifier for this processor.
        """
        self.processor_id = processor_id
        self.logger = logging.getLogger(f"{__name__}.processor.{processor_id}")
        self.metrics = {
            "processed_items": 0,
            "filtered_items": 0,
            "errors": 0,
            "last_processing_time": 0,
        }
    
    def process(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single data item.
        
        Args:
            data: The data item to process.
            
        Returns:
            The processed data item, or None if the item should be filtered out.
        """
        # Default implementation just passes through the data
        self.metrics["processed_items"] += 1
        self.metrics["last_processing_time"] = time.time()
        return data
    
    def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of data items.
        
        Args:
            batch: The batch of data items to process.
            
        Returns:
            The processed batch of data items.
        """
        # Default implementation processes each item individually
        result = []
        for data in batch:
            processed = self.process(data)
            if processed is not None:
                result.append(processed)
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics.
        
        Returns:
            A dictionary containing processor metrics.
        """
        return {
            "processor_id": self.processor_id,
            "metrics": self.metrics,
        }

class FilterProcessor(StreamProcessor):
    """Filters data based on a predicate."""
    
    def __init__(self, processor_id: str, predicate: Callable[[Dict[str, Any]], bool]):
        """Initialize the filter processor.
        
        Args:
            processor_id: Unique identifier for this processor.
            predicate: Function that returns True for data to keep, False to filter out.
        """
        super().__init__(processor_id)
        self.predicate = predicate
    
    def process(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single data item.
        
        Args:
            data: The data item to process.
            
        Returns:
            The data item if it passes the filter, None otherwise.
        """
        try:
            if self.predicate(data):
                self.metrics["processed_items"] += 1
                self.metrics["last_processing_time"] = time.time()
                return data
            else:
                self.metrics["filtered_items"] += 1
                return None
        except Exception as e:
            self.logger.error(f"Error in filter: {e}", exc_info=True)
            self.metrics["errors"] += 1
            return None

class TransformProcessor(StreamProcessor):
    """Transforms data using a transformation function."""
    
    def __init__(self, processor_id: str, transform_func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Initialize the transform processor.
        
        Args:
            processor_id: Unique identifier for this processor.
            transform_func: Function that transforms the data.
        """
        super().__init__(processor_id)
        self.transform_func = transform_func
    
    def process(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single data item.
        
        Args:
            data: The data item to process.
            
        Returns:
            The transformed data item.
        """
        try:
            result = self.transform_func(data)
            self.metrics["processed_items"] += 1
            self.metrics["last_processing_time"] = time.time()
            return result
        except Exception as e:
            self.logger.error(f"Error in transform: {e}", exc_info=True)
            self.metrics["errors"] += 1
            return data  # Return original data on error

class AggregateProcessor(StreamProcessor):
    """Aggregates data in a batch."""
    
    def __init__(self, processor_id: str, aggregate_func: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]):
        """Initialize the aggregate processor.
        
        Args:
            processor_id: Unique identifier for this processor.
            aggregate_func: Function that aggregates a batch of data.
        """
        super().__init__(processor_id)
        self.aggregate_func = aggregate_func
    
    def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of data items.
        
        Args:
            batch: The batch of data items to process.
            
        Returns:
            The aggregated batch of data items.
        """
        try:
            result = self.aggregate_func(batch)
            self.metrics["processed_items"] += len(batch)
            self.metrics["last_processing_time"] = time.time()
            return result
        except Exception as e:
            self.logger.error(f"Error in aggregate: {e}", exc_info=True)
            self.metrics["errors"] += 1
            return batch  # Return original batch on error

class StreamSink:
    """Destination for processed data."""
    
    def __init__(self, sink_id: str):
        """Initialize the stream sink.
        
        Args:
            sink_id: Unique identifier for this sink.
        """
        self.sink_id = sink_id
        self.logger = logging.getLogger(f"{__name__}.sink.{sink_id}")
        self.metrics = {
            "received_items": 0,
            "errors": 0,
            "last_sink_time": 0,
        }
    
    def sink(self, data: Dict[str, Any]) -> bool:
        """Sink a single data item.
        
        Args:
            data: The data item to sink.
            
        Returns:
            True if the data was successfully sunk, False otherwise.
        """
        # Default implementation just logs the data
        self.logger.debug(f"Sinking data: {data}")
        self.metrics["received_items"] += 1
        self.metrics["last_sink_time"] = time.time()
        return True
    
    def sink_batch(self, batch: List[Dict[str, Any]]) -> bool:
        """Sink a batch of data items.
        
        Args:
            batch: The batch of data items to sink.
            
        Returns:
            True if the batch was successfully sunk, False otherwise.
        """
        # Default implementation sinks each item individually
        success = True
        for data in batch:
            if not self.sink(data):
                success = False
        
        return success
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get sink metrics.
        
        Returns:
            A dictionary containing sink metrics.
        """
        return {
            "sink_id": self.sink_id,
            "metrics": self.metrics,
        }

class LogSink(StreamSink):
    """Sink that logs data."""
    
    def __init__(self, sink_id: str, log_level: int = logging.INFO):
        """Initialize the log sink.
        
        Args:
            sink_id: Unique identifier for this sink.
            log_level: Logging level to use.
        """
        super().__init__(sink_id)
        self.log_level = log_level
    
    def sink(self, data: Dict[str, Any]) -> bool:
        """Sink a single data item.
        
        Args:
            data: The data item to sink.
            
        Returns:
            True if the data was successfully sunk, False otherwise.
        """
        try:
            self.logger.log(self.log_level, f"Data: {data}")
            self.metrics["received_items"] += 1
            self.metrics["last_sink_time"] = time.time()
            return True
        except Exception as e:
            self.logger.error(f"Error in log sink: {e}", exc_info=True)
            self.metrics["errors"] += 1
            return False

class FileSink(StreamSink):
    """Sink that writes data to a file."""
    
    def __init__(self, sink_id: str, file_path: str, append: bool = True):
        """Initialize the file sink.
        
        Args:
            sink_id: Unique identifier for this sink.
            file_path: Path to the file to write to.
            append: Whether to append to the file or overwrite it.
        """
        super().__init__(sink_id)
        self.file_path = file_path
        self.append = append
        self.file = None
        
        # Open the file
        self._open_file()
    
    def _open_file(self):
        """Open the file for writing."""
        try:
            mode = 'a' if self.append else 'w'
            self.file = open(self.file_path, mode)
            self.logger.debug(f"Opened file {self.file_path} for writing")
        except Exception as e:
            self.logger.error(f"Error opening file {self.file_path}: {e}", exc_info=True)
            self.file = None
    
    def sink(self, data: Dict[str, Any]) -> bool:
        """Sink a single data item.
        
        Args:
            data: The data item to sink.
            
        Returns:
            True if the data was successfully sunk, False otherwise.
        """
        if self.file is None:
            self._open_file()
            if self.file is None:
                return False
        
        try:
            self.file.write(f"{data}\n")
            self.file.flush()
            self.metrics["received_items"] += 1
            self.metrics["last_sink_time"] = time.time()
            return True
        except Exception as e:
            self.logger.error(f"Error writing to file: {e}", exc_info=True)
            self.metrics["errors"] += 1
            self.file = None  # Force reopen on next attempt
            return False
    
    def sink_batch(self, batch: List[Dict[str, Any]]) -> bool:
        """Sink a batch of data items.
        
        Args:
            batch: The batch of data items to sink.
            
        Returns:
            True if the batch was successfully sunk, False otherwise.
        """
        if self.file is None:
            self._open_file()
            if self.file is None:
                return False
        
        try:
            for data in batch:
                self.file.write(f"{data}\n")
            
            self.file.flush()
            self.metrics["received_items"] += len(batch)
            self.metrics["last_sink_time"] = time.time()
            return True
        except Exception as e:
            self.logger.error(f"Error writing batch to file: {e}", exc_info=True)
            self.metrics["errors"] += 1
            self.file = None  # Force reopen on next attempt
            return False
    
    def close(self):
        """Close the file."""
        if self.file:
            try:
                self.file.close()
                self.logger.debug(f"Closed file {self.file_path}")
            except Exception as e:
                self.logger.error(f"Error closing file: {e}", exc_info=True)
            
            self.file = None

class CallbackSink(StreamSink):
    """Sink that calls a callback function."""
    
    def __init__(self, sink_id: str, callback: Callable[[Dict[str, Any]], None]):
        """Initialize the callback sink.
        
        Args:
            sink_id: Unique identifier for this sink.
            callback: Function to call with each data item.
        """
        super().__init__(sink_id)
        self.callback = callback
    
    def sink(self, data: Dict[str, Any]) -> bool:
        """Sink a single data item.
        
        Args:
            data: The data item to sink.
            
        Returns:
            True if the data was successfully sunk, False otherwise.
        """
        try:
            self.callback(data)
            self.metrics["received_items"] += 1
            self.metrics["last_sink_time"] = time.time()
            return True
        except Exception as e:
            self.logger.error(f"Error in callback: {e}", exc_info=True)
            self.metrics["errors"] += 1
            return False

class StreamingPipeline:
    """Core streaming pipeline implementation."""
    
    def __init__(self, pipeline_id: str, config: Optional[StreamingConfig] = None):
        """Initialize the streaming pipeline.
        
        Args:
            pipeline_id: Unique identifier for this pipeline.
            config: Configuration for this pipeline.
        """
        self.pipeline_id = pipeline_id
        self.config = config or StreamingConfig()
        self.streams: Dict[str, DataStream] = {}
        self.processors: Dict[str, StreamProcessor] = {}
        self.sinks: Dict[str, StreamSink] = {}
        self.logger = logging.getLogger(f"{__name__}.pipeline.{pipeline_id}")
        self.running = False
    
    def create_stream(self, stream_id: str, config: Optional[StreamingConfig] = None) -> DataStream:
        """Create a new data stream.
        
        Args:
            stream_id: Unique identifier for the stream.
            config: Configuration for the stream.
            
        Returns:
            The created data stream.
        """
        if stream_id in self.streams:
            self.logger.warning(f"Stream {stream_id} already exists")
            return self.streams[stream_id]
        
        stream = DataStream(stream_id, config or self.config)
        self.streams[stream_id] = stream
        self.logger.debug(f"Created stream {stream_id}")
        return stream
    
    def add_processor(self, stream_id: str, processor: StreamProcessor):
        """Add a processor to a stream.
        
        Args:
            stream_id: Identifier of the stream to add the processor to.
            processor: The processor to add.
        """
        if stream_id not in self.streams:
            self.logger.error(f"Stream {stream_id} does not exist")
            return
        
        self.streams[stream_id].add_processor(processor)
        self.processors[processor.processor_id] = processor
    
    def create_sink_processor(self, processor_id: str, sink: StreamSink) -> StreamProcessor:
        """Create a processor that sinks data to a sink.
        
        Args:
            processor_id: Unique identifier for the processor.
            sink: The sink to use.
            
        Returns:
            The created processor.
        """
        self.sinks[sink.sink_id] = sink
        
        # Create a processor that sinks data to the sink
        class SinkProcessor(StreamProcessor):
            def __init__(self, processor_id: str, sink: StreamSink):
                super().__init__(processor_id)
                self.sink = sink
            
            def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
                self.sink.sink(data)
                self.metrics["processed_items"] += 1
                self.metrics["last_processing_time"] = time.time()
                return data
            
            def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                self.sink.sink_batch(batch)
                self.metrics["processed_items"] += len(batch)
                self.metrics["last_processing_time"] = time.time()
                return batch
        
        processor = SinkProcessor(processor_id, sink)
        self.processors[processor_id] = processor
        return processor
    
    def start(self):
        """Start the streaming pipeline."""
        if self.running:
            self.logger.warning("Pipeline already running")
            return
        
        self.running = True
        self.logger.info(f"Starting streaming pipeline {self.pipeline_id}")
        
        # Start all streams
        for stream in self.streams.values():
            stream.start()
    
    def stop(self):
        """Stop the streaming pipeline."""
        if not self.running:
            self.logger.warning("Pipeline not running")
            return
        
        self.logger.info(f"Stopping streaming pipeline {self.pipeline_id}")
        self.running = False
        
        # Stop all streams
        for stream in self.streams.values():
            stream.stop()
        
        # Close all sinks
        for sink in self.sinks.values():
            if hasattr(sink, 'close') and callable(sink.close):
                sink.close()
        
        self.logger.info(f"Streaming pipeline {self.pipeline_id} stopped")
    
    def push(self, stream_id: str, data: Dict[str, Any]) -> bool:
        """Push data into a stream.
        
        Args:
            stream_id: Identifier of the stream to push data to.
            data: The data to push.
            
        Returns:
            True if the data was successfully pushed, False otherwise.
        """
        if stream_id not in self.streams:
            self.logger.error(f"Stream {stream_id} does not exist")
            return False
        
        return self.streams[stream_id].push(data)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics.
        
        Returns:
            A dictionary containing pipeline metrics.
        """
        return {
            "pipeline_id": self.pipeline_id,
            "running": self.running,
            "streams": {stream_id: stream.get_metrics() for stream_id, stream in self.streams.items()},
            "processors": {processor_id: processor.get_metrics() for processor_id, processor in self.processors.items()},
            "sinks": {sink_id: sink.get_metrics() for sink_id, sink in self.sinks.items()},
        }

# Example usage
if __name__ == "__main__":
    # Create a streaming pipeline
    pipeline = StreamingPipeline("example-pipeline")
    
    # Create a data stream
    stream = pipeline.create_stream("example-stream")
    
    # Create a filter processor
    filter_processor = FilterProcessor("example-filter", lambda data: data.get("value", 0) > 50)
    
    # Create a transform processor
    transform_processor = TransformProcessor("example-transform", lambda data: {**data, "value_squared": data.get("value", 0) ** 2})
    
    # Create a log sink
    log_sink = LogSink("example-log-sink")
    
    # Create a sink processor
    sink_processor = pipeline.create_sink_processor("example-sink-processor", log_sink)
    
    # Add processors to the stream
    pipeline.add_processor("example-stream", filter_processor)
    pipeline.add_processor("example-stream", transform_processor)
    pipeline.add_processor("example-stream", sink_processor)
    
    # Start the pipeline
    pipeline.start()
    
    # Push some data
    for i in range(100):
        pipeline.push("example-stream", {"value": i})
    
    # Let it run for a while
    time.sleep(5)
    
    # Stop the pipeline
    pipeline.stop()
    
    # Print metrics
    print(pipeline.get_metrics())

