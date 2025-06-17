"""
ALL-USE Learning Systems - Batch Collector

This module implements the batch collection capabilities for the ALL-USE Learning Systems,
providing components for collecting data in batches from various system components.

The batch collector is designed to:
- Collect data in configurable batch sizes
- Support scheduled collection at specified intervals
- Provide data validation and transformation
- Support multiple data sources
- Implement efficient storage mechanisms

Classes:
- BatchCollector: Core batch collection implementation
- CollectionJob: Represents a scheduled collection job
- DataValidator: Validates collected data
- DataTransformer: Transforms collected data

Version: 1.0.0
"""

import time
import logging
import threading
import schedule
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json
import os
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CollectionFrequency(Enum):
    """Collection frequencies for batch collection."""
    MINUTELY = 1
    HOURLY = 2
    DAILY = 3
    WEEKLY = 4
    MONTHLY = 5
    CUSTOM = 6

@dataclass
class BatchCollectionConfig:
    """Configuration for batch collection."""
    batch_size: int = 1000
    collection_frequency: CollectionFrequency = CollectionFrequency.HOURLY
    custom_interval_seconds: Optional[int] = None  # Used with CUSTOM frequency
    max_retries: int = 3
    retry_delay_seconds: int = 60
    validation_enabled: bool = True
    transformation_enabled: bool = True
    storage_path: str = "/tmp/batch_collection"
    file_format: str = "json"  # json, csv, etc.
    compression_enabled: bool = False
    max_batch_age_seconds: int = 3600  # Maximum age of data in a batch

class DataValidator:
    """Validates collected data."""
    
    def __init__(self, validator_id: str):
        """Initialize the data validator.
        
        Args:
            validator_id: Unique identifier for this validator.
        """
        self.validator_id = validator_id
        self.logger = logging.getLogger(f"{__name__}.validator.{validator_id}")
        self.metrics = {
            "validated_items": 0,
            "invalid_items": 0,
            "errors": 0,
        }
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate a single data item.
        
        Args:
            data: The data item to validate.
            
        Returns:
            True if the data is valid, False otherwise.
        """
        # Default implementation considers all data valid
        self.metrics["validated_items"] += 1
        return True
    
    def validate_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate a batch of data items.
        
        Args:
            batch: The batch of data items to validate.
            
        Returns:
            A list of valid data items.
        """
        valid_items = []
        for data in batch:
            try:
                if self.validate(data):
                    valid_items.append(data)
                else:
                    self.metrics["invalid_items"] += 1
            except Exception as e:
                self.logger.error(f"Error validating data: {e}", exc_info=True)
                self.metrics["errors"] += 1
        
        return valid_items
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get validator metrics.
        
        Returns:
            A dictionary containing validator metrics.
        """
        return {
            "validator_id": self.validator_id,
            "metrics": self.metrics,
        }

class SchemaValidator(DataValidator):
    """Validates data against a schema."""
    
    def __init__(self, validator_id: str, schema: Dict[str, Any]):
        """Initialize the schema validator.
        
        Args:
            validator_id: Unique identifier for this validator.
            schema: The schema to validate against.
        """
        super().__init__(validator_id)
        self.schema = schema
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate a single data item against the schema.
        
        Args:
            data: The data item to validate.
            
        Returns:
            True if the data is valid, False otherwise.
        """
        try:
            # Simple schema validation
            for field, field_type in self.schema.items():
                if field not in data:
                    self.logger.debug(f"Field {field} missing in data")
                    self.metrics["invalid_items"] += 1
                    return False
                
                if not isinstance(data[field], field_type):
                    self.logger.debug(f"Field {field} has wrong type: expected {field_type}, got {type(data[field])}")
                    self.metrics["invalid_items"] += 1
                    return False
            
            self.metrics["validated_items"] += 1
            return True
        except Exception as e:
            self.logger.error(f"Error validating data against schema: {e}", exc_info=True)
            self.metrics["errors"] += 1
            return False

class DataTransformer:
    """Transforms collected data."""
    
    def __init__(self, transformer_id: str):
        """Initialize the data transformer.
        
        Args:
            transformer_id: Unique identifier for this transformer.
        """
        self.transformer_id = transformer_id
        self.logger = logging.getLogger(f"{__name__}.transformer.{transformer_id}")
        self.metrics = {
            "transformed_items": 0,
            "errors": 0,
        }
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single data item.
        
        Args:
            data: The data item to transform.
            
        Returns:
            The transformed data item.
        """
        # Default implementation returns the data unchanged
        self.metrics["transformed_items"] += 1
        return data
    
    def transform_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform a batch of data items.
        
        Args:
            batch: The batch of data items to transform.
            
        Returns:
            The transformed batch of data items.
        """
        transformed_batch = []
        for data in batch:
            try:
                transformed_data = self.transform(data)
                transformed_batch.append(transformed_data)
            except Exception as e:
                self.logger.error(f"Error transforming data: {e}", exc_info=True)
                self.metrics["errors"] += 1
                transformed_batch.append(data)  # Use original data on error
        
        return transformed_batch
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get transformer metrics.
        
        Returns:
            A dictionary containing transformer metrics.
        """
        return {
            "transformer_id": self.transformer_id,
            "metrics": self.metrics,
        }

class EnrichmentTransformer(DataTransformer):
    """Enriches data with additional information."""
    
    def __init__(self, transformer_id: str, enrichment_func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Initialize the enrichment transformer.
        
        Args:
            transformer_id: Unique identifier for this transformer.
            enrichment_func: Function that enriches the data.
        """
        super().__init__(transformer_id)
        self.enrichment_func = enrichment_func
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single data item by enriching it.
        
        Args:
            data: The data item to transform.
            
        Returns:
            The enriched data item.
        """
        try:
            enriched_data = self.enrichment_func(data)
            self.metrics["transformed_items"] += 1
            return enriched_data
        except Exception as e:
            self.logger.error(f"Error enriching data: {e}", exc_info=True)
            self.metrics["errors"] += 1
            return data  # Return original data on error

class CollectionJob:
    """Represents a scheduled collection job."""
    
    def __init__(self, job_id: str, collector: 'BatchCollector', data_source: Callable[[], List[Dict[str, Any]]]):
        """Initialize the collection job.
        
        Args:
            job_id: Unique identifier for this job.
            collector: The batch collector to use.
            data_source: Function that returns the data to collect.
        """
        self.job_id = job_id
        self.collector = collector
        self.data_source = data_source
        self.logger = logging.getLogger(f"{__name__}.job.{job_id}")
        self.metrics = {
            "executions": 0,
            "collected_items": 0,
            "errors": 0,
            "last_execution_time": None,
            "last_execution_duration": None,
        }
        self.scheduled_job = None
    
    def execute(self):
        """Execute the collection job."""
        start_time = time.time()
        self.metrics["executions"] += 1
        self.metrics["last_execution_time"] = datetime.now().isoformat()
        
        try:
            # Collect data from source
            data = self.data_source()
            
            # Add metadata
            for item in data:
                if "timestamp" not in item:
                    item["timestamp"] = time.time()
                if "job_id" not in item:
                    item["job_id"] = self.job_id
            
            # Add to collector
            self.collector.add_batch(data)
            
            self.metrics["collected_items"] += len(data)
            self.logger.info(f"Collected {len(data)} items from job {self.job_id}")
        except Exception as e:
            self.logger.error(f"Error executing collection job: {e}", exc_info=True)
            self.metrics["errors"] += 1
        
        self.metrics["last_execution_duration"] = time.time() - start_time
    
    def schedule(self, frequency: CollectionFrequency, custom_interval_seconds: Optional[int] = None):
        """Schedule the collection job.
        
        Args:
            frequency: The frequency at which to execute the job.
            custom_interval_seconds: Custom interval in seconds (used with CUSTOM frequency).
        """
        if self.scheduled_job:
            schedule.cancel_job(self.scheduled_job)
        
        if frequency == CollectionFrequency.MINUTELY:
            self.scheduled_job = schedule.every().minute.do(self.execute)
        elif frequency == CollectionFrequency.HOURLY:
            self.scheduled_job = schedule.every().hour.do(self.execute)
        elif frequency == CollectionFrequency.DAILY:
            self.scheduled_job = schedule.every().day.do(self.execute)
        elif frequency == CollectionFrequency.WEEKLY:
            self.scheduled_job = schedule.every().week.do(self.execute)
        elif frequency == CollectionFrequency.MONTHLY:
            self.scheduled_job = schedule.every(30).days.do(self.execute)  # Approximate
        elif frequency == CollectionFrequency.CUSTOM and custom_interval_seconds:
            self.scheduled_job = schedule.every(custom_interval_seconds).seconds.do(self.execute)
        else:
            self.logger.error(f"Invalid frequency: {frequency}")
        
        self.logger.info(f"Scheduled job {self.job_id} with frequency {frequency.name}")
    
    def cancel(self):
        """Cancel the scheduled job."""
        if self.scheduled_job:
            schedule.cancel_job(self.scheduled_job)
            self.scheduled_job = None
            self.logger.info(f"Cancelled job {self.job_id}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get job metrics.
        
        Returns:
            A dictionary containing job metrics.
        """
        return {
            "job_id": self.job_id,
            "metrics": self.metrics,
        }

class BatchCollector:
    """Core batch collection implementation."""
    
    def __init__(self, collector_id: str, config: Optional[BatchCollectionConfig] = None):
        """Initialize the batch collector.
        
        Args:
            collector_id: Unique identifier for this collector.
            config: Configuration for this collector.
        """
        self.collector_id = collector_id
        self.config = config or BatchCollectionConfig()
        self.logger = logging.getLogger(f"{__name__}.collector.{collector_id}")
        self.current_batch: List[Dict[str, Any]] = []
        self.jobs: Dict[str, CollectionJob] = {}
        self.validators: Dict[str, DataValidator] = {}
        self.transformers: Dict[str, DataTransformer] = {}
        self.running = False
        self.scheduler_thread = None
        self.batch_lock = threading.Lock()
        self.metrics = {
            "collected_items": 0,
            "stored_batches": 0,
            "errors": 0,
            "last_storage_time": None,
        }
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.config.storage_path, exist_ok=True)
    
    def start(self):
        """Start the batch collector."""
        if self.running:
            self.logger.warning("Collector already running")
            return
        
        self.running = True
        self.logger.info(f"Starting batch collector {self.collector_id}")
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name=f"collector-{self.collector_id}",
            daemon=True
        )
        self.scheduler_thread.start()
    
    def stop(self):
        """Stop the batch collector."""
        if not self.running:
            self.logger.warning("Collector not running")
            return
        
        self.logger.info(f"Stopping batch collector {self.collector_id}")
        self.running = False
        
        # Wait for thread to terminate
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        
        # Store any remaining data
        self._store_current_batch()
        
        self.logger.info(f"Batch collector {self.collector_id} stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        self.logger.info(f"Scheduler loop started for collector {self.collector_id}")
        
        last_batch_check = time.time()
        
        while self.running:
            try:
                # Run scheduled jobs
                schedule.run_pending()
                
                # Check if current batch should be stored
                current_time = time.time()
                if (len(self.current_batch) >= self.config.batch_size or
                    (self.current_batch and current_time - last_batch_check >= self.config.max_batch_age_seconds)):
                    
                    self._store_current_batch()
                    last_batch_check = current_time
                
                # Sleep briefly
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                self.metrics["errors"] += 1
                time.sleep(5.0)  # Sleep longer on error
    
    def add_job(self, job_id: str, data_source: Callable[[], List[Dict[str, Any]]]) -> CollectionJob:
        """Add a collection job.
        
        Args:
            job_id: Unique identifier for the job.
            data_source: Function that returns the data to collect.
            
        Returns:
            The created collection job.
        """
        if job_id in self.jobs:
            self.logger.warning(f"Job {job_id} already exists")
            return self.jobs[job_id]
        
        job = CollectionJob(job_id, self, data_source)
        self.jobs[job_id] = job
        
        # Schedule the job based on configuration
        job.schedule(
            self.config.collection_frequency,
            self.config.custom_interval_seconds
        )
        
        self.logger.debug(f"Added job {job_id}")
        return job
    
    def remove_job(self, job_id: str):
        """Remove a collection job.
        
        Args:
            job_id: Identifier of the job to remove.
        """
        if job_id in self.jobs:
            self.jobs[job_id].cancel()
            del self.jobs[job_id]
            self.logger.debug(f"Removed job {job_id}")
    
    def add_validator(self, validator: DataValidator):
        """Add a data validator.
        
        Args:
            validator: The validator to add.
        """
        self.validators[validator.validator_id] = validator
        self.logger.debug(f"Added validator {validator.validator_id}")
    
    def add_transformer(self, transformer: DataTransformer):
        """Add a data transformer.
        
        Args:
            transformer: The transformer to add.
        """
        self.transformers[transformer.transformer_id] = transformer
        self.logger.debug(f"Added transformer {transformer.transformer_id}")
    
    def add_batch(self, batch: List[Dict[str, Any]]):
        """Add a batch of data.
        
        Args:
            batch: The batch of data to add.
        """
        if not batch:
            return
        
        # Validate data if enabled
        if self.config.validation_enabled:
            for validator in self.validators.values():
                batch = validator.validate_batch(batch)
                if not batch:
                    self.logger.warning("All data items were invalid")
                    return
        
        # Transform data if enabled
        if self.config.transformation_enabled:
            for transformer in self.transformers.values():
                batch = transformer.transform_batch(batch)
        
        # Add to current batch
        with self.batch_lock:
            self.current_batch.extend(batch)
            self.metrics["collected_items"] += len(batch)
        
        # Store batch if it's full
        if len(self.current_batch) >= self.config.batch_size:
            self._store_current_batch()
    
    def _store_current_batch(self):
        """Store the current batch."""
        with self.batch_lock:
            if not self.current_batch:
                return
            
            try:
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_id = str(uuid.uuid4())[:8]
                filename = f"{self.collector_id}_{timestamp}_{batch_id}.{self.config.file_format}"
                filepath = os.path.join(self.config.storage_path, filename)
                
                # Store based on format
                if self.config.file_format == "json":
                    with open(filepath, 'w') as f:
                        json.dump(self.current_batch, f)
                else:
                    # Default to JSON
                    with open(filepath, 'w') as f:
                        json.dump(self.current_batch, f)
                
                self.logger.info(f"Stored batch of {len(self.current_batch)} items to {filepath}")
                self.metrics["stored_batches"] += 1
                self.metrics["last_storage_time"] = datetime.now().isoformat()
                
                # Clear current batch
                self.current_batch = []
                
            except Exception as e:
                self.logger.error(f"Error storing batch: {e}", exc_info=True)
                self.metrics["errors"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collector metrics.
        
        Returns:
            A dictionary containing collector metrics.
        """
        return {
            "collector_id": self.collector_id,
            "current_batch_size": len(self.current_batch),
            "metrics": self.metrics,
            "jobs": {job_id: job.get_metrics() for job_id, job in self.jobs.items()},
            "validators": {validator_id: validator.get_metrics() for validator_id, validator in self.validators.items()},
            "transformers": {transformer_id: transformer.get_metrics() for transformer_id, transformer in self.transformers.items()},
        }

# Example usage
if __name__ == "__main__":
    # Create a batch collector
    collector = BatchCollector("example-collector")
    
    # Create a schema validator
    schema = {
        "value": int,
        "name": str,
    }
    validator = SchemaValidator("example-validator", schema)
    collector.add_validator(validator)
    
    # Create an enrichment transformer
    def enrich_data(data):
        data["enriched"] = True
        data["processed_at"] = datetime.now().isoformat()
        return data
    
    transformer = EnrichmentTransformer("example-transformer", enrich_data)
    collector.add_transformer(transformer)
    
    # Create a data source
    def data_source():
        return [
            {"value": i, "name": f"Item {i}"} 
            for i in range(10)
        ]
    
    # Add a collection job
    job = collector.add_job("example-job", data_source)
    
    # Start the collector
    collector.start()
    
    # Let it run for a while
    time.sleep(5)
    
    # Execute the job manually
    job.execute()
    
    # Let it process
    time.sleep(2)
    
    # Stop the collector
    collector.stop()
    
    # Print metrics
    print(collector.get_metrics())

