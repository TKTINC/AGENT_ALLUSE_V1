"""
ALL-USE Learning Systems - Distributed Storage

This module implements a distributed storage system for the ALL-USE Learning Systems,
providing scalable and fault-tolerant storage for large datasets.

The distributed storage system is designed to:
- Store large volumes of data across multiple nodes
- Provide high availability and fault tolerance
- Support efficient data retrieval and processing
- Enable horizontal scaling as data volumes grow
- Support various data formats and access patterns

Classes:
- DistributedStorage: Core distributed storage implementation
- StorageNode: Represents a single storage node
- DataPartition: Represents a partition of data
- ReplicationManager: Manages data replication
- PartitionManager: Manages data partitioning

Version: 1.0.0
"""

import time
import logging
import threading
import json
import os
import uuid
import hashlib
import random
import socket
import struct
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StorageMode(Enum):
    """Storage modes for the distributed storage system."""
    LOCAL = 1      # Local storage only (single node)
    REPLICATED = 2 # Replicated storage across multiple nodes
    SHARDED = 3    # Sharded storage across multiple nodes
    HYBRID = 4     # Hybrid of replication and sharding

class ConsistencyLevel(Enum):
    """Consistency levels for read and write operations."""
    ONE = 1        # At least one node must respond
    QUORUM = 2     # A majority of nodes must respond
    ALL = 3        # All nodes must respond
    LOCAL_QUORUM = 4 # A majority of nodes in the local datacenter must respond

@dataclass
class DistributedStorageConfig:
    """Configuration for the distributed storage system."""
    storage_root: str = "distributed_storage"
    replication_factor: int = 3
    num_partitions: int = 16
    storage_mode: StorageMode = StorageMode.REPLICATED
    read_consistency: ConsistencyLevel = ConsistencyLevel.QUORUM
    write_consistency: ConsistencyLevel = ConsistencyLevel.QUORUM
    node_check_interval_seconds: int = 30
    rebalance_threshold: float = 0.2  # Trigger rebalance when node utilization differs by this factor
    max_partition_size_mb: int = 1024  # Maximum size of a partition before splitting
    enable_compression: bool = True
    compression_level: int = 6  # 1-9, higher means better compression but slower
    enable_encryption: bool = False
    encryption_key: Optional[str] = None

class StorageNode:
    """Represents a single storage node in the distributed storage system."""
    
    def __init__(self, node_id: str, host: str, port: int, storage_path: str):
        """Initialize the storage node.
        
        Args:
            node_id: Unique identifier for this node.
            host: Hostname or IP address of the node.
            port: Port number the node is listening on.
            storage_path: Path to the storage directory on this node.
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.storage_path = storage_path
        self.status = "ONLINE"
        self.last_heartbeat = time.time()
        self.capacity_bytes = 0
        self.used_bytes = 0
        self.partitions: Set[str] = set()
        
        # Initialize storage path
        os.makedirs(storage_path, exist_ok=True)
        
        # Calculate capacity
        try:
            stat = os.statvfs(storage_path)
            self.capacity_bytes = stat.f_frsize * stat.f_blocks
            self.used_bytes = self.capacity_bytes - (stat.f_frsize * stat.f_bfree)
        except Exception as e:
            logger.error(f"Error calculating storage capacity for node {node_id}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary.
        
        Returns:
            Dictionary representation of the node.
        """
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "storage_path": self.storage_path,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "partitions": list(self.partitions),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StorageNode':
        """Create a node from a dictionary.
        
        Args:
            data: Dictionary representation of the node.
            
        Returns:
            The created node.
        """
        node = cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            storage_path=data["storage_path"],
        )
        node.status = data["status"]
        node.last_heartbeat = data["last_heartbeat"]
        node.capacity_bytes = data["capacity_bytes"]
        node.used_bytes = data["used_bytes"]
        node.partitions = set(data["partitions"])
        return node
    
    def update_heartbeat(self):
        """Update the last heartbeat time."""
        self.last_heartbeat = time.time()
    
    def update_usage(self):
        """Update the storage usage statistics."""
        try:
            stat = os.statvfs(self.storage_path)
            self.capacity_bytes = stat.f_frsize * stat.f_blocks
            self.used_bytes = self.capacity_bytes - (stat.f_frsize * stat.f_bfree)
        except Exception as e:
            logger.error(f"Error updating storage usage for node {self.node_id}: {e}")
    
    def is_alive(self, timeout_seconds: int = 60) -> bool:
        """Check if the node is alive.
        
        Args:
            timeout_seconds: Number of seconds after which a node is considered dead.
            
        Returns:
            True if the node is alive, False otherwise.
        """
        return (time.time() - self.last_heartbeat) < timeout_seconds
    
    def utilization(self) -> float:
        """Calculate the storage utilization of this node.
        
        Returns:
            Storage utilization as a fraction (0.0 to 1.0).
        """
        if self.capacity_bytes == 0:
            return 0.0
        return self.used_bytes / self.capacity_bytes
    
    def add_partition(self, partition_id: str):
        """Add a partition to this node.
        
        Args:
            partition_id: ID of the partition to add.
        """
        self.partitions.add(partition_id)
    
    def remove_partition(self, partition_id: str):
        """Remove a partition from this node.
        
        Args:
            partition_id: ID of the partition to remove.
        """
        self.partitions.discard(partition_id)

class DataPartition:
    """Represents a partition of data in the distributed storage system."""
    
    def __init__(self, partition_id: str, storage: 'DistributedStorage'):
        """Initialize the data partition.
        
        Args:
            partition_id: Unique identifier for this partition.
            storage: The distributed storage system this partition belongs to.
        """
        self.partition_id = partition_id
        self.storage = storage
        self.nodes: List[str] = []  # List of node IDs that have this partition
        self.size_bytes = 0
        self.num_objects = 0
        self.created_at = time.time()
        self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the partition to a dictionary.
        
        Returns:
            Dictionary representation of the partition.
        """
        return {
            "partition_id": self.partition_id,
            "nodes": self.nodes,
            "size_bytes": self.size_bytes,
            "num_objects": self.num_objects,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], storage: 'DistributedStorage') -> 'DataPartition':
        """Create a partition from a dictionary.
        
        Args:
            data: Dictionary representation of the partition.
            storage: The distributed storage system this partition belongs to.
            
        Returns:
            The created partition.
        """
        partition = cls(partition_id=data["partition_id"], storage=storage)
        partition.nodes = data["nodes"]
        partition.size_bytes = data["size_bytes"]
        partition.num_objects = data["num_objects"]
        partition.created_at = data["created_at"]
        partition.updated_at = data["updated_at"]
        return partition
    
    def add_node(self, node_id: str):
        """Add a node to this partition.
        
        Args:
            node_id: ID of the node to add.
        """
        if node_id not in self.nodes:
            self.nodes.append(node_id)
            node = self.storage.get_node(node_id)
            if node:
                node.add_partition(self.partition_id)
    
    def remove_node(self, node_id: str):
        """Remove a node from this partition.
        
        Args:
            node_id: ID of the node to remove.
        """
        if node_id in self.nodes:
            self.nodes.remove(node_id)
            node = self.storage.get_node(node_id)
            if node:
                node.remove_partition(self.partition_id)
    
    def get_path(self, node_id: str) -> str:
        """Get the path to this partition on a specific node.
        
        Args:
            node_id: ID of the node.
            
        Returns:
            Path to the partition on the node.
        """
        node = self.storage.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")
        
        return os.path.join(node.storage_path, self.partition_id)
    
    def ensure_path(self, node_id: str) -> str:
        """Ensure the partition path exists on a specific node.
        
        Args:
            node_id: ID of the node.
            
        Returns:
            Path to the partition on the node.
        """
        path = self.get_path(node_id)
        os.makedirs(path, exist_ok=True)
        return path
    
    def update_stats(self):
        """Update the statistics for this partition."""
        self.size_bytes = 0
        self.num_objects = 0
        
        for node_id in self.nodes:
            try:
                path = self.get_path(node_id)
                if os.path.exists(path):
                    # Count files and calculate size
                    for root, _, files in os.walk(path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            self.size_bytes += os.path.getsize(file_path)
                            self.num_objects += 1
            except Exception as e:
                logger.error(f"Error updating stats for partition {self.partition_id} on node {node_id}: {e}")
        
        self.updated_at = time.time()

class ReplicationManager:
    """Manages data replication across nodes."""
    
    def __init__(self, storage: 'DistributedStorage'):
        """Initialize the replication manager.
        
        Args:
            storage: The distributed storage system this manager belongs to.
        """
        self.storage = storage
        self.logger = logging.getLogger(f"{__name__}.replication_manager")
        self.running = False
        self.replication_thread = None
    
    def start(self):
        """Start the replication manager."""
        if self.running:
            self.logger.warning("Replication manager already running")
            return
        
        self.running = True
        self.logger.info("Starting replication manager")
        
        # Start replication thread
        self.replication_thread = threading.Thread(
            target=self._replication_loop,
            name="replication-manager",
            daemon=True
        )
        self.replication_thread.start()
    
    def stop(self):
        """Stop the replication manager."""
        if not self.running:
            self.logger.warning("Replication manager not running")
            return
        
        self.logger.info("Stopping replication manager")
        self.running = False
        
        # Wait for thread to terminate
        if self.replication_thread and self.replication_thread.is_alive():
            self.replication_thread.join(timeout=5.0)
        
        self.logger.info("Replication manager stopped")
    
    def _replication_loop(self):
        """Main loop for the replication manager."""
        self.logger.info("Replication loop started")
        
        while self.running:
            try:
                # Check replication status for all partitions
                self._check_replication()
                
                # Sleep for a while
                time.sleep(60)  # 1 minute
                
            except Exception as e:
                self.logger.error(f"Error in replication loop: {e}", exc_info=True)
                time.sleep(60)  # Sleep longer on error
    
    def _check_replication(self):
        """Check replication status for all partitions."""
        for partition_id, partition in self.storage.partitions.items():
            try:
                # Check if partition has enough replicas
                active_nodes = [
                    node_id for node_id in partition.nodes
                    if self.storage.get_node(node_id) and self.storage.get_node(node_id).is_alive()
                ]
                
                if len(active_nodes) < self.storage.config.replication_factor:
                    self.logger.info(f"Partition {partition_id} has {len(active_nodes)} active replicas, "
                                    f"need {self.storage.config.replication_factor}")
                    
                    # Find nodes to replicate to
                    available_nodes = [
                        node_id for node_id in self.storage.nodes.keys()
                        if node_id not in partition.nodes and self.storage.get_node(node_id).is_alive()
                    ]
                    
                    # Sort by utilization (least utilized first)
                    available_nodes.sort(
                        key=lambda node_id: self.storage.get_node(node_id).utilization()
                    )
                    
                    # Replicate to as many nodes as needed
                    nodes_needed = self.storage.config.replication_factor - len(active_nodes)
                    nodes_to_replicate = available_nodes[:nodes_needed]
                    
                    if nodes_to_replicate:
                        self._replicate_partition(partition_id, active_nodes[0], nodes_to_replicate)
            
            except Exception as e:
                self.logger.error(f"Error checking replication for partition {partition_id}: {e}", exc_info=True)
    
    def _replicate_partition(self, partition_id: str, source_node_id: str, target_node_ids: List[str]):
        """Replicate a partition from one node to others.
        
        Args:
            partition_id: ID of the partition to replicate.
            source_node_id: ID of the source node.
            target_node_ids: IDs of the target nodes.
        """
        partition = self.storage.partitions.get(partition_id)
        if not partition:
            self.logger.error(f"Partition {partition_id} not found")
            return
        
        source_node = self.storage.get_node(source_node_id)
        if not source_node:
            self.logger.error(f"Source node {source_node_id} not found")
            return
        
        source_path = partition.get_path(source_node_id)
        if not os.path.exists(source_path):
            self.logger.error(f"Source path {source_path} does not exist")
            return
        
        for target_node_id in target_node_ids:
            try:
                target_node = self.storage.get_node(target_node_id)
                if not target_node:
                    self.logger.error(f"Target node {target_node_id} not found")
                    continue
                
                target_path = partition.ensure_path(target_node_id)
                
                self.logger.info(f"Replicating partition {partition_id} from {source_node_id} to {target_node_id}")
                
                # Copy all files from source to target
                for root, _, files in os.walk(source_path):
                    rel_path = os.path.relpath(root, source_path)
                    target_dir = os.path.join(target_path, rel_path)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    for file in files:
                        source_file = os.path.join(root, file)
                        target_file = os.path.join(target_dir, file)
                        
                        # Copy file
                        with open(source_file, 'rb') as src, open(target_file, 'wb') as dst:
                            dst.write(src.read())
                
                # Update partition
                partition.add_node(target_node_id)
                
                self.logger.info(f"Replication of partition {partition_id} to {target_node_id} completed")
                
            except Exception as e:
                self.logger.error(f"Error replicating partition {partition_id} to {target_node_id}: {e}", exc_info=True)

class PartitionManager:
    """Manages data partitioning across nodes."""
    
    def __init__(self, storage: 'DistributedStorage'):
        """Initialize the partition manager.
        
        Args:
            storage: The distributed storage system this manager belongs to.
        """
        self.storage = storage
        self.logger = logging.getLogger(f"{__name__}.partition_manager")
        self.running = False
        self.partition_thread = None
    
    def start(self):
        """Start the partition manager."""
        if self.running:
            self.logger.warning("Partition manager already running")
            return
        
        self.running = True
        self.logger.info("Starting partition manager")
        
        # Start partition thread
        self.partition_thread = threading.Thread(
            target=self._partition_loop,
            name="partition-manager",
            daemon=True
        )
        self.partition_thread.start()
    
    def stop(self):
        """Stop the partition manager."""
        if not self.running:
            self.logger.warning("Partition manager not running")
            return
        
        self.logger.info("Stopping partition manager")
        self.running = False
        
        # Wait for thread to terminate
        if self.partition_thread and self.partition_thread.is_alive():
            self.partition_thread.join(timeout=5.0)
        
        self.logger.info("Partition manager stopped")
    
    def _partition_loop(self):
        """Main loop for the partition manager."""
        self.logger.info("Partition loop started")
        
        while self.running:
            try:
                # Check partition sizes
                self._check_partition_sizes()
                
                # Check node balance
                self._check_node_balance()
                
                # Sleep for a while
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in partition loop: {e}", exc_info=True)
                time.sleep(300)  # Sleep longer on error
    
    def _check_partition_sizes(self):
        """Check if any partitions need to be split."""
        for partition_id, partition in self.storage.partitions.items():
            try:
                # Update partition stats
                partition.update_stats()
                
                # Check if partition is too large
                if partition.size_bytes > self.storage.config.max_partition_size_mb * 1024 * 1024:
                    self.logger.info(f"Partition {partition_id} is too large ({partition.size_bytes} bytes), "
                                    f"need to split")
                    
                    # Split partition
                    self._split_partition(partition_id)
            
            except Exception as e:
                self.logger.error(f"Error checking partition size for {partition_id}: {e}", exc_info=True)
    
    def _split_partition(self, partition_id: str):
        """Split a partition into two.
        
        Args:
            partition_id: ID of the partition to split.
        """
        partition = self.storage.partitions.get(partition_id)
        if not partition:
            self.logger.error(f"Partition {partition_id} not found")
            return
        
        # Create two new partitions
        new_partition_id1 = f"{partition_id}_1"
        new_partition_id2 = f"{partition_id}_2"
        
        new_partition1 = DataPartition(new_partition_id1, self.storage)
        new_partition2 = DataPartition(new_partition_id2, self.storage)
        
        # Add to storage
        self.storage.partitions[new_partition_id1] = new_partition1
        self.storage.partitions[new_partition_id2] = new_partition2
        
        # Assign nodes
        for node_id in partition.nodes:
            new_partition1.add_node(node_id)
            new_partition2.add_node(node_id)
            
            # Create directories
            new_partition1.ensure_path(node_id)
            new_partition2.ensure_path(node_id)
        
        # Move data
        for node_id in partition.nodes:
            try:
                source_path = partition.get_path(node_id)
                target_path1 = new_partition1.get_path(node_id)
                target_path2 = new_partition2.get_path(node_id)
                
                # List all files
                all_files = []
                for root, _, files in os.walk(source_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, source_path)
                        all_files.append(rel_path)
                
                # Sort files
                all_files.sort()
                
                # Split files
                files1 = all_files[:len(all_files)//2]
                files2 = all_files[len(all_files)//2:]
                
                # Move files to new partitions
                for rel_path in files1:
                    source_file = os.path.join(source_path, rel_path)
                    target_file = os.path.join(target_path1, rel_path)
                    os.makedirs(os.path.dirname(target_file), exist_ok=True)
                    os.rename(source_file, target_file)
                
                for rel_path in files2:
                    source_file = os.path.join(source_path, rel_path)
                    target_file = os.path.join(target_path2, rel_path)
                    os.makedirs(os.path.dirname(target_file), exist_ok=True)
                    os.rename(source_file, target_file)
                
            except Exception as e:
                self.logger.error(f"Error moving data for partition {partition_id} on node {node_id}: {e}", exc_info=True)
        
        # Remove old partition
        for node_id in partition.nodes:
            partition.remove_node(node_id)
            
            # Remove directory
            try:
                source_path = partition.get_path(node_id)
                if os.path.exists(source_path):
                    os.rmdir(source_path)
            except Exception as e:
                self.logger.error(f"Error removing directory for partition {partition_id} on node {node_id}: {e}", exc_info=True)
        
        del self.storage.partitions[partition_id]
        
        self.logger.info(f"Split partition {partition_id} into {new_partition_id1} and {new_partition_id2}")
    
    def _check_node_balance(self):
        """Check if nodes are balanced."""
        if len(self.storage.nodes) <= 1:
            return
        
        # Calculate utilization for each node
        utilizations = {}
        for node_id, node in self.storage.nodes.items():
            if node.is_alive():
                node.update_usage()
                utilizations[node_id] = node.utilization()
        
        if not utilizations:
            return
        
        # Find min and max utilization
        min_util = min(utilizations.values())
        max_util = max(utilizations.values())
        
        # Check if rebalance is needed
        if max_util - min_util > self.storage.config.rebalance_threshold:
            self.logger.info(f"Node utilization imbalance detected: min={min_util:.2f}, max={max_util:.2f}")
            
            # Find most and least utilized nodes
            most_utilized = max(utilizations.items(), key=lambda x: x[1])[0]
            least_utilized = min(utilizations.items(), key=lambda x: x[1])[0]
            
            # Rebalance
            self._rebalance_nodes(most_utilized, least_utilized)
    
    def _rebalance_nodes(self, source_node_id: str, target_node_id: str):
        """Rebalance data from one node to another.
        
        Args:
            source_node_id: ID of the source node (most utilized).
            target_node_id: ID of the target node (least utilized).
        """
        source_node = self.storage.get_node(source_node_id)
        target_node = self.storage.get_node(target_node_id)
        
        if not source_node or not target_node:
            self.logger.error(f"Source or target node not found: {source_node_id}, {target_node_id}")
            return
        
        # Find partitions on source node but not on target node
        partitions_to_move = []
        for partition_id in source_node.partitions:
            partition = self.storage.partitions.get(partition_id)
            if partition and target_node_id not in partition.nodes:
                partitions_to_move.append(partition_id)
        
        if not partitions_to_move:
            self.logger.info(f"No partitions to move from {source_node_id} to {target_node_id}")
            return
        
        # Sort partitions by size (largest first)
        partitions_to_move.sort(
            key=lambda pid: self.storage.partitions[pid].size_bytes,
            reverse=True
        )
        
        # Move one partition
        partition_id = partitions_to_move[0]
        partition = self.storage.partitions[partition_id]
        
        self.logger.info(f"Rebalancing partition {partition_id} from {source_node_id} to {target_node_id}")
        
        # Copy data
        try:
            source_path = partition.get_path(source_node_id)
            target_path = partition.ensure_path(target_node_id)
            
            # Copy all files from source to target
            for root, _, files in os.walk(source_path):
                rel_path = os.path.relpath(root, source_path)
                target_dir = os.path.join(target_path, rel_path)
                os.makedirs(target_dir, exist_ok=True)
                
                for file in files:
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(target_dir, file)
                    
                    # Copy file
                    with open(source_file, 'rb') as src, open(target_file, 'wb') as dst:
                        dst.write(src.read())
            
            # Update partition
            partition.add_node(target_node_id)
            
            self.logger.info(f"Rebalancing of partition {partition_id} to {target_node_id} completed")
            
        except Exception as e:
            self.logger.error(f"Error rebalancing partition {partition_id} to {target_node_id}: {e}", exc_info=True)

class DistributedStorage:
    """Core distributed storage implementation."""
    
    def __init__(self, storage_id: str, config: Optional[DistributedStorageConfig] = None):
        """Initialize the distributed storage system.
        
        Args:
            storage_id: Unique identifier for this storage system.
            config: Configuration for this storage system.
        """
        self.storage_id = storage_id
        self.config = config or DistributedStorageConfig()
        self.logger = logging.getLogger(f"{__name__}.storage.{storage_id}")
        self.nodes: Dict[str, StorageNode] = {}
        self.partitions: Dict[str, DataPartition] = {}
        self.running = False
        self.maintenance_thread = None
        
        # Create managers
        self.replication_manager = ReplicationManager(self)
        self.partition_manager = PartitionManager(self)
        
        # Initialize storage
        self._init_storage()
    
    def _init_storage(self):
        """Initialize the storage system."""
        try:
            # Create storage root directory
            os.makedirs(self.config.storage_root, exist_ok=True)
            
            # Create metadata directory
            metadata_dir = os.path.join(self.config.storage_root, "metadata")
            os.makedirs(metadata_dir, exist_ok=True)
            
            # Load metadata if exists
            nodes_file = os.path.join(metadata_dir, "nodes.json")
            partitions_file = os.path.join(metadata_dir, "partitions.json")
            
            if os.path.exists(nodes_file):
                with open(nodes_file, 'r') as f:
                    nodes_data = json.load(f)
                    for node_data in nodes_data:
                        node = StorageNode.from_dict(node_data)
                        self.nodes[node.node_id] = node
            
            if os.path.exists(partitions_file):
                with open(partitions_file, 'r') as f:
                    partitions_data = json.load(f)
                    for partition_data in partitions_data:
                        partition = DataPartition.from_dict(partition_data, self)
                        self.partitions[partition.partition_id] = partition
            
            # Create default partitions if none exist
            if not self.partitions:
                for i in range(self.config.num_partitions):
                    partition_id = f"partition_{i:04d}"
                    partition = DataPartition(partition_id, self)
                    self.partitions[partition_id] = partition
            
            self.logger.info(f"Initialized storage with {len(self.nodes)} nodes and {len(self.partitions)} partitions")
            
        except Exception as e:
            self.logger.error(f"Error initializing storage: {e}", exc_info=True)
            raise
    
    def start(self):
        """Start the distributed storage system."""
        if self.running:
            self.logger.warning("Storage already running")
            return
        
        self.running = True
        self.logger.info(f"Starting distributed storage {self.storage_id}")
        
        # Start managers
        self.replication_manager.start()
        self.partition_manager.start()
        
        # Start maintenance thread
        self.maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            name=f"storage-maintenance-{self.storage_id}",
            daemon=True
        )
        self.maintenance_thread.start()
    
    def stop(self):
        """Stop the distributed storage system."""
        if not self.running:
            self.logger.warning("Storage not running")
            return
        
        self.logger.info(f"Stopping distributed storage {self.storage_id}")
        self.running = False
        
        # Stop managers
        self.replication_manager.stop()
        self.partition_manager.stop()
        
        # Wait for thread to terminate
        if self.maintenance_thread and self.maintenance_thread.is_alive():
            self.maintenance_thread.join(timeout=5.0)
        
        # Save metadata
        self._save_metadata()
        
        self.logger.info(f"Distributed storage {self.storage_id} stopped")
    
    def _maintenance_loop(self):
        """Maintenance loop for the storage system."""
        self.logger.info(f"Maintenance loop started for storage {self.storage_id}")
        
        last_save_time = time.time()
        last_node_check_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Save metadata periodically
                if current_time - last_save_time >= 300:  # 5 minutes
                    self._save_metadata()
                    last_save_time = current_time
                
                # Check node status periodically
                if current_time - last_node_check_time >= self.config.node_check_interval_seconds:
                    self._check_nodes()
                    last_node_check_time = current_time
                
                # Sleep for a while
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in maintenance loop: {e}", exc_info=True)
                time.sleep(60)  # Sleep longer on error
    
    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            metadata_dir = os.path.join(self.config.storage_root, "metadata")
            os.makedirs(metadata_dir, exist_ok=True)
            
            # Save nodes
            nodes_file = os.path.join(metadata_dir, "nodes.json")
            with open(nodes_file, 'w') as f:
                nodes_data = [node.to_dict() for node in self.nodes.values()]
                json.dump(nodes_data, f, indent=2)
            
            # Save partitions
            partitions_file = os.path.join(metadata_dir, "partitions.json")
            with open(partitions_file, 'w') as f:
                partitions_data = [partition.to_dict() for partition in self.partitions.values()]
                json.dump(partitions_data, f, indent=2)
            
            self.logger.debug("Metadata saved")
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}", exc_info=True)
    
    def _check_nodes(self):
        """Check status of all nodes."""
        for node_id, node in list(self.nodes.items()):
            if not node.is_alive():
                if node.status == "ONLINE":
                    self.logger.warning(f"Node {node_id} is not responding, marking as OFFLINE")
                    node.status = "OFFLINE"
            else:
                if node.status == "OFFLINE":
                    self.logger.info(f"Node {node_id} is back online")
                    node.status = "ONLINE"
    
    def add_node(self, host: str, port: int, storage_path: str) -> str:
        """Add a new node to the storage system.
        
        Args:
            host: Hostname or IP address of the node.
            port: Port number the node is listening on.
            storage_path: Path to the storage directory on this node.
            
        Returns:
            ID of the newly added node.
        """
        try:
            # Generate node ID
            node_id = str(uuid.uuid4())
            
            # Create node
            node = StorageNode(node_id, host, port, storage_path)
            
            # Add to storage
            self.nodes[node_id] = node
            
            self.logger.info(f"Added node {node_id} ({host}:{port})")
            
            # Save metadata
            self._save_metadata()
            
            return node_id
            
        except Exception as e:
            self.logger.error(f"Error adding node: {e}", exc_info=True)
            raise
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the storage system.
        
        Args:
            node_id: ID of the node to remove.
            
        Returns:
            True if the node was successfully removed, False otherwise.
        """
        try:
            # Check if node exists
            if node_id not in self.nodes:
                self.logger.warning(f"Node {node_id} does not exist")
                return False
            
            # Remove from partitions
            for partition in self.partitions.values():
                if node_id in partition.nodes:
                    partition.remove_node(node_id)
            
            # Remove from storage
            del self.nodes[node_id]
            
            self.logger.info(f"Removed node {node_id}")
            
            # Save metadata
            self._save_metadata()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing node: {e}", exc_info=True)
            return False
    
    def get_node(self, node_id: str) -> Optional[StorageNode]:
        """Get a node by ID.
        
        Args:
            node_id: ID of the node to get.
            
        Returns:
            The node, or None if not found.
        """
        return self.nodes.get(node_id)
    
    def list_nodes(self) -> List[Dict[str, Any]]:
        """List all nodes.
        
        Returns:
            List of node information.
        """
        return [node.to_dict() for node in self.nodes.values()]
    
    def get_partition_for_key(self, key: str) -> DataPartition:
        """Get the partition for a specific key.
        
        Args:
            key: The key to get the partition for.
            
        Returns:
            The partition for the key.
        """
        # Hash the key
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        # Get partition
        partition_index = hash_value % len(self.partitions)
        partition_id = list(self.partitions.keys())[partition_index]
        
        return self.partitions[partition_id]
    
    def put_object(self, key: str, data: bytes) -> bool:
        """Put an object into the storage system.
        
        Args:
            key: Key for the object.
            data: Object data.
            
        Returns:
            True if the object was successfully stored, False otherwise.
        """
        try:
            # Get partition for key
            partition = self.get_partition_for_key(key)
            
            # Compress data if enabled
            if self.config.enable_compression:
                import zlib
                data = zlib.compress(data, self.config.compression_level)
            
            # Encrypt data if enabled
            if self.config.enable_encryption and self.config.encryption_key:
                from cryptography.fernet import Fernet
                f = Fernet(self.config.encryption_key.encode())
                data = f.encrypt(data)
            
            # Determine nodes to write to based on consistency level
            nodes_to_write = partition.nodes
            
            if self.config.write_consistency == ConsistencyLevel.ONE:
                # Write to one node
                for node_id in nodes_to_write:
                    node = self.get_node(node_id)
                    if node and node.is_alive():
                        self._write_to_node(node_id, partition, key, data)
                        return True
                
                return False
                
            elif self.config.write_consistency == ConsistencyLevel.QUORUM:
                # Write to a quorum of nodes
                quorum_size = len(nodes_to_write) // 2 + 1
                success_count = 0
                
                for node_id in nodes_to_write:
                    node = self.get_node(node_id)
                    if node and node.is_alive():
                        if self._write_to_node(node_id, partition, key, data):
                            success_count += 1
                            if success_count >= quorum_size:
                                return True
                
                return False
                
            elif self.config.write_consistency == ConsistencyLevel.ALL:
                # Write to all nodes
                success = True
                
                for node_id in nodes_to_write:
                    node = self.get_node(node_id)
                    if node and node.is_alive():
                        if not self._write_to_node(node_id, partition, key, data):
                            success = False
                
                return success
            
            else:
                self.logger.error(f"Unsupported write consistency level: {self.config.write_consistency}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error putting object {key}: {e}", exc_info=True)
            return False
    
    def _write_to_node(self, node_id: str, partition: DataPartition, key: str, data: bytes) -> bool:
        """Write data to a specific node.
        
        Args:
            node_id: ID of the node to write to.
            partition: Partition the data belongs to.
            key: Key for the object.
            data: Object data.
            
        Returns:
            True if the write was successful, False otherwise.
        """
        try:
            # Get path
            path = partition.ensure_path(node_id)
            
            # Hash the key to create a file path
            hash_value = hashlib.md5(key.encode()).hexdigest()
            
            # Create subdirectories based on hash
            subdir = os.path.join(path, hash_value[:2], hash_value[2:4])
            os.makedirs(subdir, exist_ok=True)
            
            # Write file
            file_path = os.path.join(subdir, hash_value)
            with open(file_path, 'wb') as f:
                f.write(data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing to node {node_id}: {e}", exc_info=True)
            return False
    
    def get_object(self, key: str) -> Optional[bytes]:
        """Get an object from the storage system.
        
        Args:
            key: Key for the object.
            
        Returns:
            Object data, or None if not found.
        """
        try:
            # Get partition for key
            partition = self.get_partition_for_key(key)
            
            # Determine nodes to read from based on consistency level
            nodes_to_read = partition.nodes
            
            if self.config.read_consistency == ConsistencyLevel.ONE:
                # Read from one node
                for node_id in nodes_to_read:
                    node = self.get_node(node_id)
                    if node and node.is_alive():
                        data = self._read_from_node(node_id, partition, key)
                        if data is not None:
                            return self._process_read_data(data)
                
                return None
                
            elif self.config.read_consistency == ConsistencyLevel.QUORUM:
                # Read from a quorum of nodes
                quorum_size = len(nodes_to_read) // 2 + 1
                results = []
                
                for node_id in nodes_to_read:
                    node = self.get_node(node_id)
                    if node and node.is_alive():
                        data = self._read_from_node(node_id, partition, key)
                        if data is not None:
                            results.append(data)
                            if len(results) >= quorum_size:
                                # Return the most common result
                                return self._process_read_data(max(results, key=results.count))
                
                return None
                
            elif self.config.read_consistency == ConsistencyLevel.ALL:
                # Read from all nodes
                results = []
                
                for node_id in nodes_to_read:
                    node = self.get_node(node_id)
                    if node and node.is_alive():
                        data = self._read_from_node(node_id, partition, key)
                        if data is not None:
                            results.append(data)
                
                if len(results) == len(nodes_to_read):
                    # Return the most common result
                    return self._process_read_data(max(results, key=results.count))
                else:
                    return None
            
            else:
                self.logger.error(f"Unsupported read consistency level: {self.config.read_consistency}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error getting object {key}: {e}", exc_info=True)
            return None
    
    def _read_from_node(self, node_id: str, partition: DataPartition, key: str) -> Optional[bytes]:
        """Read data from a specific node.
        
        Args:
            node_id: ID of the node to read from.
            partition: Partition the data belongs to.
            key: Key for the object.
            
        Returns:
            Object data, or None if not found.
        """
        try:
            # Get path
            path = partition.get_path(node_id)
            
            # Hash the key to create a file path
            hash_value = hashlib.md5(key.encode()).hexdigest()
            
            # Create file path
            file_path = os.path.join(path, hash_value[:2], hash_value[2:4], hash_value)
            
            # Check if file exists
            if not os.path.exists(file_path):
                return None
            
            # Read file
            with open(file_path, 'rb') as f:
                return f.read()
            
        except Exception as e:
            self.logger.error(f"Error reading from node {node_id}: {e}", exc_info=True)
            return None
    
    def _process_read_data(self, data: bytes) -> bytes:
        """Process data read from storage.
        
        Args:
            data: Raw data read from storage.
            
        Returns:
            Processed data.
        """
        try:
            # Decrypt data if enabled
            if self.config.enable_encryption and self.config.encryption_key:
                from cryptography.fernet import Fernet
                f = Fernet(self.config.encryption_key.encode())
                data = f.decrypt(data)
            
            # Decompress data if enabled
            if self.config.enable_compression:
                import zlib
                data = zlib.decompress(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error processing read data: {e}", exc_info=True)
            return data
    
    def delete_object(self, key: str) -> bool:
        """Delete an object from the storage system.
        
        Args:
            key: Key for the object.
            
        Returns:
            True if the object was successfully deleted, False otherwise.
        """
        try:
            # Get partition for key
            partition = self.get_partition_for_key(key)
            
            # Determine nodes to delete from based on consistency level
            nodes_to_delete = partition.nodes
            
            if self.config.write_consistency == ConsistencyLevel.ONE:
                # Delete from one node
                for node_id in nodes_to_delete:
                    node = self.get_node(node_id)
                    if node and node.is_alive():
                        self._delete_from_node(node_id, partition, key)
                        return True
                
                return False
                
            elif self.config.write_consistency == ConsistencyLevel.QUORUM:
                # Delete from a quorum of nodes
                quorum_size = len(nodes_to_delete) // 2 + 1
                success_count = 0
                
                for node_id in nodes_to_delete:
                    node = self.get_node(node_id)
                    if node and node.is_alive():
                        if self._delete_from_node(node_id, partition, key):
                            success_count += 1
                            if success_count >= quorum_size:
                                return True
                
                return False
                
            elif self.config.write_consistency == ConsistencyLevel.ALL:
                # Delete from all nodes
                success = True
                
                for node_id in nodes_to_delete:
                    node = self.get_node(node_id)
                    if node and node.is_alive():
                        if not self._delete_from_node(node_id, partition, key):
                            success = False
                
                return success
            
            else:
                self.logger.error(f"Unsupported write consistency level: {self.config.write_consistency}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error deleting object {key}: {e}", exc_info=True)
            return False
    
    def _delete_from_node(self, node_id: str, partition: DataPartition, key: str) -> bool:
        """Delete data from a specific node.
        
        Args:
            node_id: ID of the node to delete from.
            partition: Partition the data belongs to.
            key: Key for the object.
            
        Returns:
            True if the delete was successful, False otherwise.
        """
        try:
            # Get path
            path = partition.get_path(node_id)
            
            # Hash the key to create a file path
            hash_value = hashlib.md5(key.encode()).hexdigest()
            
            # Create file path
            file_path = os.path.join(path, hash_value[:2], hash_value[2:4], hash_value)
            
            # Check if file exists
            if not os.path.exists(file_path):
                return True
            
            # Delete file
            os.remove(file_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting from node {node_id}: {e}", exc_info=True)
            return False
    
    def list_objects(self, prefix: str = "") -> List[str]:
        """List objects in the storage system.
        
        Args:
            prefix: Prefix to filter objects by.
            
        Returns:
            List of object keys.
        """
        # This is a simplified implementation that doesn't actually use the prefix
        # In a real implementation, we would need to store keys separately
        return []

# Example usage
if __name__ == "__main__":
    # Create a distributed storage system
    storage = DistributedStorage("example-storage")
    
    # Add some nodes
    node1_id = storage.add_node("localhost", 8001, "/tmp/distributed_storage/node1")
    node2_id = storage.add_node("localhost", 8002, "/tmp/distributed_storage/node2")
    node3_id = storage.add_node("localhost", 8003, "/tmp/distributed_storage/node3")
    
    # Start the storage system
    storage.start()
    
    # Put some objects
    storage.put_object("key1", b"value1")
    storage.put_object("key2", b"value2")
    storage.put_object("key3", b"value3")
    
    # Get objects
    value1 = storage.get_object("key1")
    value2 = storage.get_object("key2")
    value3 = storage.get_object("key3")
    
    print(f"key1: {value1}")
    print(f"key2: {value2}")
    print(f"key3: {value3}")
    
    # Delete an object
    storage.delete_object("key2")
    
    # Try to get the deleted object
    value2 = storage.get_object("key2")
    print(f"key2 after delete: {value2}")
    
    # Stop the storage system
    storage.stop()

