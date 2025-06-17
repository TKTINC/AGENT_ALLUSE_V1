#!/usr/bin/env python3
"""
ALL-USE Account Management System - Caching Framework

This module implements a sophisticated caching framework for the ALL-USE
Account Management System, providing multi-level caching capabilities
to enhance performance of frequently accessed data and operations.

The framework supports various caching strategies, automatic cache invalidation,
and performance monitoring to ensure optimal cache utilization.

Author: Manus AI
Date: June 17, 2025
"""

import os
import sys
import time
import logging
import json
import hashlib
import threading
import functools
from collections import OrderedDict
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import account management modules
from performance.performance_analyzer import PerformanceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("caching_framework")

class CacheEntry:
    """Class representing a cache entry with metadata."""
    
    def __init__(self, key, value, expiry=None, tags=None):
        """Initialize a cache entry.
        
        Args:
            key: Cache key
            value: Cached value
            expiry (datetime, optional): Expiry time
            tags (list, optional): List of tags for this entry
        """
        self.key = key
        self.value = value
        self.expiry = expiry
        self.tags = tags or []
        self.created_at = datetime.now()
        self.last_accessed = self.created_at
        self.access_count = 0
    
    def is_expired(self):
        """Check if the entry is expired.
        
        Returns:
            bool: True if expired, False otherwise
        """
        if self.expiry is None:
            return False
        
        return datetime.now() > self.expiry
    
    def access(self):
        """Record an access to this entry."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def to_dict(self):
        """Convert the entry to a dictionary.
        
        Returns:
            dict: Dictionary representation of the entry
        """
        return {
            "key": self.key,
            "value": self.value,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count
        }

class LRUCache:
    """Least Recently Used (LRU) cache implementation."""
    
    def __init__(self, capacity):
        """Initialize an LRU cache.
        
        Args:
            capacity (int): Maximum number of items in the cache
        """
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key):
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value or None if not found
        """
        if key not in self.cache:
            return None
        
        # Move to end (most recently used)
        value = self.cache.pop(key)
        self.cache[key] = value
        return value
    
    def put(self, key, value):
        """Put a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Remove existing entry
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used item
            self.cache.popitem(last=False)
        
        # Add new entry
        self.cache[key] = value
    
    def remove(self, key):
        """Remove a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if removed, False if not found
        """
        if key in self.cache:
            self.cache.pop(key)
            return True
        return False
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
    
    def __len__(self):
        """Get the number of items in the cache.
        
        Returns:
            int: Number of items
        """
        return len(self.cache)

class CacheLayer:
    """Class representing a single cache layer."""
    
    def __init__(self, name, capacity, ttl=None):
        """Initialize a cache layer.
        
        Args:
            name (str): Name of the cache layer
            capacity (int): Maximum number of items in the cache
            ttl (int, optional): Default time-to-live in seconds
        """
        self.name = name
        self.capacity = capacity
        self.default_ttl = ttl
        self.cache = LRUCache(capacity)
        self.entries = {}  # Metadata for cache entries
        self.tag_index = {}  # Index of entries by tag
        self.lock = threading.RLock()
        
        logger.info(f"Cache layer '{name}' initialized with capacity {capacity}")
    
    def get(self, key):
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value or None if not found or expired
        """
        with self.lock:
            # Get from LRU cache
            value = self.cache.get(key)
            
            if value is None:
                return None
            
            # Check if entry exists and is not expired
            if key in self.entries:
                entry = self.entries[key]
                
                if entry.is_expired():
                    # Remove expired entry
                    self._remove_entry(key)
                    return None
                
                # Update access metadata
                entry.access()
                return entry.value
            
            # Entry in LRU but not in metadata (should not happen)
            self.cache.remove(key)
            return None
    
    def put(self, key, value, ttl=None, tags=None):
        """Put a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl (int, optional): Time-to-live in seconds
            tags (list, optional): List of tags for this entry
            
        Returns:
            bool: True if successful
        """
        with self.lock:
            # Calculate expiry time
            expiry = None
            if ttl is not None:
                expiry = datetime.now() + timedelta(seconds=ttl)
            elif self.default_ttl is not None:
                expiry = datetime.now() + timedelta(seconds=self.default_ttl)
            
            # Create entry
            entry = CacheEntry(key, value, expiry, tags)
            
            # Remove old entry if exists
            if key in self.entries:
                self._remove_entry(key)
            
            # Add to LRU cache
            self.cache.put(key, value)
            
            # Add to entries
            self.entries[key] = entry
            
            # Add to tag index
            if tags:
                for tag in tags:
                    if tag not in self.tag_index:
                        self.tag_index[tag] = set()
                    self.tag_index[tag].add(key)
            
            return True
    
    def remove(self, key):
        """Remove a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if removed, False if not found
        """
        with self.lock:
            return self._remove_entry(key)
    
    def _remove_entry(self, key):
        """Internal method to remove an entry.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if removed, False if not found
        """
        if key not in self.entries:
            return False
        
        # Remove from LRU cache
        self.cache.remove(key)
        
        # Remove from tag index
        entry = self.entries[key]
        for tag in entry.tags:
            if tag in self.tag_index and key in self.tag_index[tag]:
                self.tag_index[tag].remove(key)
                if not self.tag_index[tag]:
                    del self.tag_index[tag]
        
        # Remove from entries
        del self.entries[key]
        
        return True
    
    def invalidate_by_tag(self, tag):
        """Invalidate all entries with a specific tag.
        
        Args:
            tag: Tag to invalidate
            
        Returns:
            int: Number of entries invalidated
        """
        with self.lock:
            if tag not in self.tag_index:
                return 0
            
            # Get keys to invalidate
            keys = list(self.tag_index[tag])
            
            # Remove entries
            count = 0
            for key in keys:
                if self._remove_entry(key):
                    count += 1
            
            return count
    
    def clear(self):
        """Clear the cache.
        
        Returns:
            int: Number of entries cleared
        """
        with self.lock:
            count = len(self.entries)
            
            # Clear all collections
            self.cache.clear()
            self.entries.clear()
            self.tag_index.clear()
            
            return count
    
    def cleanup_expired(self):
        """Remove all expired entries.
        
        Returns:
            int: Number of entries removed
        """
        with self.lock:
            now = datetime.now()
            expired_keys = []
            
            # Find expired entries
            for key, entry in self.entries.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            # Remove expired entries
            count = 0
            for key in expired_keys:
                if self._remove_entry(key):
                    count += 1
            
            return count
    
    def get_statistics(self):
        """Get statistics for this cache layer.
        
        Returns:
            dict: Cache statistics
        """
        with self.lock:
            total_entries = len(self.entries)
            expired_entries = sum(1 for entry in self.entries.values() if entry.is_expired())
            
            # Calculate hit rate (if any accesses)
            total_accesses = sum(entry.access_count for entry in self.entries.values())
            hit_rate = None
            if total_accesses > 0:
                hit_rate = total_accesses / (total_accesses + 1)  # Add 1 to avoid division by zero
            
            # Get most accessed entries
            most_accessed = sorted(
                self.entries.values(),
                key=lambda e: e.access_count,
                reverse=True
            )[:10]
            
            return {
                "name": self.name,
                "capacity": self.capacity,
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "utilization": total_entries / self.capacity if self.capacity > 0 else 0,
                "hit_rate": hit_rate,
                "tag_count": len(self.tag_index),
                "most_accessed": [
                    {
                        "key": entry.key,
                        "access_count": entry.access_count,
                        "last_accessed": entry.last_accessed.isoformat()
                    }
                    for entry in most_accessed
                ]
            }

class CachingFramework:
    """Multi-level caching framework for the ALL-USE Account Management System."""
    
    def __init__(self, analyzer=None):
        """Initialize the caching framework.
        
        Args:
            analyzer (PerformanceAnalyzer, optional): Performance analyzer
        """
        self.layers = {}
        self.analyzer = analyzer
        self.lock = threading.RLock()
        
        # Initialize default layers
        self._initialize_default_layers()
        
        logger.info("Caching framework initialized")
    
    def _initialize_default_layers(self):
        """Initialize default cache layers."""
        # L1: Small, fast cache for frequently accessed items
        self.add_layer("L1", 100, ttl=60)  # 1 minute TTL
        
        # L2: Medium cache for moderately accessed items
        self.add_layer("L2", 1000, ttl=300)  # 5 minutes TTL
        
        # L3: Large cache for less frequently accessed items
        self.add_layer("L3", 10000, ttl=1800)  # 30 minutes TTL
    
    def add_layer(self, name, capacity, ttl=None):
        """Add a new cache layer.
        
        Args:
            name (str): Name of the cache layer
            capacity (int): Maximum number of items in the cache
            ttl (int, optional): Default time-to-live in seconds
            
        Returns:
            bool: True if successful, False if layer already exists
        """
        with self.lock:
            if name in self.layers:
                logger.warning(f"Cache layer '{name}' already exists")
                return False
            
            self.layers[name] = CacheLayer(name, capacity, ttl)
            logger.info(f"Added cache layer '{name}' with capacity {capacity}")
            return True
    
    def remove_layer(self, name):
        """Remove a cache layer.
        
        Args:
            name (str): Name of the cache layer
            
        Returns:
            bool: True if successful, False if layer not found
        """
        with self.lock:
            if name not in self.layers:
                logger.warning(f"Cache layer '{name}' not found")
                return False
            
            del self.layers[name]
            logger.info(f"Removed cache layer '{name}'")
            return True
    
    def get(self, key, layer=None):
        """Get a value from the cache.
        
        Args:
            key: Cache key
            layer (str, optional): Specific layer to get from
            
        Returns:
            The cached value or None if not found
        """
        # Record operation start time if analyzer is available
        start_time = time.time() if self.analyzer else None
        
        try:
            # Generate cache key if not a string
            cache_key = self._generate_cache_key(key)
            
            # Try specific layer if specified
            if layer is not None:
                if layer not in self.layers:
                    logger.warning(f"Cache layer '{layer}' not found")
                    return None
                
                return self.layers[layer].get(cache_key)
            
            # Try all layers in order
            for layer_name in ["L1", "L2", "L3"]:
                if layer_name in self.layers:
                    value = self.layers[layer_name].get(cache_key)
                    if value is not None:
                        # Promote to higher layer if found in lower layer
                        if layer_name == "L2" and "L1" in self.layers:
                            self.layers["L1"].put(cache_key, value)
                        elif layer_name == "L3" and "L2" in self.layers:
                            self.layers["L2"].put(cache_key, value)
                        
                        return value
            
            return None
        finally:
            # Record operation end time if analyzer is available
            if self.analyzer and start_time:
                duration_ms = (time.time() - start_time) * 1000
                self.analyzer.record_metric("cache_get", duration_ms)
    
    def put(self, key, value, ttl=None, tags=None, layer=None):
        """Put a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl (int, optional): Time-to-live in seconds
            tags (list, optional): List of tags for this entry
            layer (str, optional): Specific layer to put in
            
        Returns:
            bool: True if successful
        """
        # Record operation start time if analyzer is available
        start_time = time.time() if self.analyzer else None
        
        try:
            # Generate cache key if not a string
            cache_key = self._generate_cache_key(key)
            
            # Put in specific layer if specified
            if layer is not None:
                if layer not in self.layers:
                    logger.warning(f"Cache layer '{layer}' not found")
                    return False
                
                return self.layers[layer].put(cache_key, value, ttl, tags)
            
            # Put in all layers
            success = False
            for layer_name in ["L1", "L2", "L3"]:
                if layer_name in self.layers:
                    if self.layers[layer_name].put(cache_key, value, ttl, tags):
                        success = True
            
            return success
        finally:
            # Record operation end time if analyzer is available
            if self.analyzer and start_time:
                duration_ms = (time.time() - start_time) * 1000
                self.analyzer.record_metric("cache_put", duration_ms)
    
    def remove(self, key, layer=None):
        """Remove a value from the cache.
        
        Args:
            key: Cache key
            layer (str, optional): Specific layer to remove from
            
        Returns:
            bool: True if removed from any layer
        """
        # Generate cache key if not a string
        cache_key = self._generate_cache_key(key)
        
        # Remove from specific layer if specified
        if layer is not None:
            if layer not in self.layers:
                logger.warning(f"Cache layer '{layer}' not found")
                return False
            
            return self.layers[layer].remove(cache_key)
        
        # Remove from all layers
        success = False
        for layer_obj in self.layers.values():
            if layer_obj.remove(cache_key):
                success = True
        
        return success
    
    def invalidate_by_tag(self, tag, layer=None):
        """Invalidate all entries with a specific tag.
        
        Args:
            tag: Tag to invalidate
            layer (str, optional): Specific layer to invalidate in
            
        Returns:
            int: Number of entries invalidated
        """
        # Invalidate in specific layer if specified
        if layer is not None:
            if layer not in self.layers:
                logger.warning(f"Cache layer '{layer}' not found")
                return 0
            
            return self.layers[layer].invalidate_by_tag(tag)
        
        # Invalidate in all layers
        count = 0
        for layer_obj in self.layers.values():
            count += layer_obj.invalidate_by_tag(tag)
        
        return count
    
    def clear(self, layer=None):
        """Clear the cache.
        
        Args:
            layer (str, optional): Specific layer to clear
            
        Returns:
            int: Number of entries cleared
        """
        # Clear specific layer if specified
        if layer is not None:
            if layer not in self.layers:
                logger.warning(f"Cache layer '{layer}' not found")
                return 0
            
            return self.layers[layer].clear()
        
        # Clear all layers
        count = 0
        for layer_obj in self.layers.values():
            count += layer_obj.clear()
        
        return count
    
    def cleanup_expired(self, layer=None):
        """Remove all expired entries.
        
        Args:
            layer (str, optional): Specific layer to clean up
            
        Returns:
            int: Number of entries removed
        """
        # Clean up specific layer if specified
        if layer is not None:
            if layer not in self.layers:
                logger.warning(f"Cache layer '{layer}' not found")
                return 0
            
            return self.layers[layer].cleanup_expired()
        
        # Clean up all layers
        count = 0
        for layer_obj in self.layers.values():
            count += layer_obj.cleanup_expired()
        
        return count
    
    def get_statistics(self, layer=None):
        """Get statistics for the cache.
        
        Args:
            layer (str, optional): Specific layer to get statistics for
            
        Returns:
            dict: Cache statistics
        """
        # Get statistics for specific layer if specified
        if layer is not None:
            if layer not in self.layers:
                logger.warning(f"Cache layer '{layer}' not found")
                return None
            
            return self.layers[layer].get_statistics()
        
        # Get statistics for all layers
        stats = {
            "layers": {},
            "total_entries": 0,
            "expired_entries": 0
        }
        
        for layer_name, layer_obj in self.layers.items():
            layer_stats = layer_obj.get_statistics()
            stats["layers"][layer_name] = layer_stats
            stats["total_entries"] += layer_stats["total_entries"]
            stats["expired_entries"] += layer_stats["expired_entries"]
        
        return stats
    
    def _generate_cache_key(self, key):
        """Generate a cache key from any object.
        
        Args:
            key: Object to use as cache key
            
        Returns:
            str: String representation of the key
        """
        if isinstance(key, str):
            return key
        
        if isinstance(key, (int, float, bool)):
            return str(key)
        
        # For complex objects, use JSON representation and hash
        try:
            key_str = json.dumps(key, sort_keys=True)
        except (TypeError, ValueError):
            # If not JSON serializable, use string representation
            key_str = str(key)
        
        # Hash the key string to ensure it's a valid cache key
        return hashlib.md5(key_str.encode()).hexdigest()

def cached(ttl=None, tags=None, layer=None):
    """Decorator for caching function results.
    
    Args:
        ttl (int, optional): Time-to-live in seconds
        tags (list, optional): List of tags for this entry
        layer (str, optional): Specific layer to use
        
    Returns:
        function: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache framework instance
            # This assumes a global instance is available
            cache = get_cache_instance()
            
            # Generate cache key
            cache_key = f"{func.__module__}.{func.__name__}:{args}:{kwargs}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key, layer)
            if cached_result is not None:
                return cached_result
            
            # Call function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.put(cache_key, result, ttl, tags, layer)
            
            return result
        return wrapper
    return decorator

# Global cache instance
_cache_instance = None

def get_cache_instance():
    """Get the global cache instance.
    
    Returns:
        CachingFramework: Global cache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CachingFramework()
    return _cache_instance

def set_cache_instance(instance):
    """Set the global cache instance.
    
    Args:
        instance (CachingFramework): Cache instance
    """
    global _cache_instance
    _cache_instance = instance

def main():
    """Main function for standalone execution."""
    print("ALL-USE Account Management System - Caching Framework")
    print("===================================================")
    print("\nThis module provides caching capabilities and should be imported by other modules.")
    print("For standalone testing, this script will perform a basic self-test.")
    
    # Create caching framework
    cache = CachingFramework()
    
    # Run self-test
    print("\nRunning caching framework self-test...")
    
    # Test basic operations
    print("\nTesting basic operations:")
    
    # Put some values
    cache.put("test_key1", "test_value1", tags=["test"])
    cache.put("test_key2", "test_value2", tags=["test"])
    cache.put("test_key3", "test_value3", tags=["other"])
    
    # Get values
    print(f"Get test_key1: {cache.get('test_key1')}")
    print(f"Get test_key2: {cache.get('test_key2')}")
    print(f"Get test_key3: {cache.get('test_key3')}")
    print(f"Get non_existent_key: {cache.get('non_existent_key')}")
    
    # Test tag invalidation
    print("\nTesting tag invalidation:")
    print(f"Invalidating tag 'test': {cache.invalidate_by_tag('test')} entries invalidated")
    print(f"Get test_key1 after invalidation: {cache.get('test_key1')}")
    print(f"Get test_key3 after invalidation: {cache.get('test_key3')}")
    
    # Test statistics
    print("\nTesting statistics:")
    stats = cache.get_statistics()
    print(f"Total entries: {stats['total_entries']}")
    print(f"Expired entries: {stats['expired_entries']}")
    
    # Test decorator
    print("\nTesting decorator:")
    
    @cached(ttl=60, tags=["decorated"])
    def slow_function(x):
        print("  Executing slow_function...")
        time.sleep(1)
        return x * 2
    
    print("First call (should execute):")
    result1 = slow_function(5)
    print(f"Result: {result1}")
    
    print("Second call (should use cache):")
    result2 = slow_function(5)
    print(f"Result: {result2}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()

