"""
ALL-USE Performance Optimizer

This module provides comprehensive performance optimization for all WS1 components,
including algorithmic improvements, memory optimization, and caching systems.
"""

import asyncio
import time
import functools
import threading
import weakref
from typing import Dict, Any, Optional, Callable, List, Tuple
from datetime import datetime, timedelta
import logging
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger('all_use_performance_optimizer')


class PerformanceOptimizer:
    """
    Core performance optimization engine for ALL-USE components.
    
    Provides:
    - Function-level performance optimization
    - Memory usage optimization
    - Caching and memoization
    - Async processing optimization
    - Resource pooling and management
    """
    
    def __init__(self):
        """Initialize the performance optimizer."""
        self.cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        self.cache_max_size = 1000
        self.cache_ttl = 300  # 5 minutes
        
        # Performance monitoring
        self.performance_stats = {}
        self.optimization_enabled = True
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Performance optimizer initialized")
    
    def optimize_function(self, cache_ttl: int = 300, async_enabled: bool = False):
        """
        Decorator for function-level performance optimization.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
            async_enabled: Enable asynchronous execution
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.optimization_enabled:
                    return func(*args, **kwargs)
                
                # Generate cache key
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Check cache first
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    self.cache_stats['hits'] += 1
                    return cached_result
                
                # Execute function with performance monitoring
                start_time = time.perf_counter()
                
                try:
                    if async_enabled and asyncio.iscoroutinefunction(func):
                        result = asyncio.run(func(*args, **kwargs))
                    else:
                        result = func(*args, **kwargs)
                    
                    execution_time = time.perf_counter() - start_time
                    
                    # Update performance stats
                    self._update_performance_stats(func.__name__, execution_time)
                    
                    # Cache result
                    self._store_in_cache(cache_key, result, cache_ttl)
                    self.cache_stats['misses'] += 1
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.perf_counter() - start_time
                    self._update_performance_stats(func.__name__, execution_time, error=True)
                    raise
            
            return wrapper
        return decorator
    
    def optimize_memory_usage(self):
        """Optimize memory usage across the system."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear expired cache entries
        expired_keys = self._clear_expired_cache()
        
        # Optimize cache size
        if len(self.cache) > self.cache_max_size:
            self._evict_cache_entries()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_saved = initial_memory - final_memory
        
        optimization_result = {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_saved_mb': memory_saved,
            'gc_collected': collected,
            'cache_expired': expired_keys,
            'cache_size': len(self.cache)
        }
        
        logger.info(f"Memory optimization completed: {memory_saved:.2f}MB saved")
        return optimization_result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        total_calls = sum(stats['calls'] for stats in self.performance_stats.values())
        total_time = sum(stats['total_time'] for stats in self.performance_stats.values())
        
        cache_hit_rate = (
            self.cache_stats['hits'] / 
            (self.cache_stats['hits'] + self.cache_stats['misses'])
            if (self.cache_stats['hits'] + self.cache_stats['misses']) > 0 else 0
        )
        
        report = {
            'summary': {
                'total_function_calls': total_calls,
                'total_execution_time_ms': total_time * 1000,
                'average_execution_time_ms': (total_time / total_calls * 1000) if total_calls > 0 else 0,
                'cache_hit_rate': cache_hit_rate,
                'cache_size': len(self.cache),
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
            },
            'function_stats': {},
            'cache_stats': self.cache_stats.copy(),
            'optimization_recommendations': self._generate_optimization_recommendations()
        }
        
        # Add detailed function statistics
        for func_name, stats in self.performance_stats.items():
            report['function_stats'][func_name] = {
                'calls': stats['calls'],
                'total_time_ms': stats['total_time'] * 1000,
                'average_time_ms': (stats['total_time'] / stats['calls'] * 1000) if stats['calls'] > 0 else 0,
                'min_time_ms': stats['min_time'] * 1000,
                'max_time_ms': stats['max_time'] * 1000,
                'error_count': stats['errors'],
                'error_rate': (stats['errors'] / stats['calls']) if stats['calls'] > 0 else 0
            }
        
        return report
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call."""
        # Create a simple hash-based cache key
        key_parts = [func_name]
        
        # Add args (convert to string representation)
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(str(type(arg).__name__))
        
        # Add kwargs
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")
            else:
                key_parts.append(f"{k}={type(v).__name__}")
        
        return "|".join(key_parts)
    
    def _get_from_cache(self, cache_key: str) -> Any:
        """Retrieve value from cache if not expired."""
        if cache_key not in self.cache:
            return None
        
        entry = self.cache[cache_key]
        if datetime.now() > entry['expires']:
            del self.cache[cache_key]
            return None
        
        entry['last_accessed'] = datetime.now()
        return entry['value']
    
    def _store_in_cache(self, cache_key: str, value: Any, ttl: int):
        """Store value in cache with TTL."""
        expires = datetime.now() + timedelta(seconds=ttl)
        self.cache[cache_key] = {
            'value': value,
            'expires': expires,
            'created': datetime.now(),
            'last_accessed': datetime.now()
        }
    
    def _clear_expired_cache(self) -> int:
        """Clear expired cache entries."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now > entry['expires']
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
    
    def _evict_cache_entries(self):
        """Evict least recently used cache entries."""
        if len(self.cache) <= self.cache_max_size:
            return
        
        # Sort by last accessed time
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        # Remove oldest entries
        entries_to_remove = len(self.cache) - self.cache_max_size
        for i in range(entries_to_remove):
            key = sorted_entries[i][0]
            del self.cache[key]
            self.cache_stats['evictions'] += 1
    
    def _update_performance_stats(self, func_name: str, execution_time: float, error: bool = False):
        """Update performance statistics for a function."""
        if func_name not in self.performance_stats:
            self.performance_stats[func_name] = {
                'calls': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'errors': 0
            }
        
        stats = self.performance_stats[func_name]
        stats['calls'] += 1
        stats['total_time'] += execution_time
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)
        
        if error:
            stats['errors'] += 1
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []
        
        # Check cache hit rate
        cache_hit_rate = (
            self.cache_stats['hits'] / 
            (self.cache_stats['hits'] + self.cache_stats['misses'])
            if (self.cache_stats['hits'] + self.cache_stats['misses']) > 0 else 0
        )
        
        if cache_hit_rate < 0.5:
            recommendations.append("Low cache hit rate detected. Consider increasing cache TTL or optimizing cache keys.")
        
        # Check for slow functions
        for func_name, stats in self.performance_stats.items():
            avg_time = stats['total_time'] / stats['calls'] if stats['calls'] > 0 else 0
            if avg_time > 0.1:  # 100ms
                recommendations.append(f"Function '{func_name}' has high average execution time ({avg_time*1000:.2f}ms). Consider optimization.")
        
        # Check error rates
        for func_name, stats in self.performance_stats.items():
            error_rate = stats['errors'] / stats['calls'] if stats['calls'] > 0 else 0
            if error_rate > 0.05:  # 5%
                recommendations.append(f"Function '{func_name}' has high error rate ({error_rate*100:.1f}%). Review error handling.")
        
        # Check memory usage
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_mb > 500:  # 500MB
            recommendations.append(f"High memory usage detected ({memory_mb:.1f}MB). Consider memory optimization.")
        
        return recommendations


class MemoryOptimizer:
    """
    Memory usage optimization for ALL-USE components.
    
    Provides:
    - Memory leak detection
    - Garbage collection optimization
    - Object pooling
    - Memory usage monitoring
    """
    
    def __init__(self):
        """Initialize the memory optimizer."""
        self.object_pools = {}
        self.memory_snapshots = []
        self.gc_stats = {'collections': 0, 'collected': 0}
        
        logger.info("Memory optimizer initialized")
    
    def create_object_pool(self, pool_name: str, factory_func: Callable, max_size: int = 100):
        """Create an object pool for expensive-to-create objects."""
        self.object_pools[pool_name] = {
            'factory': factory_func,
            'pool': [],
            'max_size': max_size,
            'created': 0,
            'reused': 0
        }
    
    def get_from_pool(self, pool_name: str):
        """Get an object from the pool."""
        if pool_name not in self.object_pools:
            raise ValueError(f"Object pool '{pool_name}' not found")
        
        pool_info = self.object_pools[pool_name]
        
        if pool_info['pool']:
            obj = pool_info['pool'].pop()
            pool_info['reused'] += 1
            return obj
        else:
            obj = pool_info['factory']()
            pool_info['created'] += 1
            return obj
    
    def return_to_pool(self, pool_name: str, obj: Any):
        """Return an object to the pool."""
        if pool_name not in self.object_pools:
            return
        
        pool_info = self.object_pools[pool_name]
        
        if len(pool_info['pool']) < pool_info['max_size']:
            # Reset object state if it has a reset method
            if hasattr(obj, 'reset'):
                obj.reset()
            pool_info['pool'].append(obj)
    
    def optimize_garbage_collection(self) -> Dict[str, Any]:
        """Optimize garbage collection settings and force collection."""
        # Get initial memory state
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Force garbage collection for all generations
        collected_counts = []
        for generation in range(3):
            collected = gc.collect(generation)
            collected_counts.append(collected)
        
        # Update stats
        self.gc_stats['collections'] += 1
        self.gc_stats['collected'] += sum(collected_counts)
        
        # Get final memory state
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_freed = initial_memory - final_memory
        
        result = {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_freed_mb': memory_freed,
            'objects_collected': collected_counts,
            'total_collected': sum(collected_counts)
        }
        
        logger.info(f"Garbage collection completed: {memory_freed:.2f}MB freed, {sum(collected_counts)} objects collected")
        return result
    
    def take_memory_snapshot(self) -> Dict[str, Any]:
        """Take a memory usage snapshot."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        snapshot = {
            'timestamp': datetime.now(),
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads(),
            'gc_stats': dict(gc.get_stats()),
            'object_pools': {
                name: {
                    'pool_size': len(pool['pool']),
                    'created': pool['created'],
                    'reused': pool['reused'],
                    'reuse_rate': pool['reused'] / (pool['created'] + pool['reused']) if (pool['created'] + pool['reused']) > 0 else 0
                }
                for name, pool in self.object_pools.items()
            }
        }
        
        self.memory_snapshots.append(snapshot)
        
        # Keep only last 100 snapshots
        if len(self.memory_snapshots) > 100:
            self.memory_snapshots = self.memory_snapshots[-100:]
        
        return snapshot
    
    def detect_memory_leaks(self) -> Dict[str, Any]:
        """Detect potential memory leaks based on snapshots."""
        if len(self.memory_snapshots) < 10:
            return {'status': 'insufficient_data', 'message': 'Need at least 10 snapshots for leak detection'}
        
        # Analyze memory trend over last 10 snapshots
        recent_snapshots = self.memory_snapshots[-10:]
        memory_values = [snapshot['rss_mb'] for snapshot in recent_snapshots]
        
        # Calculate trend (simple linear regression)
        n = len(memory_values)
        x_values = list(range(n))
        
        sum_x = sum(x_values)
        sum_y = sum(memory_values)
        sum_xy = sum(x * y for x, y in zip(x_values, memory_values))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Determine leak status
        leak_threshold = 1.0  # MB per snapshot
        
        if slope > leak_threshold:
            status = 'leak_detected'
            severity = 'high' if slope > 5.0 else 'medium' if slope > 2.0 else 'low'
        elif slope > 0.1:
            status = 'potential_leak'
            severity = 'low'
        else:
            status = 'no_leak'
            severity = 'none'
        
        return {
            'status': status,
            'severity': severity,
            'memory_trend_mb_per_snapshot': slope,
            'current_memory_mb': memory_values[-1],
            'memory_change_mb': memory_values[-1] - memory_values[0],
            'snapshots_analyzed': n,
            'recommendations': self._generate_memory_recommendations(slope, memory_values[-1])
        }
    
    def _generate_memory_recommendations(self, trend: float, current_memory: float) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        if trend > 1.0:
            recommendations.append("Memory leak detected. Review object lifecycle and ensure proper cleanup.")
        
        if current_memory > 500:
            recommendations.append("High memory usage. Consider implementing object pooling or reducing cache sizes.")
        
        if trend > 0.1:
            recommendations.append("Gradual memory increase detected. Monitor for potential leaks.")
        
        # Check object pool efficiency
        for pool_name, pool in self.object_pools.items():
            reuse_rate = pool['reused'] / (pool['created'] + pool['reused']) if (pool['created'] + pool['reused']) > 0 else 0
            if reuse_rate < 0.5:
                recommendations.append(f"Object pool '{pool_name}' has low reuse rate ({reuse_rate:.1%}). Consider adjusting pool size or usage patterns.")
        
        return recommendations


class CacheManager:
    """
    Intelligent caching system for ALL-USE components.
    
    Provides:
    - Multi-level caching
    - Cache warming and preloading
    - Cache invalidation strategies
    - Cache performance monitoring
    """
    
    def __init__(self):
        """Initialize the cache manager."""
        self.caches = {
            'l1': {},  # Fast in-memory cache
            'l2': {},  # Larger in-memory cache
            'l3': {}   # Persistent cache (simulated)
        }
        
        self.cache_configs = {
            'l1': {'max_size': 100, 'ttl': 60},      # 1 minute
            'l2': {'max_size': 1000, 'ttl': 300},    # 5 minutes
            'l3': {'max_size': 10000, 'ttl': 3600}   # 1 hour
        }
        
        self.cache_stats = {
            level: {'hits': 0, 'misses': 0, 'evictions': 0, 'size': 0}
            for level in self.caches.keys()
        }
        
        logger.info("Cache manager initialized")
    
    def get(self, key: str, default=None) -> Any:
        """Get value from cache (checks all levels)."""
        # Check L1 cache first
        value = self._get_from_level('l1', key)
        if value is not None:
            return value
        
        # Check L2 cache
        value = self._get_from_level('l2', key)
        if value is not None:
            # Promote to L1
            self._set_to_level('l1', key, value)
            return value
        
        # Check L3 cache
        value = self._get_from_level('l3', key)
        if value is not None:
            # Promote to L2 and L1
            self._set_to_level('l2', key, value)
            self._set_to_level('l1', key, value)
            return value
        
        return default
    
    def set(self, key: str, value: Any, level: str = 'l1'):
        """Set value in cache at specified level."""
        self._set_to_level(level, key, value)
        
        # Also set in lower levels for faster access
        if level == 'l3':
            self._set_to_level('l2', key, value)
            self._set_to_level('l1', key, value)
        elif level == 'l2':
            self._set_to_level('l1', key, value)
    
    def invalidate(self, key: str, all_levels: bool = True):
        """Invalidate cache entry."""
        if all_levels:
            for level in self.caches.keys():
                if key in self.caches[level]:
                    del self.caches[level][key]
                    self.cache_stats[level]['size'] -= 1
        else:
            # Only invalidate L1
            if key in self.caches['l1']:
                del self.caches['l1'][key]
                self.cache_stats['l1']['size'] -= 1
    
    def warm_cache(self, warm_func: Callable, keys: List[str]):
        """Warm cache with precomputed values."""
        warmed_count = 0
        
        for key in keys:
            try:
                value = warm_func(key)
                self.set(key, value, level='l2')  # Warm L2 cache
                warmed_count += 1
            except Exception as e:
                logger.warning(f"Failed to warm cache for key '{key}': {e}")
        
        logger.info(f"Cache warming completed: {warmed_count}/{len(keys)} entries warmed")
        return warmed_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_hits = sum(stats['hits'] for stats in self.cache_stats.values())
        total_misses = sum(stats['misses'] for stats in self.cache_stats.values())
        total_requests = total_hits + total_misses
        
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            'overall': {
                'hit_rate': overall_hit_rate,
                'total_requests': total_requests,
                'total_hits': total_hits,
                'total_misses': total_misses
            },
            'by_level': {
                level: {
                    'hit_rate': stats['hits'] / (stats['hits'] + stats['misses']) if (stats['hits'] + stats['misses']) > 0 else 0,
                    'hits': stats['hits'],
                    'misses': stats['misses'],
                    'evictions': stats['evictions'],
                    'current_size': stats['size'],
                    'max_size': self.cache_configs[level]['max_size'],
                    'utilization': stats['size'] / self.cache_configs[level]['max_size']
                }
                for level, stats in self.cache_stats.items()
            }
        }
    
    def _get_from_level(self, level: str, key: str) -> Any:
        """Get value from specific cache level."""
        if key not in self.caches[level]:
            self.cache_stats[level]['misses'] += 1
            return None
        
        entry = self.caches[level][key]
        
        # Check if expired
        if datetime.now() > entry['expires']:
            del self.caches[level][key]
            self.cache_stats[level]['size'] -= 1
            self.cache_stats[level]['misses'] += 1
            return None
        
        self.cache_stats[level]['hits'] += 1
        entry['last_accessed'] = datetime.now()
        return entry['value']
    
    def _set_to_level(self, level: str, key: str, value: Any):
        """Set value to specific cache level."""
        config = self.cache_configs[level]
        
        # Check if cache is full
        if len(self.caches[level]) >= config['max_size'] and key not in self.caches[level]:
            self._evict_from_level(level)
        
        expires = datetime.now() + timedelta(seconds=config['ttl'])
        
        if key not in self.caches[level]:
            self.cache_stats[level]['size'] += 1
        
        self.caches[level][key] = {
            'value': value,
            'expires': expires,
            'created': datetime.now(),
            'last_accessed': datetime.now()
        }
    
    def _evict_from_level(self, level: str):
        """Evict least recently used entry from cache level."""
        if not self.caches[level]:
            return
        
        # Find least recently used entry
        lru_key = min(
            self.caches[level].keys(),
            key=lambda k: self.caches[level][k]['last_accessed']
        )
        
        del self.caches[level][lru_key]
        self.cache_stats[level]['size'] -= 1
        self.cache_stats[level]['evictions'] += 1


# Global optimizer instances
performance_optimizer = PerformanceOptimizer()
memory_optimizer = MemoryOptimizer()
cache_manager = CacheManager()


# Convenience decorators
def optimize_performance(cache_ttl: int = 300, async_enabled: bool = False):
    """Convenience decorator for performance optimization."""
    return performance_optimizer.optimize_function(cache_ttl=cache_ttl, async_enabled=async_enabled)


def cached(ttl: int = 300, level: str = 'l1'):
    """Convenience decorator for caching function results."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, level=level)
            
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test the performance optimization system
    
    @optimize_performance(cache_ttl=60)
    def test_function(x: int, y: int) -> int:
        """Test function for performance optimization."""
        time.sleep(0.01)  # Simulate work
        return x + y
    
    @cached(ttl=120, level='l2')
    def expensive_calculation(n: int) -> int:
        """Test function for caching."""
        time.sleep(0.05)  # Simulate expensive calculation
        return sum(range(n))
    
    # Test performance optimization
    print("Testing performance optimization...")
    for i in range(10):
        result = test_function(i, i + 1)
        print(f"test_function({i}, {i+1}) = {result}")
    
    # Test caching
    print("\nTesting caching...")
    for i in range(5):
        result = expensive_calculation(100)
        print(f"expensive_calculation(100) = {result}")
    
    # Test memory optimization
    print("\nTesting memory optimization...")
    memory_result = memory_optimizer.optimize_garbage_collection()
    print(f"Memory optimization result: {memory_result}")
    
    # Generate performance report
    print("\nPerformance Report:")
    report = performance_optimizer.get_performance_report()
    print(f"Total function calls: {report['summary']['total_function_calls']}")
    print(f"Cache hit rate: {report['summary']['cache_hit_rate']:.2%}")
    print(f"Average execution time: {report['summary']['average_execution_time_ms']:.2f}ms")
    
    # Cache statistics
    print("\nCache Statistics:")
    cache_stats = cache_manager.get_cache_stats()
    print(f"Overall hit rate: {cache_stats['overall']['hit_rate']:.2%}")
    
    print("\nOptimization system test completed successfully!")

