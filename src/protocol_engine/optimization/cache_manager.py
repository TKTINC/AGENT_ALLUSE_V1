#!/usr/bin/env python3
"""
ALL-USE Agent - Protocol Engine Caching System
P5 of WS2: Performance Optimization and Monitoring - Phase 3

This module provides intelligent caching systems and performance enhancements
for the Protocol Engine, implementing LRU caches, result caching, and
algorithm optimizations to maximize performance while maintaining accuracy.

Caching Components:
1. LRU Cache Manager - Least Recently Used caching with size limits
2. Result Cache - Intelligent caching for computed results
3. Week Classification Cache - Specialized caching for week classifications
4. Market Analysis Cache - Caching for market condition analysis
5. Performance Enhancer - Algorithm optimizations and lazy loading
6. Cache Coordinator - Centralized cache management and optimization
"""

import time
import hashlib
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import wraps, lru_cache
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    def touch(self):
        """Update last accessed time and increment access count"""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate calculation"""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0


class LRUCache:
    """Thread-safe LRU (Least Recently Used) cache implementation"""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = OrderedDict()
        self._stats = CacheStats(max_size=max_size)
        self._lock = threading.RLock()
        
        logger.info(f"LRU Cache initialized (max_size: {max_size}, ttl: {default_ttl}s)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if entry.is_expired():
                    del self._cache[key]
                    self._stats.misses += 1
                    self._stats.evictions += 1
                    self._stats.size = len(self._cache)
                    self._stats.update_hit_rate()
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.touch()
                
                self._stats.hits += 1
                self._stats.update_hit_rate()
                return entry.value
            else:
                self._stats.misses += 1
                self._stats.update_hit_rate()
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache"""
        with self._lock:
            now = datetime.now()
            ttl_to_use = ttl if ttl is not None else self.default_ttl
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                ttl_seconds=ttl_to_use
            )
            
            if key in self._cache:
                # Update existing entry
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = entry
                
                # Evict if over capacity
                while len(self._cache) > self.max_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    self._stats.evictions += 1
            
            self._stats.size = len(self._cache)
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            evicted_count = len(self._cache)
            self._cache.clear()
            self._stats.evictions += evicted_count
            self._stats.size = 0
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count"""
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            self._stats.evictions += len(expired_keys)
            self._stats.size = len(self._cache)
            
            return len(expired_keys)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=self._stats.size,
                max_size=self._stats.max_size,
                hit_rate=self._stats.hit_rate
            )


class ResultCache:
    """Intelligent result caching with function signature awareness"""
    
    def __init__(self, max_size: int = 500, default_ttl: int = 300):
        self.cache = LRUCache(max_size, default_ttl)
        self.function_stats = defaultdict(lambda: {'calls': 0, 'cache_hits': 0})
        
        logger.info(f"Result Cache initialized (max_size: {max_size}, ttl: {default_ttl}s)")
    
    def cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function signature"""
        # Create a deterministic key from function name and arguments
        key_data = {
            'function': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        
        # Convert to JSON string and hash for consistent key
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def cached_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with caching"""
        func_name = f"{func.__module__}.{func.__name__}"
        cache_key = self.cache_key(func_name, args, kwargs)
        
        self.function_stats[func_name]['calls'] += 1
        
        # Try to get from cache
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.function_stats[func_name]['cache_hits'] += 1
            logger.debug(f"Cache hit for {func_name}")
            return cached_result
        
        # Execute function and cache result
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        
        # Cache the result
        self.cache.put(cache_key, result)
        
        logger.debug(f"Cache miss for {func_name} (executed in {execution_time*1000:.2f}ms)")
        return result
    
    def get_function_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get per-function cache statistics"""
        stats = {}
        for func_name, func_stats in self.function_stats.items():
            calls = func_stats['calls']
            hits = func_stats['cache_hits']
            stats[func_name] = {
                'total_calls': calls,
                'cache_hits': hits,
                'cache_misses': calls - hits,
                'hit_rate': hits / calls if calls > 0 else 0.0
            }
        return stats


class WeekClassificationCache:
    """Specialized cache for week classification results"""
    
    def __init__(self, max_size: int = 200, ttl: int = 600):
        self.cache = LRUCache(max_size, ttl)
        self.classification_stats = defaultdict(int)
        
        logger.info(f"Week Classification Cache initialized (max_size: {max_size}, ttl: {ttl}s)")
    
    def cache_key_from_market_condition(self, market_condition, position) -> str:
        """Generate cache key from market condition and position"""
        key_data = {
            'symbol': market_condition.symbol,
            'current_price': round(market_condition.current_price, 2),
            'previous_close': round(market_condition.previous_close, 2),
            'week_start_price': round(market_condition.week_start_price, 2),
            'movement_percentage': round(market_condition.movement_percentage, 3),
            'movement_category': market_condition.movement_category.value,
            'volatility': round(market_condition.volatility, 3),
            'volume_ratio': round(market_condition.volume_ratio, 2),
            'position': position.value if hasattr(position, 'value') else str(position)
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_classification(self, market_condition, position) -> Optional[Any]:
        """Get cached week classification"""
        cache_key = self.cache_key_from_market_condition(market_condition, position)
        result = self.cache.get(cache_key)
        
        if result:
            self.classification_stats['cache_hits'] += 1
            logger.debug(f"Week classification cache hit for {market_condition.symbol}")
        else:
            self.classification_stats['cache_misses'] += 1
        
        return result
    
    def cache_classification(self, market_condition, position, result) -> None:
        """Cache week classification result"""
        cache_key = self.cache_key_from_market_condition(market_condition, position)
        self.cache.put(cache_key, result)
        
        # Track classification type
        if hasattr(result, 'week_type'):
            week_type = result.week_type.value if hasattr(result.week_type, 'value') else str(result.week_type)
            self.classification_stats[f'type_{week_type}'] += 1
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get week classification cache statistics"""
        cache_stats = self.cache.get_stats()
        return {
            'cache_stats': cache_stats.__dict__,
            'classification_stats': dict(self.classification_stats)
        }


class MarketAnalysisCache:
    """Specialized cache for market analysis results"""
    
    def __init__(self, max_size: int = 300, ttl: int = 180):
        self.cache = LRUCache(max_size, ttl)
        self.analysis_stats = defaultdict(int)
        
        logger.info(f"Market Analysis Cache initialized (max_size: {max_size}, ttl: {ttl}s)")
    
    def cache_key_from_market_data(self, market_data: Dict[str, Any]) -> str:
        """Generate cache key from market data"""
        # Round numerical values for better cache hit rates
        normalized_data = {}
        for key, value in market_data.items():
            if isinstance(value, (int, float)):
                normalized_data[key] = round(value, 3)
            else:
                normalized_data[key] = value
        
        key_str = json.dumps(normalized_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_analysis(self, market_data: Dict[str, Any]) -> Optional[Any]:
        """Get cached market analysis"""
        cache_key = self.cache_key_from_market_data(market_data)
        result = self.cache.get(cache_key)
        
        if result:
            self.analysis_stats['cache_hits'] += 1
            logger.debug("Market analysis cache hit")
        else:
            self.analysis_stats['cache_misses'] += 1
        
        return result
    
    def cache_analysis(self, market_data: Dict[str, Any], result) -> None:
        """Cache market analysis result"""
        cache_key = self.cache_key_from_market_data(market_data)
        self.cache.put(cache_key, result)
        
        # Track analysis condition
        if hasattr(result, 'condition'):
            self.analysis_stats[f'condition_{result.condition}'] += 1
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get market analysis cache statistics"""
        cache_stats = self.cache.get_stats()
        return {
            'cache_stats': cache_stats.__dict__,
            'analysis_stats': dict(self.analysis_stats)
        }


def cached_function(cache_instance: ResultCache, ttl: Optional[int] = None):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cache_instance.cached_call(func, *args, **kwargs)
        
        # Add cache management methods to the wrapped function
        wrapper.cache_invalidate = lambda: cache_instance.cache.clear()
        wrapper.cache_stats = lambda: cache_instance.get_function_stats().get(
            f"{func.__module__}.{func.__name__}", {}
        )
        
        return wrapper
    return decorator


class PerformanceEnhancer:
    """Algorithm optimizations and performance enhancements"""
    
    def __init__(self):
        self.optimization_stats = defaultdict(int)
        self.lazy_loaded_components = {}
        
        logger.info("Performance Enhancer initialized")
    
    def lazy_load_component(self, component_name: str, loader_func: Callable) -> Any:
        """Lazy load components on first access"""
        if component_name not in self.lazy_loaded_components:
            start_time = time.perf_counter()
            component = loader_func()
            load_time = time.perf_counter() - start_time
            
            self.lazy_loaded_components[component_name] = component
            self.optimization_stats[f'lazy_load_{component_name}'] = load_time * 1000
            
            logger.info(f"Lazy loaded {component_name} in {load_time*1000:.2f}ms")
        
        return self.lazy_loaded_components[component_name]
    
    def optimize_numerical_calculation(self, calculation_func: Callable, *args, **kwargs) -> Any:
        """Optimize numerical calculations with caching and vectorization"""
        # Use numpy for vectorized operations if available
        try:
            import numpy as np
            
            # Convert lists to numpy arrays for faster computation
            optimized_args = []
            for arg in args:
                if isinstance(arg, list) and len(arg) > 10:
                    optimized_args.append(np.array(arg))
                else:
                    optimized_args.append(arg)
            
            return calculation_func(*optimized_args, **kwargs)
        except ImportError:
            # Fall back to regular calculation
            return calculation_func(*args, **kwargs)
    
    def batch_process(self, items: List[Any], processor_func: Callable, batch_size: int = 50) -> List[Any]:
        """Process items in batches for better performance"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_start = time.perf_counter()
            
            batch_results = [processor_func(item) for item in batch]
            results.extend(batch_results)
            
            batch_time = time.perf_counter() - batch_start
            self.optimization_stats['batch_processing_time'] += batch_time * 1000
            
        self.optimization_stats['batches_processed'] += len(items) // batch_size + (1 if len(items) % batch_size else 0)
        
        return results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get performance optimization statistics"""
        return dict(self.optimization_stats)


class CacheCoordinator:
    """Centralized cache management and coordination"""
    
    def __init__(self):
        self.result_cache = ResultCache(max_size=1000, default_ttl=300)
        self.week_classification_cache = WeekClassificationCache(max_size=200, ttl=600)
        self.market_analysis_cache = MarketAnalysisCache(max_size=300, ttl=180)
        self.performance_enhancer = PerformanceEnhancer()
        
        self.global_stats = {
            'cache_operations': 0,
            'total_cache_hits': 0,
            'total_cache_misses': 0,
            'optimization_operations': 0
        }
        
        logger.info("Cache Coordinator initialized with all caching systems")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive caching statistics"""
        result_stats = self.result_cache.get_function_stats()
        week_stats = self.week_classification_cache.get_classification_stats()
        market_stats = self.market_analysis_cache.get_analysis_stats()
        optimization_stats = self.performance_enhancer.get_optimization_stats()
        
        # Calculate total hit rates
        total_hits = 0
        total_misses = 0
        
        for func_stats in result_stats.values():
            total_hits += func_stats['cache_hits']
            total_misses += func_stats['cache_misses']
        
        total_hits += week_stats['classification_stats'].get('cache_hits', 0)
        total_misses += week_stats['classification_stats'].get('cache_misses', 0)
        
        total_hits += market_stats['analysis_stats'].get('cache_hits', 0)
        total_misses += market_stats['analysis_stats'].get('cache_misses', 0)
        
        overall_hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0
        
        return {
            'overall_hit_rate': overall_hit_rate,
            'total_cache_hits': total_hits,
            'total_cache_misses': total_misses,
            'result_cache': result_stats,
            'week_classification_cache': week_stats,
            'market_analysis_cache': market_stats,
            'performance_optimizations': optimization_stats,
            'global_stats': self.global_stats
        }
    
    def cleanup_all_caches(self) -> Dict[str, int]:
        """Cleanup expired entries from all caches"""
        cleanup_results = {
            'result_cache': self.result_cache.cache.cleanup_expired(),
            'week_classification_cache': self.week_classification_cache.cache.cleanup_expired(),
            'market_analysis_cache': self.market_analysis_cache.cache.cleanup_expired()
        }
        
        total_cleaned = sum(cleanup_results.values())
        logger.info(f"Cache cleanup completed: {total_cleaned} expired entries removed")
        
        return cleanup_results
    
    def invalidate_all_caches(self):
        """Invalidate all caches"""
        self.result_cache.cache.clear()
        self.week_classification_cache.cache.clear()
        self.market_analysis_cache.cache.clear()
        
        logger.info("All caches invalidated")
    
    def optimize_cache_sizes(self):
        """Dynamically optimize cache sizes based on usage patterns"""
        stats = self.get_comprehensive_stats()
        
        # Adjust cache sizes based on hit rates
        week_hit_rate = stats['week_classification_cache']['cache_stats'].get('hit_rate', 0)
        market_hit_rate = stats['market_analysis_cache']['cache_stats'].get('hit_rate', 0)
        
        if week_hit_rate > 0.8 and self.week_classification_cache.cache.max_size < 500:
            self.week_classification_cache.cache.max_size = min(500, self.week_classification_cache.cache.max_size * 1.5)
            logger.info(f"Increased week classification cache size to {self.week_classification_cache.cache.max_size}")
        
        if market_hit_rate > 0.8 and self.market_analysis_cache.cache.max_size < 600:
            self.market_analysis_cache.cache.max_size = min(600, self.market_analysis_cache.cache.max_size * 1.5)
            logger.info(f"Increased market analysis cache size to {self.market_analysis_cache.cache.max_size}")


# Global cache coordinator instance
_global_cache_coordinator = None


def get_cache_coordinator() -> CacheCoordinator:
    """Get the global cache coordinator instance"""
    global _global_cache_coordinator
    if _global_cache_coordinator is None:
        _global_cache_coordinator = CacheCoordinator()
        logger.info("Global cache coordinator initialized")
    
    return _global_cache_coordinator


# Convenience functions for easy integration
def cached_week_classification(market_condition, position, classifier_func: Callable) -> Any:
    """Cache-enabled week classification"""
    coordinator = get_cache_coordinator()
    
    # Try cache first
    cached_result = coordinator.week_classification_cache.get_classification(market_condition, position)
    if cached_result is not None:
        return cached_result
    
    # Execute classification and cache result
    result = classifier_func(market_condition, position)
    coordinator.week_classification_cache.cache_classification(market_condition, position, result)
    
    return result


def cached_market_analysis(market_data: Dict[str, Any], analyzer_func: Callable) -> Any:
    """Cache-enabled market analysis"""
    coordinator = get_cache_coordinator()
    
    # Try cache first
    cached_result = coordinator.market_analysis_cache.get_analysis(market_data)
    if cached_result is not None:
        return cached_result
    
    # Execute analysis and cache result
    result = analyzer_func(market_data)
    coordinator.market_analysis_cache.cache_analysis(market_data, result)
    
    return result


if __name__ == '__main__':
    print("🚀 Testing Caching Systems and Performance Enhancements (P5 of WS2 - Phase 3)")
    print("=" * 80)
    
    # Initialize cache coordinator
    coordinator = get_cache_coordinator()
    
    print("\n📊 Testing LRU Cache:")
    lru_cache = LRUCache(max_size=5, default_ttl=10)
    
    # Test cache operations
    for i in range(10):
        lru_cache.put(f"key_{i}", f"value_{i}")
    
    # Test cache hits and misses
    for i in range(15):
        result = lru_cache.get(f"key_{i}")
        print(f"   key_{i}: {'HIT' if result else 'MISS'}")
    
    cache_stats = lru_cache.get_stats()
    print(f"   Cache Stats: {cache_stats.hits} hits, {cache_stats.misses} misses, {cache_stats.hit_rate:.2%} hit rate")
    
    print("\n🧠 Testing Result Cache:")
    result_cache = ResultCache(max_size=100, default_ttl=60)
    
    # Test function caching
    @cached_function(result_cache)
    def expensive_calculation(x, y):
        time.sleep(0.001)  # Simulate expensive operation
        return x * y + x ** 2
    
    # Test cache performance
    start_time = time.perf_counter()
    for i in range(20):
        result = expensive_calculation(i % 5, i % 3)  # Repeated calculations
    first_run_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    for i in range(20):
        result = expensive_calculation(i % 5, i % 3)  # Same calculations (should hit cache)
    second_run_time = time.perf_counter() - start_time
    
    print(f"   First run: {first_run_time*1000:.2f}ms")
    print(f"   Second run: {second_run_time*1000:.2f}ms")
    print(f"   Speedup: {first_run_time/second_run_time:.1f}x")
    
    print("\n📈 Testing Performance Enhancements:")
    enhancer = PerformanceEnhancer()
    
    # Test batch processing
    items = list(range(100))
    
    def simple_processor(x):
        return x ** 2
    
    start_time = time.perf_counter()
    batch_results = enhancer.batch_process(items, simple_processor, batch_size=20)
    batch_time = time.perf_counter() - start_time
    
    print(f"   Batch processing: {len(batch_results)} items in {batch_time*1000:.2f}ms")
    
    # Test lazy loading
    def expensive_component():
        time.sleep(0.01)  # Simulate expensive initialization
        return {"initialized": True, "data": list(range(1000))}
    
    start_time = time.perf_counter()
    component = enhancer.lazy_load_component("test_component", expensive_component)
    first_load_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    component = enhancer.lazy_load_component("test_component", expensive_component)
    second_load_time = time.perf_counter() - start_time
    
    print(f"   Lazy loading - First: {first_load_time*1000:.2f}ms, Second: {second_load_time*1000:.2f}ms")
    
    print("\n📊 Comprehensive Cache Statistics:")
    comprehensive_stats = coordinator.get_comprehensive_stats()
    print(f"   Overall Hit Rate: {comprehensive_stats['overall_hit_rate']:.2%}")
    print(f"   Total Cache Hits: {comprehensive_stats['total_cache_hits']}")
    print(f"   Total Cache Misses: {comprehensive_stats['total_cache_misses']}")
    
    # Test cache cleanup
    print("\n🧹 Testing Cache Cleanup:")
    cleanup_results = coordinator.cleanup_all_caches()
    print(f"   Cleanup Results: {cleanup_results}")
    
    print("\n✅ P5 of WS2 - Phase 3: Caching Systems and Performance Enhancements COMPLETE")
    print("🚀 Ready for Phase 4: Monitoring and Metrics Collection Framework")

