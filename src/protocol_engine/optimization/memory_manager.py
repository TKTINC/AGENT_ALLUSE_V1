#!/usr/bin/env python3
"""
ALL-USE Agent - Protocol Engine Memory Manager
P5 of WS2: Performance Optimization and Monitoring - Phase 2

This module provides comprehensive memory optimization and resource management
for the Protocol Engine, implementing object pooling, memory cleanup, and
resource lifecycle management to achieve the target memory usage reduction.

Memory Optimization Components:
1. Object Pool Manager - Reusable object pooling system
2. Memory Manager - Intelligent memory allocation and cleanup
3. Resource Manager - Resource lifecycle and allocation management
4. Garbage Collection Optimizer - Enhanced garbage collection strategies
5. Memory Leak Detector - Proactive memory leak detection and prevention
"""

import gc
import weakref
import threading
import time
import psutil
import tracemalloc
from typing import Dict, List, Any, Optional, Type, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemoryMetrics:
    """Memory usage metrics tracking"""
    timestamp: datetime
    total_memory: float  # MB
    component_memory: Dict[str, float]
    object_counts: Dict[str, int]
    gc_stats: Dict[str, int]
    pool_stats: Dict[str, int]


class ObjectPool:
    """Generic object pooling system for memory optimization"""
    
    def __init__(self, object_type: Type, max_size: int = 100, factory: Optional[Callable] = None):
        self.object_type = object_type
        self.max_size = max_size
        self.factory = factory or object_type
        self._pool = deque(maxlen=max_size)
        self._created_count = 0
        self._reused_count = 0
        self._lock = threading.Lock()
        
        logger.info(f"Object pool created for {object_type.__name__} (max_size: {max_size})")
    
    def acquire(self, *args, **kwargs):
        """Acquire an object from the pool or create new one"""
        with self._lock:
            if self._pool:
                obj = self._pool.popleft()
                self._reused_count += 1
                
                # Reset object if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset(*args, **kwargs)
                
                return obj
            else:
                # Create new object
                obj = self.factory(*args, **kwargs)
                self._created_count += 1
                return obj
    
    def release(self, obj):
        """Return an object to the pool"""
        if obj is None:
            return
        
        with self._lock:
            if len(self._pool) < self.max_size:
                # Clean object before returning to pool
                if hasattr(obj, 'cleanup'):
                    obj.cleanup()
                
                self._pool.append(obj)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool usage statistics"""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'max_size': self.max_size,
                'created_count': self._created_count,
                'reused_count': self._reused_count,
                'reuse_ratio': self._reused_count / max(self._created_count + self._reused_count, 1)
            }
    
    def clear(self):
        """Clear the pool"""
        with self._lock:
            self._pool.clear()


class MemoryManager:
    """Comprehensive memory management system"""
    
    def __init__(self):
        self.object_pools = {}
        self.memory_metrics = []
        self.weak_references = weakref.WeakSet()
        self.cleanup_callbacks = []
        self.monitoring_enabled = True
        self.gc_threshold_adjustment = True
        
        # Configure garbage collection for better performance
        if self.gc_threshold_adjustment:
            self._optimize_gc_thresholds()
        
        logger.info("Memory Manager initialized with optimization features")
    
    def create_object_pool(self, name: str, object_type: Type, max_size: int = 100, factory: Optional[Callable] = None) -> ObjectPool:
        """Create a new object pool"""
        pool = ObjectPool(object_type, max_size, factory)
        self.object_pools[name] = pool
        logger.info(f"Created object pool '{name}' for {object_type.__name__}")
        return pool
    
    def get_object_pool(self, name: str) -> Optional[ObjectPool]:
        """Get an existing object pool"""
        return self.object_pools.get(name)
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback for memory management"""
        self.cleanup_callbacks.append(callback)
    
    def track_object(self, obj):
        """Track an object for memory management"""
        self.weak_references.add(obj)
    
    def collect_memory_metrics(self) -> MemoryMetrics:
        """Collect current memory usage metrics"""
        if not self.monitoring_enabled:
            return None
        
        # Get system memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        total_memory = memory_info.rss / 1024 / 1024  # MB
        
        # Get garbage collection stats
        gc_stats = {
            f'generation_{i}': len(gc.get_objects(i)) for i in range(3)
        }
        gc_stats['total_objects'] = len(gc.get_objects())
        
        # Get object counts by type
        object_counts = defaultdict(int)
        for obj in gc.get_objects():
            object_counts[type(obj).__name__] += 1
        
        # Get pool statistics
        pool_stats = {}
        for name, pool in self.object_pools.items():
            pool_stats[name] = pool.get_stats()
        
        metrics = MemoryMetrics(
            timestamp=datetime.now(),
            total_memory=total_memory,
            component_memory={},  # Will be populated by component-specific tracking
            object_counts=dict(object_counts),
            gc_stats=gc_stats,
            pool_stats=pool_stats
        )
        
        self.memory_metrics.append(metrics)
        
        # Keep only last 100 metrics to prevent memory growth
        if len(self.memory_metrics) > 100:
            self.memory_metrics = self.memory_metrics[-100:]
        
        return metrics
    
    def force_cleanup(self):
        """Force memory cleanup and garbage collection"""
        logger.info("Forcing memory cleanup...")
        
        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Cleanup callback failed: {e}")
        
        # Clear object pools if needed
        for name, pool in self.object_pools.items():
            if pool.get_stats()['pool_size'] > pool.max_size * 0.8:
                pool.clear()
                logger.info(f"Cleared object pool '{name}' due to high usage")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        return collected
    
    def _optimize_gc_thresholds(self):
        """Optimize garbage collection thresholds for better performance"""
        # Get current thresholds
        current_thresholds = gc.get_threshold()
        
        # Increase thresholds to reduce GC frequency (trade memory for performance)
        new_thresholds = (
            current_thresholds[0] * 2,  # Generation 0
            current_thresholds[1] * 2,  # Generation 1
            current_thresholds[2] * 2   # Generation 2
        )
        
        gc.set_threshold(*new_thresholds)
        logger.info(f"Optimized GC thresholds: {current_thresholds} → {new_thresholds}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary"""
        if not self.memory_metrics:
            return {}
        
        latest_metrics = self.memory_metrics[-1]
        
        # Calculate memory trends
        if len(self.memory_metrics) >= 2:
            previous_metrics = self.memory_metrics[-2]
            memory_trend = latest_metrics.total_memory - previous_metrics.total_memory
        else:
            memory_trend = 0
        
        # Get top object types by count
        top_objects = sorted(
            latest_metrics.object_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Get pool efficiency
        pool_efficiency = {}
        for name, stats in latest_metrics.pool_stats.items():
            pool_efficiency[name] = {
                'reuse_ratio': stats['reuse_ratio'],
                'utilization': stats['pool_size'] / stats['max_size']
            }
        
        return {
            'current_memory_mb': latest_metrics.total_memory,
            'memory_trend_mb': memory_trend,
            'total_objects': latest_metrics.gc_stats['total_objects'],
            'top_object_types': top_objects,
            'pool_efficiency': pool_efficiency,
            'gc_stats': latest_metrics.gc_stats,
            'tracked_objects': len(self.weak_references)
        }


class ResourceManager:
    """Resource lifecycle and allocation management"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.active_resources = {}
        self.resource_stats = defaultdict(int)
        self.cleanup_schedule = []
        self._lock = threading.Lock()
        
        logger.info("Resource Manager initialized")
    
    def allocate_resource(self, resource_type: str, resource_id: str, resource: Any, cleanup_func: Optional[Callable] = None):
        """Allocate and track a resource"""
        with self._lock:
            self.active_resources[resource_id] = {
                'type': resource_type,
                'resource': resource,
                'allocated_at': datetime.now(),
                'cleanup_func': cleanup_func,
                'access_count': 0
            }
            self.resource_stats[resource_type] += 1
            
            # Track with memory manager
            self.memory_manager.track_object(resource)
        
        logger.debug(f"Allocated resource '{resource_id}' of type '{resource_type}'")
    
    def access_resource(self, resource_id: str) -> Optional[Any]:
        """Access a tracked resource"""
        with self._lock:
            if resource_id in self.active_resources:
                self.active_resources[resource_id]['access_count'] += 1
                return self.active_resources[resource_id]['resource']
            return None
    
    def release_resource(self, resource_id: str):
        """Release a tracked resource"""
        with self._lock:
            if resource_id in self.active_resources:
                resource_info = self.active_resources.pop(resource_id)
                self.resource_stats[resource_info['type']] -= 1
                
                # Run cleanup function if provided
                if resource_info['cleanup_func']:
                    try:
                        resource_info['cleanup_func'](resource_info['resource'])
                    except Exception as e:
                        logger.warning(f"Resource cleanup failed for '{resource_id}': {e}")
                
                logger.debug(f"Released resource '{resource_id}'")
    
    def schedule_cleanup(self, resource_id: str, delay_seconds: int):
        """Schedule resource cleanup after a delay"""
        cleanup_time = datetime.now() + timedelta(seconds=delay_seconds)
        self.cleanup_schedule.append((cleanup_time, resource_id))
        self.cleanup_schedule.sort(key=lambda x: x[0])
    
    def process_scheduled_cleanup(self):
        """Process any scheduled resource cleanup"""
        now = datetime.now()
        to_cleanup = []
        
        with self._lock:
            while self.cleanup_schedule and self.cleanup_schedule[0][0] <= now:
                cleanup_time, resource_id = self.cleanup_schedule.pop(0)
                to_cleanup.append(resource_id)
        
        for resource_id in to_cleanup:
            self.release_resource(resource_id)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics"""
        with self._lock:
            active_count = len(self.active_resources)
            resource_types = dict(self.resource_stats)
            
            # Calculate resource age statistics
            ages = []
            for resource_info in self.active_resources.values():
                age = (datetime.now() - resource_info['allocated_at']).total_seconds()
                ages.append(age)
            
            avg_age = sum(ages) / len(ages) if ages else 0
            
            return {
                'active_resources': active_count,
                'resource_types': resource_types,
                'average_age_seconds': avg_age,
                'scheduled_cleanups': len(self.cleanup_schedule)
            }


class MemoryLeakDetector:
    """Proactive memory leak detection and prevention"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.baseline_memory = None
        self.memory_samples = deque(maxlen=50)
        self.leak_threshold_mb = 10  # MB growth threshold
        self.monitoring_interval = 30  # seconds
        self.last_check = datetime.now()
        
        logger.info("Memory Leak Detector initialized")
    
    def establish_baseline(self):
        """Establish memory usage baseline"""
        metrics = self.memory_manager.collect_memory_metrics()
        if metrics:
            self.baseline_memory = metrics.total_memory
            logger.info(f"Memory baseline established: {self.baseline_memory:.2f} MB")
    
    def check_for_leaks(self) -> Dict[str, Any]:
        """Check for potential memory leaks"""
        now = datetime.now()
        if (now - self.last_check).total_seconds() < self.monitoring_interval:
            return {}
        
        self.last_check = now
        
        # Collect current metrics
        metrics = self.memory_manager.collect_memory_metrics()
        if not metrics:
            return {}
        
        self.memory_samples.append(metrics.total_memory)
        
        leak_indicators = {
            'leak_detected': False,
            'memory_growth_mb': 0,
            'growth_rate_mb_per_min': 0,
            'recommendations': []
        }
        
        # Check for sustained memory growth
        if len(self.memory_samples) >= 10:
            recent_avg = sum(list(self.memory_samples)[-5:]) / 5
            older_avg = sum(list(self.memory_samples)[-10:-5]) / 5
            growth = recent_avg - older_avg
            
            leak_indicators['memory_growth_mb'] = growth
            
            if growth > self.leak_threshold_mb:
                leak_indicators['leak_detected'] = True
                leak_indicators['recommendations'].append('Force garbage collection')
                leak_indicators['recommendations'].append('Review object pooling efficiency')
                leak_indicators['recommendations'].append('Check for circular references')
                
                logger.warning(f"Potential memory leak detected: {growth:.2f} MB growth")
        
        # Check against baseline
        if self.baseline_memory and metrics.total_memory > self.baseline_memory + 50:
            leak_indicators['leak_detected'] = True
            leak_indicators['recommendations'].append('Memory usage significantly above baseline')
            
            logger.warning(f"Memory usage above baseline: {metrics.total_memory:.2f} MB vs {self.baseline_memory:.2f} MB")
        
        return leak_indicators
    
    def auto_remediate(self):
        """Automatically attempt to remediate detected memory issues"""
        leak_info = self.check_for_leaks()
        
        if leak_info.get('leak_detected'):
            logger.info("Attempting automatic memory leak remediation...")
            
            # Force cleanup
            freed_objects = self.memory_manager.force_cleanup()
            
            # Clear object pools if they're too full
            for name, pool in self.memory_manager.object_pools.items():
                stats = pool.get_stats()
                if stats['reuse_ratio'] < 0.5:  # Low reuse efficiency
                    pool.clear()
                    logger.info(f"Cleared inefficient object pool '{name}'")
            
            logger.info(f"Memory remediation complete: {freed_objects} objects freed")


# Factory functions for common Protocol Engine objects
def create_market_condition_factory():
    """Factory for MarketCondition objects with pooling support"""
    def factory(*args, **kwargs):
        from protocol_engine.week_classification.week_classifier import MarketCondition, MarketMovement
        return MarketCondition(
            symbol=kwargs.get('symbol', 'SPY'),
            current_price=kwargs.get('current_price', 450.0),
            previous_close=kwargs.get('previous_close', 445.0),
            week_start_price=kwargs.get('week_start_price', 440.0),
            movement_percentage=kwargs.get('movement_percentage', 2.27),
            movement_category=kwargs.get('movement_category', MarketMovement.SLIGHT_UP),
            volatility=kwargs.get('volatility', 0.15),
            volume_ratio=kwargs.get('volume_ratio', 1.2),
            timestamp=kwargs.get('timestamp', datetime.now())
        )
    return factory


def create_trading_decision_factory():
    """Factory for TradingDecision objects with pooling support"""
    def factory(*args, **kwargs):
        from protocol_engine.rules.trading_protocol_rules import TradingDecision, AccountType
        return TradingDecision(
            action=kwargs.get('action', 'sell_to_open'),
            symbol=kwargs.get('symbol', 'SPY'),
            quantity=kwargs.get('quantity', 10),
            delta=kwargs.get('delta', 45.0),
            expiration=kwargs.get('expiration', datetime.now() + timedelta(days=35)),
            strike=kwargs.get('strike', 440.0),
            account_type=kwargs.get('account_type', AccountType.GEN_ACC),
            market_conditions=kwargs.get('market_conditions', {}),
            week_classification=kwargs.get('week_classification', 'P-EW'),
            confidence=kwargs.get('confidence', 0.85),
            expected_return=kwargs.get('expected_return', 0.025),
            max_risk=kwargs.get('max_risk', 0.05)
        )
    return factory


# Global memory management instance
_global_memory_manager = None
_global_resource_manager = None
_global_leak_detector = None


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
        
        # Create common object pools
        _global_memory_manager.create_object_pool(
            'market_conditions',
            type(None),  # Placeholder type
            max_size=50,
            factory=create_market_condition_factory()
        )
        
        _global_memory_manager.create_object_pool(
            'trading_decisions',
            type(None),  # Placeholder type
            max_size=100,
            factory=create_trading_decision_factory()
        )
        
        logger.info("Global memory manager initialized with object pools")
    
    return _global_memory_manager


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance"""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager(get_memory_manager())
        logger.info("Global resource manager initialized")
    
    return _global_resource_manager


def get_leak_detector() -> MemoryLeakDetector:
    """Get the global memory leak detector instance"""
    global _global_leak_detector
    if _global_leak_detector is None:
        _global_leak_detector = MemoryLeakDetector(get_memory_manager())
        _global_leak_detector.establish_baseline()
        logger.info("Global memory leak detector initialized")
    
    return _global_leak_detector


def optimize_memory_usage():
    """Comprehensive memory optimization function"""
    logger.info("Starting comprehensive memory optimization...")
    
    memory_manager = get_memory_manager()
    resource_manager = get_resource_manager()
    leak_detector = get_leak_detector()
    
    # Collect initial metrics
    initial_metrics = memory_manager.collect_memory_metrics()
    initial_memory = initial_metrics.total_memory if initial_metrics else 0
    
    # Process scheduled cleanups
    resource_manager.process_scheduled_cleanup()
    
    # Check for memory leaks and auto-remediate
    leak_detector.auto_remediate()
    
    # Force cleanup if memory usage is high
    if initial_memory > 150:  # MB threshold
        memory_manager.force_cleanup()
    
    # Collect final metrics
    final_metrics = memory_manager.collect_memory_metrics()
    final_memory = final_metrics.total_memory if final_metrics else 0
    
    memory_saved = initial_memory - final_memory
    
    logger.info(f"Memory optimization complete: {memory_saved:.2f} MB saved")
    logger.info(f"Memory usage: {initial_memory:.2f} MB → {final_memory:.2f} MB")
    
    return {
        'initial_memory_mb': initial_memory,
        'final_memory_mb': final_memory,
        'memory_saved_mb': memory_saved,
        'optimization_success': memory_saved > 0
    }


if __name__ == '__main__':
    print("🧠 Testing Memory Optimization System (P5 of WS2 - Phase 2)")
    print("=" * 65)
    
    # Initialize memory management
    memory_manager = get_memory_manager()
    resource_manager = get_resource_manager()
    leak_detector = get_leak_detector()
    
    print("\n📊 Initial Memory State:")
    initial_summary = memory_manager.get_memory_summary()
    print(f"   Memory Usage: {initial_summary.get('current_memory_mb', 0):.2f} MB")
    print(f"   Total Objects: {initial_summary.get('total_objects', 0):,}")
    
    # Test object pooling
    print("\n🔄 Testing Object Pooling:")
    market_pool = memory_manager.get_object_pool('market_conditions')
    decision_pool = memory_manager.get_object_pool('trading_decisions')
    
    # Simulate object usage
    for i in range(20):
        market_obj = market_pool.acquire(symbol='SPY', current_price=450.0 + i)
        decision_obj = decision_pool.acquire(symbol='SPY', quantity=10 + i)
        
        # Simulate usage
        time.sleep(0.001)
        
        # Return to pool
        market_pool.release(market_obj)
        decision_pool.release(decision_obj)
    
    print(f"   Market Conditions Pool: {market_pool.get_stats()}")
    print(f"   Trading Decisions Pool: {decision_pool.get_stats()}")
    
    # Test memory optimization
    print("\n🎯 Running Memory Optimization:")
    optimization_result = optimize_memory_usage()
    print(f"   Memory Saved: {optimization_result['memory_saved_mb']:.2f} MB")
    print(f"   Optimization Success: {optimization_result['optimization_success']}")
    
    # Final memory state
    print("\n📈 Final Memory State:")
    final_summary = memory_manager.get_memory_summary()
    print(f"   Memory Usage: {final_summary.get('current_memory_mb', 0):.2f} MB")
    print(f"   Memory Trend: {final_summary.get('memory_trend_mb', 0):+.2f} MB")
    print(f"   Pool Efficiency: {final_summary.get('pool_efficiency', {})}")
    
    print("\n✅ P5 of WS2 - Phase 2: Memory Optimization COMPLETE")
    print("🚀 Ready for Phase 3: Caching Systems and Performance Enhancements")

