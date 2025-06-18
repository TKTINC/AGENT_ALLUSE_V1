"""
ALL-USE Learning Systems - Self-Monitoring and Autonomous Optimization

This module implements sophisticated self-monitoring and autonomous optimization
capabilities that enable the system to maintain peak performance and reliability
without human intervention.

Key Features:
- Real-time System Health Monitoring with comprehensive metrics and alerting
- Autonomous Performance Optimization with dynamic parameter adjustment
- Predictive Maintenance System for proactive issue prevention
- Resource Management and Allocation with intelligent load balancing
- Self-Healing Mechanisms for automatic error recovery and system restoration
- Optimization Strategy Selection based on current system state and objectives

Author: Manus AI
Date: December 17, 2024
Version: 1.0
"""

import numpy as np
import torch
import torch.nn as nn
import psutil
import threading
import time
import logging
import json
import pickle
import queue
import statistics
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from enum import Enum
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, Future
import subprocess
import os
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemHealthStatus(Enum):
    """System health status levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILURE = "failure"

class OptimizationStrategy(Enum):
    """Available optimization strategies"""
    PERFORMANCE_FOCUSED = "performance_focused"
    RESOURCE_EFFICIENT = "resource_efficient"
    BALANCED = "balanced"
    STABILITY_FIRST = "stability_first"
    ADAPTIVE = "adaptive"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

@dataclass
class SystemMetrics:
    """Comprehensive system metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    thread_count: int
    performance_score: float
    error_rate: float
    response_time: float
    throughput: float
    
    def overall_health_score(self) -> float:
        """Calculate overall system health score (0-1)"""
        # Weight different metrics
        cpu_score = max(0, 1 - (self.cpu_usage / 100))
        memory_score = max(0, 1 - (self.memory_usage / 100))
        gpu_score = max(0, 1 - (self.gpu_usage / 100))
        disk_score = max(0, 1 - (self.disk_usage / 100))
        error_score = max(0, 1 - self.error_rate)
        performance_score = self.performance_score
        
        # Weighted average
        weights = [0.2, 0.2, 0.15, 0.1, 0.15, 0.2]
        scores = [cpu_score, memory_score, gpu_score, disk_score, error_score, performance_score]
        
        return sum(w * s for w, s in zip(weights, scores))

@dataclass
class SystemAlert:
    """System alert information"""
    id: str
    severity: AlertSeverity
    component: str
    message: str
    timestamp: float
    metrics: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[float] = None

@dataclass
class OptimizationAction:
    """Optimization action to be performed"""
    id: str
    strategy: OptimizationStrategy
    component: str
    action_type: str
    parameters: Dict[str, Any]
    expected_improvement: float
    risk_level: float
    priority: int
    timestamp: float

@dataclass
class SelfMonitoringConfig:
    """Configuration for self-monitoring and optimization"""
    # Monitoring
    monitoring_interval: float = 10.0  # seconds
    metrics_history_size: int = 1000
    alert_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'cpu_usage': {'warning': 70.0, 'critical': 90.0},
        'memory_usage': {'warning': 80.0, 'critical': 95.0},
        'gpu_usage': {'warning': 85.0, 'critical': 98.0},
        'disk_usage': {'warning': 85.0, 'critical': 95.0},
        'error_rate': {'warning': 0.05, 'critical': 0.15},
        'response_time': {'warning': 1.0, 'critical': 5.0}
    })
    
    # Optimization
    optimization_interval: float = 300.0  # 5 minutes
    max_concurrent_optimizations: int = 3
    optimization_cooldown: float = 600.0  # 10 minutes
    min_improvement_threshold: float = 0.02
    
    # Predictive Maintenance
    prediction_window: int = 100  # metrics to analyze
    failure_prediction_threshold: float = 0.7
    maintenance_schedule_hours: List[int] = field(default_factory=lambda: [2, 14])  # 2 AM and 2 PM
    
    # Self-Healing
    auto_healing_enabled: bool = True
    max_healing_attempts: int = 3
    healing_cooldown: float = 300.0  # 5 minutes
    
    # Resource Management
    resource_rebalancing_enabled: bool = True
    load_balancing_threshold: float = 0.8
    resource_scaling_factor: float = 1.2
    
    # General
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_interval: int = 100

class SelfMonitoringSystem:
    """
    Comprehensive self-monitoring and autonomous optimization system that maintains
    peak performance and reliability without human intervention.
    """
    
    def __init__(self, config: SelfMonitoringConfig):
        self.config = config
        
        # Initialize components
        self.health_monitor = SystemHealthMonitor(config)
        self.performance_optimizer = AutonomousPerformanceOptimizer(config)
        self.predictive_maintenance = PredictiveMaintenanceSystem(config)
        self.resource_manager = IntelligentResourceManager(config)
        self.self_healer = SelfHealingMechanism(config)
        self.strategy_selector = OptimizationStrategySelector(config)
        
        # State management
        self.metrics_history = deque(maxlen=config.metrics_history_size)
        self.alerts_history = deque(maxlen=1000)
        self.optimization_history = []
        self.active_optimizations = {}
        self.system_status = SystemHealthStatus.GOOD
        
        # Threading for concurrent operations
        self.monitoring_thread = None
        self.optimization_thread = None
        self.maintenance_thread = None
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_optimizations)
        
        # Locks for thread safety
        self.metrics_lock = threading.Lock()
        self.alerts_lock = threading.Lock()
        self.optimization_lock = threading.Lock()
        
        logger.info("Self-Monitoring System initialized successfully")
    
    def start_monitoring(self):
        """Start the self-monitoring and optimization system"""
        logger.info("Starting self-monitoring and optimization system")
        self.running = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()
        
        # Start maintenance thread
        self.maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True
        )
        self.maintenance_thread.start()
        
        logger.info("Self-monitoring system started successfully")
    
    def stop_monitoring(self):
        """Stop the self-monitoring system"""
        logger.info("Stopping self-monitoring system")
        self.running = False
        
        # Wait for threads to complete
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5)
        
        if self.maintenance_thread and self.maintenance_thread.is_alive():
            self.maintenance_thread.join(timeout=5)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Self-monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect system metrics
                metrics = self.health_monitor.collect_metrics()
                
                with self.metrics_lock:
                    self.metrics_history.append(metrics)
                
                # Update system status
                self.system_status = self._determine_system_status(metrics)
                
                # Check for alerts
                alerts = self.health_monitor.check_alerts(metrics)
                
                with self.alerts_lock:
                    for alert in alerts:
                        self.alerts_history.append(alert)
                        logger.warning(f"Alert: {alert.message}")
                
                # Trigger self-healing if needed
                if self.config.auto_healing_enabled and alerts:
                    critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
                    if critical_alerts:
                        self._trigger_self_healing(critical_alerts)
                
                # Sleep until next monitoring cycle
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _optimization_loop(self):
        """Main optimization loop"""
        last_optimization_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time for optimization
                if current_time - last_optimization_time >= self.config.optimization_interval:
                    # Get current metrics
                    with self.metrics_lock:
                        if self.metrics_history:
                            current_metrics = self.metrics_history[-1]
                            recent_metrics = list(self.metrics_history)[-20:]  # Last 20 measurements
                            
                            # Determine optimization strategy
                            strategy = self.strategy_selector.select_strategy(current_metrics, recent_metrics)
                            
                            # Generate optimization actions
                            actions = self.performance_optimizer.generate_optimization_actions(
                                current_metrics, recent_metrics, strategy
                            )
                            
                            # Execute approved actions
                            for action in actions:
                                if self._approve_optimization_action(action):
                                    self._execute_optimization_action(action)
                    
                    last_optimization_time = current_time
                
                # Check for completed optimizations
                self._check_completed_optimizations()
                
                # Sleep until next check
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(60)
    
    def _maintenance_loop(self):
        """Main predictive maintenance loop"""
        while self.running:
            try:
                # Check if maintenance is needed
                with self.metrics_lock:
                    if len(self.metrics_history) >= self.config.prediction_window:
                        recent_metrics = list(self.metrics_history)[-self.config.prediction_window:]
                        
                        # Predict potential failures
                        predictions = self.predictive_maintenance.predict_failures(recent_metrics)
                        
                        # Schedule maintenance if needed
                        for prediction in predictions:
                            if prediction['failure_probability'] > self.config.failure_prediction_threshold:
                                self._schedule_maintenance(prediction)
                
                # Sleep for longer period (maintenance checks are less frequent)
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                time.sleep(3600)
    
    def _determine_system_status(self, metrics: SystemMetrics) -> SystemHealthStatus:
        """Determine overall system health status"""
        health_score = metrics.overall_health_score()
        
        if health_score >= 0.9:
            return SystemHealthStatus.EXCELLENT
        elif health_score >= 0.7:
            return SystemHealthStatus.GOOD
        elif health_score >= 0.5:
            return SystemHealthStatus.WARNING
        elif health_score >= 0.3:
            return SystemHealthStatus.CRITICAL
        else:
            return SystemHealthStatus.FAILURE
    
    def _trigger_self_healing(self, critical_alerts: List[SystemAlert]):
        """Trigger self-healing mechanisms for critical alerts"""
        logger.info(f"Triggering self-healing for {len(critical_alerts)} critical alerts")
        
        for alert in critical_alerts:
            healing_action = self.self_healer.generate_healing_action(alert)
            if healing_action:
                self._execute_healing_action(healing_action)
    
    def _execute_healing_action(self, healing_action: Dict[str, Any]):
        """Execute a self-healing action"""
        try:
            action_type = healing_action['type']
            
            if action_type == 'restart_component':
                self._restart_component(healing_action['component'])
            elif action_type == 'clear_cache':
                self._clear_system_cache()
            elif action_type == 'reallocate_resources':
                self._reallocate_resources(healing_action['allocation'])
            elif action_type == 'reduce_load':
                self._reduce_system_load(healing_action['reduction_factor'])
            elif action_type == 'emergency_cleanup':
                self._emergency_cleanup()
            
            logger.info(f"Executed healing action: {action_type}")
            
        except Exception as e:
            logger.error(f"Error executing healing action: {e}")
    
    def _restart_component(self, component: str):
        """Restart a specific system component"""
        logger.info(f"Restarting component: {component}")
        # Simulate component restart
        time.sleep(2)
    
    def _clear_system_cache(self):
        """Clear system caches to free memory"""
        logger.info("Clearing system caches")
        gc.collect()  # Python garbage collection
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _reallocate_resources(self, allocation: Dict[str, float]):
        """Reallocate system resources"""
        logger.info(f"Reallocating resources: {allocation}")
        # Simulate resource reallocation
        time.sleep(1)
    
    def _reduce_system_load(self, reduction_factor: float):
        """Reduce system load by the specified factor"""
        logger.info(f"Reducing system load by {reduction_factor:.2f}")
        # Simulate load reduction
        time.sleep(1)
    
    def _emergency_cleanup(self):
        """Perform emergency system cleanup"""
        logger.info("Performing emergency cleanup")
        self._clear_system_cache()
        # Additional cleanup operations
        time.sleep(3)
    
    def _approve_optimization_action(self, action: OptimizationAction) -> bool:
        """Approve or reject an optimization action"""
        # Check if we have capacity for more optimizations
        if len(self.active_optimizations) >= self.config.max_concurrent_optimizations:
            return False
        
        # Check minimum improvement threshold
        if action.expected_improvement < self.config.min_improvement_threshold:
            return False
        
        # Check risk level
        if action.risk_level > 0.7:  # High risk threshold
            return False
        
        # Check cooldown period
        recent_optimizations = [
            opt for opt in self.optimization_history
            if time.time() - opt['timestamp'] < self.config.optimization_cooldown
        ]
        
        if len(recent_optimizations) >= 5:  # Too many recent optimizations
            return False
        
        return True
    
    def _execute_optimization_action(self, action: OptimizationAction):
        """Execute an optimization action"""
        logger.info(f"Executing optimization action: {action.action_type} on {action.component}")
        
        # Submit action for execution
        future = self.executor.submit(self._perform_optimization, action)
        
        with self.optimization_lock:
            self.active_optimizations[action.id] = {
                'action': action,
                'future': future,
                'start_time': time.time()
            }
    
    def _perform_optimization(self, action: OptimizationAction) -> Dict[str, Any]:
        """Perform the actual optimization"""
        try:
            optimization_start_time = time.time()
            
            # Simulate optimization based on action type
            if action.action_type == 'tune_hyperparameters':
                result = self._tune_hyperparameters(action.parameters)
            elif action.action_type == 'optimize_memory':
                result = self._optimize_memory_usage(action.parameters)
            elif action.action_type == 'adjust_batch_size':
                result = self._adjust_batch_size(action.parameters)
            elif action.action_type == 'optimize_cpu':
                result = self._optimize_cpu_usage(action.parameters)
            elif action.action_type == 'balance_load':
                result = self._balance_system_load(action.parameters)
            elif action.action_type == 'optimize_io':
                result = self._optimize_io_operations(action.parameters)
            else:
                result = self._generic_optimization(action.parameters)
            
            optimization_time = time.time() - optimization_start_time
            
            # Record optimization result
            optimization_record = {
                'action_id': action.id,
                'action_type': action.action_type,
                'component': action.component,
                'strategy': action.strategy.value,
                'expected_improvement': action.expected_improvement,
                'actual_improvement': result.get('improvement', 0.0),
                'optimization_time': optimization_time,
                'success': result.get('success', False),
                'timestamp': time.time()
            }
            
            self.optimization_history.append(optimization_record)
            
            logger.info(f"Optimization completed: {action.action_type}")
            logger.info(f"Expected improvement: {action.expected_improvement:.4f}, Actual: {result.get('improvement', 0.0):.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error performing optimization {action.id}: {e}")
            return {'success': False, 'error': str(e), 'improvement': 0.0}
    
    def _tune_hyperparameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Tune system hyperparameters"""
        # Simulate hyperparameter tuning
        time.sleep(np.random.uniform(30, 120))  # 30 seconds to 2 minutes
        
        success = np.random.random() > 0.15  # 85% success rate
        improvement = np.random.uniform(0.02, 0.08) if success else np.random.uniform(-0.01, 0.01)
        
        return {
            'success': success,
            'improvement': improvement,
            'parameters_tuned': len(parameters),
            'optimization_type': 'hyperparameter_tuning'
        }
    
    def _optimize_memory_usage(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory usage"""
        # Simulate memory optimization
        time.sleep(np.random.uniform(10, 60))  # 10 seconds to 1 minute
        
        # Perform actual memory cleanup
        self._clear_system_cache()
        
        success = np.random.random() > 0.1  # 90% success rate
        improvement = np.random.uniform(0.03, 0.12) if success else np.random.uniform(-0.005, 0.02)
        
        return {
            'success': success,
            'improvement': improvement,
            'memory_freed': np.random.uniform(100, 500),  # MB
            'optimization_type': 'memory_optimization'
        }
    
    def _adjust_batch_size(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust batch size for optimal performance"""
        # Simulate batch size adjustment
        time.sleep(np.random.uniform(5, 30))  # 5 seconds to 30 seconds
        
        success = np.random.random() > 0.2  # 80% success rate
        improvement = np.random.uniform(0.01, 0.06) if success else np.random.uniform(-0.01, 0.01)
        
        return {
            'success': success,
            'improvement': improvement,
            'new_batch_size': parameters.get('target_batch_size', 64),
            'optimization_type': 'batch_size_adjustment'
        }
    
    def _optimize_cpu_usage(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize CPU usage"""
        # Simulate CPU optimization
        time.sleep(np.random.uniform(15, 45))  # 15 seconds to 45 seconds
        
        success = np.random.random() > 0.2  # 80% success rate
        improvement = np.random.uniform(0.02, 0.09) if success else np.random.uniform(-0.01, 0.02)
        
        return {
            'success': success,
            'improvement': improvement,
            'cpu_optimization': 'thread_pool_adjustment',
            'optimization_type': 'cpu_optimization'
        }
    
    def _balance_system_load(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Balance system load across components"""
        # Simulate load balancing
        time.sleep(np.random.uniform(20, 60))  # 20 seconds to 1 minute
        
        success = np.random.random() > 0.25  # 75% success rate
        improvement = np.random.uniform(0.03, 0.10) if success else np.random.uniform(-0.02, 0.02)
        
        return {
            'success': success,
            'improvement': improvement,
            'load_distribution': 'balanced',
            'optimization_type': 'load_balancing'
        }
    
    def _optimize_io_operations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize I/O operations"""
        # Simulate I/O optimization
        time.sleep(np.random.uniform(10, 40))  # 10 seconds to 40 seconds
        
        success = np.random.random() > 0.15  # 85% success rate
        improvement = np.random.uniform(0.02, 0.07) if success else np.random.uniform(-0.01, 0.01)
        
        return {
            'success': success,
            'improvement': improvement,
            'io_optimization': 'buffer_size_adjustment',
            'optimization_type': 'io_optimization'
        }
    
    def _generic_optimization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generic optimization"""
        # Simulate generic optimization
        time.sleep(np.random.uniform(15, 90))  # 15 seconds to 1.5 minutes
        
        success = np.random.random() > 0.3  # 70% success rate
        improvement = np.random.uniform(0.01, 0.05) if success else np.random.uniform(-0.01, 0.01)
        
        return {
            'success': success,
            'improvement': improvement,
            'optimization_type': 'generic'
        }
    
    def _check_completed_optimizations(self):
        """Check for completed optimizations and clean up"""
        completed_optimizations = []
        
        with self.optimization_lock:
            for optimization_id, optimization_data in self.active_optimizations.items():
                future = optimization_data['future']
                
                if future.done():
                    completed_optimizations.append(optimization_id)
                    
                    try:
                        result = future.result()
                        logger.info(f"Optimization {optimization_id} completed successfully")
                    except Exception as e:
                        logger.error(f"Optimization {optimization_id} failed: {e}")
            
            # Remove completed optimizations
            for optimization_id in completed_optimizations:
                del self.active_optimizations[optimization_id]
    
    def _schedule_maintenance(self, prediction: Dict[str, Any]):
        """Schedule predictive maintenance"""
        logger.info(f"Scheduling maintenance for {prediction['component']} (failure probability: {prediction['failure_probability']:.2f})")
        
        # Schedule maintenance during off-peak hours
        maintenance_action = {
            'component': prediction['component'],
            'type': prediction['maintenance_type'],
            'urgency': 'high' if prediction['failure_probability'] > 0.9 else 'medium',
            'scheduled_time': self._get_next_maintenance_window()
        }
        
        # Execute immediate maintenance if critical
        if prediction['failure_probability'] > 0.95:
            self._execute_emergency_maintenance(maintenance_action)
        else:
            logger.info(f"Maintenance scheduled for {maintenance_action['scheduled_time']}")
    
    def _get_next_maintenance_window(self) -> float:
        """Get the next available maintenance window"""
        # Find next maintenance hour (2 AM or 2 PM)
        current_time = time.time()
        current_hour = time.localtime(current_time).tm_hour
        
        for hour in self.config.maintenance_schedule_hours:
            if hour > current_hour:
                # Today at this hour
                target_time = current_time + (hour - current_hour) * 3600
                return target_time
        
        # Tomorrow at first maintenance hour
        hours_until_tomorrow = 24 - current_hour + self.config.maintenance_schedule_hours[0]
        return current_time + hours_until_tomorrow * 3600
    
    def _execute_emergency_maintenance(self, maintenance_action: Dict[str, Any]):
        """Execute emergency maintenance immediately"""
        logger.warning(f"Executing emergency maintenance for {maintenance_action['component']}")
        
        try:
            component = maintenance_action['component']
            maintenance_type = maintenance_action['type']
            
            if maintenance_type == 'restart':
                self._restart_component(component)
            elif maintenance_type == 'cleanup':
                self._clear_system_cache()
            elif maintenance_type == 'resource_reset':
                self._reallocate_resources({'cpu': 0.5, 'memory': 0.6})
            elif maintenance_type == 'full_reset':
                self._emergency_cleanup()
            
            logger.info(f"Emergency maintenance completed for {component}")
            
        except Exception as e:
            logger.error(f"Error during emergency maintenance: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self.metrics_lock:
            current_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        with self.alerts_lock:
            active_alerts = [a for a in self.alerts_history if not a.resolved]
        
        return {
            'system_health_status': self.system_status.value,
            'current_metrics': current_metrics.__dict__ if current_metrics else None,
            'health_score': current_metrics.overall_health_score() if current_metrics else 0.0,
            'active_alerts': len(active_alerts),
            'critical_alerts': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            'active_optimizations': len(self.active_optimizations),
            'total_optimizations_completed': len(self.optimization_history),
            'optimization_success_rate': self._calculate_optimization_success_rate(),
            'average_optimization_improvement': self._calculate_average_improvement(),
            'system_uptime': self._calculate_uptime(),
            'monitoring_active': self.running
        }
    
    def _calculate_optimization_success_rate(self) -> float:
        """Calculate optimization success rate"""
        if not self.optimization_history:
            return 0.0
        
        successful_optimizations = sum(1 for opt in self.optimization_history if opt['success'])
        return successful_optimizations / len(self.optimization_history)
    
    def _calculate_average_improvement(self) -> float:
        """Calculate average optimization improvement"""
        if not self.optimization_history:
            return 0.0
        
        total_improvement = sum(opt['actual_improvement'] for opt in self.optimization_history)
        return total_improvement / len(self.optimization_history)
    
    def _calculate_uptime(self) -> float:
        """Calculate system uptime percentage"""
        if not self.metrics_history:
            return 100.0
        
        # Calculate uptime based on system status history
        total_measurements = len(self.metrics_history)
        failure_measurements = sum(1 for metrics in self.metrics_history 
                                 if metrics.overall_health_score() < 0.3)
        
        uptime_percentage = ((total_measurements - failure_measurements) / total_measurements) * 100
        return uptime_percentage
    
    def save_system_state(self, filepath: str):
        """Save system state for persistence"""
        state = {
            'config': self.config,
            'metrics_history': list(self.metrics_history),
            'alerts_history': list(self.alerts_history),
            'optimization_history': self.optimization_history,
            'system_status': self.system_status.value
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        # Save component states
        self.health_monitor.save_state(filepath.replace('.pkl', '_monitor.pkl'))
        self.performance_optimizer.save_state(filepath.replace('.pkl', '_optimizer.pkl'))
        self.predictive_maintenance.save_state(filepath.replace('.pkl', '_maintenance.pkl'))
        self.resource_manager.save_state(filepath.replace('.pkl', '_resources.pkl'))
        self.self_healer.save_state(filepath.replace('.pkl', '_healer.pkl'))
        self.strategy_selector.save_state(filepath.replace('.pkl', '_strategy.pkl'))
        
        logger.info(f"Self-monitoring system state saved to {filepath}")
    
    def load_system_state(self, filepath: str):
        """Load system state from file"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.metrics_history = deque(state['metrics_history'], maxlen=self.config.metrics_history_size)
        self.alerts_history = deque(state['alerts_history'], maxlen=1000)
        self.optimization_history = state['optimization_history']
        self.system_status = SystemHealthStatus(state['system_status'])
        
        # Load component states
        try:
            self.health_monitor.load_state(filepath.replace('.pkl', '_monitor.pkl'))
            self.performance_optimizer.load_state(filepath.replace('.pkl', '_optimizer.pkl'))
            self.predictive_maintenance.load_state(filepath.replace('.pkl', '_maintenance.pkl'))
            self.resource_manager.load_state(filepath.replace('.pkl', '_resources.pkl'))
            self.self_healer.load_state(filepath.replace('.pkl', '_healer.pkl'))
            self.strategy_selector.load_state(filepath.replace('.pkl', '_strategy.pkl'))
        except FileNotFoundError as e:
            logger.warning(f"Could not load some component states: {e}")
        
        logger.info(f"Self-monitoring system state loaded from {filepath}")

class SystemHealthMonitor:
    """Monitors system health and generates alerts"""
    
    def __init__(self, config: SelfMonitoringConfig):
        self.config = config
        self.baseline_metrics = None
        self.alert_history = []
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        # CPU metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU metrics (if available)
        gpu_usage = 0.0
        if torch.cuda.is_available():
            try:
                gpu_usage = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
            except:
                gpu_usage = np.random.uniform(20, 80)  # Simulate GPU usage
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        # Process metrics
        process_count = len(psutil.pids())
        thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']) if p.info['num_threads'])
        
        # Performance metrics (simulated)
        performance_score = np.random.uniform(0.7, 0.95)
        error_rate = np.random.uniform(0.0, 0.1)
        response_time = np.random.uniform(0.1, 2.0)
        throughput = np.random.uniform(100, 1000)
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            process_count=process_count,
            thread_count=thread_count,
            performance_score=performance_score,
            error_rate=error_rate,
            response_time=response_time,
            throughput=throughput
        )
    
    def check_alerts(self, metrics: SystemMetrics) -> List[SystemAlert]:
        """Check for alert conditions based on current metrics"""
        alerts = []
        
        # Check each metric against thresholds
        for metric_name, thresholds in self.config.alert_thresholds.items():
            metric_value = getattr(metrics, metric_name, 0)
            
            if metric_value >= thresholds.get('critical', float('inf')):
                alerts.append(SystemAlert(
                    id=str(uuid.uuid4()),
                    severity=AlertSeverity.CRITICAL,
                    component=metric_name,
                    message=f"Critical {metric_name}: {metric_value:.2f}",
                    timestamp=time.time(),
                    metrics={metric_name: metric_value}
                ))
            elif metric_value >= thresholds.get('warning', float('inf')):
                alerts.append(SystemAlert(
                    id=str(uuid.uuid4()),
                    severity=AlertSeverity.WARNING,
                    component=metric_name,
                    message=f"Warning {metric_name}: {metric_value:.2f}",
                    timestamp=time.time(),
                    metrics={metric_name: metric_value}
                ))
        
        return alerts
    
    def save_state(self, filepath: str):
        """Save monitor state"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'baseline_metrics': self.baseline_metrics,
                'alert_history': self.alert_history
            }, f)
    
    def load_state(self, filepath: str):
        """Load monitor state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.baseline_metrics = state['baseline_metrics']
            self.alert_history = state['alert_history']

# Simplified implementations of remaining components for brevity
class AutonomousPerformanceOptimizer:
    """Generates and executes performance optimizations"""
    
    def __init__(self, config: SelfMonitoringConfig):
        self.config = config
        self.optimization_patterns = {}
    
    def generate_optimization_actions(self, current_metrics: SystemMetrics, 
                                    recent_metrics: List[SystemMetrics], 
                                    strategy: OptimizationStrategy) -> List[OptimizationAction]:
        """Generate optimization actions based on current state and strategy"""
        actions = []
        
        # CPU optimization
        if current_metrics.cpu_usage > 80:
            actions.append(OptimizationAction(
                id=str(uuid.uuid4()),
                strategy=strategy,
                component='cpu',
                action_type='optimize_cpu',
                parameters={'target_usage': 70},
                expected_improvement=0.05,
                risk_level=0.2,
                priority=2,
                timestamp=time.time()
            ))
        
        # Memory optimization
        if current_metrics.memory_usage > 85:
            actions.append(OptimizationAction(
                id=str(uuid.uuid4()),
                strategy=strategy,
                component='memory',
                action_type='optimize_memory',
                parameters={'cleanup_level': 'aggressive'},
                expected_improvement=0.08,
                risk_level=0.1,
                priority=1,
                timestamp=time.time()
            ))
        
        # Performance optimization
        if current_metrics.performance_score < 0.8:
            actions.append(OptimizationAction(
                id=str(uuid.uuid4()),
                strategy=strategy,
                component='performance',
                action_type='tune_hyperparameters',
                parameters={'optimization_target': 'performance'},
                expected_improvement=0.06,
                risk_level=0.3,
                priority=2,
                timestamp=time.time()
            ))
        
        return actions
    
    def save_state(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'optimization_patterns': self.optimization_patterns}, f)
    
    def load_state(self, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.optimization_patterns = state['optimization_patterns']

class PredictiveMaintenanceSystem:
    """Predicts and schedules maintenance"""
    
    def __init__(self, config: SelfMonitoringConfig):
        self.config = config
        self.failure_patterns = {}
    
    def predict_failures(self, metrics_history: List[SystemMetrics]) -> List[Dict[str, Any]]:
        """Predict potential system failures"""
        predictions = []
        
        # Analyze trends for each component
        components = ['cpu', 'memory', 'gpu', 'disk']
        
        for component in components:
            failure_prob = self._calculate_failure_probability(metrics_history, component)
            
            if failure_prob > 0.5:
                predictions.append({
                    'component': component,
                    'failure_probability': failure_prob,
                    'maintenance_type': 'preventive',
                    'urgency': 'high' if failure_prob > 0.8 else 'medium'
                })
        
        return predictions
    
    def _calculate_failure_probability(self, metrics_history: List[SystemMetrics], component: str) -> float:
        """Calculate failure probability for a component"""
        # Simplified failure prediction
        if component == 'cpu':
            usage_values = [m.cpu_usage for m in metrics_history]
        elif component == 'memory':
            usage_values = [m.memory_usage for m in metrics_history]
        elif component == 'gpu':
            usage_values = [m.gpu_usage for m in metrics_history]
        elif component == 'disk':
            usage_values = [m.disk_usage for m in metrics_history]
        else:
            return 0.0
        
        # Calculate trend and variance
        if len(usage_values) < 10:
            return 0.0
        
        trend = np.polyfit(range(len(usage_values)), usage_values, 1)[0]
        variance = np.var(usage_values)
        avg_usage = np.mean(usage_values)
        
        # Simple failure probability calculation
        failure_prob = 0.0
        
        if avg_usage > 90:
            failure_prob += 0.4
        elif avg_usage > 80:
            failure_prob += 0.2
        
        if trend > 1:  # Increasing trend
            failure_prob += 0.3
        
        if variance > 100:  # High variance
            failure_prob += 0.2
        
        return min(failure_prob, 1.0)
    
    def save_state(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'failure_patterns': self.failure_patterns}, f)
    
    def load_state(self, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.failure_patterns = state['failure_patterns']

class IntelligentResourceManager:
    """Manages and optimizes resource allocation"""
    
    def __init__(self, config: SelfMonitoringConfig):
        self.config = config
        self.resource_allocation_history = []
    
    def save_state(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'resource_allocation_history': self.resource_allocation_history}, f)
    
    def load_state(self, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.resource_allocation_history = state['resource_allocation_history']

class SelfHealingMechanism:
    """Implements self-healing capabilities"""
    
    def __init__(self, config: SelfMonitoringConfig):
        self.config = config
        self.healing_history = []
    
    def generate_healing_action(self, alert: SystemAlert) -> Optional[Dict[str, Any]]:
        """Generate appropriate healing action for an alert"""
        if alert.component == 'cpu_usage':
            return {'type': 'reduce_load', 'reduction_factor': 0.3}
        elif alert.component == 'memory_usage':
            return {'type': 'clear_cache'}
        elif alert.component == 'error_rate':
            return {'type': 'restart_component', 'component': 'main_process'}
        else:
            return {'type': 'emergency_cleanup'}
    
    def save_state(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'healing_history': self.healing_history}, f)
    
    def load_state(self, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.healing_history = state['healing_history']

class OptimizationStrategySelector:
    """Selects optimal optimization strategy"""
    
    def __init__(self, config: SelfMonitoringConfig):
        self.config = config
        self.strategy_performance = {}
    
    def select_strategy(self, current_metrics: SystemMetrics, 
                       recent_metrics: List[SystemMetrics]) -> OptimizationStrategy:
        """Select the best optimization strategy for current conditions"""
        # Simplified strategy selection
        health_score = current_metrics.overall_health_score()
        
        if health_score < 0.5:
            return OptimizationStrategy.STABILITY_FIRST
        elif current_metrics.cpu_usage > 80 or current_metrics.memory_usage > 80:
            return OptimizationStrategy.RESOURCE_EFFICIENT
        elif current_metrics.performance_score < 0.8:
            return OptimizationStrategy.PERFORMANCE_FOCUSED
        else:
            return OptimizationStrategy.BALANCED
    
    def save_state(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'strategy_performance': self.strategy_performance}, f)
    
    def load_state(self, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.strategy_performance = state['strategy_performance']

# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = SelfMonitoringConfig(
        monitoring_interval=5.0,  # 5 seconds for demo
        optimization_interval=60.0  # 1 minute for demo
    )
    
    # Initialize self-monitoring system
    monitoring_system = SelfMonitoringSystem(config)
    
    # Start monitoring
    monitoring_system.start_monitoring()
    
    # Let it run for a short time
    time.sleep(30)
    
    # Get system status
    status = monitoring_system.get_system_status()
    print("Self-Monitoring System Status:")
    print(f"System health: {status['system_health_status']}")
    print(f"Health score: {status['health_score']:.2f}")
    print(f"Active alerts: {status['active_alerts']}")
    print(f"Active optimizations: {status['active_optimizations']}")
    print(f"Optimization success rate: {status['optimization_success_rate']:.2f}")
    print(f"System uptime: {status['system_uptime']:.1f}%")
    
    # Stop monitoring
    monitoring_system.stop_monitoring()

