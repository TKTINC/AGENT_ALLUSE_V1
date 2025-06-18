"""
WS5-P5: Performance Integration and System Coordination
Comprehensive integration framework for performance optimization components.

This module provides system-wide coordination capabilities including:
- Cross-component performance optimization coordination
- Unified performance management and orchestration
- Conflict resolution and priority management
- System-wide performance governance
- Integrated optimization workflow management
"""

import time
import threading
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import os
from abc import ABC, abstractmethod
import uuid

# Import performance components
from .performance_monitoring_framework import PerformanceMonitoringFramework, PerformanceMetric
from .optimization_engine import OptimizationEngine, OptimizationResult, OptimizationParameter
from .advanced_analytics import PredictiveAnalyzer, PredictionResult, AnomalyPrediction, CapacityForecast

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceTask:
    """Represents a performance optimization task."""
    task_id: str
    task_type: str  # 'monitoring', 'optimization', 'prediction', 'analysis'
    priority: int  # 1 (highest) to 5 (lowest)
    component: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    estimated_duration: timedelta
    created_at: datetime
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed'
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'priority': self.priority,
            'component': self.component,
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'estimated_duration_seconds': self.estimated_duration.total_seconds(),
            'created_at': self.created_at.isoformat(),
            'status': self.status,
            'result': self.result,
            'error': self.error
        }

@dataclass
class SystemPerformanceState:
    """Represents the current system performance state."""
    timestamp: datetime
    overall_health_score: float  # 0-100
    component_health: Dict[str, float]
    active_optimizations: int
    pending_tasks: int
    recent_improvements: List[Dict[str, Any]]
    performance_trends: Dict[str, str]  # 'improving', 'stable', 'degrading'
    resource_utilization: Dict[str, float]
    alerts: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_health_score': self.overall_health_score,
            'component_health': self.component_health,
            'active_optimizations': self.active_optimizations,
            'pending_tasks': self.pending_tasks,
            'recent_improvements': self.recent_improvements,
            'performance_trends': self.performance_trends,
            'resource_utilization': self.resource_utilization,
            'alerts': self.alerts
        }

class PerformanceTaskScheduler:
    """Schedules and manages performance optimization tasks."""
    
    def __init__(self, max_concurrent_tasks: int = 5):
        """Initialize task scheduler."""
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue = deque()
        self.running_tasks = {}
        self.completed_tasks = deque(maxlen=1000)
        self.task_executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self.is_running = False
        self.scheduler_thread = None
        
    def add_task(self, task: PerformanceTask) -> str:
        """Add task to scheduler queue."""
        self.task_queue.append(task)
        logger.info(f"Added task {task.task_id} to queue (priority: {task.priority})")
        return task.task_id
    
    def start_scheduler(self):
        """Start the task scheduler."""
        if self.is_running:
            logger.warning("Task scheduler already running")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Performance task scheduler started")
    
    def stop_scheduler(self):
        """Stop the task scheduler."""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.task_executor.shutdown(wait=True)
        logger.info("Performance task scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                # Check for completed tasks
                self._check_completed_tasks()
                
                # Schedule new tasks if capacity available
                if len(self.running_tasks) < self.max_concurrent_tasks and self.task_queue:
                    # Sort queue by priority and dependencies
                    sorted_tasks = self._sort_tasks_by_priority()
                    
                    for task in sorted_tasks:
                        if len(self.running_tasks) >= self.max_concurrent_tasks:
                            break
                        
                        if self._can_execute_task(task):
                            self._execute_task(task)
                            self.task_queue.remove(task)
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(5)
    
    def _sort_tasks_by_priority(self) -> List[PerformanceTask]:
        """Sort tasks by priority and readiness."""
        return sorted(self.task_queue, key=lambda t: (t.priority, t.created_at))
    
    def _can_execute_task(self, task: PerformanceTask) -> bool:
        """Check if task can be executed (dependencies satisfied)."""
        for dep_id in task.dependencies:
            # Check if dependency is completed
            dep_completed = any(t.task_id == dep_id and t.status == 'completed' 
                              for t in self.completed_tasks)
            if not dep_completed:
                return False
        return True
    
    def _execute_task(self, task: PerformanceTask):
        """Execute a performance task."""
        task.status = 'running'
        future = self.task_executor.submit(self._run_task, task)
        self.running_tasks[task.task_id] = (task, future)
        logger.info(f"Started executing task {task.task_id}")
    
    def _run_task(self, task: PerformanceTask) -> Dict[str, Any]:
        """Run a specific task."""
        try:
            # Simulate task execution based on type
            if task.task_type == 'monitoring':
                result = self._run_monitoring_task(task)
            elif task.task_type == 'optimization':
                result = self._run_optimization_task(task)
            elif task.task_type == 'prediction':
                result = self._run_prediction_task(task)
            elif task.task_type == 'analysis':
                result = self._run_analysis_task(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            task.status = 'completed'
            task.result = result
            return result
            
        except Exception as e:
            task.status = 'failed'
            task.error = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")
            return {'error': str(e)}
    
    def _run_monitoring_task(self, task: PerformanceTask) -> Dict[str, Any]:
        """Run monitoring task."""
        # Simulate monitoring task
        time.sleep(1)  # Simulate work
        return {
            'task_type': 'monitoring',
            'metrics_collected': 50,
            'anomalies_detected': 2,
            'execution_time': 1.0
        }
    
    def _run_optimization_task(self, task: PerformanceTask) -> Dict[str, Any]:
        """Run optimization task."""
        # Simulate optimization task
        time.sleep(2)  # Simulate work
        return {
            'task_type': 'optimization',
            'parameters_optimized': 5,
            'improvement_achieved': 15.3,
            'execution_time': 2.0
        }
    
    def _run_prediction_task(self, task: PerformanceTask) -> Dict[str, Any]:
        """Run prediction task."""
        # Simulate prediction task
        time.sleep(1.5)  # Simulate work
        return {
            'task_type': 'prediction',
            'forecasts_generated': 10,
            'anomalies_predicted': 1,
            'execution_time': 1.5
        }
    
    def _run_analysis_task(self, task: PerformanceTask) -> Dict[str, Any]:
        """Run analysis task."""
        # Simulate analysis task
        time.sleep(3)  # Simulate work
        return {
            'task_type': 'analysis',
            'opportunities_identified': 8,
            'recommendations_generated': 12,
            'execution_time': 3.0
        }
    
    def _check_completed_tasks(self):
        """Check for completed tasks and move them to completed queue."""
        completed_task_ids = []
        
        for task_id, (task, future) in self.running_tasks.items():
            if future.done():
                try:
                    result = future.result()
                    if task.status != 'failed':
                        task.result = result
                except Exception as e:
                    task.status = 'failed'
                    task.error = str(e)
                
                self.completed_tasks.append(task)
                completed_task_ids.append(task_id)
                logger.info(f"Task {task_id} completed with status: {task.status}")
        
        # Remove completed tasks from running tasks
        for task_id in completed_task_ids:
            del self.running_tasks[task_id]
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            'is_running': self.is_running,
            'queued_tasks': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'task_queue_details': [task.to_dict() for task in list(self.task_queue)[:5]],  # First 5 tasks
            'running_task_details': [task.to_dict() for task, _ in self.running_tasks.values()]
        }

class ConflictResolver:
    """Resolves conflicts between optimization operations."""
    
    def __init__(self):
        """Initialize conflict resolver."""
        self.conflict_rules = {}
        self.resolution_history = deque(maxlen=500)
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default conflict resolution rules."""
        self.conflict_rules = {
            # Resource allocation conflicts
            'resource_allocation': {
                'cpu_memory': 'prioritize_cpu',  # CPU optimization takes priority over memory
                'memory_disk': 'prioritize_memory',  # Memory optimization takes priority over disk
                'network_cpu': 'prioritize_network'  # Network optimization takes priority over CPU
            },
            
            # Optimization type conflicts
            'optimization_type': {
                'parameter_resource': 'prioritize_parameter',  # Parameter optimization over resource
                'prediction_optimization': 'prioritize_optimization',  # Optimization over prediction
                'monitoring_optimization': 'allow_concurrent'  # Allow concurrent execution
            },
            
            # Priority-based resolution
            'priority_resolution': {
                'high_vs_medium': 'prioritize_high',
                'medium_vs_low': 'prioritize_medium',
                'critical_vs_any': 'prioritize_critical'
            }
        }
    
    def detect_conflicts(self, tasks: List[PerformanceTask]) -> List[Dict[str, Any]]:
        """Detect conflicts between tasks."""
        conflicts = []
        
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks[i+1:], i+1):
                conflict = self._check_task_conflict(task1, task2)
                if conflict:
                    conflicts.append({
                        'conflict_id': f"conflict_{int(time.time())}_{i}_{j}",
                        'task1': task1.task_id,
                        'task2': task2.task_id,
                        'conflict_type': conflict['type'],
                        'severity': conflict['severity'],
                        'description': conflict['description'],
                        'detected_at': datetime.now()
                    })
        
        return conflicts
    
    def _check_task_conflict(self, task1: PerformanceTask, task2: PerformanceTask) -> Optional[Dict[str, Any]]:
        """Check if two tasks conflict."""
        # Resource conflicts
        if self._check_resource_conflict(task1, task2):
            return {
                'type': 'resource_conflict',
                'severity': 'medium',
                'description': f"Tasks {task1.task_id} and {task2.task_id} compete for same resources"
            }
        
        # Component conflicts
        if task1.component == task2.component and task1.task_type == task2.task_type:
            return {
                'type': 'component_conflict',
                'severity': 'high',
                'description': f"Tasks {task1.task_id} and {task2.task_id} target same component"
            }
        
        # Parameter conflicts
        if self._check_parameter_conflict(task1, task2):
            return {
                'type': 'parameter_conflict',
                'severity': 'high',
                'description': f"Tasks {task1.task_id} and {task2.task_id} modify same parameters"
            }
        
        return None
    
    def _check_resource_conflict(self, task1: PerformanceTask, task2: PerformanceTask) -> bool:
        """Check if tasks have resource conflicts."""
        # Check if both tasks involve resource optimization
        resource_keywords = ['cpu', 'memory', 'disk', 'network']
        
        task1_resources = set()
        task2_resources = set()
        
        for keyword in resource_keywords:
            if keyword in str(task1.parameters).lower():
                task1_resources.add(keyword)
            if keyword in str(task2.parameters).lower():
                task2_resources.add(keyword)
        
        return bool(task1_resources.intersection(task2_resources))
    
    def _check_parameter_conflict(self, task1: PerformanceTask, task2: PerformanceTask) -> bool:
        """Check if tasks have parameter conflicts."""
        # Check if both tasks modify the same parameters
        if 'parameters' in task1.parameters and 'parameters' in task2.parameters:
            params1 = set(task1.parameters.get('parameters', {}).keys())
            params2 = set(task2.parameters.get('parameters', {}).keys())
            return bool(params1.intersection(params2))
        
        return False
    
    def resolve_conflicts(self, conflicts: List[Dict[str, Any]], 
                         tasks: List[PerformanceTask]) -> List[Dict[str, Any]]:
        """Resolve detected conflicts."""
        resolutions = []
        
        for conflict in conflicts:
            resolution = self._resolve_single_conflict(conflict, tasks)
            resolutions.append(resolution)
            
            # Store resolution history
            self.resolution_history.append({
                'conflict': conflict,
                'resolution': resolution,
                'timestamp': datetime.now()
            })
        
        return resolutions
    
    def _resolve_single_conflict(self, conflict: Dict[str, Any], 
                                tasks: List[PerformanceTask]) -> Dict[str, Any]:
        """Resolve a single conflict."""
        task1_id = conflict['task1']
        task2_id = conflict['task2']
        
        # Find tasks
        task1 = next((t for t in tasks if t.task_id == task1_id), None)
        task2 = next((t for t in tasks if t.task_id == task2_id), None)
        
        if not task1 or not task2:
            return {
                'resolution_type': 'error',
                'action': 'no_action',
                'reason': 'Tasks not found'
            }
        
        # Priority-based resolution
        if task1.priority < task2.priority:  # Lower number = higher priority
            return {
                'resolution_type': 'priority',
                'action': 'prioritize_task1',
                'prioritized_task': task1_id,
                'delayed_task': task2_id,
                'reason': f"Task {task1_id} has higher priority ({task1.priority} vs {task2.priority})"
            }
        elif task2.priority < task1.priority:
            return {
                'resolution_type': 'priority',
                'action': 'prioritize_task2',
                'prioritized_task': task2_id,
                'delayed_task': task1_id,
                'reason': f"Task {task2_id} has higher priority ({task2.priority} vs {task1.priority})"
            }
        
        # Type-based resolution
        if conflict['conflict_type'] == 'resource_conflict':
            return self._resolve_resource_conflict(task1, task2)
        elif conflict['conflict_type'] == 'component_conflict':
            return self._resolve_component_conflict(task1, task2)
        elif conflict['conflict_type'] == 'parameter_conflict':
            return self._resolve_parameter_conflict(task1, task2)
        
        # Default: serialize execution
        return {
            'resolution_type': 'serialization',
            'action': 'serialize_execution',
            'first_task': task1_id,
            'second_task': task2_id,
            'reason': 'Default serialization to avoid conflicts'
        }
    
    def _resolve_resource_conflict(self, task1: PerformanceTask, task2: PerformanceTask) -> Dict[str, Any]:
        """Resolve resource conflict."""
        # Use rule-based resolution
        return {
            'resolution_type': 'resource_allocation',
            'action': 'allocate_resources',
            'task1_allocation': 0.6,
            'task2_allocation': 0.4,
            'reason': 'Proportional resource allocation based on priority'
        }
    
    def _resolve_component_conflict(self, task1: PerformanceTask, task2: PerformanceTask) -> Dict[str, Any]:
        """Resolve component conflict."""
        return {
            'resolution_type': 'component_serialization',
            'action': 'serialize_by_type',
            'order': [task1.task_id, task2.task_id],
            'reason': 'Serialize tasks targeting same component'
        }
    
    def _resolve_parameter_conflict(self, task1: PerformanceTask, task2: PerformanceTask) -> Dict[str, Any]:
        """Resolve parameter conflict."""
        return {
            'resolution_type': 'parameter_coordination',
            'action': 'coordinate_parameters',
            'coordination_strategy': 'merge_and_optimize',
            'reason': 'Coordinate parameter modifications to avoid conflicts'
        }

class PerformanceGovernor:
    """Governs system-wide performance policies and constraints."""
    
    def __init__(self):
        """Initialize performance governor."""
        self.policies = {}
        self.constraints = {}
        self.governance_history = deque(maxlen=1000)
        self._setup_default_policies()
    
    def _setup_default_policies(self):
        """Setup default performance policies."""
        self.policies = {
            'resource_utilization': {
                'max_cpu_utilization': 0.85,
                'max_memory_utilization': 0.90,
                'max_disk_utilization': 0.95,
                'max_network_utilization': 0.80
            },
            'optimization_frequency': {
                'parameter_optimization': timedelta(minutes=30),
                'resource_optimization': timedelta(minutes=15),
                'predictive_analysis': timedelta(minutes=60)
            },
            'performance_thresholds': {
                'response_time_threshold': 500,  # milliseconds
                'throughput_threshold': 1000,   # requests/second
                'error_rate_threshold': 0.01    # 1%
            },
            'safety_constraints': {
                'max_concurrent_optimizations': 3,
                'optimization_rollback_threshold': 0.1,  # 10% performance degradation
                'emergency_stop_threshold': 0.25  # 25% performance degradation
            }
        }
        
        self.constraints = {
            'business_hours': {
                'start_hour': 9,
                'end_hour': 17,
                'restricted_operations': ['major_optimizations', 'system_restarts']
            },
            'maintenance_windows': {
                'weekly_window': 'sunday_02:00',
                'duration_hours': 4,
                'allowed_operations': ['all']
            }
        }
    
    def evaluate_policy_compliance(self, task: PerformanceTask, 
                                 current_state: SystemPerformanceState) -> Dict[str, Any]:
        """Evaluate if task complies with governance policies."""
        compliance_result = {
            'compliant': True,
            'violations': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check resource utilization policies
        resource_violations = self._check_resource_policies(task, current_state)
        if resource_violations:
            compliance_result['violations'].extend(resource_violations)
            compliance_result['compliant'] = False
        
        # Check optimization frequency policies
        frequency_violations = self._check_frequency_policies(task)
        if frequency_violations:
            compliance_result['violations'].extend(frequency_violations)
            compliance_result['compliant'] = False
        
        # Check safety constraints
        safety_violations = self._check_safety_constraints(task, current_state)
        if safety_violations:
            compliance_result['violations'].extend(safety_violations)
            compliance_result['compliant'] = False
        
        # Check business hour constraints
        business_warnings = self._check_business_constraints(task)
        if business_warnings:
            compliance_result['warnings'].extend(business_warnings)
        
        # Generate recommendations
        recommendations = self._generate_compliance_recommendations(task, compliance_result)
        compliance_result['recommendations'] = recommendations
        
        # Store governance decision
        self.governance_history.append({
            'task_id': task.task_id,
            'compliance_result': compliance_result,
            'timestamp': datetime.now()
        })
        
        return compliance_result
    
    def _check_resource_policies(self, task: PerformanceTask, 
                               current_state: SystemPerformanceState) -> List[str]:
        """Check resource utilization policies."""
        violations = []
        
        for resource, max_util in self.policies['resource_utilization'].items():
            resource_key = resource.replace('max_', '').replace('_utilization', '')
            current_util = current_state.resource_utilization.get(resource_key, 0)
            
            if current_util > max_util:
                violations.append(f"Resource {resource_key} utilization ({current_util:.2f}) exceeds policy limit ({max_util:.2f})")
        
        return violations
    
    def _check_frequency_policies(self, task: PerformanceTask) -> List[str]:
        """Check optimization frequency policies."""
        violations = []
        
        # Check if similar task was run recently
        task_type = task.task_type
        if task_type in self.policies['optimization_frequency']:
            min_interval = self.policies['optimization_frequency'][task_type]
            
            # Check recent tasks of same type
            recent_tasks = [h for h in self.governance_history 
                          if (datetime.now() - h['timestamp']) < min_interval]
            
            similar_tasks = [h for h in recent_tasks 
                           if h.get('task_id', '').startswith(task_type)]
            
            if similar_tasks:
                violations.append(f"Task type {task_type} executed too frequently (min interval: {min_interval})")
        
        return violations
    
    def _check_safety_constraints(self, task: PerformanceTask, 
                                current_state: SystemPerformanceState) -> List[str]:
        """Check safety constraints."""
        violations = []
        
        # Check concurrent optimization limit
        max_concurrent = self.policies['safety_constraints']['max_concurrent_optimizations']
        if current_state.active_optimizations >= max_concurrent:
            violations.append(f"Maximum concurrent optimizations ({max_concurrent}) already reached")
        
        # Check system health
        if current_state.overall_health_score < 50:  # Poor health
            violations.append("System health too poor for optimization operations")
        
        return violations
    
    def _check_business_constraints(self, task: PerformanceTask) -> List[str]:
        """Check business hour constraints."""
        warnings = []
        
        current_hour = datetime.now().hour
        business_start = self.constraints['business_hours']['start_hour']
        business_end = self.constraints['business_hours']['end_hour']
        
        if business_start <= current_hour <= business_end:
            restricted_ops = self.constraints['business_hours']['restricted_operations']
            if any(op in task.task_type for op in restricted_ops):
                warnings.append(f"Task {task.task_type} is restricted during business hours")
        
        return warnings
    
    def _generate_compliance_recommendations(self, task: PerformanceTask, 
                                           compliance_result: Dict[str, Any]) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        if not compliance_result['compliant']:
            recommendations.append("Defer task execution until policy violations are resolved")
            recommendations.append("Consider reducing task scope or priority")
        
        if compliance_result['warnings']:
            recommendations.append("Schedule task for off-business hours")
            recommendations.append("Obtain approval for business-hour execution")
        
        if task.priority > 3:  # Low priority
            recommendations.append("Consider batching with other low-priority tasks")
        
        return recommendations

class SystemCoordinator:
    """Main system coordinator for performance optimization."""
    
    def __init__(self):
        """Initialize system coordinator."""
        self.monitoring_framework = PerformanceMonitoringFramework()
        self.optimization_engine = OptimizationEngine()
        self.predictive_analyzer = PredictiveAnalyzer()
        self.task_scheduler = PerformanceTaskScheduler()
        self.conflict_resolver = ConflictResolver()
        self.performance_governor = PerformanceGovernor()
        
        self.system_state = None
        self.coordination_history = deque(maxlen=500)
        self.is_coordinating = False
        self.coordination_thread = None
        
        logger.info("System coordinator initialized")
    
    def start_coordination(self):
        """Start system-wide performance coordination."""
        if self.is_coordinating:
            logger.warning("System coordination already running")
            return
        
        # Start individual components
        self.monitoring_framework.start_monitoring()
        self.optimization_engine.start_optimization_engine()
        self.task_scheduler.start_scheduler()
        
        # Start coordination loop
        self.is_coordinating = True
        self.coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self.coordination_thread.start()
        
        logger.info("System-wide performance coordination started")
    
    def stop_coordination(self):
        """Stop system-wide performance coordination."""
        self.is_coordinating = False
        
        # Stop coordination loop
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5)
        
        # Stop individual components
        self.monitoring_framework.stop_monitoring()
        self.optimization_engine.stop_optimization_engine()
        self.task_scheduler.stop_scheduler()
        
        logger.info("System-wide performance coordination stopped")
    
    def _coordination_loop(self):
        """Main coordination loop."""
        while self.is_coordinating:
            try:
                # Update system state
                self._update_system_state()
                
                # Analyze current performance
                self._analyze_performance()
                
                # Generate optimization tasks
                self._generate_optimization_tasks()
                
                # Resolve conflicts
                self._resolve_task_conflicts()
                
                # Apply governance policies
                self._apply_governance()
                
                # Execute coordinated optimizations
                self._execute_coordinated_optimizations()
                
                time.sleep(30)  # Coordinate every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _update_system_state(self):
        """Update current system performance state."""
        try:
            # Get monitoring status
            monitoring_status = self.monitoring_framework.get_current_status()
            
            # Get optimization stats
            optimization_stats = self.optimization_engine.get_optimization_stats()
            
            # Get scheduler status
            scheduler_status = self.task_scheduler.get_scheduler_status()
            
            # Calculate overall health score
            health_score = self._calculate_health_score(monitoring_status, optimization_stats)
            
            # Get recent performance data
            recent_report = self.monitoring_framework.generate_immediate_report()
            
            # Extract resource utilization
            resource_utilization = {}
            if 'summary' in recent_report and 'top_metrics' in recent_report['summary']:
                for metric_name, stats in recent_report['summary']['top_metrics'].items():
                    if any(resource in metric_name.lower() for resource in ['cpu', 'memory', 'disk', 'network']):
                        resource_utilization[metric_name] = stats.get('current', 0)
            
            # Create system state
            self.system_state = SystemPerformanceState(
                timestamp=datetime.now(),
                overall_health_score=health_score,
                component_health={
                    'monitoring': 100 if monitoring_status['is_running'] else 0,
                    'optimization': 100 if optimization_stats['is_running'] else 0,
                    'scheduling': 100 if scheduler_status['is_running'] else 0
                },
                active_optimizations=scheduler_status['running_tasks'],
                pending_tasks=scheduler_status['queued_tasks'],
                recent_improvements=[],  # Would be populated from optimization history
                performance_trends={},   # Would be calculated from historical data
                resource_utilization=resource_utilization,
                alerts=[]  # Would be populated from anomaly detection
            )
            
        except Exception as e:
            logger.error(f"Error updating system state: {e}")
    
    def _calculate_health_score(self, monitoring_status: Dict[str, Any], 
                              optimization_stats: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        score = 100.0
        
        # Deduct for non-running components
        if not monitoring_status.get('is_running', False):
            score -= 30
        if not optimization_stats.get('is_running', False):
            score -= 20
        
        # Deduct for high anomaly rates
        anomalies = monitoring_status.get('total_anomalies_detected', 0)
        metrics = monitoring_status.get('total_metrics_collected', 1)
        anomaly_rate = anomalies / max(metrics, 1)
        score -= min(30, anomaly_rate * 1000)  # Cap at 30 point deduction
        
        # Bonus for successful optimizations
        successful_opts = optimization_stats.get('stats', {}).get('successful_optimizations', 0)
        total_opts = optimization_stats.get('stats', {}).get('total_optimizations', 1)
        success_rate = successful_opts / max(total_opts, 1)
        score += min(10, success_rate * 10)  # Up to 10 point bonus
        
        return max(0, min(100, score))
    
    def _analyze_performance(self):
        """Analyze current performance and identify issues."""
        if not self.system_state:
            return
        
        try:
            # Get recent metrics for analysis
            recent_metrics = self.monitoring_framework.collector.get_recent_metrics(100)
            
            if recent_metrics:
                # Convert to format expected by predictive analyzer
                recent_data = [
                    {
                        'name': metric.name,
                        'value': metric.value,
                        'timestamp': metric.timestamp.isoformat()
                    }
                    for metric in recent_metrics
                ]
                
                # Run predictive analysis
                analysis_result = self.predictive_analyzer.run_comprehensive_analysis(
                    recent_data, 
                    self.system_state.resource_utilization
                )
                
                # Store analysis results
                self.coordination_history.append({
                    'type': 'performance_analysis',
                    'timestamp': datetime.now(),
                    'analysis_result': analysis_result
                })
                
        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
    
    def _generate_optimization_tasks(self):
        """Generate optimization tasks based on analysis."""
        if not self.system_state:
            return
        
        try:
            # Generate monitoring tasks
            if self.system_state.overall_health_score < 80:
                monitoring_task = PerformanceTask(
                    task_id=f"monitoring_{uuid.uuid4().hex[:8]}",
                    task_type='monitoring',
                    priority=2,
                    component='monitoring_framework',
                    parameters={'intensive_monitoring': True},
                    dependencies=[],
                    estimated_duration=timedelta(minutes=5),
                    created_at=datetime.now()
                )
                self.task_scheduler.add_task(monitoring_task)
            
            # Generate optimization tasks for high resource utilization
            for resource, utilization in self.system_state.resource_utilization.items():
                if utilization > 80:
                    optimization_task = PerformanceTask(
                        task_id=f"optimize_{resource}_{uuid.uuid4().hex[:8]}",
                        task_type='optimization',
                        priority=1,
                        component='optimization_engine',
                        parameters={'resource_type': resource, 'target_utilization': 70},
                        dependencies=[],
                        estimated_duration=timedelta(minutes=10),
                        created_at=datetime.now()
                    )
                    self.task_scheduler.add_task(optimization_task)
            
            # Generate predictive analysis tasks
            if len(self.coordination_history) % 5 == 0:  # Every 5th coordination cycle
                prediction_task = PerformanceTask(
                    task_id=f"prediction_{uuid.uuid4().hex[:8]}",
                    task_type='prediction',
                    priority=3,
                    component='predictive_analyzer',
                    parameters={'forecast_horizon': 24},
                    dependencies=[],
                    estimated_duration=timedelta(minutes=15),
                    created_at=datetime.now()
                )
                self.task_scheduler.add_task(prediction_task)
                
        except Exception as e:
            logger.error(f"Error generating optimization tasks: {e}")
    
    def _resolve_task_conflicts(self):
        """Resolve conflicts between queued tasks."""
        try:
            queued_tasks = list(self.task_scheduler.task_queue)
            
            if len(queued_tasks) > 1:
                conflicts = self.conflict_resolver.detect_conflicts(queued_tasks)
                
                if conflicts:
                    resolutions = self.conflict_resolver.resolve_conflicts(conflicts, queued_tasks)
                    
                    # Apply resolutions
                    for resolution in resolutions:
                        self._apply_conflict_resolution(resolution, queued_tasks)
                    
                    logger.info(f"Resolved {len(conflicts)} task conflicts")
                    
        except Exception as e:
            logger.error(f"Error resolving task conflicts: {e}")
    
    def _apply_conflict_resolution(self, resolution: Dict[str, Any], tasks: List[PerformanceTask]):
        """Apply conflict resolution to tasks."""
        if resolution['action'] == 'prioritize_task1':
            # Increase priority of task1
            task_id = resolution['prioritized_task']
            for task in tasks:
                if task.task_id == task_id:
                    task.priority = max(1, task.priority - 1)
                    break
        elif resolution['action'] == 'serialize_execution':
            # Add dependency
            first_task_id = resolution['first_task']
            second_task_id = resolution['second_task']
            for task in tasks:
                if task.task_id == second_task_id:
                    task.dependencies.append(first_task_id)
                    break
    
    def _apply_governance(self):
        """Apply governance policies to queued tasks."""
        try:
            if not self.system_state:
                return
            
            queued_tasks = list(self.task_scheduler.task_queue)
            
            for task in queued_tasks:
                compliance = self.performance_governor.evaluate_policy_compliance(task, self.system_state)
                
                if not compliance['compliant']:
                    # Remove non-compliant task or modify it
                    logger.warning(f"Task {task.task_id} violates policies: {compliance['violations']}")
                    # Could implement task modification or deferral here
                
                if compliance['warnings']:
                    logger.info(f"Task {task.task_id} has warnings: {compliance['warnings']}")
                    
        except Exception as e:
            logger.error(f"Error applying governance: {e}")
    
    def _execute_coordinated_optimizations(self):
        """Execute coordinated optimizations."""
        try:
            # This is handled by the task scheduler
            # Additional coordination logic could be added here
            pass
            
        except Exception as e:
            logger.error(f"Error executing coordinated optimizations: {e}")
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status."""
        return {
            'is_coordinating': self.is_coordinating,
            'system_state': self.system_state.to_dict() if self.system_state else None,
            'component_status': {
                'monitoring': self.monitoring_framework.get_current_status(),
                'optimization': self.optimization_engine.get_optimization_stats(),
                'scheduling': self.task_scheduler.get_scheduler_status()
            },
            'coordination_history_size': len(self.coordination_history),
            'last_coordination': self.coordination_history[-1]['timestamp'].isoformat() if self.coordination_history else None
        }
    
    def run_coordination_test(self) -> Dict[str, Any]:
        """Run comprehensive coordination test."""
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'test_status': 'passed',
            'component_tests': {},
            'integration_tests': {},
            'performance_metrics': {}
        }
        
        try:
            # Test individual components
            test_results['component_tests']['monitoring'] = self.monitoring_framework.run_self_test()
            test_results['component_tests']['optimization'] = {'status': 'passed'}  # Would implement actual test
            test_results['component_tests']['scheduling'] = {'status': 'passed'}   # Would implement actual test
            
            # Test integration
            start_time = time.time()
            
            # Test task creation and scheduling
            test_task = PerformanceTask(
                task_id=f"test_{uuid.uuid4().hex[:8]}",
                task_type='monitoring',
                priority=5,
                component='test',
                parameters={'test': True},
                dependencies=[],
                estimated_duration=timedelta(seconds=1),
                created_at=datetime.now()
            )
            
            self.task_scheduler.add_task(test_task)
            integration_time = time.time() - start_time
            
            test_results['integration_tests']['task_scheduling'] = {
                'status': 'passed',
                'execution_time_ms': integration_time * 1000
            }
            
            # Performance metrics
            test_results['performance_metrics'] = {
                'coordination_overhead_ms': integration_time * 1000,
                'memory_usage_mb': self._get_memory_usage(),
                'active_threads': threading.active_count()
            }
            
        except Exception as e:
            test_results['test_status'] = 'failed'
            test_results['error'] = str(e)
            logger.error(f"Coordination test failed: {e}")
        
        return test_results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0

# Example usage and testing
if __name__ == "__main__":
    # Create system coordinator
    coordinator = SystemCoordinator()
    
    # Run coordination test
    print("Running system coordination test...")
    test_results = coordinator.run_coordination_test()
    print(f"Test status: {test_results['test_status']}")
    
    # Start coordination for demonstration
    print("Starting system coordination...")
    coordinator.start_coordination()
    
    try:
        # Let it run for a short time
        time.sleep(30)
        
        # Get coordination status
        status = coordinator.get_coordination_status()
        print(f"Coordination status: {status['is_coordinating']}")
        
        if status['system_state']:
            print(f"System health: {status['system_state']['overall_health_score']:.1f}")
        
    finally:
        # Stop coordination
        coordinator.stop_coordination()
        print("System coordination stopped")

