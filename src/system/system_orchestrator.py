"""
ALL-USE System Orchestrator

This module provides the central coordination system for all WS1 components,
including component registration, service discovery, configuration management,
and lifecycle coordination.
"""

import asyncio
import threading
import time
import logging
import signal
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import weakref
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all WS1 components
from agent_core.enhanced_agent import EnhancedALLUSEAgent
from agent_core.enhanced_cognitive_framework import EnhancedIntentDetector, EnhancedEntityExtractor
from agent_core.enhanced_memory_manager import EnhancedMemoryManager
from trading_engine.market_analyzer import MarketAnalyzer
from trading_engine.position_sizer import PositionSizer
from trading_engine.delta_selector import DeltaSelector
from risk_management.portfolio_risk_monitor import PortfolioRiskMonitor
from risk_management.drawdown_protection import DrawdownProtectionSystem
from optimization.portfolio_optimizer import PortfolioOptimizer
from optimization.performance_optimizer import PerformanceOptimizer
from monitoring.monitoring_system import metrics_collector, health_monitor, alert_manager
from production.production_infrastructure import (
    config_manager, production_logger, health_checker, deployment_manager
)

logger = logging.getLogger('all_use_orchestrator')


@dataclass
class ComponentInfo:
    """Information about a registered component."""
    name: str
    instance: Any
    component_type: str
    dependencies: List[str]
    status: str  # 'registered', 'initializing', 'running', 'stopped', 'error'
    health_check: Optional[Callable[[], bool]]
    startup_order: int
    shutdown_order: int


@dataclass
class SystemStatus:
    """Overall system status information."""
    status: str  # 'initializing', 'running', 'degraded', 'stopped', 'error'
    components: Dict[str, str]
    startup_time: Optional[datetime]
    uptime_seconds: float
    health_summary: Dict[str, Any]
    performance_summary: Dict[str, Any]


class ComponentInterface(ABC):
    """Abstract interface for system components."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the component."""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the component."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the component."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check component health."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        pass


class SystemOrchestrator:
    """
    Central coordination system for all WS1 components.
    
    Provides:
    - Component registration and discovery
    - Dependency management and injection
    - Lifecycle coordination (startup/shutdown)
    - Configuration management
    - Health monitoring integration
    - Performance monitoring integration
    """
    
    def __init__(self, config_path: str = "config"):
        """Initialize the system orchestrator."""
        self.config_path = config_path
        self.components = {}
        self.component_instances = {}
        self.dependency_graph = {}
        self.startup_order = []
        self.shutdown_order = []
        
        # System state
        self.system_status = SystemStatus(
            status='initializing',
            components={},
            startup_time=None,
            uptime_seconds=0,
            health_summary={},
            performance_summary={}
        )
        
        # Event system
        self.event_handlers = {}
        self.event_queue = asyncio.Queue()
        
        # Lifecycle management
        self.is_running = False
        self.is_shutting_down = False
        self.startup_complete = asyncio.Event()
        self.shutdown_complete = asyncio.Event()
        
        # Background tasks
        self.background_tasks = []
        
        logger.info("System orchestrator initialized")
    
    def register_component(self, name: str, component_class: Type, 
                          dependencies: List[str] = None,
                          startup_order: int = 100,
                          shutdown_order: int = None,
                          health_check: Callable[[], bool] = None,
                          **kwargs) -> None:
        """Register a component with the orchestrator."""
        dependencies = dependencies or []
        shutdown_order = shutdown_order or (1000 - startup_order)
        
        component_info = ComponentInfo(
            name=name,
            instance=None,
            component_type=component_class.__name__,
            dependencies=dependencies,
            status='registered',
            health_check=health_check,
            startup_order=startup_order,
            shutdown_order=shutdown_order
        )
        
        self.components[name] = {
            'info': component_info,
            'class': component_class,
            'kwargs': kwargs
        }
        
        logger.info(f"Component '{name}' registered with dependencies: {dependencies}")
    
    def get_component(self, name: str) -> Any:
        """Get a component instance by name."""
        if name in self.component_instances:
            return self.component_instances[name]
        
        raise ValueError(f"Component '{name}' not found or not initialized")
    
    def inject_dependencies(self, component_name: str, instance: Any) -> None:
        """Inject dependencies into a component instance."""
        component_info = self.components[component_name]['info']
        
        for dep_name in component_info.dependencies:
            if dep_name in self.component_instances:
                dep_instance = self.component_instances[dep_name]
                
                # Try to inject dependency using various methods
                if hasattr(instance, f'set_{dep_name}'):
                    getattr(instance, f'set_{dep_name}')(dep_instance)
                elif hasattr(instance, dep_name):
                    setattr(instance, dep_name, dep_instance)
                elif hasattr(instance, 'inject_dependency'):
                    instance.inject_dependency(dep_name, dep_instance)
                else:
                    logger.warning(f"Could not inject dependency '{dep_name}' into '{component_name}'")
            else:
                logger.error(f"Dependency '{dep_name}' not available for '{component_name}'")
    
    def resolve_dependencies(self) -> List[str]:
        """Resolve component dependencies and return startup order."""
        # Simple topological sort for dependency resolution
        visited = set()
        temp_visited = set()
        startup_order = []
        
        def visit(component_name: str):
            if component_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{component_name}'")
            
            if component_name not in visited:
                temp_visited.add(component_name)
                
                component_info = self.components[component_name]['info']
                for dep_name in component_info.dependencies:
                    if dep_name in self.components:
                        visit(dep_name)
                    else:
                        logger.warning(f"Dependency '{dep_name}' not registered")
                
                temp_visited.remove(component_name)
                visited.add(component_name)
                startup_order.append(component_name)
        
        for component_name in self.components.keys():
            if component_name not in visited:
                visit(component_name)
        
        # Sort by startup order within dependency groups
        startup_order.sort(key=lambda name: self.components[name]['info'].startup_order)
        
        return startup_order
    
    async def initialize_component(self, name: str) -> bool:
        """Initialize a single component."""
        try:
            component_data = self.components[name]
            component_class = component_data['class']
            kwargs = component_data['kwargs']
            
            logger.info(f"Initializing component '{name}'")
            
            # Create component instance
            instance = component_class(**kwargs)
            self.component_instances[name] = instance
            
            # Update component status
            component_data['info'].instance = instance
            component_data['info'].status = 'initializing'
            
            # Inject dependencies
            self.inject_dependencies(name, instance)
            
            # Initialize component if it supports async initialization
            if hasattr(instance, 'initialize') and asyncio.iscoroutinefunction(instance.initialize):
                success = await instance.initialize()
            elif hasattr(instance, 'initialize'):
                success = instance.initialize()
            else:
                success = True
            
            if success:
                component_data['info'].status = 'running'
                logger.info(f"Component '{name}' initialized successfully")
                return True
            else:
                component_data['info'].status = 'error'
                logger.error(f"Component '{name}' initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing component '{name}': {e}")
            if name in self.components:
                self.components[name]['info'].status = 'error'
            return False
    
    async def start_component(self, name: str) -> bool:
        """Start a single component."""
        try:
            if name not in self.component_instances:
                logger.error(f"Component '{name}' not initialized")
                return False
            
            instance = self.component_instances[name]
            
            # Start component if it supports async start
            if hasattr(instance, 'start') and asyncio.iscoroutinefunction(instance.start):
                success = await instance.start()
            elif hasattr(instance, 'start'):
                success = instance.start()
            else:
                success = True
            
            if success:
                self.components[name]['info'].status = 'running'
                logger.info(f"Component '{name}' started successfully")
                return True
            else:
                self.components[name]['info'].status = 'error'
                logger.error(f"Component '{name}' start failed")
                return False
                
        except Exception as e:
            logger.error(f"Error starting component '{name}': {e}")
            if name in self.components:
                self.components[name]['info'].status = 'error'
            return False
    
    async def stop_component(self, name: str) -> bool:
        """Stop a single component."""
        try:
            if name not in self.component_instances:
                return True  # Already stopped
            
            instance = self.component_instances[name]
            
            logger.info(f"Stopping component '{name}'")
            
            # Stop component if it supports async stop
            if hasattr(instance, 'stop') and asyncio.iscoroutinefunction(instance.stop):
                success = await instance.stop()
            elif hasattr(instance, 'stop'):
                success = instance.stop()
            else:
                success = True
            
            if success:
                self.components[name]['info'].status = 'stopped'
                logger.info(f"Component '{name}' stopped successfully")
            else:
                logger.error(f"Component '{name}' stop failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error stopping component '{name}': {e}")
            return False
    
    async def startup_system(self) -> bool:
        """Start up the entire system."""
        try:
            logger.info("Starting system startup sequence")
            self.system_status.status = 'initializing'
            
            # Resolve dependencies and get startup order
            self.startup_order = self.resolve_dependencies()
            logger.info(f"Component startup order: {self.startup_order}")
            
            # Initialize and start components in order
            for component_name in self.startup_order:
                success = await self.initialize_component(component_name)
                if not success:
                    logger.error(f"Failed to initialize component '{component_name}'")
                    self.system_status.status = 'error'
                    return False
                
                success = await self.start_component(component_name)
                if not success:
                    logger.error(f"Failed to start component '{component_name}'")
                    self.system_status.status = 'error'
                    return False
            
            # Update system status
            self.system_status.status = 'running'
            self.system_status.startup_time = datetime.now()
            self.is_running = True
            self.startup_complete.set()
            
            # Start background monitoring
            await self.start_background_tasks()
            
            logger.info("System startup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System startup failed: {e}")
            self.system_status.status = 'error'
            return False
    
    async def shutdown_system(self) -> bool:
        """Shut down the entire system."""
        try:
            if self.is_shutting_down:
                return True
            
            self.is_shutting_down = True
            logger.info("Starting system shutdown sequence")
            
            # Stop background tasks
            await self.stop_background_tasks()
            
            # Create shutdown order (reverse of startup order)
            self.shutdown_order = list(reversed(self.startup_order))
            logger.info(f"Component shutdown order: {self.shutdown_order}")
            
            # Stop components in reverse order
            for component_name in self.shutdown_order:
                await self.stop_component(component_name)
            
            # Update system status
            self.system_status.status = 'stopped'
            self.is_running = False
            self.shutdown_complete.set()
            
            logger.info("System shutdown completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System shutdown failed: {e}")
            return False
    
    async def start_background_tasks(self):
        """Start background monitoring and maintenance tasks."""
        # Health monitoring task
        health_task = asyncio.create_task(self.health_monitoring_loop())
        self.background_tasks.append(health_task)
        
        # Performance monitoring task
        perf_task = asyncio.create_task(self.performance_monitoring_loop())
        self.background_tasks.append(perf_task)
        
        # Event processing task
        event_task = asyncio.create_task(self.event_processing_loop())
        self.background_tasks.append(event_task)
        
        logger.info("Background tasks started")
    
    async def stop_background_tasks(self):
        """Stop all background tasks."""
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.background_tasks.clear()
        logger.info("Background tasks stopped")
    
    async def health_monitoring_loop(self):
        """Background health monitoring loop."""
        while self.is_running and not self.is_shutting_down:
            try:
                # Check component health
                health_summary = {}
                overall_healthy = True
                
                for name, component_data in self.components.items():
                    component_info = component_data['info']
                    
                    if component_info.instance and component_info.health_check:
                        try:
                            is_healthy = component_info.health_check()
                            health_summary[name] = 'healthy' if is_healthy else 'unhealthy'
                            
                            if not is_healthy:
                                overall_healthy = False
                                
                        except Exception as e:
                            health_summary[name] = 'error'
                            overall_healthy = False
                            logger.error(f"Health check failed for '{name}': {e}")
                    else:
                        health_summary[name] = 'unknown'
                
                # Update system health status
                self.system_status.health_summary = health_summary
                
                if not overall_healthy and self.system_status.status == 'running':
                    self.system_status.status = 'degraded'
                elif overall_healthy and self.system_status.status == 'degraded':
                    self.system_status.status = 'running'
                
                # Update uptime
                if self.system_status.startup_time:
                    self.system_status.uptime_seconds = (
                        datetime.now() - self.system_status.startup_time
                    ).total_seconds()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def performance_monitoring_loop(self):
        """Background performance monitoring loop."""
        while self.is_running and not self.is_shutting_down:
            try:
                # Collect performance metrics
                performance_summary = {}
                
                # Get metrics from monitoring system
                if 'monitoring' in self.component_instances:
                    monitoring = self.component_instances['monitoring']
                    if hasattr(monitoring, 'get_performance_summary'):
                        performance_summary = monitoring.get_performance_summary()
                
                self.system_status.performance_summary = performance_summary
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def event_processing_loop(self):
        """Background event processing loop."""
        while self.is_running and not self.is_shutting_down:
            try:
                # Process events from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Handle event
                await self.handle_event(event)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
    
    async def handle_event(self, event: Dict[str, Any]):
        """Handle system events."""
        event_type = event.get('type')
        
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler for '{event_type}': {e}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
    
    async def emit_event(self, event_type: str, data: Dict[str, Any] = None):
        """Emit a system event."""
        event = {
            'type': event_type,
            'timestamp': datetime.now(),
            'data': data or {}
        }
        
        await self.event_queue.put(event)
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status."""
        # Update component statuses
        component_statuses = {}
        for name, component_data in self.components.items():
            component_statuses[name] = component_data['info'].status
        
        self.system_status.components = component_statuses
        return self.system_status
    
    def get_component_status(self, name: str) -> Dict[str, Any]:
        """Get status of a specific component."""
        if name not in self.components:
            return {'error': f"Component '{name}' not found"}
        
        component_info = self.components[name]['info']
        instance = component_info.instance
        
        status = {
            'name': name,
            'type': component_info.component_type,
            'status': component_info.status,
            'dependencies': component_info.dependencies
        }
        
        # Get component-specific status if available
        if instance and hasattr(instance, 'get_status'):
            try:
                component_status = instance.get_status()
                status.update(component_status)
            except Exception as e:
                status['status_error'] = str(e)
        
        return status
    
    @asynccontextmanager
    async def managed_lifecycle(self):
        """Context manager for managed system lifecycle."""
        try:
            success = await self.startup_system()
            if not success:
                raise RuntimeError("System startup failed")
            
            yield self
            
        finally:
            await self.shutdown_system()


# Global orchestrator instance
orchestrator = SystemOrchestrator()


def register_all_ws1_components():
    """Register all WS1 components with the orchestrator."""
    
    # Register core components (no dependencies)
    orchestrator.register_component(
        'config_manager',
        type(config_manager),
        startup_order=10
    )
    
    orchestrator.register_component(
        'production_logger',
        type(production_logger),
        dependencies=['config_manager'],
        startup_order=20
    )
    
    orchestrator.register_component(
        'metrics_collector',
        type(metrics_collector),
        startup_order=30
    )
    
    orchestrator.register_component(
        'health_monitor',
        type(health_monitor),
        startup_order=40
    )
    
    orchestrator.register_component(
        'alert_manager',
        type(alert_manager),
        startup_order=50
    )
    
    # Register optimization components
    orchestrator.register_component(
        'performance_optimizer',
        PerformanceOptimizer,
        startup_order=60
    )
    
    # Register trading engine components
    orchestrator.register_component(
        'market_analyzer',
        MarketAnalyzer,
        startup_order=70
    )
    
    orchestrator.register_component(
        'position_sizer',
        PositionSizer,
        startup_order=80
    )
    
    orchestrator.register_component(
        'delta_selector',
        DeltaSelector,
        startup_order=90
    )
    
    # Register risk management components
    orchestrator.register_component(
        'portfolio_risk_monitor',
        PortfolioRiskMonitor,
        startup_order=100
    )
    
    orchestrator.register_component(
        'drawdown_protection',
        DrawdownProtectionSystem,
        startup_order=110
    )
    
    orchestrator.register_component(
        'portfolio_optimizer',
        PortfolioOptimizer,
        startup_order=120
    )
    
    # Register agent core components (depend on other components)
    orchestrator.register_component(
        'memory_manager',
        EnhancedMemoryManager,
        startup_order=130
    )
    
    orchestrator.register_component(
        'intent_detector',
        EnhancedIntentDetector,
        startup_order=140
    )
    
    orchestrator.register_component(
        'entity_extractor',
        EnhancedEntityExtractor,
        startup_order=150
    )
    
    orchestrator.register_component(
        'enhanced_agent',
        EnhancedALLUSEAgent,
        dependencies=[
            'memory_manager',
            'intent_detector',
            'entity_extractor',
            'market_analyzer',
            'position_sizer',
            'delta_selector',
            'portfolio_risk_monitor'
        ],
        startup_order=200
    )


if __name__ == "__main__":
    async def main():
        # Register all components
        register_all_ws1_components()
        
        # Test system orchestration
        async with orchestrator.managed_lifecycle():
            print("System started successfully!")
            
            # Get system status
            status = orchestrator.get_system_status()
            print(f"System status: {status.status}")
            print(f"Components: {status.components}")
            
            # Test component access
            try:
                agent = orchestrator.get_component('enhanced_agent')
                print(f"Agent component: {type(agent).__name__}")
            except Exception as e:
                print(f"Error accessing agent: {e}")
            
            # Wait a bit to see background monitoring
            await asyncio.sleep(5)
            
            print("System test completed!")
    
    # Run the test
    asyncio.run(main())

