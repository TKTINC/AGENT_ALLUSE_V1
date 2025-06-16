"""
ALL-USE Production Infrastructure

This module provides production-ready infrastructure for ALL-USE components,
including configuration management, logging, health checks, and deployment management.
"""

import os
import json
import logging
import logging.handlers
import threading
import time
import signal
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil
import yaml
from contextlib import contextmanager

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class HealthStatus:
    """Health status information."""
    status: str  # 'healthy', 'degraded', 'unhealthy'
    timestamp: datetime
    details: Dict[str, Any]
    version: str


class ConfigManager:
    """
    Configuration management for ALL-USE production deployment.
    
    Provides:
    - Environment-specific configuration
    - Configuration validation
    - Dynamic configuration updates
    - Configuration versioning
    """
    
    def __init__(self, config_dir: str = "config", environment: str = None):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv('ALL_USE_ENV', 'development')
        self.config = {}
        self.config_watchers = []
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(exist_ok=True)
        
        self._load_configuration()
        
        logging.info(f"Configuration manager initialized for environment: {self.environment}")
    
    def _load_configuration(self):
        """Load configuration from files."""
        # Load base configuration
        base_config_file = self.config_dir / "base.yaml"
        if base_config_file.exists():
            with open(base_config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        
        # Load environment-specific configuration
        env_config_file = self.config_dir / f"{self.environment}.yaml"
        if env_config_file.exists():
            with open(env_config_file, 'r') as f:
                env_config = yaml.safe_load(f) or {}
                self._merge_config(self.config, env_config)
        
        # Override with environment variables
        self._load_env_overrides()
        
        # Validate configuration
        self._validate_configuration()
    
    def _merge_config(self, base: Dict, override: Dict):
        """Merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        # Look for ALL_USE_* environment variables
        for key, value in os.environ.items():
            if key.startswith('ALL_USE_'):
                config_key = key[8:].lower().replace('_', '.')
                self._set_nested_value(self.config, config_key, value)
    
    def _set_nested_value(self, config: Dict, key_path: str, value: str):
        """Set nested configuration value using dot notation."""
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Try to convert value to appropriate type
        try:
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                value = float(value)
        except:
            pass  # Keep as string
        
        current[keys[-1]] = value
    
    def _validate_configuration(self):
        """Validate configuration values."""
        required_keys = [
            'app.name',
            'app.version',
            'logging.level',
            'monitoring.enabled'
        ]
        
        for key in required_keys:
            if not self.get(key):
                raise ValueError(f"Required configuration key '{key}' is missing")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        current = self.config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        self._set_nested_value(self.config, key, value)
        
        # Notify watchers
        for watcher in self.config_watchers:
            try:
                watcher(key, value)
            except Exception as e:
                logging.error(f"Error in configuration watcher: {e}")
    
    def add_watcher(self, callback: Callable[[str, Any], None]):
        """Add configuration change watcher."""
        self.config_watchers.append(callback)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self.config.copy()
    
    def save_config(self, filename: str = None):
        """Save current configuration to file."""
        if not filename:
            filename = f"{self.environment}.yaml"
        
        config_file = self.config_dir / filename
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


class ProductionLogger:
    """
    Production-ready logging system for ALL-USE components.
    
    Provides:
    - Structured logging
    - Log rotation
    - Multiple output formats
    - Performance monitoring
    """
    
    def __init__(self, config: ConfigManager):
        """Initialize production logger."""
        self.config = config
        self.loggers = {}
        
        self._setup_logging()
        
        logging.info("Production logger initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get('logging.level', 'INFO').upper()
        log_format = self.config.get('logging.format', 'json')
        log_dir = Path(self.config.get('logging.directory', 'logs'))
        
        # Create log directory
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if self.config.get('logging.console.enabled', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level))
            
            if log_format == 'json':
                console_handler.setFormatter(JSONFormatter())
            else:
                console_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.config.get('logging.file.enabled', True):
            log_file = log_dir / f"{self.config.get('app.name', 'all_use')}.log"
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.get('logging.file.max_size', 10 * 1024 * 1024),  # 10MB
                backupCount=self.config.get('logging.file.backup_count', 5)
            )
            file_handler.setLevel(getattr(logging, log_level))
            
            if log_format == 'json':
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            
            root_logger.addHandler(file_handler)
        
        # Error file handler
        if self.config.get('logging.error_file.enabled', True):
            error_file = log_dir / f"{self.config.get('app.name', 'all_use')}_error.log"
            
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=self.config.get('logging.error_file.max_size', 5 * 1024 * 1024),  # 5MB
                backupCount=self.config.get('logging.error_file.backup_count', 3)
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(JSONFormatter())
            
            root_logger.addHandler(error_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger instance."""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        
        return self.loggers[name]


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info'):
                log_entry[key] = value
        
        return json.dumps(log_entry)


class HealthChecker:
    """
    Production health checking system for ALL-USE components.
    
    Provides:
    - HTTP health endpoints
    - Readiness and liveness probes
    - Dependency health checking
    - Health status aggregation
    """
    
    def __init__(self, config: ConfigManager):
        """Initialize health checker."""
        self.config = config
        self.health_checks = {}
        self.last_health_check = None
        self.health_status = HealthStatus(
            status='unknown',
            timestamp=datetime.now(),
            details={},
            version=config.get('app.version', '1.0.0')
        )
        
        # Register default health checks
        self._register_default_checks()
        
        logging.info("Health checker initialized")
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check('system_memory', self._check_system_memory)
        self.register_check('system_disk', self._check_system_disk)
        self.register_check('system_cpu', self._check_system_cpu)
    
    def register_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self.health_checks[name] = check_func
        logging.info(f"Health check '{name}' registered")
    
    def run_health_checks(self) -> HealthStatus:
        """Run all health checks and return status."""
        start_time = time.time()
        check_results = {}
        overall_healthy = True
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                check_results[name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'healthy': result
                }
                
                if not result:
                    overall_healthy = False
                    
            except Exception as e:
                check_results[name] = {
                    'status': 'error',
                    'healthy': False,
                    'error': str(e)
                }
                overall_healthy = False
        
        # Determine overall status
        if overall_healthy:
            status = 'healthy'
        else:
            # Check if any critical checks failed
            critical_failed = any(
                not result['healthy'] for name, result in check_results.items()
                if name in ['system_memory', 'system_disk']
            )
            status = 'unhealthy' if critical_failed else 'degraded'
        
        self.health_status = HealthStatus(
            status=status,
            timestamp=datetime.now(),
            details={
                'checks': check_results,
                'check_duration_ms': (time.time() - start_time) * 1000,
                'uptime_seconds': time.time() - psutil.Process().create_time()
            },
            version=self.config.get('app.version', '1.0.0')
        )
        
        self.last_health_check = datetime.now()
        return self.health_status
    
    def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        # Run health checks if they haven't been run recently
        if (not self.last_health_check or 
            (datetime.now() - self.last_health_check).seconds > 30):
            self.run_health_checks()
        
        return self.health_status
    
    def _check_system_memory(self) -> bool:
        """Check system memory usage."""
        memory = psutil.virtual_memory()
        threshold = self.config.get('health.memory_threshold', 90)
        return memory.percent < threshold
    
    def _check_system_disk(self) -> bool:
        """Check system disk usage."""
        disk = psutil.disk_usage('/')
        threshold = self.config.get('health.disk_threshold', 90)
        return disk.percent < threshold
    
    def _check_system_cpu(self) -> bool:
        """Check system CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        threshold = self.config.get('health.cpu_threshold', 90)
        return cpu_percent < threshold


class DeploymentManager:
    """
    Deployment and lifecycle management for ALL-USE components.
    
    Provides:
    - Graceful startup and shutdown
    - Signal handling
    - Process management
    - Deployment validation
    """
    
    def __init__(self, config: ConfigManager):
        """Initialize deployment manager."""
        self.config = config
        self.shutdown_handlers = []
        self.startup_handlers = []
        self.is_shutting_down = False
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logging.info("Deployment manager initialized")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def add_startup_handler(self, handler: Callable[[], None]):
        """Add startup handler."""
        self.startup_handlers.append(handler)
    
    def add_shutdown_handler(self, handler: Callable[[], None]):
        """Add shutdown handler."""
        self.shutdown_handlers.append(handler)
    
    def startup(self):
        """Execute startup sequence."""
        logging.info("Starting ALL-USE application")
        
        for handler in self.startup_handlers:
            try:
                handler()
            except Exception as e:
                logging.error(f"Error in startup handler: {e}")
                raise
        
        logging.info("ALL-USE application started successfully")
    
    def shutdown(self):
        """Execute graceful shutdown sequence."""
        if self.is_shutting_down:
            return
        
        self.is_shutting_down = True
        logging.info("Initiating graceful shutdown")
        
        for handler in reversed(self.shutdown_handlers):
            try:
                handler()
            except Exception as e:
                logging.error(f"Error in shutdown handler: {e}")
        
        logging.info("Graceful shutdown completed")
        sys.exit(0)
    
    @contextmanager
    def managed_lifecycle(self):
        """Context manager for managed application lifecycle."""
        try:
            self.startup()
            yield
        finally:
            if not self.is_shutting_down:
                self.shutdown()


# Global production infrastructure instances
config_manager = None
production_logger = None
health_checker = None
deployment_manager = None


def initialize_production_infrastructure(config_dir: str = "config", environment: str = None):
    """Initialize production infrastructure."""
    global config_manager, production_logger, health_checker, deployment_manager
    
    # Initialize configuration
    config_manager = ConfigManager(config_dir, environment)
    
    # Initialize logging
    production_logger = ProductionLogger(config_manager)
    
    # Initialize health checking
    health_checker = HealthChecker(config_manager)
    
    # Initialize deployment management
    deployment_manager = DeploymentManager(config_manager)
    
    logging.info("Production infrastructure initialized")
    
    return {
        'config': config_manager,
        'logger': production_logger,
        'health': health_checker,
        'deployment': deployment_manager
    }


if __name__ == "__main__":
    # Test production infrastructure
    
    # Create test configuration
    os.makedirs("config", exist_ok=True)
    
    base_config = {
        'app': {
            'name': 'all_use_test',
            'version': '1.0.0'
        },
        'logging': {
            'level': 'INFO',
            'format': 'json',
            'console': {'enabled': True},
            'file': {'enabled': True}
        },
        'monitoring': {
            'enabled': True
        },
        'health': {
            'memory_threshold': 90,
            'disk_threshold': 90,
            'cpu_threshold': 90
        }
    }
    
    with open("config/base.yaml", 'w') as f:
        yaml.dump(base_config, f)
    
    # Initialize infrastructure
    infrastructure = initialize_production_infrastructure()
    
    # Test configuration
    print("Testing configuration...")
    print(f"App name: {config_manager.get('app.name')}")
    print(f"Log level: {config_manager.get('logging.level')}")
    
    # Test logging
    print("\nTesting logging...")
    logger = production_logger.get_logger('test')
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    
    # Test health checking
    print("\nTesting health checking...")
    health_status = health_checker.run_health_checks()
    print(f"Health status: {health_status.status}")
    print(f"Health details: {health_status.details}")
    
    # Test deployment manager
    print("\nTesting deployment manager...")
    
    def test_startup():
        print("Startup handler executed")
    
    def test_shutdown():
        print("Shutdown handler executed")
    
    deployment_manager.add_startup_handler(test_startup)
    deployment_manager.add_shutdown_handler(test_shutdown)
    
    with deployment_manager.managed_lifecycle():
        print("Application running...")
        time.sleep(1)
    
    print("\nProduction infrastructure test completed successfully!")

