#!/usr/bin/env python3
"""
ALL-USE Account Management System - Integration Validator

This module provides validation mechanisms for integration points in the
ALL-USE Account Management System, ensuring that integrations are functioning
correctly and meeting their requirements.

The integration validator implements:
- Integration health checks
- Integration connectivity validation
- Data consistency validation
- Performance validation
- Security validation

Author: Manus AI
Date: June 17, 2025
Version: 1.0
"""

import logging
import time
import json
import threading
import datetime
import statistics
from typing import Dict, List, Any, Callable, Optional, Union, Tuple

from .integration_framework import (
    IntegrationManager, IntegrationConfiguration, IntegrationType,
    IntegrationError, get_integration_manager
)
from .component_integrator import get_component_integrator
from .external_system_adapter import get_external_system_adapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationResult:
    """
    Represents the result of an integration validation.
    
    This class encapsulates the outcome of a validation check,
    including success/failure status, metrics, and details.
    """
    
    def __init__(
        self,
        integration_id: str,
        validation_type: str,
        success: bool,
        message: str,
        details: Dict[str, Any] = None,
        metrics: Dict[str, Any] = None
    ):
        """
        Initialize a validation result.
        
        Args:
            integration_id: ID of the integration that was validated
            validation_type: Type of validation performed
            success: Whether the validation was successful
            message: Description of the validation result
            details: Additional details about the validation
            metrics: Performance or other metrics from the validation
        """
        self.integration_id = integration_id
        self.validation_type = validation_type
        self.success = success
        self.message = message
        self.details = details or {}
        self.metrics = metrics or {}
        self.timestamp = datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation result to a dictionary."""
        return {
            "integration_id": self.integration_id,
            "validation_type": self.validation_type,
            "success": self.success,
            "message": self.message,
            "details": self.details,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat()
        }


class IntegrationValidator:
    """
    Validates integration points in the account management system.
    
    This class provides mechanisms for validating that integrations
    are functioning correctly and meeting their requirements.
    """
    
    def __init__(self):
        """Initialize the integration validator."""
        self.integration_manager = get_integration_manager()
        self.component_integrator = get_component_integrator()
        self.external_system_adapter = get_external_system_adapter()
        self.validation_results: List[ValidationResult] = []
        logger.info("Integration Validator initialized")
    
    def validate_integration_health(self, integration_id: str) -> ValidationResult:
        """
        Validate the health of an integration.
        
        Args:
            integration_id: ID of the integration to validate
            
        Returns:
            ValidationResult with the outcome of the health check
        """
        integration = self.integration_manager.registry.get(integration_id)
        if not integration:
            return ValidationResult(
                integration_id=integration_id,
                validation_type="health",
                success=False,
                message=f"Integration with ID {integration_id} not found"
            )
        
        # Check if the integration is enabled
        if not integration.config.enabled:
            return ValidationResult(
                integration_id=integration_id,
                validation_type="health",
                success=False,
                message=f"Integration {integration.config.name} is disabled",
                details={"status": integration.status.value}
            )
        
        # Check the integration status
        if integration.status.value != "active":
            return ValidationResult(
                integration_id=integration_id,
                validation_type="health",
                success=False,
                message=f"Integration {integration.config.name} has status {integration.status.value}",
                details={"status": integration.status.value}
            )
        
        # Check the circuit breaker state
        if integration.circuit_breaker.get_state() != "CLOSED":
            return ValidationResult(
                integration_id=integration_id,
                validation_type="health",
                success=False,
                message=f"Circuit breaker for {integration.config.name} is {integration.circuit_breaker.get_state()}",
                details={"circuit_breaker_state": integration.circuit_breaker.get_state()}
            )
        
        # Get metrics for the integration
        metrics = integration.get_metrics()
        
        return ValidationResult(
            integration_id=integration_id,
            validation_type="health",
            success=True,
            message=f"Integration {integration.config.name} is healthy",
            details={"status": integration.status.value},
            metrics=metrics.get("metrics", {})
        )
    
    def validate_connectivity(self, integration_id: str) -> ValidationResult:
        """
        Validate connectivity for an integration.
        
        Args:
            integration_id: ID of the integration to validate
            
        Returns:
            ValidationResult with the outcome of the connectivity check
        """
        integration = self.integration_manager.registry.get(integration_id)
        if not integration:
            return ValidationResult(
                integration_id=integration_id,
                validation_type="connectivity",
                success=False,
                message=f"Integration with ID {integration_id} not found"
            )
        
        # Define a simple ping operation
        def ping_operation():
            return {"status": "success", "timestamp": datetime.datetime.now().isoformat()}
        
        # Try to execute the ping operation
        try:
            start_time = time.time()
            result = self.integration_manager.execute_integration(integration_id, ping_operation)
            end_time = time.time()
            
            # Calculate response time
            response_time_ms = (end_time - start_time) * 1000
            
            return ValidationResult(
                integration_id=integration_id,
                validation_type="connectivity",
                success=True,
                message=f"Successfully connected to {integration.config.name}",
                details={"result": result},
                metrics={"response_time_ms": response_time_ms}
            )
        except IntegrationError as e:
            return ValidationResult(
                integration_id=integration_id,
                validation_type="connectivity",
                success=False,
                message=f"Failed to connect to {integration.config.name}: {str(e)}",
                details={"error": str(e), "error_code": e.error_code if hasattr(e, 'error_code') else None}
            )
    
    def validate_data_consistency(self, integration_id: str, test_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate data consistency for an integration.
        
        Args:
            integration_id: ID of the integration to validate
            test_data: Test data to use for validation
            
        Returns:
            ValidationResult with the outcome of the data consistency check
        """
        integration = self.integration_manager.registry.get(integration_id)
        if not integration:
            return ValidationResult(
                integration_id=integration_id,
                validation_type="data_consistency",
                success=False,
                message=f"Integration with ID {integration_id} not found"
            )
        
        # Define a data echo operation that returns the input data
        def echo_operation(data):
            return data
        
        # Try to execute the echo operation
        try:
            result = self.integration_manager.execute_integration(integration_id, echo_operation, test_data)
            
            # Check if the result matches the input
            if result == test_data:
                return ValidationResult(
                    integration_id=integration_id,
                    validation_type="data_consistency",
                    success=True,
                    message=f"Data consistency validated for {integration.config.name}",
                    details={"input": test_data, "output": result}
                )
            else:
                return ValidationResult(
                    integration_id=integration_id,
                    validation_type="data_consistency",
                    success=False,
                    message=f"Data consistency check failed for {integration.config.name}: output does not match input",
                    details={"input": test_data, "output": result}
                )
        except IntegrationError as e:
            return ValidationResult(
                integration_id=integration_id,
                validation_type="data_consistency",
                success=False,
                message=f"Data consistency check failed for {integration.config.name}: {str(e)}",
                details={"error": str(e), "error_code": e.error_code if hasattr(e, 'error_code') else None}
            )
    
    def validate_performance(self, integration_id: str, iterations: int = 10) -> ValidationResult:
        """
        Validate performance for an integration.
        
        Args:
            integration_id: ID of the integration to validate
            iterations: Number of test iterations to run
            
        Returns:
            ValidationResult with the outcome of the performance check
        """
        integration = self.integration_manager.registry.get(integration_id)
        if not integration:
            return ValidationResult(
                integration_id=integration_id,
                validation_type="performance",
                success=False,
                message=f"Integration with ID {integration_id} not found"
            )
        
        # Define a simple operation for performance testing
        def noop_operation():
            return {"status": "success"}
        
        # Run the operation multiple times and measure performance
        response_times = []
        failures = 0
        
        for _ in range(iterations):
            try:
                start_time = time.time()
                self.integration_manager.execute_integration(integration_id, noop_operation)
                end_time = time.time()
                
                # Calculate response time
                response_time_ms = (end_time - start_time) * 1000
                response_times.append(response_time_ms)
            except IntegrationError:
                failures += 1
        
        # Calculate performance metrics
        if not response_times:
            return ValidationResult(
                integration_id=integration_id,
                validation_type="performance",
                success=False,
                message=f"Performance validation failed for {integration.config.name}: all iterations failed",
                details={"failures": failures, "iterations": iterations}
            )
        
        avg_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
        
        # Check if performance meets requirements
        # For this example, we'll consider < 100ms as good performance
        success = avg_response_time < 100 and failures == 0
        
        return ValidationResult(
            integration_id=integration_id,
            validation_type="performance",
            success=success,
            message=f"Performance validation {'succeeded' if success else 'failed'} for {integration.config.name}",
            details={"failures": failures, "iterations": iterations},
            metrics={
                "avg_response_time_ms": avg_response_time,
                "min_response_time_ms": min_response_time,
                "max_response_time_ms": max_response_time,
                "p95_response_time_ms": p95_response_time,
                "success_rate": (iterations - failures) / iterations
            }
        )
    
    def validate_security(self, integration_id: str) -> ValidationResult:
        """
        Validate security for an integration.
        
        Args:
            integration_id: ID of the integration to validate
            
        Returns:
            ValidationResult with the outcome of the security check
        """
        integration = self.integration_manager.registry.get(integration_id)
        if not integration:
            return ValidationResult(
                integration_id=integration_id,
                validation_type="security",
                success=False,
                message=f"Integration with ID {integration_id} not found"
            )
        
        # For this example, we'll perform some basic security checks
        security_checks = []
        
        # Check 1: Ensure timeout is set
        timeout_check = {
            "name": "timeout_configured",
            "success": integration.config.timeout_seconds > 0,
            "message": f"Timeout is {'configured' if integration.config.timeout_seconds > 0 else 'not configured'}"
        }
        security_checks.append(timeout_check)
        
        # Check 2: Ensure retry count is reasonable
        retry_check = {
            "name": "retry_count_reasonable",
            "success": 0 < integration.config.retry_count <= 5,
            "message": f"Retry count is {'reasonable' if 0 < integration.config.retry_count <= 5 else 'not reasonable'}"
        }
        security_checks.append(retry_check)
        
        # Check 3: Ensure circuit breaker is configured
        circuit_breaker_check = {
            "name": "circuit_breaker_configured",
            "success": integration.config.circuit_breaker_threshold > 0,
            "message": f"Circuit breaker is {'configured' if integration.config.circuit_breaker_threshold > 0 else 'not configured'}"
        }
        security_checks.append(circuit_breaker_check)
        
        # Determine overall success
        all_checks_passed = all(check["success"] for check in security_checks)
        
        return ValidationResult(
            integration_id=integration_id,
            validation_type="security",
            success=all_checks_passed,
            message=f"Security validation {'succeeded' if all_checks_passed else 'failed'} for {integration.config.name}",
            details={"checks": security_checks}
        )
    
    def validate_all_integrations(self) -> Dict[str, List[ValidationResult]]:
        """
        Validate all integrations in the system.
        
        Returns:
            Dictionary mapping integration IDs to lists of validation results
        """
        results = {}
        
        # Get all integrations
        integrations = self.integration_manager.registry.get_all()
        
        for integration in integrations:
            integration_id = integration.config.integration_id
            results[integration_id] = []
            
            # Run all validation types
            results[integration_id].append(self.validate_integration_health(integration_id))
            results[integration_id].append(self.validate_connectivity(integration_id))
            results[integration_id].append(self.validate_data_consistency(integration_id, {"test": "data"}))
            results[integration_id].append(self.validate_performance(integration_id))
            results[integration_id].append(self.validate_security(integration_id))
            
            # Store the results
            self.validation_results.extend(results[integration_id])
        
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all validation results.
        
        Returns:
            Dictionary containing validation summary
        """
        if not self.validation_results:
            return {
                "status": "NO_DATA",
                "message": "No validation results available",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        total_validations = len(self.validation_results)
        successful_validations = sum(1 for result in self.validation_results if result.success)
        failed_validations = total_validations - successful_validations
        
        # Group results by integration
        results_by_integration = {}
        for result in self.validation_results:
            if result.integration_id not in results_by_integration:
                results_by_integration[result.integration_id] = []
            results_by_integration[result.integration_id].append(result.to_dict())
        
        # Determine overall status
        status = "HEALTHY"
        if failed_validations > 0:
            status = "DEGRADED" if successful_validations > 0 else "CRITICAL"
        
        return {
            "status": status,
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "failed_validations": failed_validations,
            "success_rate": successful_validations / total_validations if total_validations > 0 else 0,
            "results_by_integration": results_by_integration,
            "timestamp": datetime.datetime.now().isoformat()
        }


# Singleton instance of the integration validator
_integration_validator = None

def get_integration_validator() -> IntegrationValidator:
    """
    Get the singleton instance of the integration validator.
    
    Returns:
        The integration validator instance
    """
    global _integration_validator
    if _integration_validator is None:
        _integration_validator = IntegrationValidator()
    return _integration_validator


# Example usage
if __name__ == "__main__":
    # Create the integration validator
    validator = get_integration_validator()
    
    # Validate a specific integration
    integration_id = "account_model_db"  # This should be registered by the component integrator
    
    health_result = validator.validate_integration_health(integration_id)
    print(f"Health validation: {health_result.success}, Message: {health_result.message}")
    
    connectivity_result = validator.validate_connectivity(integration_id)
    print(f"Connectivity validation: {connectivity_result.success}, Message: {connectivity_result.message}")
    
    # Validate all integrations
    all_results = validator.validate_all_integrations()
    print(f"Validated {len(all_results)} integrations")
    
    # Get validation summary
    summary = validator.get_validation_summary()
    print(f"Validation summary: {json.dumps(summary, indent=2)}")

