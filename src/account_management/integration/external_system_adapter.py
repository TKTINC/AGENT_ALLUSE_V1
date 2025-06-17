#!/usr/bin/env python3
"""
ALL-USE Account Management System - External System Adapter

This module provides integration adapters for external systems that interact
with the ALL-USE Account Management System, including strategy engine,
market integration, user management, notification, and reporting systems.

The external system adapter implements:
- Strategy Engine integration
- Market Integration system integration
- User Management system integration
- Notification System integration
- Reporting System integration

Author: Manus AI
Date: June 17, 2025
Version: 1.0
"""

import logging
import time
import json
import threading
import requests
import datetime
from typing import Dict, List, Any, Callable, Optional, Union, Tuple

from .integration_framework import (
    IntegrationManager, IntegrationConfiguration, IntegrationType,
    IntegrationError, IntegrationConnectionError, get_integration_manager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExternalSystemAdapter:
    """
    Provides integration adapters for external systems.
    
    This class implements adapters for various external systems that interact
    with the account management system, ensuring seamless integration.
    """
    
    def __init__(self):
        """Initialize the external system adapter."""
        self.integration_manager = get_integration_manager()
        self._register_external_integrations()
        logger.info("External System Adapter initialized")
    
    def _register_external_integrations(self):
        """Register all external system integrations with the integration manager."""
        # Strategy Engine integration
        self.integration_manager.register_integration(
            IntegrationConfiguration(
                integration_id="strategy_engine",
                integration_type=IntegrationType.EXTERNAL,
                name="Strategy Engine Integration",
                description="Integration with the strategy engine for account strategy execution",
                config={
                    "api_url": "http://strategy-engine.alluse.internal/api/v1",
                    "timeout_seconds": 30
                },
                timeout_seconds=30,
                retry_count=3,
                retry_delay_seconds=2,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout_seconds=60
            )
        )
        
        # Market Integration system
        self.integration_manager.register_integration(
            IntegrationConfiguration(
                integration_id="market_integration",
                integration_type=IntegrationType.EXTERNAL,
                name="Market Integration System",
                description="Integration with the market integration system for market data and trading",
                config={
                    "api_url": "http://market-integration.alluse.internal/api/v1",
                    "timeout_seconds": 20
                },
                timeout_seconds=20,
                retry_count=3,
                retry_delay_seconds=2,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout_seconds=60
            )
        )
        
        # User Management system
        self.integration_manager.register_integration(
            IntegrationConfiguration(
                integration_id="user_management",
                integration_type=IntegrationType.EXTERNAL,
                name="User Management System",
                description="Integration with the user management system for user authentication and authorization",
                config={
                    "api_url": "http://user-management.alluse.internal/api/v1",
                    "timeout_seconds": 10
                },
                timeout_seconds=10,
                retry_count=3,
                retry_delay_seconds=1,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout_seconds=60
            )
        )
        
        # Notification System
        self.integration_manager.register_integration(
            IntegrationConfiguration(
                integration_id="notification_system",
                integration_type=IntegrationType.EXTERNAL,
                name="Notification System",
                description="Integration with the notification system for alerts and messages",
                config={
                    "api_url": "http://notification.alluse.internal/api/v1",
                    "timeout_seconds": 5
                },
                timeout_seconds=5,
                retry_count=3,
                retry_delay_seconds=1,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout_seconds=60
            )
        )
        
        # Reporting System
        self.integration_manager.register_integration(
            IntegrationConfiguration(
                integration_id="reporting_system",
                integration_type=IntegrationType.EXTERNAL,
                name="Reporting System",
                description="Integration with the reporting system for account reports and analytics",
                config={
                    "api_url": "http://reporting.alluse.internal/api/v1",
                    "timeout_seconds": 15
                },
                timeout_seconds=15,
                retry_count=3,
                retry_delay_seconds=2,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout_seconds=60
            )
        )
    
    def _make_api_request(self, integration_id: str, method: str, endpoint: str, 
                         data: Dict = None, params: Dict = None) -> Dict:
        """
        Make an API request to an external system.
        
        Args:
            integration_id: ID of the integration to use
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data (for POST, PUT)
            params: Query parameters (for GET)
            
        Returns:
            API response as dictionary
            
        Raises:
            IntegrationError: If the API request fails
        """
        # Get the integration configuration
        integration = self.integration_manager.registry.get(integration_id)
        if not integration:
            raise IntegrationError(f"Integration with ID {integration_id} not found")
        
        # Get the API URL from the integration configuration
        api_url = integration.config.config.get("api_url")
        if not api_url:
            raise IntegrationError(f"API URL not configured for integration {integration_id}")
        
        # Build the full URL
        url = f"{api_url}/{endpoint.lstrip('/')}"
        
        # Define the request function
        def request_operation():
            try:
                headers = {
                    "Content-Type": "application/json",
                    "X-Integration-ID": integration_id
                }
                
                timeout = integration.config.config.get("timeout_seconds", 30)
                
                if method.upper() == "GET":
                    response = requests.get(url, params=params, headers=headers, timeout=timeout)
                elif method.upper() == "POST":
                    response = requests.post(url, json=data, headers=headers, timeout=timeout)
                elif method.upper() == "PUT":
                    response = requests.put(url, json=data, headers=headers, timeout=timeout)
                elif method.upper() == "DELETE":
                    response = requests.delete(url, json=data, headers=headers, timeout=timeout)
                else:
                    raise IntegrationError(f"Unsupported HTTP method: {method}")
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse and return the JSON response
                return response.json()
                
            except requests.exceptions.RequestException as e:
                # Convert requests exceptions to integration errors
                if isinstance(e, requests.exceptions.Timeout):
                    raise IntegrationError(f"Request timed out: {str(e)}", error_code="TIMEOUT")
                elif isinstance(e, requests.exceptions.ConnectionError):
                    raise IntegrationConnectionError(f"Connection error: {str(e)}", error_code="CONNECTION_ERROR")
                elif isinstance(e, requests.exceptions.HTTPError):
                    status_code = e.response.status_code if hasattr(e, 'response') else "unknown"
                    raise IntegrationError(
                        f"HTTP error {status_code}: {str(e)}", 
                        error_code=f"HTTP_{status_code}",
                        details={"status_code": status_code}
                    )
                else:
                    raise IntegrationError(f"Request error: {str(e)}")
        
        # Execute the request through the integration manager
        return self.integration_manager.execute_integration(integration_id, request_operation)
    
    def strategy_engine_request(self, method: str, endpoint: str, data: Dict = None, params: Dict = None) -> Dict:
        """
        Make a request to the strategy engine.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data (for POST, PUT)
            params: Query parameters (for GET)
            
        Returns:
            Strategy engine API response
            
        Raises:
            IntegrationError: If the request fails
        """
        return self._make_api_request("strategy_engine", method, endpoint, data, params)
    
    def market_integration_request(self, method: str, endpoint: str, data: Dict = None, params: Dict = None) -> Dict:
        """
        Make a request to the market integration system.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data (for POST, PUT)
            params: Query parameters (for GET)
            
        Returns:
            Market integration API response
            
        Raises:
            IntegrationError: If the request fails
        """
        return self._make_api_request("market_integration", method, endpoint, data, params)
    
    def user_management_request(self, method: str, endpoint: str, data: Dict = None, params: Dict = None) -> Dict:
        """
        Make a request to the user management system.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data (for POST, PUT)
            params: Query parameters (for GET)
            
        Returns:
            User management API response
            
        Raises:
            IntegrationError: If the request fails
        """
        return self._make_api_request("user_management", method, endpoint, data, params)
    
    def notification_system_request(self, method: str, endpoint: str, data: Dict = None, params: Dict = None) -> Dict:
        """
        Make a request to the notification system.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data (for POST, PUT)
            params: Query parameters (for GET)
            
        Returns:
            Notification system API response
            
        Raises:
            IntegrationError: If the request fails
        """
        return self._make_api_request("notification_system", method, endpoint, data, params)
    
    def reporting_system_request(self, method: str, endpoint: str, data: Dict = None, params: Dict = None) -> Dict:
        """
        Make a request to the reporting system.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data (for POST, PUT)
            params: Query parameters (for GET)
            
        Returns:
            Reporting system API response
            
        Raises:
            IntegrationError: If the request fails
        """
        return self._make_api_request("reporting_system", method, endpoint, data, params)
    
    def get_external_system_health(self) -> Dict[str, Any]:
        """
        Get the health status of all external system integrations.
        
        Returns:
            Dict containing health status information
        """
        # Get all external integrations
        external_integrations = self.integration_manager.registry.get_by_type(IntegrationType.EXTERNAL)
        
        # Extract health information
        total = len(external_integrations)
        health_data = {
            "total_integrations": total,
            "status": "HEALTHY",
            "systems": {}
        }
        
        for integration in external_integrations:
            system_name = integration.config.name
            system_status = integration.status.value
            circuit_breaker_state = integration.circuit_breaker.get_state()
            
            health_data["systems"][integration.config.integration_id] = {
                "name": system_name,
                "status": system_status,
                "circuit_breaker_state": circuit_breaker_state
            }
            
            # Update overall status
            if system_status == "FAILED":
                health_data["status"] = "CRITICAL"
            elif system_status == "DEGRADED" and health_data["status"] != "CRITICAL":
                health_data["status"] = "DEGRADED"
        
        health_data["timestamp"] = datetime.datetime.now().isoformat()
        return health_data
    
    def get_external_system_metrics(self) -> List[Dict[str, Any]]:
        """
        Get metrics for all external system integrations.
        
        Returns:
            List of dictionaries containing metrics for each integration
        """
        # Get all external integrations
        external_integrations = self.integration_manager.registry.get_by_type(IntegrationType.EXTERNAL)
        
        # Extract metrics
        return [integration.get_metrics() for integration in external_integrations]


# Singleton instance of the external system adapter
_external_system_adapter = None

def get_external_system_adapter() -> ExternalSystemAdapter:
    """
    Get the singleton instance of the external system adapter.
    
    Returns:
        The external system adapter instance
    """
    global _external_system_adapter
    if _external_system_adapter is None:
        _external_system_adapter = ExternalSystemAdapter()
    return _external_system_adapter


# Example usage
if __name__ == "__main__":
    # Create the external system adapter
    adapter = get_external_system_adapter()
    
    # Mock the requests module for testing
    import unittest.mock
    
    with unittest.mock.patch('requests.get') as mock_get:
        # Configure the mock to return a successful response
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"status": "success", "data": {"account_id": "ACC123456"}}
        mock_response.raise_for_status = unittest.mock.Mock()
        mock_get.return_value = mock_response
        
        # Make a request to the strategy engine
        try:
            result = adapter.strategy_engine_request(
                "GET", 
                "/accounts/ACC123456/strategy", 
                params={"include_history": "true"}
            )
            print(f"Result: {result}")
        except IntegrationError as e:
            print(f"Error: {e}")
    
    # Get health and metrics
    print(f"Health: {json.dumps(adapter.get_external_system_health(), indent=2)}")
    print(f"Metrics: {json.dumps(adapter.get_external_system_metrics(), indent=2)}")

