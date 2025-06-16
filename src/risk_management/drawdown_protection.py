"""
ALL-USE Risk Management: Drawdown Protection Module

This module implements automated drawdown protection and risk adjustment mechanisms
for the ALL-USE trading system. It provides real-time drawdown monitoring,
automated position adjustments, and emergency protection protocols.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
import sys
import os
from dataclasses import dataclass
import asyncio
import threading
import time

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocol_engine.all_use_parameters import ALLUSEParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('all_use_drawdown_protection.log')
    ]
)

logger = logging.getLogger('all_use_drawdown_protection')

class ProtectionLevel(Enum):
    """Enumeration of protection levels."""
    NONE = "None"
    LIGHT = "Light"
    MODERATE = "Moderate"
    AGGRESSIVE = "Aggressive"
    EMERGENCY = "Emergency"

class AdjustmentAction(Enum):
    """Enumeration of adjustment actions."""
    REDUCE_POSITION = "Reduce_Position"
    CLOSE_POSITION = "Close_Position"
    HEDGE_POSITION = "Hedge_Position"
    STOP_TRADING = "Stop_Trading"
    EMERGENCY_EXIT = "Emergency_Exit"

@dataclass
class DrawdownEvent:
    """Drawdown event data structure."""
    portfolio_id: str
    timestamp: datetime
    drawdown_percentage: float
    portfolio_value: float
    peak_value: float
    duration_days: int
    protection_level: ProtectionLevel
    actions_taken: List[AdjustmentAction]

@dataclass
class RiskAdjustment:
    """Risk adjustment data structure."""
    portfolio_id: str
    timestamp: datetime
    adjustment_type: AdjustmentAction
    symbol: str
    original_size: float
    adjusted_size: float
    adjustment_percentage: float
    reason: str
    expected_risk_reduction: float

class DrawdownProtectionSystem:
    """
    Advanced drawdown protection and risk adjustment system for ALL-USE trading.
    
    This class provides:
    - Real-time drawdown monitoring and detection
    - Automated position size adjustments
    - Emergency stop-loss mechanisms
    - Adaptive recovery strategies
    - Risk-based position management
    """
    
    def __init__(self, adjustment_callback: Optional[Callable] = None):
        """Initialize the drawdown protection system."""
        self.parameters = ALLUSEParameters
        self.adjustment_callback = adjustment_callback
        
        # Protection configuration
        self.protection_config = {
            'drawdown_thresholds': {
                'light': 0.05,      # 5% drawdown
                'moderate': 0.10,   # 10% drawdown
                'aggressive': 0.15, # 15% drawdown
                'emergency': 0.20   # 20% drawdown
            },
            'adjustment_factors': {
                'light': 0.20,      # Reduce positions by 20%
                'moderate': 0.40,   # Reduce positions by 40%
                'aggressive': 0.60, # Reduce positions by 60%
                'emergency': 0.80   # Reduce positions by 80%
            },
            'recovery_thresholds': {
                'partial': 0.50,    # 50% recovery from max drawdown
                'full': 0.90        # 90% recovery from max drawdown
            },
            'monitoring_interval': 10,  # seconds
            'cooldown_period': 300,     # 5 minutes between adjustments
            'max_adjustments_per_day': 5
        }
        
        # Protection state
        self.portfolios = {}
        self.drawdown_history = {}
        self.adjustment_history = {}
        self.protection_active = {}
        self.last_adjustment = {}
        self.daily_adjustment_count = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        
        logger.info("Drawdown protection system initialized")
    
    def add_portfolio(self, portfolio_id: str, initial_value: float, 
                     protection_level: ProtectionLevel = ProtectionLevel.MODERATE) -> None:
        """
        Add a portfolio to drawdown protection.
        
        Args:
            portfolio_id: Unique portfolio identifier
            initial_value: Initial portfolio value
            protection_level: Protection level to apply
        """
        logger.info(f"Adding portfolio {portfolio_id} to drawdown protection")
        
        self.portfolios[portfolio_id] = {
            'initial_value': initial_value,
            'current_value': initial_value,
            'peak_value': initial_value,
            'protection_level': protection_level,
            'positions': {},
            'last_update': datetime.now()
        }
        
        self.drawdown_history[portfolio_id] = []
        self.adjustment_history[portfolio_id] = []
        self.protection_active[portfolio_id] = False
        self.last_adjustment[portfolio_id] = None
        self.daily_adjustment_count[portfolio_id] = 0
    
    def update_portfolio_value(self, portfolio_id: str, current_value: float, 
                              positions: Dict[str, Any]) -> None:
        """
        Update portfolio value and positions for drawdown monitoring.
        
        Args:
            portfolio_id: Portfolio identifier
            current_value: Current portfolio value
            positions: Current portfolio positions
        """
        if portfolio_id not in self.portfolios:
            logger.warning(f"Portfolio {portfolio_id} not found in protection system")
            return
        
        portfolio = self.portfolios[portfolio_id]
        
        # Update portfolio data
        portfolio['current_value'] = current_value
        portfolio['positions'] = positions
        portfolio['last_update'] = datetime.now()
        
        # Update peak value
        if current_value > portfolio['peak_value']:
            portfolio['peak_value'] = current_value
        
        # Check for drawdown protection triggers
        self._check_drawdown_protection(portfolio_id)
    
    def start_monitoring(self) -> None:
        """Start real-time drawdown monitoring."""
        if self.monitoring_active:
            logger.warning("Drawdown monitoring already active")
            return
        
        logger.info("Starting real-time drawdown monitoring")
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop real-time drawdown monitoring."""
        logger.info("Stopping real-time drawdown monitoring")
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for drawdown protection."""
        while self.monitoring_active:
            try:
                # Check all portfolios for drawdown protection
                for portfolio_id in self.portfolios.keys():
                    self._check_drawdown_protection(portfolio_id)
                
                # Reset daily adjustment counts at midnight
                self._reset_daily_counters()
                
                # Sleep until next check
                time.sleep(self.protection_config['monitoring_interval'])
                
            except Exception as e:
                logger.error(f"Error in drawdown monitoring loop: {str(e)}")
                time.sleep(5)
    
    def _check_drawdown_protection(self, portfolio_id: str) -> None:
        """
        Check if drawdown protection should be triggered.
        
        Args:
            portfolio_id: Portfolio identifier
        """
        portfolio = self.portfolios[portfolio_id]
        current_value = portfolio['current_value']
        peak_value = portfolio['peak_value']
        
        # Calculate current drawdown
        drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
        
        # Determine protection level needed
        protection_needed = self._determine_protection_level(drawdown)
        
        # Check if protection should be triggered
        if protection_needed != ProtectionLevel.NONE and not self.protection_active[portfolio_id]:
            self._trigger_drawdown_protection(portfolio_id, drawdown, protection_needed)
        
        # Check for recovery
        elif self.protection_active[portfolio_id]:
            self._check_recovery(portfolio_id, drawdown)
    
    def _determine_protection_level(self, drawdown: float) -> ProtectionLevel:
        """
        Determine the appropriate protection level based on drawdown.
        
        Args:
            drawdown: Current drawdown percentage
            
        Returns:
            ProtectionLevel enum
        """
        thresholds = self.protection_config['drawdown_thresholds']
        
        if drawdown >= thresholds['emergency']:
            return ProtectionLevel.EMERGENCY
        elif drawdown >= thresholds['aggressive']:
            return ProtectionLevel.AGGRESSIVE
        elif drawdown >= thresholds['moderate']:
            return ProtectionLevel.MODERATE
        elif drawdown >= thresholds['light']:
            return ProtectionLevel.LIGHT
        else:
            return ProtectionLevel.NONE
    
    def _trigger_drawdown_protection(self, portfolio_id: str, drawdown: float, 
                                   protection_level: ProtectionLevel) -> None:
        """
        Trigger drawdown protection measures.
        
        Args:
            portfolio_id: Portfolio identifier
            drawdown: Current drawdown percentage
            protection_level: Protection level to apply
        """
        logger.warning(f"Triggering {protection_level.value} drawdown protection for {portfolio_id}: {drawdown:.2%}")
        
        portfolio = self.portfolios[portfolio_id]
        
        # Check cooldown period
        if self._is_in_cooldown(portfolio_id):
            logger.info(f"Drawdown protection in cooldown for {portfolio_id}")
            return
        
        # Check daily adjustment limit
        if self._exceeds_daily_limit(portfolio_id):
            logger.warning(f"Daily adjustment limit exceeded for {portfolio_id}")
            return
        
        # Apply protection measures
        actions_taken = self._apply_protection_measures(portfolio_id, protection_level)
        
        # Record drawdown event
        drawdown_event = DrawdownEvent(
            portfolio_id=portfolio_id,
            timestamp=datetime.now(),
            drawdown_percentage=drawdown,
            portfolio_value=portfolio['current_value'],
            peak_value=portfolio['peak_value'],
            duration_days=self._calculate_drawdown_duration(portfolio_id),
            protection_level=protection_level,
            actions_taken=actions_taken
        )
        
        self.drawdown_history[portfolio_id].append(drawdown_event)
        self.protection_active[portfolio_id] = True
        self.last_adjustment[portfolio_id] = datetime.now()
        self.daily_adjustment_count[portfolio_id] += 1
        
        # Keep only recent history
        if len(self.drawdown_history[portfolio_id]) > 100:
            self.drawdown_history[portfolio_id] = self.drawdown_history[portfolio_id][-100:]
    
    def _apply_protection_measures(self, portfolio_id: str, 
                                 protection_level: ProtectionLevel) -> List[AdjustmentAction]:
        """
        Apply specific protection measures based on protection level.
        
        Args:
            portfolio_id: Portfolio identifier
            protection_level: Protection level to apply
            
        Returns:
            List of actions taken
        """
        portfolio = self.portfolios[portfolio_id]
        positions = portfolio['positions']
        actions_taken = []
        
        # Get adjustment factor
        adjustment_factor = self.protection_config['adjustment_factors'].get(
            protection_level.value.lower(), 0.5
        )
        
        if protection_level == ProtectionLevel.EMERGENCY:
            # Emergency measures: Close high-risk positions
            actions_taken.extend(self._emergency_position_closure(portfolio_id))
        
        elif protection_level in [ProtectionLevel.AGGRESSIVE, ProtectionLevel.MODERATE]:
            # Aggressive/Moderate: Reduce position sizes
            actions_taken.extend(self._reduce_position_sizes(portfolio_id, adjustment_factor))
        
        elif protection_level == ProtectionLevel.LIGHT:
            # Light: Reduce only high-risk positions
            actions_taken.extend(self._reduce_high_risk_positions(portfolio_id, adjustment_factor))
        
        return actions_taken
    
    def _emergency_position_closure(self, portfolio_id: str) -> List[AdjustmentAction]:
        """
        Implement emergency position closure measures.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            List of actions taken
        """
        portfolio = self.portfolios[portfolio_id]
        positions = portfolio['positions']
        actions_taken = []
        
        # Close positions with highest risk first
        risk_sorted_positions = self._sort_positions_by_risk(positions)
        
        for symbol, position in risk_sorted_positions[:3]:  # Close top 3 riskiest
            original_size = position.get('quantity', 0)
            
            if original_size > 0:
                adjustment = RiskAdjustment(
                    portfolio_id=portfolio_id,
                    timestamp=datetime.now(),
                    adjustment_type=AdjustmentAction.CLOSE_POSITION,
                    symbol=symbol,
                    original_size=original_size,
                    adjusted_size=0,
                    adjustment_percentage=1.0,
                    reason=f"Emergency drawdown protection",
                    expected_risk_reduction=0.8
                )
                
                self.adjustment_history[portfolio_id].append(adjustment)
                actions_taken.append(AdjustmentAction.CLOSE_POSITION)
                
                # Trigger callback if provided
                if self.adjustment_callback:
                    try:
                        self.adjustment_callback(adjustment)
                    except Exception as e:
                        logger.error(f"Error in adjustment callback: {str(e)}")
                
                logger.warning(f"Emergency closure of {symbol} position for {portfolio_id}")
        
        return actions_taken
    
    def _reduce_position_sizes(self, portfolio_id: str, adjustment_factor: float) -> List[AdjustmentAction]:
        """
        Reduce position sizes across the portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            adjustment_factor: Factor by which to reduce positions
            
        Returns:
            List of actions taken
        """
        portfolio = self.portfolios[portfolio_id]
        positions = portfolio['positions']
        actions_taken = []
        
        for symbol, position in positions.items():
            original_size = position.get('quantity', 0)
            
            if original_size > 0:
                adjusted_size = original_size * (1 - adjustment_factor)
                
                adjustment = RiskAdjustment(
                    portfolio_id=portfolio_id,
                    timestamp=datetime.now(),
                    adjustment_type=AdjustmentAction.REDUCE_POSITION,
                    symbol=symbol,
                    original_size=original_size,
                    adjusted_size=adjusted_size,
                    adjustment_percentage=adjustment_factor,
                    reason=f"Drawdown protection: {adjustment_factor:.0%} reduction",
                    expected_risk_reduction=adjustment_factor * 0.8
                )
                
                self.adjustment_history[portfolio_id].append(adjustment)
                actions_taken.append(AdjustmentAction.REDUCE_POSITION)
                
                # Trigger callback if provided
                if self.adjustment_callback:
                    try:
                        self.adjustment_callback(adjustment)
                    except Exception as e:
                        logger.error(f"Error in adjustment callback: {str(e)}")
                
                logger.info(f"Reduced {symbol} position by {adjustment_factor:.0%} for {portfolio_id}")
        
        return actions_taken
    
    def _reduce_high_risk_positions(self, portfolio_id: str, adjustment_factor: float) -> List[AdjustmentAction]:
        """
        Reduce only high-risk positions.
        
        Args:
            portfolio_id: Portfolio identifier
            adjustment_factor: Factor by which to reduce positions
            
        Returns:
            List of actions taken
        """
        portfolio = self.portfolios[portfolio_id]
        positions = portfolio['positions']
        actions_taken = []
        
        # Identify high-risk positions (simplified)
        high_risk_symbols = self._identify_high_risk_positions(positions)
        
        for symbol in high_risk_symbols:
            if symbol in positions:
                position = positions[symbol]
                original_size = position.get('quantity', 0)
                
                if original_size > 0:
                    adjusted_size = original_size * (1 - adjustment_factor)
                    
                    adjustment = RiskAdjustment(
                        portfolio_id=portfolio_id,
                        timestamp=datetime.now(),
                        adjustment_type=AdjustmentAction.REDUCE_POSITION,
                        symbol=symbol,
                        original_size=original_size,
                        adjusted_size=adjusted_size,
                        adjustment_percentage=adjustment_factor,
                        reason=f"High-risk position reduction: {adjustment_factor:.0%}",
                        expected_risk_reduction=adjustment_factor * 0.6
                    )
                    
                    self.adjustment_history[portfolio_id].append(adjustment)
                    actions_taken.append(AdjustmentAction.REDUCE_POSITION)
                    
                    # Trigger callback if provided
                    if self.adjustment_callback:
                        try:
                            self.adjustment_callback(adjustment)
                        except Exception as e:
                            logger.error(f"Error in adjustment callback: {str(e)}")
                    
                    logger.info(f"Reduced high-risk {symbol} position by {adjustment_factor:.0%} for {portfolio_id}")
        
        return actions_taken
    
    def _check_recovery(self, portfolio_id: str, current_drawdown: float) -> None:
        """
        Check if portfolio has recovered enough to reduce protection.
        
        Args:
            portfolio_id: Portfolio identifier
            current_drawdown: Current drawdown percentage
        """
        # Get maximum drawdown from recent history
        recent_events = [
            event for event in self.drawdown_history[portfolio_id]
            if (datetime.now() - event.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        if not recent_events:
            return
        
        max_recent_drawdown = max(event.drawdown_percentage for event in recent_events)
        recovery_percentage = (max_recent_drawdown - current_drawdown) / max_recent_drawdown if max_recent_drawdown > 0 else 0
        
        # Check for significant recovery
        if recovery_percentage >= self.protection_config['recovery_thresholds']['full']:
            self._deactivate_protection(portfolio_id, "Full recovery achieved")
        elif recovery_percentage >= self.protection_config['recovery_thresholds']['partial']:
            self._reduce_protection_level(portfolio_id, "Partial recovery achieved")
    
    def _deactivate_protection(self, portfolio_id: str, reason: str) -> None:
        """
        Deactivate drawdown protection for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            reason: Reason for deactivation
        """
        logger.info(f"Deactivating drawdown protection for {portfolio_id}: {reason}")
        self.protection_active[portfolio_id] = False
    
    def _reduce_protection_level(self, portfolio_id: str, reason: str) -> None:
        """
        Reduce protection level for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            reason: Reason for reduction
        """
        logger.info(f"Reducing protection level for {portfolio_id}: {reason}")
        # Implementation would reduce the aggressiveness of protection measures
    
    def _sort_positions_by_risk(self, positions: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """
        Sort positions by risk level (highest risk first).
        
        Args:
            positions: Portfolio positions
            
        Returns:
            List of (symbol, position) tuples sorted by risk
        """
        # Simplified risk scoring based on volatility and position size
        risk_scores = []
        
        for symbol, position in positions.items():
            volatility = self._get_symbol_volatility(symbol)
            market_value = position.get('market_value', 0)
            
            # Risk score combines volatility and position size
            risk_score = volatility * market_value
            risk_scores.append((risk_score, symbol, position))
        
        # Sort by risk score (highest first)
        risk_scores.sort(reverse=True)
        
        return [(symbol, position) for _, symbol, position in risk_scores]
    
    def _identify_high_risk_positions(self, positions: Dict[str, Any]) -> List[str]:
        """
        Identify high-risk positions in the portfolio.
        
        Args:
            positions: Portfolio positions
            
        Returns:
            List of high-risk symbols
        """
        high_risk_symbols = []
        
        for symbol, position in positions.items():
            volatility = self._get_symbol_volatility(symbol)
            
            # Consider high volatility stocks as high risk
            if volatility > 0.35:  # 35% annualized volatility threshold
                high_risk_symbols.append(symbol)
        
        return high_risk_symbols
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """Get volatility for a symbol (annualized)."""
        volatility_estimates = {
            'AAPL': 0.25, 'MSFT': 0.22, 'AMZN': 0.30,
            'GOOGL': 0.28, 'TSLA': 0.45, 'NVDA': 0.40,
            'META': 0.35, 'NFLX': 0.38
        }
        return volatility_estimates.get(symbol, 0.30)
    
    def _calculate_drawdown_duration(self, portfolio_id: str) -> int:
        """
        Calculate the duration of the current drawdown in days.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Duration in days
        """
        # Simplified calculation - would need historical data for accuracy
        return 1  # Default to 1 day
    
    def _is_in_cooldown(self, portfolio_id: str) -> bool:
        """
        Check if portfolio is in cooldown period.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            True if in cooldown
        """
        last_adj = self.last_adjustment.get(portfolio_id)
        if not last_adj:
            return False
        
        cooldown_seconds = self.protection_config['cooldown_period']
        return (datetime.now() - last_adj).total_seconds() < cooldown_seconds
    
    def _exceeds_daily_limit(self, portfolio_id: str) -> bool:
        """
        Check if daily adjustment limit is exceeded.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            True if limit exceeded
        """
        daily_count = self.daily_adjustment_count.get(portfolio_id, 0)
        max_daily = self.protection_config['max_adjustments_per_day']
        return daily_count >= max_daily
    
    def _reset_daily_counters(self) -> None:
        """Reset daily adjustment counters at midnight."""
        now = datetime.now()
        if now.hour == 0 and now.minute == 0:
            for portfolio_id in self.daily_adjustment_count:
                self.daily_adjustment_count[portfolio_id] = 0
    
    def get_protection_status(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Get comprehensive protection status for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Dict containing protection status
        """
        if portfolio_id not in self.portfolios:
            return {'error': f'Portfolio {portfolio_id} not found'}
        
        portfolio = self.portfolios[portfolio_id]
        current_drawdown = (portfolio['peak_value'] - portfolio['current_value']) / portfolio['peak_value'] if portfolio['peak_value'] > 0 else 0
        
        # Get recent adjustments
        recent_adjustments = [
            adj for adj in self.adjustment_history.get(portfolio_id, [])
            if (datetime.now() - adj.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        # Get recent drawdown events
        recent_events = [
            event for event in self.drawdown_history.get(portfolio_id, [])
            if (datetime.now() - event.timestamp).total_seconds() < 86400  # Last day
        ]
        
        return {
            'portfolio_id': portfolio_id,
            'timestamp': datetime.now(),
            'protection_active': self.protection_active.get(portfolio_id, False),
            'current_drawdown': current_drawdown,
            'protection_level_needed': self._determine_protection_level(current_drawdown).value,
            'portfolio_metrics': {
                'current_value': portfolio['current_value'],
                'peak_value': portfolio['peak_value'],
                'initial_value': portfolio['initial_value']
            },
            'recent_adjustments': len(recent_adjustments),
            'recent_events': len(recent_events),
            'daily_adjustment_count': self.daily_adjustment_count.get(portfolio_id, 0),
            'in_cooldown': self._is_in_cooldown(portfolio_id),
            'adjustment_details': [
                {
                    'timestamp': adj.timestamp,
                    'action': adj.adjustment_type.value,
                    'symbol': adj.symbol,
                    'adjustment_percentage': adj.adjustment_percentage,
                    'reason': adj.reason
                }
                for adj in recent_adjustments[-5:]  # Last 5 adjustments
            ]
        }
    
    def get_all_portfolios_status(self) -> Dict[str, Any]:
        """
        Get protection status for all monitored portfolios.
        
        Returns:
            Dict containing status for all portfolios
        """
        statuses = {}
        
        for portfolio_id in self.portfolios.keys():
            statuses[portfolio_id] = self.get_protection_status(portfolio_id)
        
        # Overall system metrics
        total_active = sum(1 for active in self.protection_active.values() if active)
        total_adjustments = sum(len(history) for history in self.adjustment_history.values())
        
        return {
            'timestamp': datetime.now(),
            'total_portfolios': len(self.portfolios),
            'monitoring_active': self.monitoring_active,
            'total_active_protections': total_active,
            'total_adjustments_today': sum(self.daily_adjustment_count.values()),
            'total_historical_adjustments': total_adjustments,
            'portfolios': statuses
        }


# Example usage and testing
if __name__ == "__main__":
    def adjustment_handler(adjustment: RiskAdjustment):
        """Example adjustment handler."""
        print(f"ADJUSTMENT: {adjustment.adjustment_type.value} - {adjustment.symbol} by {adjustment.adjustment_percentage:.0%}")
    
    # Create drawdown protection system
    protection = DrawdownProtectionSystem(adjustment_callback=adjustment_handler)
    
    # Add test portfolio
    protection.add_portfolio('test_portfolio', 100000.0, ProtectionLevel.MODERATE)
    
    # Simulate portfolio value updates with drawdown
    test_values = [100000, 95000, 90000, 85000, 80000]  # 20% drawdown
    test_positions = {
        'TSLA': {'quantity': 100, 'market_value': 25000},
        'AAPL': {'quantity': 200, 'market_value': 36000},
        'NVDA': {'quantity': 50, 'market_value': 40000}
    }
    
    print("=== Drawdown Protection System Test ===")
    
    for i, value in enumerate(test_values):
        print(f"\nStep {i+1}: Portfolio Value = ${value:,}")
        
        # Update portfolio value
        protection.update_portfolio_value('test_portfolio', value, test_positions)
        
        # Get protection status
        status = protection.get_protection_status('test_portfolio')
        
        print(f"Current Drawdown: {status['current_drawdown']:.2%}")
        print(f"Protection Active: {status['protection_active']}")
        print(f"Protection Level Needed: {status['protection_level_needed']}")
        print(f"Recent Adjustments: {status['recent_adjustments']}")
        
        # Simulate time passing
        time.sleep(0.1)
    
    # Test recovery scenario
    print(f"\n=== Recovery Test ===")
    recovery_values = [82000, 85000, 90000, 95000]
    
    for i, value in enumerate(recovery_values):
        print(f"\nRecovery Step {i+1}: Portfolio Value = ${value:,}")
        
        protection.update_portfolio_value('test_portfolio', value, test_positions)
        status = protection.get_protection_status('test_portfolio')
        
        print(f"Current Drawdown: {status['current_drawdown']:.2%}")
        print(f"Protection Active: {status['protection_active']}")
    
    # Get overall system status
    overall_status = protection.get_all_portfolios_status()
    print(f"\nTotal Active Protections: {overall_status['total_active_protections']}")
    print(f"Total Adjustments Today: {overall_status['total_adjustments_today']}")

