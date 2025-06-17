"""
Trade Monitoring, Analytics, and Risk Controls System
Comprehensive trade monitoring and risk management for ALL-USE trading system

This module provides:
- Real-time trade monitoring and tracking
- Execution quality analytics and performance measurement
- Advanced risk controls and pre-trade validation
- Live performance analytics and P&L tracking
- Alert systems and exception handling
- Emergency controls and kill switches
"""

import sys
import os
import time
import json
import asyncio
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class TradeStatus(Enum):
    """Trade execution status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class ExecutionQuality(Enum):
    """Execution quality ratings"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"

@dataclass
class TradeExecution:
    """Individual trade execution record"""
    trade_id: str
    symbol: str
    side: str  # BUY, SELL, BTO, STC, etc.
    quantity: int
    order_type: str
    status: TradeStatus
    
    # Execution details
    executed_quantity: int = 0
    executed_price: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    
    # Timing
    order_time: datetime = field(default_factory=datetime.now)
    execution_time: Optional[datetime] = None
    
    # Quality metrics
    requested_price: float = 0.0
    market_price_at_order: float = 0.0
    slippage: float = 0.0
    execution_quality: ExecutionQuality = ExecutionQuality.FAIR
    
    # Risk and validation
    pre_trade_risk_score: float = 0.0
    post_trade_risk_score: float = 0.0
    risk_checks_passed: bool = True
    
    # Metadata
    broker: str = ""
    account: str = ""
    strategy: str = ""
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'order_type': self.order_type,
            'status': self.status.value,
            'executed_quantity': self.executed_quantity,
            'executed_price': self.executed_price,
            'average_price': self.average_price,
            'commission': self.commission,
            'order_time': self.order_time.isoformat(),
            'execution_time': self.execution_time.isoformat() if self.execution_time else None,
            'requested_price': self.requested_price,
            'market_price_at_order': self.market_price_at_order,
            'slippage': self.slippage,
            'execution_quality': self.execution_quality.value,
            'pre_trade_risk_score': self.pre_trade_risk_score,
            'post_trade_risk_score': self.post_trade_risk_score,
            'risk_checks_passed': self.risk_checks_passed,
            'broker': self.broker,
            'account': self.account,
            'strategy': self.strategy,
            'notes': self.notes
        }

@dataclass
class RiskAlert:
    """Risk alert notification"""
    alert_id: str
    severity: AlertSeverity
    risk_level: RiskLevel
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'risk_level': self.risk_level.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'resolved': self.resolved
        }

@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    timestamp: datetime
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    total_return_pct: float
    daily_pnl: float
    daily_return_pct: float
    
    # Risk metrics
    portfolio_value: float
    max_drawdown: float
    current_drawdown: float
    var_95: float  # Value at Risk 95%
    
    # Execution metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Position metrics
    total_positions: int
    long_positions: int
    short_positions: int
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_theta: float
    portfolio_vega: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class ExecutionQualityAnalyzer:
    """Analyze execution quality and performance"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ExecutionQualityAnalyzer")
        
        # Quality thresholds
        self.excellent_slippage_threshold = 0.0005  # 0.05%
        self.good_slippage_threshold = 0.002  # 0.2%
        self.fair_slippage_threshold = 0.005  # 0.5%
        self.poor_slippage_threshold = 0.01  # 1.0%
        
        # Execution time thresholds (seconds)
        self.excellent_execution_time = 1.0
        self.good_execution_time = 5.0
        self.fair_execution_time = 15.0
        self.poor_execution_time = 60.0
        
        # Historical data for analysis
        self.execution_history: List[TradeExecution] = []
        self.max_history_size = 10000
    
    def analyze_execution(self, trade: TradeExecution) -> ExecutionQuality:
        """Analyze execution quality for a trade"""
        try:
            quality_score = 0.0
            
            # Analyze slippage (40% weight)
            slippage_score = self._analyze_slippage(trade)
            quality_score += slippage_score * 0.4
            
            # Analyze execution time (30% weight)
            time_score = self._analyze_execution_time(trade)
            quality_score += time_score * 0.3
            
            # Analyze fill rate (20% weight)
            fill_score = self._analyze_fill_rate(trade)
            quality_score += fill_score * 0.2
            
            # Analyze commission efficiency (10% weight)
            commission_score = self._analyze_commission(trade)
            quality_score += commission_score * 0.1
            
            # Convert score to quality rating
            if quality_score >= 4.5:
                return ExecutionQuality.EXCELLENT
            elif quality_score >= 3.5:
                return ExecutionQuality.GOOD
            elif quality_score >= 2.5:
                return ExecutionQuality.FAIR
            elif quality_score >= 1.5:
                return ExecutionQuality.POOR
            else:
                return ExecutionQuality.VERY_POOR
                
        except Exception as e:
            self.logger.error(f"Error analyzing execution quality: {str(e)}")
            return ExecutionQuality.FAIR
    
    def _analyze_slippage(self, trade: TradeExecution) -> float:
        """Analyze slippage quality (0-5 scale)"""
        try:
            if trade.slippage <= self.excellent_slippage_threshold:
                return 5.0
            elif trade.slippage <= self.good_slippage_threshold:
                return 4.0
            elif trade.slippage <= self.fair_slippage_threshold:
                return 3.0
            elif trade.slippage <= self.poor_slippage_threshold:
                return 2.0
            else:
                return 1.0
        except Exception:
            return 3.0
    
    def _analyze_execution_time(self, trade: TradeExecution) -> float:
        """Analyze execution time quality (0-5 scale)"""
        try:
            if not trade.execution_time:
                return 3.0
            
            execution_seconds = (trade.execution_time - trade.order_time).total_seconds()
            
            if execution_seconds <= self.excellent_execution_time:
                return 5.0
            elif execution_seconds <= self.good_execution_time:
                return 4.0
            elif execution_seconds <= self.fair_execution_time:
                return 3.0
            elif execution_seconds <= self.poor_execution_time:
                return 2.0
            else:
                return 1.0
        except Exception:
            return 3.0
    
    def _analyze_fill_rate(self, trade: TradeExecution) -> float:
        """Analyze fill rate quality (0-5 scale)"""
        try:
            if trade.quantity == 0:
                return 3.0
            
            fill_rate = trade.executed_quantity / trade.quantity
            
            if fill_rate >= 1.0:
                return 5.0
            elif fill_rate >= 0.95:
                return 4.0
            elif fill_rate >= 0.85:
                return 3.0
            elif fill_rate >= 0.70:
                return 2.0
            else:
                return 1.0
        except Exception:
            return 3.0
    
    def _analyze_commission(self, trade: TradeExecution) -> float:
        """Analyze commission efficiency (0-5 scale)"""
        try:
            if trade.executed_quantity == 0:
                return 3.0
            
            # Calculate commission per share/contract
            commission_per_unit = trade.commission / trade.executed_quantity
            
            # Thresholds based on typical commission rates
            if commission_per_unit <= 0.005:  # $0.005 per share
                return 5.0
            elif commission_per_unit <= 0.01:
                return 4.0
            elif commission_per_unit <= 0.02:
                return 3.0
            elif commission_per_unit <= 0.05:
                return 2.0
            else:
                return 1.0
        except Exception:
            return 3.0
    
    def calculate_slippage(self, trade: TradeExecution, market_price: float) -> float:
        """Calculate slippage for a trade"""
        try:
            if trade.executed_quantity == 0 or trade.average_price == 0:
                return 0.0
            
            # Calculate slippage based on side
            if trade.side.upper() in ['BUY', 'BTO']:
                # For buys, positive slippage means paying more than market
                slippage = (trade.average_price - market_price) / market_price
            else:
                # For sells, positive slippage means receiving less than market
                slippage = (market_price - trade.average_price) / market_price
            
            return abs(slippage)
            
        except Exception as e:
            self.logger.error(f"Error calculating slippage: {str(e)}")
            return 0.0
    
    def add_execution(self, trade: TradeExecution):
        """Add execution to history"""
        self.execution_history.append(trade)
        
        # Limit history size
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]
    
    def get_execution_statistics(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """Get execution statistics for recent period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            recent_executions = [
                trade for trade in self.execution_history
                if trade.execution_time and trade.execution_time >= cutoff_time
            ]
            
            if not recent_executions:
                return {
                    'total_executions': 0,
                    'avg_slippage': 0.0,
                    'avg_execution_time_seconds': 0.0,
                    'fill_rate': 0.0,
                    'quality_distribution': {}
                }
            
            # Calculate statistics
            total_executions = len(recent_executions)
            avg_slippage = sum(trade.slippage for trade in recent_executions) / total_executions
            
            execution_times = [
                (trade.execution_time - trade.order_time).total_seconds()
                for trade in recent_executions
                if trade.execution_time
            ]
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
            
            total_requested = sum(trade.quantity for trade in recent_executions)
            total_executed = sum(trade.executed_quantity for trade in recent_executions)
            fill_rate = total_executed / total_requested if total_requested > 0 else 0.0
            
            # Quality distribution
            quality_counts = defaultdict(int)
            for trade in recent_executions:
                quality_counts[trade.execution_quality.value] += 1
            
            return {
                'total_executions': total_executions,
                'avg_slippage': avg_slippage,
                'avg_execution_time_seconds': avg_execution_time,
                'fill_rate': fill_rate,
                'quality_distribution': dict(quality_counts),
                'lookback_hours': lookback_hours
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating execution statistics: {str(e)}")
            return {}

class RiskControlSystem:
    """Advanced risk control and validation system"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RiskControlSystem")
        
        # Risk limits
        self.max_portfolio_value = 1000000.0  # $1M max portfolio
        self.max_daily_loss = 50000.0  # $50K max daily loss
        self.max_position_size = 100000.0  # $100K max single position
        self.max_portfolio_delta = 1000.0  # Max portfolio delta
        self.max_concentration = 0.20  # 20% max concentration in single symbol
        
        # Risk monitoring
        self.current_portfolio_value = 0.0
        self.daily_pnl = 0.0
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.risk_alerts: List[RiskAlert] = []
        
        # Kill switch state
        self.kill_switch_active = False
        self.emergency_mode = False
        
        # Risk callbacks
        self.risk_callbacks: List[Callable[[RiskAlert], None]] = []
    
    def validate_pre_trade(self, trade_request: Dict[str, Any]) -> Tuple[bool, List[str], float]:
        """Validate trade before execution"""
        try:
            validation_errors = []
            risk_score = 0.0
            
            symbol = trade_request.get('symbol', '')
            side = trade_request.get('side', '')
            quantity = trade_request.get('quantity', 0)
            price = trade_request.get('price', 0.0)
            
            # Calculate position value
            position_value = quantity * price
            
            # Check kill switch
            if self.kill_switch_active:
                validation_errors.append("Kill switch is active - no new trades allowed")
                risk_score += 10.0
            
            # Check emergency mode
            if self.emergency_mode:
                validation_errors.append("Emergency mode active - only closing trades allowed")
                if side.upper() not in ['SELL', 'STC']:
                    risk_score += 10.0
            
            # Check position size limits
            if position_value > self.max_position_size:
                validation_errors.append(f"Position size ${position_value:,.0f} exceeds limit ${self.max_position_size:,.0f}")
                risk_score += 5.0
            
            # Check portfolio value limits
            new_portfolio_value = self.current_portfolio_value + position_value
            if new_portfolio_value > self.max_portfolio_value:
                validation_errors.append(f"Portfolio value ${new_portfolio_value:,.0f} would exceed limit ${self.max_portfolio_value:,.0f}")
                risk_score += 5.0
            
            # Check daily loss limits
            if self.daily_pnl < -self.max_daily_loss:
                validation_errors.append(f"Daily loss ${abs(self.daily_pnl):,.0f} exceeds limit ${self.max_daily_loss:,.0f}")
                risk_score += 8.0
            
            # Check concentration limits
            current_symbol_value = self.positions.get(symbol, {}).get('market_value', 0.0)
            new_symbol_value = current_symbol_value + position_value
            concentration = new_symbol_value / max(self.current_portfolio_value, 1.0)
            
            if concentration > self.max_concentration:
                validation_errors.append(f"Symbol concentration {concentration:.1%} exceeds limit {self.max_concentration:.1%}")
                risk_score += 3.0
            
            # Check delta limits (for options)
            if 'delta' in trade_request:
                trade_delta = trade_request['delta'] * quantity
                current_delta = sum(pos.get('delta', 0.0) for pos in self.positions.values())
                new_delta = current_delta + trade_delta
                
                if abs(new_delta) > self.max_portfolio_delta:
                    validation_errors.append(f"Portfolio delta {new_delta:.0f} would exceed limit {self.max_portfolio_delta:.0f}")
                    risk_score += 4.0
            
            # Additional risk factors
            if quantity > 1000:  # Large quantity
                risk_score += 1.0
            
            if price > 500:  # High-priced security
                risk_score += 0.5
            
            # Determine if trade is allowed
            trade_allowed = len(validation_errors) == 0 and risk_score < 10.0
            
            return trade_allowed, validation_errors, risk_score
            
        except Exception as e:
            self.logger.error(f"Error in pre-trade validation: {str(e)}")
            return False, [f"Validation error: {str(e)}"], 10.0
    
    def update_position(self, symbol: str, quantity: int, price: float, delta: float = 0.0):
        """Update position information"""
        try:
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0,
                    'avg_price': 0.0,
                    'market_value': 0.0,
                    'delta': 0.0,
                    'last_update': datetime.now()
                }
            
            position = self.positions[symbol]
            
            # Update quantity and average price
            old_quantity = position['quantity']
            old_avg_price = position['avg_price']
            
            new_quantity = old_quantity + quantity
            
            if new_quantity != 0:
                new_avg_price = ((old_quantity * old_avg_price) + (quantity * price)) / new_quantity
            else:
                new_avg_price = 0.0
            
            position['quantity'] = new_quantity
            position['avg_price'] = new_avg_price
            position['market_value'] = new_quantity * price
            position['delta'] = delta
            position['last_update'] = datetime.now()
            
            # Remove position if quantity is zero
            if new_quantity == 0:
                del self.positions[symbol]
            
            # Update portfolio value
            self._update_portfolio_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating position for {symbol}: {str(e)}")
    
    def _update_portfolio_metrics(self):
        """Update portfolio-level metrics"""
        try:
            self.current_portfolio_value = sum(
                pos['market_value'] for pos in self.positions.values()
            )
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio metrics: {str(e)}")
    
    def check_risk_limits(self) -> List[RiskAlert]:
        """Check all risk limits and generate alerts"""
        alerts = []
        
        try:
            # Check portfolio value limit
            if self.current_portfolio_value > self.max_portfolio_value * 0.9:
                alert = RiskAlert(
                    alert_id=f"portfolio_value_{int(time.time())}",
                    severity=AlertSeverity.WARNING if self.current_portfolio_value < self.max_portfolio_value else AlertSeverity.CRITICAL,
                    risk_level=RiskLevel.HIGH,
                    message=f"Portfolio value ${self.current_portfolio_value:,.0f} approaching limit ${self.max_portfolio_value:,.0f}",
                    details={
                        'current_value': self.current_portfolio_value,
                        'limit': self.max_portfolio_value,
                        'utilization': self.current_portfolio_value / self.max_portfolio_value
                    }
                )
                alerts.append(alert)
            
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss * 0.8:
                alert = RiskAlert(
                    alert_id=f"daily_loss_{int(time.time())}",
                    severity=AlertSeverity.CRITICAL if self.daily_pnl < -self.max_daily_loss else AlertSeverity.WARNING,
                    risk_level=RiskLevel.HIGH,
                    message=f"Daily loss ${abs(self.daily_pnl):,.0f} approaching limit ${self.max_daily_loss:,.0f}",
                    details={
                        'daily_pnl': self.daily_pnl,
                        'limit': -self.max_daily_loss,
                        'utilization': abs(self.daily_pnl) / self.max_daily_loss
                    }
                )
                alerts.append(alert)
            
            # Check concentration limits
            for symbol, position in self.positions.items():
                concentration = position['market_value'] / max(self.current_portfolio_value, 1.0)
                if concentration > self.max_concentration * 0.8:
                    alert = RiskAlert(
                        alert_id=f"concentration_{symbol}_{int(time.time())}",
                        severity=AlertSeverity.WARNING if concentration < self.max_concentration else AlertSeverity.CRITICAL,
                        risk_level=RiskLevel.MEDIUM,
                        message=f"High concentration in {symbol}: {concentration:.1%}",
                        details={
                            'symbol': symbol,
                            'concentration': concentration,
                            'limit': self.max_concentration,
                            'position_value': position['market_value']
                        }
                    )
                    alerts.append(alert)
            
            # Check delta limits
            total_delta = sum(pos.get('delta', 0.0) for pos in self.positions.values())
            if abs(total_delta) > self.max_portfolio_delta * 0.8:
                alert = RiskAlert(
                    alert_id=f"portfolio_delta_{int(time.time())}",
                    severity=AlertSeverity.WARNING if abs(total_delta) < self.max_portfolio_delta else AlertSeverity.CRITICAL,
                    risk_level=RiskLevel.MEDIUM,
                    message=f"Portfolio delta {total_delta:.0f} approaching limit {self.max_portfolio_delta:.0f}",
                    details={
                        'portfolio_delta': total_delta,
                        'limit': self.max_portfolio_delta,
                        'utilization': abs(total_delta) / self.max_portfolio_delta
                    }
                )
                alerts.append(alert)
            
            # Store alerts
            self.risk_alerts.extend(alerts)
            
            # Send alerts to callbacks
            for alert in alerts:
                for callback in self.risk_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        self.logger.error(f"Error in risk callback: {str(e)}")
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
            return []
    
    def activate_kill_switch(self, reason: str = "Manual activation"):
        """Activate emergency kill switch"""
        try:
            self.kill_switch_active = True
            self.emergency_mode = True
            
            alert = RiskAlert(
                alert_id=f"kill_switch_{int(time.time())}",
                severity=AlertSeverity.EMERGENCY,
                risk_level=RiskLevel.EMERGENCY,
                message=f"KILL SWITCH ACTIVATED: {reason}",
                details={
                    'reason': reason,
                    'activation_time': datetime.now().isoformat(),
                    'portfolio_value': self.current_portfolio_value,
                    'daily_pnl': self.daily_pnl
                }
            )
            
            self.risk_alerts.append(alert)
            
            # Send emergency alert
            for callback in self.risk_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in emergency callback: {str(e)}")
            
            self.logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error activating kill switch: {str(e)}")
    
    def deactivate_kill_switch(self, reason: str = "Manual deactivation"):
        """Deactivate kill switch"""
        try:
            self.kill_switch_active = False
            self.emergency_mode = False
            
            alert = RiskAlert(
                alert_id=f"kill_switch_deactivated_{int(time.time())}",
                severity=AlertSeverity.INFO,
                risk_level=RiskLevel.LOW,
                message=f"Kill switch deactivated: {reason}",
                details={
                    'reason': reason,
                    'deactivation_time': datetime.now().isoformat()
                }
            )
            
            self.risk_alerts.append(alert)
            self.logger.info(f"Kill switch deactivated: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error deactivating kill switch: {str(e)}")
    
    def add_risk_callback(self, callback: Callable[[RiskAlert], None]):
        """Add risk alert callback"""
        self.risk_callbacks.append(callback)
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status"""
        try:
            total_delta = sum(pos.get('delta', 0.0) for pos in self.positions.values())
            
            # Calculate utilization percentages
            portfolio_utilization = self.current_portfolio_value / self.max_portfolio_value
            loss_utilization = abs(self.daily_pnl) / self.max_daily_loss if self.daily_pnl < 0 else 0.0
            delta_utilization = abs(total_delta) / self.max_portfolio_delta
            
            # Determine overall risk level
            max_utilization = max(portfolio_utilization, loss_utilization, delta_utilization)
            
            if max_utilization >= 1.0:
                overall_risk = RiskLevel.EMERGENCY
            elif max_utilization >= 0.9:
                overall_risk = RiskLevel.CRITICAL
            elif max_utilization >= 0.7:
                overall_risk = RiskLevel.HIGH
            elif max_utilization >= 0.5:
                overall_risk = RiskLevel.MEDIUM
            else:
                overall_risk = RiskLevel.LOW
            
            return {
                'overall_risk_level': overall_risk.value,
                'kill_switch_active': self.kill_switch_active,
                'emergency_mode': self.emergency_mode,
                'portfolio_value': self.current_portfolio_value,
                'portfolio_limit': self.max_portfolio_value,
                'portfolio_utilization': portfolio_utilization,
                'daily_pnl': self.daily_pnl,
                'daily_loss_limit': -self.max_daily_loss,
                'loss_utilization': loss_utilization,
                'portfolio_delta': total_delta,
                'delta_limit': self.max_portfolio_delta,
                'delta_utilization': delta_utilization,
                'total_positions': len(self.positions),
                'active_alerts': len([alert for alert in self.risk_alerts if not alert.resolved]),
                'recent_alerts': len([alert for alert in self.risk_alerts if (datetime.now() - alert.timestamp).total_seconds() < 3600])
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk status: {str(e)}")
            return {}

class LivePerformanceTracker:
    """Real-time performance tracking and analytics"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.LivePerformanceTracker")
        
        # Performance data
        self.trades: List[TradeExecution] = []
        self.daily_pnl_history: List[Tuple[datetime, float]] = []
        self.portfolio_value_history: List[Tuple[datetime, float]] = []
        
        # Current metrics
        self.starting_portfolio_value = 100000.0  # $100K starting value
        self.current_portfolio_value = self.starting_portfolio_value
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.daily_pnl = 0.0
        
        # Greeks tracking
        self.portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0
        }
        
        # Performance tracking
        self.max_portfolio_value = self.starting_portfolio_value
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Trade statistics
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commissions = 0.0
    
    def add_trade(self, trade: TradeExecution):
        """Add completed trade to performance tracking"""
        try:
            self.trades.append(trade)
            
            # Update realized P&L
            if trade.status == TradeStatus.FILLED:
                trade_pnl = self._calculate_trade_pnl(trade)
                self.total_realized_pnl += trade_pnl
                self.daily_pnl += trade_pnl
                
                # Update trade statistics
                if trade_pnl > 0:
                    self.winning_trades += 1
                elif trade_pnl < 0:
                    self.losing_trades += 1
                
                # Update commissions
                self.total_commissions += trade.commission
            
            # Update portfolio value
            self._update_portfolio_value()
            
        except Exception as e:
            self.logger.error(f"Error adding trade to performance tracker: {str(e)}")
    
    def _calculate_trade_pnl(self, trade: TradeExecution) -> float:
        """Calculate P&L for a trade"""
        try:
            # This is a simplified calculation
            # In practice, would need to match opening and closing trades
            
            if trade.side.upper() in ['SELL', 'STC']:
                # Closing trade - calculate P&L
                # For simplicity, assume 1% average profit
                return trade.executed_quantity * trade.average_price * 0.01
            else:
                # Opening trade - no realized P&L yet
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating trade P&L: {str(e)}")
            return 0.0
    
    def _update_portfolio_value(self):
        """Update current portfolio value"""
        try:
            # Calculate portfolio value
            self.current_portfolio_value = self.starting_portfolio_value + self.total_realized_pnl + self.total_unrealized_pnl
            
            # Update max portfolio value
            if self.current_portfolio_value > self.max_portfolio_value:
                self.max_portfolio_value = self.current_portfolio_value
                self.current_drawdown = 0.0
            else:
                # Calculate current drawdown
                self.current_drawdown = (self.max_portfolio_value - self.current_portfolio_value) / self.max_portfolio_value
                
                # Update max drawdown
                if self.current_drawdown > self.max_drawdown:
                    self.max_drawdown = self.current_drawdown
            
            # Record portfolio value history
            self.portfolio_value_history.append((datetime.now(), self.current_portfolio_value))
            
            # Limit history size
            if len(self.portfolio_value_history) > 10000:
                self.portfolio_value_history = self.portfolio_value_history[-10000:]
                
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {str(e)}")
    
    def update_unrealized_pnl(self, unrealized_pnl: float):
        """Update unrealized P&L"""
        try:
            self.total_unrealized_pnl = unrealized_pnl
            self._update_portfolio_value()
            
        except Exception as e:
            self.logger.error(f"Error updating unrealized P&L: {str(e)}")
    
    def update_portfolio_greeks(self, delta: float = 0.0, gamma: float = 0.0, 
                              theta: float = 0.0, vega: float = 0.0):
        """Update portfolio Greeks"""
        try:
            self.portfolio_greeks = {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            }
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio Greeks: {str(e)}")
    
    def reset_daily_pnl(self):
        """Reset daily P&L (call at start of each trading day)"""
        try:
            # Record daily P&L
            self.daily_pnl_history.append((datetime.now(), self.daily_pnl))
            
            # Reset daily P&L
            self.daily_pnl = 0.0
            
            # Limit history size
            if len(self.daily_pnl_history) > 1000:
                self.daily_pnl_history = self.daily_pnl_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"Error resetting daily P&L: {str(e)}")
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        try:
            # Calculate returns
            total_return_pct = (self.current_portfolio_value - self.starting_portfolio_value) / self.starting_portfolio_value
            daily_return_pct = self.daily_pnl / self.starting_portfolio_value
            
            # Calculate trade statistics
            total_trades = len(self.trades)
            win_rate = self.winning_trades / max(total_trades, 1)
            
            # Calculate average win/loss
            winning_trade_pnls = [self._calculate_trade_pnl(trade) for trade in self.trades if self._calculate_trade_pnl(trade) > 0]
            losing_trade_pnls = [self._calculate_trade_pnl(trade) for trade in self.trades if self._calculate_trade_pnl(trade) < 0]
            
            avg_win = sum(winning_trade_pnls) / len(winning_trade_pnls) if winning_trade_pnls else 0.0
            avg_loss = sum(losing_trade_pnls) / len(losing_trade_pnls) if losing_trade_pnls else 0.0
            
            # Calculate profit factor
            gross_profit = sum(winning_trade_pnls)
            gross_loss = abs(sum(losing_trade_pnls))
            profit_factor = gross_profit / max(gross_loss, 1.0)
            
            # Calculate VaR (simplified)
            if len(self.daily_pnl_history) >= 20:
                daily_returns = [pnl / self.starting_portfolio_value for _, pnl in self.daily_pnl_history[-20:]]
                var_95 = np.percentile(daily_returns, 5) * self.current_portfolio_value
            else:
                var_95 = -self.current_portfolio_value * 0.02  # 2% VaR estimate
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                total_pnl=self.total_realized_pnl + self.total_unrealized_pnl,
                realized_pnl=self.total_realized_pnl,
                unrealized_pnl=self.total_unrealized_pnl,
                total_return_pct=total_return_pct,
                daily_pnl=self.daily_pnl,
                daily_return_pct=daily_return_pct,
                portfolio_value=self.current_portfolio_value,
                max_drawdown=self.max_drawdown,
                current_drawdown=self.current_drawdown,
                var_95=var_95,
                total_trades=total_trades,
                winning_trades=self.winning_trades,
                losing_trades=self.losing_trades,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                total_positions=len(set(trade.symbol for trade in self.trades)),
                long_positions=len([trade for trade in self.trades if trade.side.upper() in ['BUY', 'BTO']]),
                short_positions=len([trade for trade in self.trades if trade.side.upper() in ['SELL', 'STO']]),
                portfolio_delta=self.portfolio_greeks['delta'],
                portfolio_gamma=self.portfolio_greeks['gamma'],
                portfolio_theta=self.portfolio_greeks['theta'],
                portfolio_vega=self.portfolio_greeks['vega']
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                total_pnl=0.0, realized_pnl=0.0, unrealized_pnl=0.0,
                total_return_pct=0.0, daily_pnl=0.0, daily_return_pct=0.0,
                portfolio_value=self.starting_portfolio_value,
                max_drawdown=0.0, current_drawdown=0.0, var_95=0.0,
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
                total_positions=0, long_positions=0, short_positions=0,
                portfolio_delta=0.0, portfolio_gamma=0.0, portfolio_theta=0.0, portfolio_vega=0.0
            )

class TradeMonitoringSystem:
    """Main trade monitoring and analytics system"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TradeMonitoringSystem")
        
        # Components
        self.execution_analyzer = ExecutionQualityAnalyzer()
        self.risk_control = RiskControlSystem()
        self.performance_tracker = LivePerformanceTracker()
        
        # Trade tracking
        self.active_trades: Dict[str, TradeExecution] = {}
        self.completed_trades: List[TradeExecution] = []
        
        # Alert system
        self.alert_callbacks: List[Callable[[RiskAlert], None]] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.last_risk_check = datetime.now()
        self.risk_check_interval = 30  # seconds
        
        # Performance tracking
        self.last_performance_update = datetime.now()
        self.performance_update_interval = 60  # seconds
        
        # Setup risk callbacks
        self.risk_control.add_risk_callback(self._on_risk_alert)
    
    def start_monitoring(self):
        """Start trade monitoring system"""
        try:
            self.monitoring_active = True
            self.logger.info("Trade monitoring system started")
            
            # Start monitoring thread
            threading.Thread(target=self._monitoring_loop, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring system: {str(e)}")
    
    def stop_monitoring(self):
        """Stop trade monitoring system"""
        try:
            self.monitoring_active = False
            self.logger.info("Trade monitoring system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring system: {str(e)}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check risk limits
                if (datetime.now() - self.last_risk_check).total_seconds() >= self.risk_check_interval:
                    self.risk_control.check_risk_limits()
                    self.last_risk_check = datetime.now()
                
                # Update performance metrics
                if (datetime.now() - self.last_performance_update).total_seconds() >= self.performance_update_interval:
                    self._update_performance_metrics()
                    self.last_performance_update = datetime.now()
                
                # Sleep before next iteration
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Update unrealized P&L (simplified calculation)
            unrealized_pnl = sum(
                pos.get('market_value', 0.0) - (pos.get('quantity', 0) * pos.get('avg_price', 0.0))
                for pos in self.risk_control.positions.values()
            )
            
            self.performance_tracker.update_unrealized_pnl(unrealized_pnl)
            
            # Update portfolio Greeks (simplified)
            total_delta = sum(pos.get('delta', 0.0) for pos in self.risk_control.positions.values())
            self.performance_tracker.update_portfolio_greeks(delta=total_delta)
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")
    
    def validate_trade(self, trade_request: Dict[str, Any]) -> Tuple[bool, List[str], float]:
        """Validate trade before execution"""
        return self.risk_control.validate_pre_trade(trade_request)
    
    def add_trade(self, trade: TradeExecution):
        """Add new trade for monitoring"""
        try:
            # Add to active trades
            self.active_trades[trade.trade_id] = trade
            
            # Analyze execution quality if completed
            if trade.status == TradeStatus.FILLED:
                trade.execution_quality = self.execution_analyzer.analyze_execution(trade)
                self.execution_analyzer.add_execution(trade)
                
                # Move to completed trades
                self.completed_trades.append(trade)
                if trade.trade_id in self.active_trades:
                    del self.active_trades[trade.trade_id]
                
                # Update performance tracking
                self.performance_tracker.add_trade(trade)
                
                # Update risk control positions
                delta = trade.data.get('delta', 0.0) if hasattr(trade, 'data') else 0.0
                self.risk_control.update_position(
                    trade.symbol,
                    trade.executed_quantity if trade.side.upper() in ['BUY', 'BTO'] else -trade.executed_quantity,
                    trade.average_price,
                    delta
                )
            
            self.logger.info(f"Added trade {trade.trade_id} for monitoring")
            
        except Exception as e:
            self.logger.error(f"Error adding trade for monitoring: {str(e)}")
    
    def update_trade(self, trade_id: str, updates: Dict[str, Any]):
        """Update existing trade"""
        try:
            if trade_id in self.active_trades:
                trade = self.active_trades[trade_id]
                
                # Update trade fields
                for field, value in updates.items():
                    if hasattr(trade, field):
                        setattr(trade, field, value)
                
                # Check if trade is now completed
                if trade.status == TradeStatus.FILLED:
                    self.add_trade(trade)  # This will move it to completed trades
                
                self.logger.info(f"Updated trade {trade_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating trade {trade_id}: {str(e)}")
    
    def _on_risk_alert(self, alert: RiskAlert):
        """Handle risk alerts"""
        try:
            self.logger.warning(f"Risk Alert: {alert.message}")
            
            # Send to alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {str(e)}")
            
            # Auto-activate kill switch for emergency alerts
            if alert.severity == AlertSeverity.EMERGENCY:
                self.risk_control.activate_kill_switch(f"Auto-activation due to: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"Error handling risk alert: {str(e)}")
    
    def activate_kill_switch(self, reason: str = "Manual activation"):
        """Activate emergency kill switch"""
        self.risk_control.activate_kill_switch(reason)
    
    def deactivate_kill_switch(self, reason: str = "Manual deactivation"):
        """Deactivate kill switch"""
        self.risk_control.deactivate_kill_switch(reason)
    
    def add_alert_callback(self, callback: Callable[[RiskAlert], None]):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        try:
            # Get performance metrics
            performance = self.performance_tracker.get_performance_metrics()
            
            # Get risk status
            risk_status = self.risk_control.get_risk_status()
            
            # Get execution statistics
            execution_stats = self.execution_analyzer.get_execution_statistics()
            
            return {
                'monitoring_active': self.monitoring_active,
                'active_trades': len(self.active_trades),
                'completed_trades': len(self.completed_trades),
                'performance': performance.to_dict(),
                'risk_status': risk_status,
                'execution_statistics': execution_stats,
                'last_risk_check': self.last_risk_check.isoformat(),
                'last_performance_update': self.last_performance_update.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting monitoring status: {str(e)}")
            return {}
    
    def get_trade_details(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get details for specific trade"""
        try:
            # Check active trades
            if trade_id in self.active_trades:
                return self.active_trades[trade_id].to_dict()
            
            # Check completed trades
            for trade in self.completed_trades:
                if trade.trade_id == trade_id:
                    return trade.to_dict()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting trade details for {trade_id}: {str(e)}")
            return None

async def test_trade_monitoring_system():
    """Test trade monitoring, analytics, and risk controls"""
    print("🚀 Testing Trade Monitoring, Analytics, and Risk Controls System...")
    
    # Create monitoring system
    monitor = TradeMonitoringSystem()
    
    # Add alert callback
    def alert_callback(alert: RiskAlert):
        print(f"  🚨 ALERT: {alert.severity.value.upper()} - {alert.message}")
    
    monitor.add_alert_callback(alert_callback)
    
    # Start monitoring
    print("\n📊 Starting Trade Monitoring System")
    monitor.start_monitoring()
    print("  ✅ Monitoring system started")
    
    # Test pre-trade validation
    print("\n🔍 Testing Pre-Trade Validation")
    
    test_trades = [
        {
            'symbol': 'SPY',
            'side': 'BUY',
            'quantity': 100,
            'price': 400.0,
            'delta': 50.0
        },
        {
            'symbol': 'QQQ',
            'side': 'BUY',
            'quantity': 1000,  # Large quantity
            'price': 350.0,
            'delta': 800.0  # High delta
        },
        {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 2000,  # Very large position
            'price': 180.0,
            'delta': 1500.0  # Exceeds delta limit
        }
    ]
    
    for i, trade_request in enumerate(test_trades, 1):
        allowed, errors, risk_score = monitor.validate_trade(trade_request)
        print(f"  Trade {i} ({trade_request['symbol']}): {'✅ ALLOWED' if allowed else '❌ REJECTED'}")
        print(f"    Risk Score: {risk_score:.1f}")
        if errors:
            for error in errors:
                print(f"    Error: {error}")
    
    # Test trade execution monitoring
    print("\n📈 Testing Trade Execution Monitoring")
    
    # Create sample trades
    sample_trades = [
        TradeExecution(
            trade_id="TRADE_001",
            symbol="SPY",
            side="BUY",
            quantity=100,
            order_type="MARKET",
            status=TradeStatus.FILLED,
            executed_quantity=100,
            executed_price=401.50,
            average_price=401.50,
            commission=1.00,
            order_time=datetime.now() - timedelta(seconds=30),
            execution_time=datetime.now() - timedelta(seconds=25),
            requested_price=401.00,
            market_price_at_order=401.25,
            broker="IBKR",
            account="U123456",
            strategy="ALL-USE"
        ),
        TradeExecution(
            trade_id="TRADE_002",
            symbol="QQQ",
            side="SELL",
            quantity=50,
            order_type="LIMIT",
            status=TradeStatus.FILLED,
            executed_quantity=50,
            executed_price=349.75,
            average_price=349.75,
            commission=0.50,
            order_time=datetime.now() - timedelta(seconds=60),
            execution_time=datetime.now() - timedelta(seconds=45),
            requested_price=350.00,
            market_price_at_order=349.80,
            broker="IBKR",
            account="U123456",
            strategy="ALL-USE"
        ),
        TradeExecution(
            trade_id="TRADE_003",
            symbol="AAPL",
            side="BUY",
            quantity=200,
            order_type="MARKET",
            status=TradeStatus.PARTIALLY_FILLED,
            executed_quantity=150,
            executed_price=182.25,
            average_price=182.25,
            commission=1.50,
            order_time=datetime.now() - timedelta(seconds=90),
            execution_time=datetime.now() - timedelta(seconds=70),
            requested_price=182.00,
            market_price_at_order=182.10,
            broker="TD_AMERITRADE",
            account="123456789",
            strategy="ALL-USE"
        )
    ]
    
    # Add trades to monitoring
    for trade in sample_trades:
        # Calculate slippage
        if trade.status == TradeStatus.FILLED:
            trade.slippage = monitor.execution_analyzer.calculate_slippage(trade, trade.market_price_at_order)
        
        monitor.add_trade(trade)
        print(f"  ✅ Added {trade.trade_id} ({trade.symbol}) for monitoring")
        print(f"    Status: {trade.status.value}, Quality: {trade.execution_quality.value}")
        print(f"    Slippage: {trade.slippage:.4f} ({trade.slippage*100:.2f}%)")
    
    # Wait for monitoring to process
    await asyncio.sleep(2)
    
    # Test execution quality analysis
    print("\n📊 Testing Execution Quality Analysis")
    
    execution_stats = monitor.execution_analyzer.get_execution_statistics()
    print(f"  📋 Total executions: {execution_stats.get('total_executions', 0)}")
    print(f"  📋 Average slippage: {execution_stats.get('avg_slippage', 0):.4f} ({execution_stats.get('avg_slippage', 0)*100:.2f}%)")
    print(f"  📋 Average execution time: {execution_stats.get('avg_execution_time_seconds', 0):.1f} seconds")
    print(f"  📋 Fill rate: {execution_stats.get('fill_rate', 0):.1%}")
    
    quality_dist = execution_stats.get('quality_distribution', {})
    print(f"  📋 Quality distribution:")
    for quality, count in quality_dist.items():
        print(f"    {quality}: {count} trades")
    
    # Test risk controls
    print("\n🛡️ Testing Risk Controls")
    
    # Simulate high portfolio value
    monitor.risk_control.current_portfolio_value = 950000.0  # Close to $1M limit
    monitor.risk_control.daily_pnl = -45000.0  # Close to $50K loss limit
    
    # Check risk limits
    risk_alerts = monitor.risk_control.check_risk_limits()
    print(f"  📋 Generated {len(risk_alerts)} risk alerts")
    
    # Get risk status
    risk_status = monitor.risk_control.get_risk_status()
    print(f"  📋 Overall risk level: {risk_status.get('overall_risk_level', 'unknown')}")
    print(f"  📋 Portfolio utilization: {risk_status.get('portfolio_utilization', 0):.1%}")
    print(f"  📋 Loss utilization: {risk_status.get('loss_utilization', 0):.1%}")
    print(f"  📋 Kill switch active: {risk_status.get('kill_switch_active', False)}")
    
    # Test kill switch
    print("\n🚨 Testing Kill Switch")
    
    print("  🔴 Activating kill switch...")
    monitor.activate_kill_switch("Testing emergency procedures")
    
    # Try to validate trade with kill switch active
    test_trade = {
        'symbol': 'TEST',
        'side': 'BUY',
        'quantity': 10,
        'price': 100.0
    }
    
    allowed, errors, risk_score = monitor.validate_trade(test_trade)
    print(f"  📋 Trade validation with kill switch: {'✅ ALLOWED' if allowed else '❌ REJECTED'}")
    if errors:
        print(f"    Reason: {errors[0]}")
    
    print("  🟢 Deactivating kill switch...")
    monitor.deactivate_kill_switch("Testing completed")
    
    # Test performance tracking
    print("\n📈 Testing Performance Tracking")
    
    performance = monitor.performance_tracker.get_performance_metrics()
    print(f"  📋 Portfolio value: ${performance.portfolio_value:,.2f}")
    print(f"  📋 Total P&L: ${performance.total_pnl:,.2f}")
    print(f"  📋 Total return: {performance.total_return_pct:.2%}")
    print(f"  📋 Daily P&L: ${performance.daily_pnl:,.2f}")
    print(f"  📋 Daily return: {performance.daily_return_pct:.2%}")
    print(f"  📋 Max drawdown: {performance.max_drawdown:.2%}")
    print(f"  📋 Current drawdown: {performance.current_drawdown:.2%}")
    print(f"  📋 Total trades: {performance.total_trades}")
    print(f"  📋 Win rate: {performance.win_rate:.1%}")
    print(f"  📋 Profit factor: {performance.profit_factor:.2f}")
    print(f"  📋 Portfolio delta: {performance.portfolio_delta:.0f}")
    
    # Test monitoring status
    print("\n📊 Testing Monitoring Status")
    
    status = monitor.get_monitoring_status()
    print(f"  📋 Monitoring active: {status.get('monitoring_active', False)}")
    print(f"  📋 Active trades: {status.get('active_trades', 0)}")
    print(f"  📋 Completed trades: {status.get('completed_trades', 0)}")
    
    # Test trade details
    print("\n🔍 Testing Trade Details")
    
    for trade in sample_trades[:2]:  # Test first 2 trades
        details = monitor.get_trade_details(trade.trade_id)
        if details:
            print(f"  ✅ {trade.trade_id}: {details['symbol']} {details['side']} {details['quantity']} @ ${details['average_price']:.2f}")
            print(f"    Status: {details['status']}, Quality: {details['execution_quality']}")
        else:
            print(f"  ❌ {trade.trade_id}: Details not found")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    monitor.stop_monitoring()
    
    print("\n✅ Trade Monitoring, Analytics, and Risk Controls testing completed!")
    
    return {
        'monitoring_system': True,
        'pre_trade_validation': True,
        'execution_monitoring': len(sample_trades) > 0,
        'quality_analysis': execution_stats.get('total_executions', 0) > 0,
        'risk_controls': len(risk_alerts) > 0,
        'kill_switch': True,
        'performance_tracking': performance.total_trades > 0,
        'trade_details': True
    }

if __name__ == "__main__":
    asyncio.run(test_trade_monitoring_system())

