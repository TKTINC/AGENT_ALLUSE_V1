"""
Position Management Decision Trees for ALL-USE Protocol Engine
Implements sophisticated position management logic with decision trees for complete position lifecycle management

This module provides comprehensive position management capabilities including
entry, monitoring, adjustment, and exit decision trees for optimal position
lifecycle management in the ALL-USE protocol.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionStatus(Enum):
    """Position status classifications"""
    PENDING = "pending"           # Position planned but not entered
    ACTIVE = "active"             # Position is open and active
    MONITORING = "monitoring"     # Position under close monitoring
    ADJUSTMENT_NEEDED = "adjustment_needed"  # Position needs adjustment
    CLOSING = "closing"           # Position being closed
    CLOSED = "closed"             # Position closed
    EXPIRED = "expired"           # Position expired
    ASSIGNED = "assigned"         # Position was assigned

class PositionType(Enum):
    """Types of positions"""
    SHORT_PUT = "short_put"
    SHORT_CALL = "short_call"
    LONG_PUT = "long_put"
    LONG_CALL = "long_call"
    PUT_SPREAD = "put_spread"
    CALL_SPREAD = "call_spread"
    IRON_CONDOR = "iron_condor"
    STRADDLE = "straddle"
    STRANGLE = "strangle"

class ActionTrigger(Enum):
    """Triggers for position actions"""
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TIME_DECAY = "time_decay"
    DELTA_CHANGE = "delta_change"
    VOLATILITY_CHANGE = "volatility_change"
    MARKET_CONDITION = "market_condition"
    ASSIGNMENT_RISK = "assignment_risk"
    EXPIRATION_APPROACH = "expiration_approach"

class ManagementAction(Enum):
    """Position management actions"""
    HOLD = "hold"
    CLOSE = "close"
    ROLL = "roll"
    ADJUST_DELTA = "adjust_delta"
    ADD_HEDGE = "add_hedge"
    REDUCE_SIZE = "reduce_size"
    INCREASE_SIZE = "increase_size"
    CONVERT_STRATEGY = "convert_strategy"
    TAKE_ASSIGNMENT = "take_assignment"
    AVOID_ASSIGNMENT = "avoid_assignment"

@dataclass
class Position:
    """Position data structure"""
    position_id: str
    symbol: str
    position_type: PositionType
    quantity: int
    entry_price: float
    current_price: float
    strike: float
    expiration: datetime
    delta: float
    gamma: float
    theta: float
    vega: float
    implied_volatility: float
    days_to_expiration: int
    profit_loss: float
    profit_loss_percentage: float
    entry_date: datetime
    status: PositionStatus
    week_classification: str
    account_type: str
    market_conditions: Dict[str, Any]

@dataclass
class ManagementDecision:
    """Position management decision"""
    position_id: str
    action: ManagementAction
    trigger: ActionTrigger
    priority: str
    confidence: float
    rationale: str
    expected_outcome: str
    risk_assessment: str
    execution_timeline: str
    parameters: Dict[str, Any]
    alternatives: List[str]
    monitoring_points: List[str]

@dataclass
class PositionAlert:
    """Position monitoring alert"""
    position_id: str
    alert_type: str
    severity: str
    message: str
    trigger_value: float
    threshold_value: float
    recommended_action: str
    timestamp: datetime

class DecisionNode:
    """Decision tree node for position management"""
    
    def __init__(self, condition: str, action: Optional[ManagementAction] = None):
        self.condition = condition
        self.action = action
        self.children = {}
        self.thresholds = {}
    
    def add_child(self, condition_value: Any, child_node: 'DecisionNode', threshold: Optional[float] = None):
        """Add a child node with optional threshold"""
        self.children[condition_value] = child_node
        if threshold is not None:
            self.thresholds[condition_value] = threshold
    
    def evaluate(self, position: Position, context: Dict[str, Any]) -> Optional[ManagementDecision]:
        """Evaluate the decision tree for a position"""
        if not self.children:
            if self.action:
                return self._create_decision(position, self.action, context)
            return None
        
        # Get the condition value from position or context
        condition_value = self._get_condition_value(position, context)
        
        # Find matching child
        for child_key, child_node in self.children.items():
            if self._matches_condition(condition_value, child_key):
                return child_node.evaluate(position, context)
        
        return None
    
    def _get_condition_value(self, position: Position, context: Dict[str, Any]) -> Any:
        """Extract condition value from position or context"""
        condition_map = {
            'profit_loss_percentage': position.profit_loss_percentage,
            'days_to_expiration': position.days_to_expiration,
            'delta': abs(position.delta),
            'implied_volatility': position.implied_volatility,
            'position_status': position.status.value,
            'market_condition': position.market_conditions.get('condition', 'neutral'),
            'volatility_regime': context.get('volatility_regime', 'normal'),
            'assignment_risk': context.get('assignment_risk', 'low')
        }
        
        return condition_map.get(self.condition, None)
    
    def _matches_condition(self, value: Any, condition_key: Any) -> bool:
        """Check if value matches condition key"""
        if isinstance(condition_key, str) and isinstance(value, str):
            return value == condition_key
        elif isinstance(condition_key, tuple) and len(condition_key) == 2:
            # Range condition (min, max)
            return condition_key[0] <= value <= condition_key[1]
        elif condition_key in self.thresholds:
            # Threshold condition
            return value >= self.thresholds[condition_key]
        else:
            return value == condition_key
    
    def _create_decision(self, position: Position, action: ManagementAction, 
                        context: Dict[str, Any]) -> ManagementDecision:
        """Create a management decision"""
        trigger = self._determine_trigger(position, context)
        priority = self._determine_priority(action, trigger)
        confidence = self._calculate_confidence(position, action, context)
        rationale = self._generate_rationale(position, action, trigger)
        
        return ManagementDecision(
            position_id=position.position_id,
            action=action,
            trigger=trigger,
            priority=priority,
            confidence=confidence,
            rationale=rationale,
            expected_outcome=self._get_expected_outcome(action),
            risk_assessment=self._assess_risk(position, action),
            execution_timeline=self._get_execution_timeline(action),
            parameters=self._get_action_parameters(position, action),
            alternatives=self._get_alternatives(action),
            monitoring_points=self._get_monitoring_points(position, action)
        )
    
    def _determine_trigger(self, position: Position, context: Dict[str, Any]) -> ActionTrigger:
        """Determine what triggered the action"""
        if position.profit_loss_percentage >= 0.5:  # 50% profit
            return ActionTrigger.PROFIT_TARGET
        elif position.profit_loss_percentage <= -0.2:  # 20% loss
            return ActionTrigger.STOP_LOSS
        elif position.days_to_expiration <= 7:
            return ActionTrigger.EXPIRATION_APPROACH
        elif abs(position.delta) > 50:
            return ActionTrigger.DELTA_CHANGE
        else:
            return ActionTrigger.TIME_DECAY
    
    def _determine_priority(self, action: ManagementAction, trigger: ActionTrigger) -> str:
        """Determine action priority"""
        critical_triggers = [ActionTrigger.STOP_LOSS, ActionTrigger.ASSIGNMENT_RISK]
        high_triggers = [ActionTrigger.PROFIT_TARGET, ActionTrigger.EXPIRATION_APPROACH]
        
        if trigger in critical_triggers:
            return "critical"
        elif trigger in high_triggers:
            return "high"
        else:
            return "medium"
    
    def _calculate_confidence(self, position: Position, action: ManagementAction, 
                            context: Dict[str, Any]) -> float:
        """Calculate confidence in the decision"""
        base_confidence = 0.8
        
        # Adjust based on position P&L clarity
        if abs(position.profit_loss_percentage) > 0.3:
            base_confidence += 0.1
        
        # Adjust based on time to expiration
        if position.days_to_expiration < 10:
            base_confidence += 0.05
        
        # Adjust based on market conditions
        market_condition = position.market_conditions.get('condition', 'neutral')
        if market_condition in ['extremely_bullish', 'extremely_bearish']:
            base_confidence += 0.05
        
        return min(0.95, base_confidence)
    
    def _generate_rationale(self, position: Position, action: ManagementAction, 
                          trigger: ActionTrigger) -> str:
        """Generate rationale for the decision"""
        rationale_map = {
            ManagementAction.CLOSE: f"Close position due to {trigger.value} trigger",
            ManagementAction.ROLL: f"Roll position to manage {trigger.value}",
            ManagementAction.HOLD: f"Hold position despite {trigger.value}",
            ManagementAction.ADJUST_DELTA: f"Adjust delta due to {trigger.value}",
            ManagementAction.ADD_HEDGE: f"Add hedge protection for {trigger.value}"
        }
        
        base_rationale = rationale_map.get(action, f"Execute {action.value}")
        
        # Add position-specific details
        if position.profit_loss_percentage > 0:
            base_rationale += f" (Current profit: {position.profit_loss_percentage:.1%})"
        else:
            base_rationale += f" (Current loss: {position.profit_loss_percentage:.1%})"
        
        return base_rationale
    
    def _get_expected_outcome(self, action: ManagementAction) -> str:
        """Get expected outcome for action"""
        outcome_map = {
            ManagementAction.CLOSE: "Realize current profit/loss and free up capital",
            ManagementAction.ROLL: "Extend position duration and collect additional premium",
            ManagementAction.HOLD: "Allow position to continue toward expiration",
            ManagementAction.ADJUST_DELTA: "Optimize position delta for current market",
            ManagementAction.ADD_HEDGE: "Reduce position risk through hedging"
        }
        
        return outcome_map.get(action, "Execute planned action")
    
    def _assess_risk(self, position: Position, action: ManagementAction) -> str:
        """Assess risk of the action"""
        if action == ManagementAction.CLOSE:
            return "Low risk - crystallizes current P&L"
        elif action == ManagementAction.ROLL:
            return "Medium risk - extends market exposure"
        elif action == ManagementAction.HOLD:
            if position.days_to_expiration < 7:
                return "High risk - assignment risk increasing"
            else:
                return "Medium risk - continued market exposure"
        else:
            return "Medium risk - position modification"
    
    def _get_execution_timeline(self, action: ManagementAction) -> str:
        """Get execution timeline for action"""
        timeline_map = {
            ManagementAction.CLOSE: "Immediate",
            ManagementAction.ROLL: "Before expiration",
            ManagementAction.HOLD: "Monitor daily",
            ManagementAction.ADJUST_DELTA: "Within 1-2 days",
            ManagementAction.ADD_HEDGE: "Within 1 day"
        }
        
        return timeline_map.get(action, "As appropriate")
    
    def _get_action_parameters(self, position: Position, action: ManagementAction) -> Dict[str, Any]:
        """Get parameters for the action"""
        if action == ManagementAction.ROLL:
            return {
                'new_expiration': position.expiration + timedelta(days=30),
                'delta_adjustment': 5,
                'strike_adjustment': 'closer_to_money'
            }
        elif action == ManagementAction.ADJUST_DELTA:
            return {
                'target_delta': 30,
                'adjustment_method': 'roll_strike'
            }
        else:
            return {}
    
    def _get_alternatives(self, action: ManagementAction) -> List[str]:
        """Get alternative actions"""
        alternatives_map = {
            ManagementAction.CLOSE: ["Roll position", "Add hedge"],
            ManagementAction.ROLL: ["Close position", "Hold position"],
            ManagementAction.HOLD: ["Close position", "Adjust delta"],
            ManagementAction.ADJUST_DELTA: ["Close position", "Add hedge"],
            ManagementAction.ADD_HEDGE: ["Close position", "Reduce size"]
        }
        
        return alternatives_map.get(action, [])
    
    def _get_monitoring_points(self, position: Position, action: ManagementAction) -> List[str]:
        """Get monitoring points for the action"""
        base_points = [
            "Daily P&L tracking",
            "Delta changes",
            "Time decay progression"
        ]
        
        if position.days_to_expiration <= 14:
            base_points.append("Assignment risk monitoring")
        
        if action == ManagementAction.ROLL:
            base_points.append("New position establishment")
        
        return base_points

class PositionManager:
    """
    Advanced Position Management System for ALL-USE Protocol
    
    Provides sophisticated position management capabilities including:
    - Complete position lifecycle management
    - Decision tree-based action recommendations
    - Real-time position monitoring and alerts
    - Risk assessment and optimization
    - Automated position adjustments
    """
    
    def __init__(self):
        """Initialize the position manager"""
        self.logger = logging.getLogger(__name__)
        
        # Position registry
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Position] = []
        
        # Decision trees
        self.profit_management_tree = self._build_profit_management_tree()
        self.loss_management_tree = self._build_loss_management_tree()
        self.expiration_management_tree = self._build_expiration_management_tree()
        self.delta_management_tree = self._build_delta_management_tree()
        
        # Monitoring thresholds
        self.monitoring_thresholds = {
            'profit_target': 0.5,      # 50% profit target
            'stop_loss': -0.2,         # 20% stop loss
            'delta_threshold': 50,     # Delta > 50
            'dte_warning': 14,         # 14 days to expiration warning
            'dte_critical': 7,         # 7 days to expiration critical
            'iv_change': 0.1,          # 10% IV change
            'assignment_risk': 0.8     # 80% assignment probability
        }
        
        # Active alerts
        self.active_alerts: List[PositionAlert] = []
        
        self.logger.info("Position Manager initialized")
    
    def add_position(self, position: Position) -> None:
        """Add a new position to management"""
        self.positions[position.position_id] = position
        self.logger.info(f"Added position: {position.position_id} ({position.symbol})")
    
    def update_position(self, position_id: str, updates: Dict[str, Any]) -> None:
        """Update position data"""
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")
        
        position = self.positions[position_id]
        
        for field, value in updates.items():
            if hasattr(position, field):
                setattr(position, field, value)
        
        # Recalculate derived fields
        self._update_derived_fields(position)
        
        self.logger.info(f"Updated position: {position_id}")
    
    def get_management_decision(self, position_id: str, 
                              context: Optional[Dict[str, Any]] = None) -> Optional[ManagementDecision]:
        """
        Get management decision for a position
        
        Args:
            position_id: Position identifier
            context: Additional context for decision making
            
        Returns:
            ManagementDecision: Recommended action or None
        """
        try:
            if position_id not in self.positions:
                raise ValueError(f"Position {position_id} not found")
            
            position = self.positions[position_id]
            
            if context is None:
                context = {}
            
            # Determine which decision tree to use
            if position.profit_loss_percentage > 0.1:  # 10% profit
                decision = self.profit_management_tree.evaluate(position, context)
            elif position.profit_loss_percentage < -0.1:  # 10% loss
                decision = self.loss_management_tree.evaluate(position, context)
            elif position.days_to_expiration <= 14:
                decision = self.expiration_management_tree.evaluate(position, context)
            elif abs(position.delta) > 40:
                decision = self.delta_management_tree.evaluate(position, context)
            else:
                # Default to hold
                decision = ManagementDecision(
                    position_id=position_id,
                    action=ManagementAction.HOLD,
                    trigger=ActionTrigger.TIME_DECAY,
                    priority="low",
                    confidence=0.8,
                    rationale="Position within normal parameters",
                    expected_outcome="Continue monitoring position",
                    risk_assessment="Low risk - normal position management",
                    execution_timeline="Monitor daily",
                    parameters={},
                    alternatives=["Close position", "Adjust delta"],
                    monitoring_points=["Daily P&L", "Delta changes", "Time decay"]
                )
            
            self.logger.info(f"Generated decision for {position_id}: {decision.action.value}")
            return decision
            
        except Exception as e:
            self.logger.error(f"Error getting management decision: {str(e)}")
            raise
    
    def monitor_positions(self) -> List[PositionAlert]:
        """
        Monitor all positions and generate alerts
        
        Returns:
            List of position alerts
        """
        try:
            alerts = []
            
            for position_id, position in self.positions.items():
                if position.status not in [PositionStatus.ACTIVE, PositionStatus.MONITORING]:
                    continue
                
                # Check profit target
                if position.profit_loss_percentage >= self.monitoring_thresholds['profit_target']:
                    alert = PositionAlert(
                        position_id=position_id,
                        alert_type="profit_target",
                        severity="medium",
                        message=f"Position reached profit target: {position.profit_loss_percentage:.1%}",
                        trigger_value=position.profit_loss_percentage,
                        threshold_value=self.monitoring_thresholds['profit_target'],
                        recommended_action="Consider closing position",
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
                
                # Check stop loss
                if position.profit_loss_percentage <= self.monitoring_thresholds['stop_loss']:
                    alert = PositionAlert(
                        position_id=position_id,
                        alert_type="stop_loss",
                        severity="high",
                        message=f"Position hit stop loss: {position.profit_loss_percentage:.1%}",
                        trigger_value=position.profit_loss_percentage,
                        threshold_value=self.monitoring_thresholds['stop_loss'],
                        recommended_action="Consider closing or rolling position",
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
                
                # Check delta threshold
                if abs(position.delta) >= self.monitoring_thresholds['delta_threshold']:
                    alert = PositionAlert(
                        position_id=position_id,
                        alert_type="delta_threshold",
                        severity="medium",
                        message=f"Position delta exceeded threshold: {position.delta:.1f}",
                        trigger_value=abs(position.delta),
                        threshold_value=self.monitoring_thresholds['delta_threshold'],
                        recommended_action="Consider delta adjustment",
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
                
                # Check expiration approach
                if position.days_to_expiration <= self.monitoring_thresholds['dte_critical']:
                    alert = PositionAlert(
                        position_id=position_id,
                        alert_type="expiration_critical",
                        severity="high",
                        message=f"Position expires in {position.days_to_expiration} days",
                        trigger_value=position.days_to_expiration,
                        threshold_value=self.monitoring_thresholds['dte_critical'],
                        recommended_action="Immediate action required",
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
                elif position.days_to_expiration <= self.monitoring_thresholds['dte_warning']:
                    alert = PositionAlert(
                        position_id=position_id,
                        alert_type="expiration_warning",
                        severity="medium",
                        message=f"Position expires in {position.days_to_expiration} days",
                        trigger_value=position.days_to_expiration,
                        threshold_value=self.monitoring_thresholds['dte_warning'],
                        recommended_action="Plan exit strategy",
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
            
            self.active_alerts = alerts
            self.logger.info(f"Generated {len(alerts)} position alerts")
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {str(e)}")
            raise
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio-level position summary"""
        active_positions = [p for p in self.positions.values() 
                          if p.status == PositionStatus.ACTIVE]
        
        if not active_positions:
            return {
                'total_positions': 0,
                'total_pnl': 0.0,
                'total_pnl_percentage': 0.0,
                'positions_by_status': {},
                'positions_by_type': {},
                'average_dte': 0,
                'delta_exposure': 0.0
            }
        
        total_pnl = sum(p.profit_loss for p in active_positions)
        total_value = sum(abs(p.entry_price * p.quantity) for p in active_positions)
        total_pnl_percentage = total_pnl / total_value if total_value > 0 else 0
        
        # Group by status
        status_counts = {}
        for position in self.positions.values():
            status = position.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Group by type
        type_counts = {}
        for position in active_positions:
            pos_type = position.position_type.value
            type_counts[pos_type] = type_counts.get(pos_type, 0) + 1
        
        # Calculate averages
        average_dte = sum(p.days_to_expiration for p in active_positions) / len(active_positions)
        delta_exposure = sum(p.delta * p.quantity for p in active_positions)
        
        return {
            'total_positions': len(active_positions),
            'total_pnl': total_pnl,
            'total_pnl_percentage': total_pnl_percentage,
            'positions_by_status': status_counts,
            'positions_by_type': type_counts,
            'average_dte': average_dte,
            'delta_exposure': delta_exposure,
            'active_alerts': len(self.active_alerts)
        }
    
    def _build_profit_management_tree(self) -> DecisionNode:
        """Build decision tree for profit management"""
        root = DecisionNode("profit_loss_percentage")
        
        # High profit (>50%)
        high_profit = DecisionNode("days_to_expiration")
        high_profit.add_child((0, 7), DecisionNode("", ManagementAction.CLOSE))  # Close if expiring soon
        high_profit.add_child((8, 30), DecisionNode("", ManagementAction.CLOSE))  # Close to lock in profit
        high_profit.add_child((31, 365), DecisionNode("", ManagementAction.HOLD))  # Hold if plenty of time
        
        # Medium profit (25-50%)
        medium_profit = DecisionNode("days_to_expiration")
        medium_profit.add_child((0, 14), DecisionNode("", ManagementAction.CLOSE))
        medium_profit.add_child((15, 365), DecisionNode("", ManagementAction.HOLD))
        
        # Low profit (10-25%)
        low_profit = DecisionNode("", ManagementAction.HOLD)
        
        root.add_child((0.5, 1.0), high_profit)      # 50-100% profit
        root.add_child((0.25, 0.5), medium_profit)   # 25-50% profit
        root.add_child((0.1, 0.25), low_profit)      # 10-25% profit
        
        return root
    
    def _build_loss_management_tree(self) -> DecisionNode:
        """Build decision tree for loss management"""
        root = DecisionNode("profit_loss_percentage")
        
        # Large loss (<-50%)
        large_loss = DecisionNode("days_to_expiration")
        large_loss.add_child((0, 14), DecisionNode("", ManagementAction.CLOSE))  # Close to limit loss
        large_loss.add_child((15, 365), DecisionNode("", ManagementAction.ROLL))  # Roll to recover
        
        # Medium loss (-20% to -50%)
        medium_loss = DecisionNode("days_to_expiration")
        medium_loss.add_child((0, 7), DecisionNode("", ManagementAction.CLOSE))
        medium_loss.add_child((8, 21), DecisionNode("", ManagementAction.ROLL))
        medium_loss.add_child((22, 365), DecisionNode("", ManagementAction.HOLD))
        
        # Small loss (-10% to -20%)
        small_loss = DecisionNode("", ManagementAction.HOLD)
        
        root.add_child((-1.0, -0.5), large_loss)     # >50% loss
        root.add_child((-0.5, -0.2), medium_loss)    # 20-50% loss
        root.add_child((-0.2, -0.1), small_loss)     # 10-20% loss
        
        return root
    
    def _build_expiration_management_tree(self) -> DecisionNode:
        """Build decision tree for expiration management"""
        root = DecisionNode("days_to_expiration")
        
        # Critical expiration (0-7 days)
        critical_expiration = DecisionNode("profit_loss_percentage")
        critical_expiration.add_child((0.1, 1.0), DecisionNode("", ManagementAction.CLOSE))  # Profit
        critical_expiration.add_child((-1.0, 0.1), DecisionNode("", ManagementAction.ROLL))  # Loss/Break-even
        
        # Warning expiration (8-14 days)
        warning_expiration = DecisionNode("profit_loss_percentage")
        warning_expiration.add_child((0.25, 1.0), DecisionNode("", ManagementAction.CLOSE))
        warning_expiration.add_child((-1.0, 0.25), DecisionNode("", ManagementAction.HOLD))
        
        # Normal expiration (>14 days)
        normal_expiration = DecisionNode("", ManagementAction.HOLD)
        
        root.add_child((0, 7), critical_expiration)
        root.add_child((8, 14), warning_expiration)
        root.add_child((15, 365), normal_expiration)
        
        return root
    
    def _build_delta_management_tree(self) -> DecisionNode:
        """Build decision tree for delta management"""
        root = DecisionNode("delta")
        
        # High delta (>50)
        high_delta = DecisionNode("profit_loss_percentage")
        high_delta.add_child((0.0, 1.0), DecisionNode("", ManagementAction.CLOSE))  # Profit
        high_delta.add_child((-1.0, 0.0), DecisionNode("", ManagementAction.ROLL))  # Loss
        
        # Medium delta (30-50)
        medium_delta = DecisionNode("", ManagementAction.HOLD)
        
        # Low delta (<30)
        low_delta = DecisionNode("", ManagementAction.HOLD)
        
        root.add_child((50, 100), high_delta)
        root.add_child((30, 50), medium_delta)
        root.add_child((0, 30), low_delta)
        
        return root
    
    def _update_derived_fields(self, position: Position) -> None:
        """Update derived fields for a position"""
        # Calculate days to expiration
        position.days_to_expiration = (position.expiration.date() - datetime.now().date()).days
        
        # Calculate profit/loss percentage
        if position.entry_price != 0:
            position.profit_loss_percentage = (position.current_price - position.entry_price) / position.entry_price
        else:
            position.profit_loss_percentage = 0.0
        
        # Calculate absolute profit/loss
        position.profit_loss = (position.current_price - position.entry_price) * position.quantity

def test_position_manager():
    """Test the position manager"""
    print("Testing Position Manager...")
    
    manager = PositionManager()
    
    # Create test positions
    test_positions = [
        Position(
            position_id="POS001",
            symbol="SPY",
            position_type=PositionType.SHORT_PUT,
            quantity=10,
            entry_price=2.50,
            current_price=1.25,  # 50% profit
            strike=450.0,
            expiration=datetime.now() + timedelta(days=25),
            delta=-25,
            gamma=0.05,
            theta=-0.15,
            vega=0.8,
            implied_volatility=0.18,
            days_to_expiration=25,
            profit_loss=0.0,
            profit_loss_percentage=0.0,
            entry_date=datetime.now() - timedelta(days=10),
            status=PositionStatus.ACTIVE,
            week_classification='P-EW',
            account_type='GEN_ACC',
            market_conditions={'condition': 'bullish', 'volatility': 0.18}
        ),
        Position(
            position_id="POS002",
            symbol="TSLA",
            position_type=PositionType.SHORT_CALL,
            quantity=5,
            entry_price=8.00,
            current_price=12.00,  # 50% loss
            strike=250.0,
            expiration=datetime.now() + timedelta(days=5),  # Expiring soon
            delta=60,  # High delta
            gamma=0.08,
            theta=-0.25,
            vega=1.2,
            implied_volatility=0.35,
            days_to_expiration=5,
            profit_loss=0.0,
            profit_loss_percentage=0.0,
            entry_date=datetime.now() - timedelta(days=30),
            status=PositionStatus.ACTIVE,
            week_classification='C-RO',
            account_type='REV_ACC',
            market_conditions={'condition': 'bearish', 'volatility': 0.35}
        ),
        Position(
            position_id="POS003",
            symbol="NVDA",
            position_type=PositionType.SHORT_PUT,
            quantity=8,
            entry_price=5.00,
            current_price=4.50,  # 10% profit
            strike=500.0,
            expiration=datetime.now() + timedelta(days=45),
            delta=-20,
            gamma=0.03,
            theta=-0.10,
            vega=0.6,
            implied_volatility=0.25,
            days_to_expiration=45,
            profit_loss=0.0,
            profit_loss_percentage=0.0,
            entry_date=datetime.now() - timedelta(days=5),
            status=PositionStatus.ACTIVE,
            week_classification='P-AWL',
            account_type='COM_ACC',
            market_conditions={'condition': 'neutral', 'volatility': 0.25}
        )
    ]
    
    # Add positions to manager
    for position in test_positions:
        manager.add_position(position)
    
    print(f"Added {len(test_positions)} test positions")
    
    # Test management decisions
    for position in test_positions:
        print(f"\n--- Position {position.position_id}: {position.symbol} ---")
        print(f"Type: {position.position_type.value}")
        print(f"P&L: {position.profit_loss_percentage:.1%}")
        print(f"DTE: {position.days_to_expiration}")
        print(f"Delta: {position.delta}")
        
        decision = manager.get_management_decision(position.position_id)
        
        if decision:
            print(f"Recommended Action: {decision.action.value}")
            print(f"Trigger: {decision.trigger.value}")
            print(f"Priority: {decision.priority}")
            print(f"Confidence: {decision.confidence:.1%}")
            print(f"Rationale: {decision.rationale}")
            print(f"Expected Outcome: {decision.expected_outcome}")
            print(f"Risk Assessment: {decision.risk_assessment}")
            print(f"Execution Timeline: {decision.execution_timeline}")
            
            if decision.alternatives:
                print(f"Alternatives: {', '.join(decision.alternatives)}")
        else:
            print("No specific action recommended")
    
    # Test position monitoring
    print("\n--- Position Monitoring ---")
    alerts = manager.monitor_positions()
    
    print(f"Generated {len(alerts)} alerts:")
    for alert in alerts:
        print(f"  - {alert.alert_type} ({alert.severity}): {alert.message}")
        print(f"    Recommended: {alert.recommended_action}")
    
    # Test portfolio summary
    print("\n--- Portfolio Summary ---")
    summary = manager.get_portfolio_summary()
    
    print(f"Total Positions: {summary['total_positions']}")
    print(f"Total P&L: ${summary['total_pnl']:.2f} ({summary['total_pnl_percentage']:.1%})")
    print(f"Average DTE: {summary['average_dte']:.1f} days")
    print(f"Delta Exposure: {summary['delta_exposure']:.1f}")
    print(f"Active Alerts: {summary['active_alerts']}")
    
    print("Position Types:")
    for pos_type, count in summary['positions_by_type'].items():
        print(f"  {pos_type}: {count}")
    
    print("\n✅ Position Manager test completed successfully!")

if __name__ == "__main__":
    test_position_manager()

