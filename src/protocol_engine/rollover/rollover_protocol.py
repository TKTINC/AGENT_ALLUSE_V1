"""
Enhanced Roll-Over Protocols and Recovery Strategies for ALL-USE Protocol Engine
Implements sophisticated roll-over decision logic and recovery strategies for adverse market scenarios

This module provides comprehensive roll-over protocols and recovery strategies
to optimize position management during challenging market conditions and
maximize the probability of successful trade outcomes.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RolloverTrigger(Enum):
    """Triggers for rollover decisions"""
    TIME_BASED = "time_based"           # Based on time to expiration
    PROFIT_BASED = "profit_based"       # Based on profit/loss levels
    DELTA_BASED = "delta_based"         # Based on delta changes
    VOLATILITY_BASED = "volatility_based"  # Based on volatility changes
    MARKET_CONDITION = "market_condition"  # Based on market regime change
    ASSIGNMENT_RISK = "assignment_risk"    # Based on assignment probability
    RECOVERY_STRATEGY = "recovery_strategy"  # Part of recovery plan

class RolloverType(Enum):
    """Types of rollover strategies"""
    TIME_ROLL = "time_roll"             # Roll to later expiration
    STRIKE_ROLL = "strike_roll"         # Roll to different strike
    TIME_AND_STRIKE = "time_and_strike" # Roll both time and strike
    DEFENSIVE_ROLL = "defensive_roll"   # Defensive position adjustment
    AGGRESSIVE_ROLL = "aggressive_roll" # Aggressive recovery attempt
    CREDIT_ROLL = "credit_roll"         # Roll for additional credit
    DEBIT_ROLL = "debit_roll"          # Roll accepting debit

class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    ROLL_OUT = "roll_out"               # Roll to later expiration
    ROLL_DOWN = "roll_down"             # Roll to lower strike (puts)
    ROLL_UP = "roll_up"                 # Roll to higher strike (calls)
    CONVERT_TO_SPREAD = "convert_to_spread"  # Convert to spread
    ADD_HEDGE = "add_hedge"             # Add hedging position
    CLOSE_AND_REOPEN = "close_and_reopen"   # Close and reopen new position
    TAKE_ASSIGNMENT = "take_assignment"      # Accept assignment
    EMERGENCY_EXIT = "emergency_exit"        # Emergency position closure

class RecoveryPhase(Enum):
    """Phases of recovery process"""
    ASSESSMENT = "assessment"           # Assess situation
    PLANNING = "planning"               # Plan recovery strategy
    EXECUTION = "execution"             # Execute recovery plan
    MONITORING = "monitoring"           # Monitor recovery progress
    COMPLETION = "completion"           # Recovery completed

@dataclass
class RolloverAnalysis:
    """Analysis for rollover decision"""
    current_position_id: str
    trigger: RolloverTrigger
    rollover_type: RolloverType
    new_expiration: datetime
    new_strike: Optional[float]
    expected_credit: float
    expected_debit: float
    net_credit_debit: float
    probability_success: float
    risk_reduction: float
    time_extension: int
    rationale: str
    confidence: float

@dataclass
class RecoveryPlan:
    """Comprehensive recovery plan"""
    position_id: str
    strategy: RecoveryStrategy
    phase: RecoveryPhase
    target_recovery: float
    max_acceptable_loss: float
    time_horizon: int
    steps: List[Dict[str, Any]]
    success_probability: float
    risk_assessment: str
    monitoring_criteria: List[str]
    exit_criteria: List[str]
    alternative_strategies: List[str]

@dataclass
class RolloverExecution:
    """Rollover execution details"""
    original_position_id: str
    new_position_id: str
    execution_timestamp: datetime
    rollover_type: RolloverType
    original_expiration: datetime
    new_expiration: datetime
    original_strike: float
    new_strike: float
    credit_received: float
    debit_paid: float
    net_result: float
    execution_status: str
    notes: str

class RolloverProtocolEngine:
    """
    Enhanced Roll-Over Protocol Engine for ALL-USE Protocol
    
    Provides sophisticated rollover decision logic including:
    - Time-based and condition-based rollover triggers
    - Multiple rollover strategy types
    - Risk assessment and optimization
    - Credit/debit analysis
    - Success probability calculation
    """
    
    def __init__(self):
        """Initialize the rollover protocol engine"""
        self.logger = logging.getLogger(__name__)
        
        # Rollover criteria and thresholds
        self.rollover_criteria = {
            'time_threshold': 21,           # Days to expiration for time-based rolls
            'profit_threshold': 0.5,        # 50% profit for profit-based rolls
            'loss_threshold': -0.2,         # 20% loss for defensive rolls
            'delta_threshold': 50,          # Delta threshold for delta-based rolls
            'iv_change_threshold': 0.15,    # 15% IV change threshold
            'assignment_prob_threshold': 0.8 # 80% assignment probability
        }
        
        # Rollover preferences by account type
        self.account_preferences = {
            'GEN_ACC': {
                'preferred_dte': 35,
                'max_debit_tolerance': 0.1,     # 10% of original credit
                'risk_tolerance': 'medium',
                'aggressive_recovery': False
            },
            'REV_ACC': {
                'preferred_dte': 30,
                'max_debit_tolerance': 0.05,    # 5% of original credit
                'risk_tolerance': 'low',
                'aggressive_recovery': False
            },
            'COM_ACC': {
                'preferred_dte': 40,
                'max_debit_tolerance': 0.15,    # 15% of original credit
                'risk_tolerance': 'high',
                'aggressive_recovery': True
            }
        }
        
        # Success probability models
        self.success_models = {
            'time_roll': {
                'base_probability': 0.75,
                'time_factor': 0.02,        # +2% per week extended
                'volatility_factor': -0.1,  # -10% in high vol
                'market_factor': 0.05       # +5% in favorable market
            },
            'strike_roll': {
                'base_probability': 0.65,
                'distance_factor': 0.03,    # +3% per delta moved
                'credit_factor': 0.1,       # +10% if credit received
                'market_factor': 0.08
            }
        }
        
        # Historical rollover data for learning
        self.rollover_history: List[RolloverExecution] = []
        
        self.logger.info("Rollover Protocol Engine initialized")
    
    def analyze_rollover_opportunity(self, position: Dict[str, Any], 
                                   market_context: Dict[str, Any]) -> Optional[RolloverAnalysis]:
        """
        Analyze rollover opportunity for a position
        
        Args:
            position: Position data
            market_context: Current market context
            
        Returns:
            RolloverAnalysis: Rollover analysis or None if not recommended
        """
        try:
            # Determine rollover trigger
            trigger = self._identify_rollover_trigger(position, market_context)
            
            if not trigger:
                return None
            
            # Determine rollover type
            rollover_type = self._determine_rollover_type(position, trigger, market_context)
            
            # Calculate new parameters
            new_expiration = self._calculate_new_expiration(position, rollover_type)
            new_strike = self._calculate_new_strike(position, rollover_type, market_context)
            
            # Estimate credit/debit
            credit_estimate = self._estimate_rollover_credit(position, new_expiration, new_strike)
            debit_estimate = self._estimate_rollover_debit(position, new_expiration, new_strike)
            net_result = credit_estimate - debit_estimate
            
            # Calculate success probability
            success_probability = self._calculate_success_probability(
                position, rollover_type, new_expiration, new_strike, market_context
            )
            
            # Assess risk reduction
            risk_reduction = self._assess_risk_reduction(position, new_expiration, new_strike)
            
            # Calculate time extension
            current_dte = (datetime.fromisoformat(position['expiration']) - datetime.now()).days
            new_dte = (new_expiration - datetime.now()).days
            time_extension = new_dte - current_dte
            
            # Generate rationale
            rationale = self._generate_rollover_rationale(
                trigger, rollover_type, net_result, success_probability
            )
            
            # Calculate confidence
            confidence = self._calculate_rollover_confidence(
                position, rollover_type, success_probability, market_context
            )
            
            analysis = RolloverAnalysis(
                current_position_id=position['position_id'],
                trigger=trigger,
                rollover_type=rollover_type,
                new_expiration=new_expiration,
                new_strike=new_strike,
                expected_credit=credit_estimate,
                expected_debit=debit_estimate,
                net_credit_debit=net_result,
                probability_success=success_probability,
                risk_reduction=risk_reduction,
                time_extension=time_extension,
                rationale=rationale,
                confidence=confidence
            )
            
            self.logger.info(f"Rollover analysis for {position['position_id']}: {rollover_type.value}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing rollover opportunity: {str(e)}")
            raise
    
    def _identify_rollover_trigger(self, position: Dict[str, Any], 
                                 market_context: Dict[str, Any]) -> Optional[RolloverTrigger]:
        """Identify what triggers the rollover consideration"""
        dte = (datetime.fromisoformat(position['expiration']) - datetime.now()).days
        pnl_pct = position.get('profit_loss_percentage', 0)
        delta = abs(position.get('delta', 0))
        
        # Time-based trigger
        if dte <= self.rollover_criteria['time_threshold']:
            return RolloverTrigger.TIME_BASED
        
        # Loss-based trigger
        if pnl_pct <= self.rollover_criteria['loss_threshold']:
            return RolloverTrigger.PROFIT_BASED
        
        # Delta-based trigger
        if delta >= self.rollover_criteria['delta_threshold']:
            return RolloverTrigger.DELTA_BASED
        
        # Assignment risk trigger
        assignment_prob = market_context.get('assignment_probability', 0)
        if assignment_prob >= self.rollover_criteria['assignment_prob_threshold']:
            return RolloverTrigger.ASSIGNMENT_RISK
        
        # Volatility-based trigger
        iv_change = market_context.get('iv_change', 0)
        if abs(iv_change) >= self.rollover_criteria['iv_change_threshold']:
            return RolloverTrigger.VOLATILITY_BASED
        
        return None
    
    def _determine_rollover_type(self, position: Dict[str, Any], trigger: RolloverTrigger,
                               market_context: Dict[str, Any]) -> RolloverType:
        """Determine the type of rollover strategy"""
        pnl_pct = position.get('profit_loss_percentage', 0)
        account_type = position.get('account_type', 'GEN_ACC')
        preferences = self.account_preferences[account_type]
        
        if trigger == RolloverTrigger.TIME_BASED:
            if pnl_pct > 0:
                return RolloverType.TIME_ROLL  # Simple time extension
            else:
                return RolloverType.TIME_AND_STRIKE  # Time and strike adjustment
        
        elif trigger == RolloverTrigger.PROFIT_BASED:
            if pnl_pct <= -0.3:  # Large loss
                if preferences['aggressive_recovery']:
                    return RolloverType.AGGRESSIVE_ROLL
                else:
                    return RolloverType.DEFENSIVE_ROLL
            else:
                return RolloverType.TIME_AND_STRIKE
        
        elif trigger == RolloverTrigger.DELTA_BASED:
            return RolloverType.STRIKE_ROLL  # Adjust strike to manage delta
        
        elif trigger == RolloverTrigger.ASSIGNMENT_RISK:
            return RolloverType.DEFENSIVE_ROLL  # Defensive adjustment
        
        else:
            return RolloverType.TIME_ROLL  # Default to time roll
    
    def _calculate_new_expiration(self, position: Dict[str, Any], 
                                rollover_type: RolloverType) -> datetime:
        """Calculate new expiration date"""
        account_type = position.get('account_type', 'GEN_ACC')
        preferred_dte = self.account_preferences[account_type]['preferred_dte']
        
        if rollover_type in [RolloverType.TIME_ROLL, RolloverType.TIME_AND_STRIKE]:
            return datetime.now() + timedelta(days=preferred_dte)
        elif rollover_type == RolloverType.DEFENSIVE_ROLL:
            return datetime.now() + timedelta(days=preferred_dte + 14)  # Extra time for recovery
        elif rollover_type == RolloverType.AGGRESSIVE_ROLL:
            return datetime.now() + timedelta(days=preferred_dte - 7)   # Shorter time for quicker recovery
        else:
            return datetime.now() + timedelta(days=preferred_dte)
    
    def _calculate_new_strike(self, position: Dict[str, Any], rollover_type: RolloverType,
                            market_context: Dict[str, Any]) -> Optional[float]:
        """Calculate new strike price"""
        current_strike = position.get('strike', 0)
        current_price = market_context.get('current_price', current_strike)
        position_type = position.get('position_type', 'short_put')
        
        if rollover_type == RolloverType.TIME_ROLL:
            return current_strike  # Keep same strike
        
        elif rollover_type == RolloverType.STRIKE_ROLL:
            if position_type == 'short_put':
                # Roll down for puts (further OTM)
                return current_strike * 0.95
            else:  # short_call
                # Roll up for calls (further OTM)
                return current_strike * 1.05
        
        elif rollover_type == RolloverType.DEFENSIVE_ROLL:
            if position_type == 'short_put':
                return current_strike * 0.90  # Significantly further OTM
            else:
                return current_strike * 1.10
        
        elif rollover_type == RolloverType.AGGRESSIVE_ROLL:
            if position_type == 'short_put':
                return current_price * 0.98  # Closer to current price for more premium
            else:
                return current_price * 1.02
        
        else:
            # TIME_AND_STRIKE - moderate adjustment
            if position_type == 'short_put':
                return current_strike * 0.97
            else:
                return current_strike * 1.03
    
    def _estimate_rollover_credit(self, position: Dict[str, Any], 
                                new_expiration: datetime, new_strike: Optional[float]) -> float:
        """Estimate credit from rollover"""
        # Simplified estimation - in practice would use options pricing model
        base_credit = position.get('entry_price', 0) * 0.3  # 30% of original premium
        
        # Time value adjustment
        dte = (new_expiration - datetime.now()).days
        time_value = base_credit * (dte / 30) * 0.1  # 10% per month
        
        # Strike adjustment
        strike_adjustment = 0
        if new_strike and new_strike != position.get('strike', 0):
            strike_adjustment = base_credit * 0.2  # 20% adjustment for strike change
        
        return base_credit + time_value + strike_adjustment
    
    def _estimate_rollover_debit(self, position: Dict[str, Any], 
                               new_expiration: datetime, new_strike: Optional[float]) -> float:
        """Estimate debit for rollover"""
        # Cost to close current position
        current_value = position.get('current_price', 0)
        
        # Add transaction costs
        transaction_cost = current_value * 0.02  # 2% transaction cost
        
        return current_value + transaction_cost
    
    def _calculate_success_probability(self, position: Dict[str, Any], rollover_type: RolloverType,
                                     new_expiration: datetime, new_strike: Optional[float],
                                     market_context: Dict[str, Any]) -> float:
        """Calculate probability of successful rollover"""
        model = self.success_models.get(rollover_type.value.split('_')[0], 
                                       self.success_models['time_roll'])
        
        base_prob = model['base_probability']
        
        # Time factor
        dte = (new_expiration - datetime.now()).days
        time_adjustment = model['time_factor'] * (dte / 7)  # Per week
        
        # Volatility factor
        volatility_regime = market_context.get('volatility_regime', 'normal')
        vol_adjustment = 0
        if volatility_regime in ['high', 'very_high']:
            vol_adjustment = model['volatility_factor']
        
        # Market factor
        market_condition = market_context.get('market_condition', 'neutral')
        market_adjustment = 0
        if market_condition in ['bullish', 'extremely_bullish']:
            market_adjustment = model['market_factor']
        elif market_condition in ['bearish', 'extremely_bearish']:
            market_adjustment = -model['market_factor']
        
        total_probability = base_prob + time_adjustment + vol_adjustment + market_adjustment
        return max(0.1, min(0.95, total_probability))
    
    def _assess_risk_reduction(self, position: Dict[str, Any], 
                             new_expiration: datetime, new_strike: Optional[float]) -> float:
        """Assess risk reduction from rollover"""
        base_risk_reduction = 0.2  # 20% base risk reduction
        
        # Time extension reduces risk
        dte = (new_expiration - datetime.now()).days
        time_risk_reduction = min(0.3, dte / 100)  # Up to 30% for longer time
        
        # Strike adjustment reduces risk
        strike_risk_reduction = 0
        if new_strike and new_strike != position.get('strike', 0):
            strike_risk_reduction = 0.15  # 15% for strike adjustment
        
        return base_risk_reduction + time_risk_reduction + strike_risk_reduction
    
    def _generate_rollover_rationale(self, trigger: RolloverTrigger, rollover_type: RolloverType,
                                   net_result: float, success_probability: float) -> str:
        """Generate rationale for rollover recommendation"""
        trigger_reasons = {
            RolloverTrigger.TIME_BASED: "approaching expiration",
            RolloverTrigger.PROFIT_BASED: "profit/loss management",
            RolloverTrigger.DELTA_BASED: "delta adjustment needed",
            RolloverTrigger.ASSIGNMENT_RISK: "high assignment risk",
            RolloverTrigger.VOLATILITY_BASED: "volatility change"
        }
        
        base_reason = trigger_reasons.get(trigger, "position optimization")
        
        if net_result > 0:
            credit_info = f"with expected net credit of ${net_result:.2f}"
        else:
            credit_info = f"with expected net debit of ${abs(net_result):.2f}"
        
        return (f"Recommend {rollover_type.value} due to {base_reason}, "
                f"{credit_info}. Success probability: {success_probability:.1%}")
    
    def _calculate_rollover_confidence(self, position: Dict[str, Any], rollover_type: RolloverType,
                                     success_probability: float, market_context: Dict[str, Any]) -> float:
        """Calculate confidence in rollover recommendation"""
        base_confidence = 0.8
        
        # Adjust based on success probability
        prob_adjustment = (success_probability - 0.5) * 0.4  # Scale to ±0.2
        
        # Adjust based on market clarity
        market_condition = market_context.get('market_condition', 'neutral')
        if market_condition in ['extremely_bullish', 'extremely_bearish']:
            clarity_adjustment = 0.1
        elif market_condition == 'neutral':
            clarity_adjustment = -0.05
        else:
            clarity_adjustment = 0.05
        
        # Adjust based on rollover type complexity
        complexity_adjustment = 0
        if rollover_type in [RolloverType.TIME_ROLL]:
            complexity_adjustment = 0.05  # Simple rollover
        elif rollover_type in [RolloverType.AGGRESSIVE_ROLL]:
            complexity_adjustment = -0.1  # Complex rollover
        
        total_confidence = base_confidence + prob_adjustment + clarity_adjustment + complexity_adjustment
        return max(0.5, min(0.95, total_confidence))

class RecoveryStrategyEngine:
    """
    Recovery Strategy Engine for adverse market scenarios
    
    Provides comprehensive recovery strategies for positions in distress
    including systematic approaches to loss mitigation and position recovery.
    """
    
    def __init__(self):
        """Initialize the recovery strategy engine"""
        self.logger = logging.getLogger(__name__)
        
        # Recovery thresholds
        self.recovery_thresholds = {
            'minor_loss': -0.15,        # 15% loss
            'moderate_loss': -0.30,     # 30% loss
            'major_loss': -0.50,        # 50% loss
            'critical_loss': -0.75      # 75% loss
        }
        
        # Recovery strategies by loss level
        self.recovery_strategies = {
            'minor_loss': [RecoveryStrategy.ROLL_OUT, RecoveryStrategy.ADD_HEDGE],
            'moderate_loss': [RecoveryStrategy.ROLL_DOWN, RecoveryStrategy.CONVERT_TO_SPREAD],
            'major_loss': [RecoveryStrategy.CLOSE_AND_REOPEN, RecoveryStrategy.TAKE_ASSIGNMENT],
            'critical_loss': [RecoveryStrategy.EMERGENCY_EXIT]
        }
        
        # Recovery success rates by strategy
        self.success_rates = {
            RecoveryStrategy.ROLL_OUT: 0.70,
            RecoveryStrategy.ROLL_DOWN: 0.65,
            RecoveryStrategy.ADD_HEDGE: 0.75,
            RecoveryStrategy.CONVERT_TO_SPREAD: 0.60,
            RecoveryStrategy.CLOSE_AND_REOPEN: 0.55,
            RecoveryStrategy.TAKE_ASSIGNMENT: 0.80,
            RecoveryStrategy.EMERGENCY_EXIT: 1.00  # Always "succeeds" by limiting loss
        }
        
        self.logger.info("Recovery Strategy Engine initialized")
    
    def create_recovery_plan(self, position: Dict[str, Any], 
                           market_context: Dict[str, Any]) -> RecoveryPlan:
        """
        Create comprehensive recovery plan for a distressed position
        
        Args:
            position: Position in distress
            market_context: Current market context
            
        Returns:
            RecoveryPlan: Comprehensive recovery strategy
        """
        try:
            pnl_pct = position.get('profit_loss_percentage', 0)
            
            # Classify loss level
            loss_level = self._classify_loss_level(pnl_pct)
            
            # Select primary recovery strategy
            primary_strategy = self._select_recovery_strategy(loss_level, position, market_context)
            
            # Calculate recovery targets
            target_recovery = self._calculate_recovery_target(pnl_pct, primary_strategy)
            max_acceptable_loss = self._calculate_max_acceptable_loss(pnl_pct, position)
            
            # Determine time horizon
            time_horizon = self._determine_recovery_time_horizon(primary_strategy, position)
            
            # Create recovery steps
            steps = self._create_recovery_steps(primary_strategy, position, market_context)
            
            # Calculate success probability
            success_probability = self._calculate_recovery_success_probability(
                primary_strategy, position, market_context
            )
            
            # Assess risk
            risk_assessment = self._assess_recovery_risk(primary_strategy, position)
            
            # Define monitoring and exit criteria
            monitoring_criteria = self._define_monitoring_criteria(primary_strategy)
            exit_criteria = self._define_exit_criteria(primary_strategy, target_recovery, max_acceptable_loss)
            
            # Identify alternative strategies
            alternative_strategies = self._identify_alternative_strategies(loss_level, primary_strategy)
            
            plan = RecoveryPlan(
                position_id=position['position_id'],
                strategy=primary_strategy,
                phase=RecoveryPhase.PLANNING,
                target_recovery=target_recovery,
                max_acceptable_loss=max_acceptable_loss,
                time_horizon=time_horizon,
                steps=steps,
                success_probability=success_probability,
                risk_assessment=risk_assessment,
                monitoring_criteria=monitoring_criteria,
                exit_criteria=exit_criteria,
                alternative_strategies=alternative_strategies
            )
            
            self.logger.info(f"Created recovery plan for {position['position_id']}: {primary_strategy.value}")
            return plan
            
        except Exception as e:
            self.logger.error(f"Error creating recovery plan: {str(e)}")
            raise
    
    def _classify_loss_level(self, pnl_pct: float) -> str:
        """Classify the loss level"""
        if pnl_pct >= self.recovery_thresholds['minor_loss']:
            return 'minor_loss'
        elif pnl_pct >= self.recovery_thresholds['moderate_loss']:
            return 'moderate_loss'
        elif pnl_pct >= self.recovery_thresholds['major_loss']:
            return 'major_loss'
        else:
            return 'critical_loss'
    
    def _select_recovery_strategy(self, loss_level: str, position: Dict[str, Any],
                                market_context: Dict[str, Any]) -> RecoveryStrategy:
        """Select the most appropriate recovery strategy"""
        available_strategies = self.recovery_strategies[loss_level]
        
        # Consider position characteristics
        dte = (datetime.fromisoformat(position['expiration']) - datetime.now()).days
        position_type = position.get('position_type', 'short_put')
        account_type = position.get('account_type', 'GEN_ACC')
        
        # Strategy selection logic
        if loss_level == 'critical_loss':
            return RecoveryStrategy.EMERGENCY_EXIT
        
        elif loss_level == 'major_loss':
            if dte < 14:  # Close to expiration
                return RecoveryStrategy.TAKE_ASSIGNMENT
            else:
                return RecoveryStrategy.CLOSE_AND_REOPEN
        
        elif loss_level == 'moderate_loss':
            if position_type == 'short_put':
                return RecoveryStrategy.ROLL_DOWN
            else:
                return RecoveryStrategy.CONVERT_TO_SPREAD
        
        else:  # minor_loss
            if dte > 21:
                return RecoveryStrategy.ROLL_OUT
            else:
                return RecoveryStrategy.ADD_HEDGE
    
    def _calculate_recovery_target(self, current_pnl_pct: float, 
                                 strategy: RecoveryStrategy) -> float:
        """Calculate realistic recovery target"""
        if strategy == RecoveryStrategy.EMERGENCY_EXIT:
            return current_pnl_pct  # No recovery expected
        
        elif strategy == RecoveryStrategy.TAKE_ASSIGNMENT:
            return current_pnl_pct * 0.5  # 50% recovery through assignment
        
        elif strategy in [RecoveryStrategy.ROLL_OUT, RecoveryStrategy.ROLL_DOWN]:
            return current_pnl_pct * 0.3  # 30% recovery through rolling
        
        else:
            return current_pnl_pct * 0.7  # 70% recovery for other strategies
    
    def _calculate_max_acceptable_loss(self, current_pnl_pct: float, 
                                     position: Dict[str, Any]) -> float:
        """Calculate maximum acceptable loss"""
        account_type = position.get('account_type', 'GEN_ACC')
        
        # Account-specific loss tolerance
        loss_tolerance = {
            'GEN_ACC': 1.2,    # 20% additional loss acceptable
            'REV_ACC': 1.1,    # 10% additional loss acceptable
            'COM_ACC': 1.3     # 30% additional loss acceptable
        }
        
        multiplier = loss_tolerance[account_type]
        return current_pnl_pct * multiplier
    
    def _determine_recovery_time_horizon(self, strategy: RecoveryStrategy, 
                                       position: Dict[str, Any]) -> int:
        """Determine time horizon for recovery"""
        dte = (datetime.fromisoformat(position['expiration']) - datetime.now()).days
        
        if strategy == RecoveryStrategy.EMERGENCY_EXIT:
            return 1  # Immediate
        elif strategy == RecoveryStrategy.TAKE_ASSIGNMENT:
            return dte  # Until expiration
        elif strategy in [RecoveryStrategy.ROLL_OUT, RecoveryStrategy.ROLL_DOWN]:
            return 30  # One month
        else:
            return 14  # Two weeks
    
    def _create_recovery_steps(self, strategy: RecoveryStrategy, position: Dict[str, Any],
                             market_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create detailed recovery steps"""
        steps = []
        
        if strategy == RecoveryStrategy.ROLL_OUT:
            steps = [
                {
                    'step': 1,
                    'action': 'Close current position',
                    'timeline': 'Immediate',
                    'expected_cost': position.get('current_price', 0)
                },
                {
                    'step': 2,
                    'action': 'Open new position with later expiration',
                    'timeline': 'Same day',
                    'expected_credit': position.get('entry_price', 0) * 0.8
                }
            ]
        
        elif strategy == RecoveryStrategy.EMERGENCY_EXIT:
            steps = [
                {
                    'step': 1,
                    'action': 'Close position immediately',
                    'timeline': 'Immediate',
                    'expected_cost': position.get('current_price', 0)
                }
            ]
        
        # Add more strategy-specific steps as needed
        
        return steps
    
    def _calculate_recovery_success_probability(self, strategy: RecoveryStrategy,
                                              position: Dict[str, Any],
                                              market_context: Dict[str, Any]) -> float:
        """Calculate probability of recovery success"""
        base_probability = self.success_rates[strategy]
        
        # Adjust based on market conditions
        market_condition = market_context.get('market_condition', 'neutral')
        if market_condition in ['bullish', 'extremely_bullish']:
            market_adjustment = 0.1
        elif market_condition in ['bearish', 'extremely_bearish']:
            market_adjustment = -0.1
        else:
            market_adjustment = 0
        
        # Adjust based on time to expiration
        dte = (datetime.fromisoformat(position['expiration']) - datetime.now()).days
        if dte > 30:
            time_adjustment = 0.1
        elif dte < 7:
            time_adjustment = -0.15
        else:
            time_adjustment = 0
        
        total_probability = base_probability + market_adjustment + time_adjustment
        return max(0.1, min(0.95, total_probability))
    
    def _assess_recovery_risk(self, strategy: RecoveryStrategy, position: Dict[str, Any]) -> str:
        """Assess risk of recovery strategy"""
        risk_levels = {
            RecoveryStrategy.EMERGENCY_EXIT: "Low - crystallizes current loss",
            RecoveryStrategy.TAKE_ASSIGNMENT: "Medium - requires capital for assignment",
            RecoveryStrategy.ROLL_OUT: "Medium - extends market exposure",
            RecoveryStrategy.ROLL_DOWN: "Medium-High - closer to money",
            RecoveryStrategy.ADD_HEDGE: "Medium - additional position complexity",
            RecoveryStrategy.CONVERT_TO_SPREAD: "High - complex position management",
            RecoveryStrategy.CLOSE_AND_REOPEN: "High - market timing risk"
        }
        
        return risk_levels.get(strategy, "Medium - standard recovery risk")
    
    def _define_monitoring_criteria(self, strategy: RecoveryStrategy) -> List[str]:
        """Define monitoring criteria for recovery strategy"""
        base_criteria = [
            "Daily P&L tracking",
            "Market condition changes",
            "Volatility regime shifts"
        ]
        
        strategy_specific = {
            RecoveryStrategy.ROLL_OUT: ["Time decay progression", "New position delta"],
            RecoveryStrategy.TAKE_ASSIGNMENT: ["Assignment probability", "Underlying price movement"],
            RecoveryStrategy.ADD_HEDGE: ["Hedge effectiveness", "Correlation stability"]
        }
        
        return base_criteria + strategy_specific.get(strategy, [])
    
    def _define_exit_criteria(self, strategy: RecoveryStrategy, target_recovery: float,
                            max_acceptable_loss: float) -> List[str]:
        """Define exit criteria for recovery strategy"""
        criteria = [
            f"Target recovery achieved: {target_recovery:.1%}",
            f"Maximum loss reached: {max_acceptable_loss:.1%}",
            "Market conditions fundamentally change"
        ]
        
        if strategy in [RecoveryStrategy.ROLL_OUT, RecoveryStrategy.ROLL_DOWN]:
            criteria.append("New position reaches 50% profit")
        
        return criteria
    
    def _identify_alternative_strategies(self, loss_level: str, 
                                       primary_strategy: RecoveryStrategy) -> List[str]:
        """Identify alternative recovery strategies"""
        all_strategies = self.recovery_strategies[loss_level]
        alternatives = [s.value for s in all_strategies if s != primary_strategy]
        
        # Add general alternatives
        if primary_strategy != RecoveryStrategy.EMERGENCY_EXIT:
            alternatives.append(RecoveryStrategy.EMERGENCY_EXIT.value)
        
        return alternatives

def test_rollover_and_recovery_systems():
    """Test the rollover and recovery systems"""
    print("Testing Rollover and Recovery Systems...")
    
    rollover_engine = RolloverProtocolEngine()
    recovery_engine = RecoveryStrategyEngine()
    
    # Test positions
    test_positions = [
        {
            'position_id': 'POS001',
            'symbol': 'SPY',
            'position_type': 'short_put',
            'strike': 450.0,
            'expiration': (datetime.now() + timedelta(days=15)).isoformat(),
            'entry_price': 2.50,
            'current_price': 1.25,
            'profit_loss_percentage': 0.5,  # 50% profit
            'delta': -25,
            'account_type': 'GEN_ACC'
        },
        {
            'position_id': 'POS002',
            'symbol': 'TSLA',
            'position_type': 'short_call',
            'strike': 250.0,
            'expiration': (datetime.now() + timedelta(days=30)).isoformat(),
            'entry_price': 8.00,
            'current_price': 12.00,
            'profit_loss_percentage': -0.35,  # 35% loss
            'delta': 60,
            'account_type': 'REV_ACC'
        }
    ]
    
    market_context = {
        'current_price': 455.0,
        'market_condition': 'bullish',
        'volatility_regime': 'normal',
        'assignment_probability': 0.15,
        'iv_change': 0.05
    }
    
    # Test rollover analysis
    print("\n--- Rollover Analysis ---")
    for position in test_positions:
        print(f"\nPosition {position['position_id']}: {position['symbol']}")
        print(f"P&L: {position['profit_loss_percentage']:.1%}")
        
        analysis = rollover_engine.analyze_rollover_opportunity(position, market_context)
        
        if analysis:
            print(f"Rollover Recommended: {analysis.rollover_type.value}")
            print(f"Trigger: {analysis.trigger.value}")
            print(f"New Expiration: {analysis.new_expiration.strftime('%Y-%m-%d')}")
            print(f"New Strike: {analysis.new_strike}")
            print(f"Net Credit/Debit: ${analysis.net_credit_debit:.2f}")
            print(f"Success Probability: {analysis.probability_success:.1%}")
            print(f"Risk Reduction: {analysis.risk_reduction:.1%}")
            print(f"Confidence: {analysis.confidence:.1%}")
            print(f"Rationale: {analysis.rationale}")
        else:
            print("No rollover recommended")
    
    # Test recovery strategies
    print("\n--- Recovery Strategy Analysis ---")
    distressed_position = {
        'position_id': 'POS003',
        'symbol': 'NVDA',
        'position_type': 'short_put',
        'strike': 500.0,
        'expiration': (datetime.now() + timedelta(days=20)).isoformat(),
        'entry_price': 5.00,
        'current_price': 8.50,
        'profit_loss_percentage': -0.40,  # 40% loss
        'delta': -55,
        'account_type': 'COM_ACC'
    }
    
    print(f"\nDistressed Position: {distressed_position['symbol']}")
    print(f"Loss: {distressed_position['profit_loss_percentage']:.1%}")
    
    recovery_plan = recovery_engine.create_recovery_plan(distressed_position, market_context)
    
    print(f"Recovery Strategy: {recovery_plan.strategy.value}")
    print(f"Target Recovery: {recovery_plan.target_recovery:.1%}")
    print(f"Max Acceptable Loss: {recovery_plan.max_acceptable_loss:.1%}")
    print(f"Time Horizon: {recovery_plan.time_horizon} days")
    print(f"Success Probability: {recovery_plan.success_probability:.1%}")
    print(f"Risk Assessment: {recovery_plan.risk_assessment}")
    
    print("Recovery Steps:")
    for step in recovery_plan.steps:
        print(f"  Step {step['step']}: {step['action']} ({step['timeline']})")
    
    print("Monitoring Criteria:")
    for criterion in recovery_plan.monitoring_criteria:
        print(f"  - {criterion}")
    
    print("Exit Criteria:")
    for criterion in recovery_plan.exit_criteria:
        print(f"  - {criterion}")
    
    print("Alternative Strategies:")
    for alt in recovery_plan.alternative_strategies:
        print(f"  - {alt}")
    
    print("\n✅ Rollover and Recovery Systems test completed successfully!")

if __name__ == "__main__":
    test_rollover_and_recovery_systems()

