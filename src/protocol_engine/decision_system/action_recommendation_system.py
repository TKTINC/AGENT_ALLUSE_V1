"""
Decision Trees and Action Recommendation System for ALL-USE Protocol Engine
Provides intelligent decision trees and action recommendations based on market conditions and week classifications

This module implements sophisticated decision trees that translate market analysis
and week classifications into specific trading actions and recommendations.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of trading actions"""
    SELL_PUT = "sell_put"
    SELL_CALL = "sell_call"
    ROLL_POSITION = "roll_position"
    CLOSE_POSITION = "close_position"
    ADJUST_DELTA = "adjust_delta"
    INCREASE_POSITION = "increase_position"
    DECREASE_POSITION = "decrease_position"
    HOLD_POSITION = "hold_position"
    ENTER_PROTECTIVE = "enter_protective"
    EXIT_PROTECTIVE = "exit_protective"
    REBALANCE_PORTFOLIO = "rebalance_portfolio"
    WAIT_SIGNAL = "wait_signal"

class Priority(Enum):
    """Action priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"

class RiskLevel(Enum):
    """Risk levels for actions"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class TradingAction:
    """Individual trading action recommendation"""
    action_type: ActionType
    priority: Priority
    risk_level: RiskLevel
    description: str
    rationale: str
    parameters: Dict[str, Any]
    expected_outcome: str
    success_probability: float
    estimated_return: float
    max_risk: float
    time_horizon: str
    prerequisites: List[str]
    alternatives: List[str]

@dataclass
class ActionPlan:
    """Complete action plan with multiple recommendations"""
    primary_action: TradingAction
    secondary_actions: List[TradingAction]
    risk_management_actions: List[TradingAction]
    contingency_actions: List[TradingAction]
    overall_strategy: str
    confidence_score: float
    expected_portfolio_impact: Dict[str, float]
    timeline: Dict[str, str]
    monitoring_points: List[str]
    exit_criteria: List[str]

class DecisionNode:
    """Decision tree node for action recommendations"""
    
    def __init__(self, condition: str, action: Optional[TradingAction] = None):
        self.condition = condition
        self.action = action
        self.children = {}
        self.probability_weights = {}
    
    def add_child(self, condition_value: str, child_node: 'DecisionNode', weight: float = 1.0):
        """Add a child node with probability weight"""
        self.children[condition_value] = child_node
        self.probability_weights[condition_value] = weight
    
    def evaluate(self, context: Dict[str, Any]) -> Optional[TradingAction]:
        """Evaluate the decision tree with given context"""
        if not self.children:
            return self.action
        
        condition_value = context.get(self.condition)
        if condition_value in self.children:
            return self.children[condition_value].evaluate(context)
        
        # Find best match based on probability weights
        best_match = max(self.children.keys(), 
                        key=lambda k: self.probability_weights.get(k, 0))
        return self.children[best_match].evaluate(context)

class ActionRecommendationSystem:
    """
    Advanced action recommendation system for ALL-USE protocol
    
    Provides intelligent decision trees and action recommendations based on:
    - Week classification results
    - Market condition analysis
    - Portfolio state
    - Risk parameters
    - Historical performance
    """
    
    def __init__(self):
        """Initialize the action recommendation system"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize decision trees
        self.week_classification_tree = self._build_week_classification_tree()
        self.market_condition_tree = self._build_market_condition_tree()
        self.risk_management_tree = self._build_risk_management_tree()
        self.portfolio_optimization_tree = self._build_portfolio_optimization_tree()
        
        # Action templates
        self.action_templates = self._initialize_action_templates()
        
        # Risk parameters
        self.risk_parameters = {
            'max_position_size': 0.1,  # 10% max per position
            'max_portfolio_risk': 0.15,  # 15% max portfolio risk
            'volatility_threshold': 0.35,  # High volatility threshold
            'drawdown_threshold': 0.05,  # 5% drawdown threshold
            'correlation_threshold': 0.7,  # High correlation threshold
        }
        
        self.logger.info("Action Recommendation System initialized")
    
    def generate_action_plan(self, context: Dict[str, Any]) -> ActionPlan:
        """
        Generate comprehensive action plan based on current context
        
        Args:
            context: Dictionary containing market analysis, week classification, portfolio state
            
        Returns:
            ActionPlan: Complete action plan with recommendations
        """
        try:
            # Extract context components
            week_classification = context.get('week_classification')
            market_analysis = context.get('market_analysis')
            portfolio_state = context.get('portfolio_state', {})
            account_type = context.get('account_type', 'GEN_ACC')
            
            # Generate primary action from week classification
            primary_action = self._get_primary_action(week_classification, market_analysis, account_type)
            
            # Generate secondary actions
            secondary_actions = self._get_secondary_actions(context)
            
            # Generate risk management actions
            risk_actions = self._get_risk_management_actions(context)
            
            # Generate contingency actions
            contingency_actions = self._get_contingency_actions(context)
            
            # Determine overall strategy
            overall_strategy = self._determine_overall_strategy(
                week_classification, market_analysis, primary_action
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_action_confidence(context, primary_action)
            
            # Estimate portfolio impact
            portfolio_impact = self._estimate_portfolio_impact(
                primary_action, secondary_actions, portfolio_state
            )
            
            # Create timeline
            timeline = self._create_action_timeline(primary_action, secondary_actions)
            
            # Define monitoring points
            monitoring_points = self._define_monitoring_points(context, primary_action)
            
            # Define exit criteria
            exit_criteria = self._define_exit_criteria(context, primary_action)
            
            action_plan = ActionPlan(
                primary_action=primary_action,
                secondary_actions=secondary_actions,
                risk_management_actions=risk_actions,
                contingency_actions=contingency_actions,
                overall_strategy=overall_strategy,
                confidence_score=confidence_score,
                expected_portfolio_impact=portfolio_impact,
                timeline=timeline,
                monitoring_points=monitoring_points,
                exit_criteria=exit_criteria
            )
            
            self.logger.info(f"Action plan generated: {primary_action.action_type.value} with {confidence_score:.1%} confidence")
            return action_plan
            
        except Exception as e:
            self.logger.error(f"Error generating action plan: {str(e)}")
            raise
    
    def _build_week_classification_tree(self) -> DecisionNode:
        """Build decision tree for week classification actions"""
        root = DecisionNode("week_type")
        
        # Put scenarios
        p_ew_action = TradingAction(
            action_type=ActionType.SELL_PUT,
            priority=Priority.HIGH,
            risk_level=RiskLevel.LOW,
            description="Sell puts for premium collection",
            rationale="Market conditions favor put expiration worthless",
            parameters={"delta_range": (15, 25), "dte_range": (30, 45)},
            expected_outcome="Premium collection with high probability",
            success_probability=0.85,
            estimated_return=0.02,
            max_risk=0.05,
            time_horizon="1-4 weeks",
            prerequisites=["Sufficient buying power", "Low volatility environment"],
            alternatives=["Sell call spreads", "Iron condors"]
        )
        
        p_awl_action = TradingAction(
            action_type=ActionType.SELL_PUT,
            priority=Priority.HIGH,
            risk_level=RiskLevel.MODERATE,
            description="Sell puts with assignment acceptance",
            rationale="Prepared for assignment within acceptable limits",
            parameters={"delta_range": (25, 35), "dte_range": (30, 45)},
            expected_outcome="Premium collection or favorable assignment",
            success_probability=0.80,
            estimated_return=0.02,
            max_risk=0.08,
            time_horizon="1-4 weeks",
            prerequisites=["Cash available for assignment", "Bullish on underlying"],
            alternatives=["Cash-secured puts", "Protective puts"]
        )
        
        p_ro_action = TradingAction(
            action_type=ActionType.ROLL_POSITION,
            priority=Priority.CRITICAL,
            risk_level=RiskLevel.HIGH,
            description="Roll put positions to avoid assignment",
            rationale="Market decline requires position adjustment",
            parameters={"roll_delta": 5, "extend_dte": 30},
            expected_outcome="Avoid assignment and collect additional premium",
            success_probability=0.70,
            estimated_return=0.015,
            max_risk=0.12,
            time_horizon="Immediate",
            prerequisites=["Liquid options market", "Sufficient margin"],
            alternatives=["Accept assignment", "Close position"]
        )
        
        p_aol_action = TradingAction(
            action_type=ActionType.CLOSE_POSITION,
            priority=Priority.CRITICAL,
            risk_level=RiskLevel.VERY_HIGH,
            description="Accept assignment and manage position",
            rationale="Assignment unavoidable, focus on recovery",
            parameters={"assignment_management": True, "recovery_strategy": "covered_calls"},
            expected_outcome="Minimize losses and plan recovery",
            success_probability=0.60,
            estimated_return=0.02,
            max_risk=0.20,
            time_horizon="Long-term",
            prerequisites=["Sufficient capital", "Long-term outlook"],
            alternatives=["Immediate sale", "Protective strategies"]
        )
        
        p_dd_action = TradingAction(
            action_type=ActionType.ENTER_PROTECTIVE,
            priority=Priority.CRITICAL,
            risk_level=RiskLevel.VERY_HIGH,
            description="Enter protective mode for deep drawdown",
            rationale="Extreme market conditions require defensive positioning",
            parameters={"protection_level": "maximum", "hedge_ratio": 0.8},
            expected_outcome="Limit further losses",
            success_probability=0.75,
            estimated_return=-0.005,
            max_risk=0.25,
            time_horizon="Until recovery",
            prerequisites=["Available hedging instruments", "Risk tolerance"],
            alternatives=["Full liquidation", "Partial hedging"]
        )
        
        # Call scenarios
        c_wap_action = TradingAction(
            action_type=ActionType.SELL_CALL,
            priority=Priority.HIGH,
            risk_level=RiskLevel.MODERATE,
            description="Sell calls to capture appreciation profit",
            rationale="Moderate appreciation allows profitable call selling",
            parameters={"delta_range": (25, 35), "dte_range": (30, 45)},
            expected_outcome="Premium plus appreciation profit",
            success_probability=0.80,
            estimated_return=0.035,
            max_risk=0.10,
            time_horizon="1-4 weeks",
            prerequisites=["Long position or cash", "Moderate volatility"],
            alternatives=["Covered calls", "Call spreads"]
        )
        
        c_wap_plus_action = TradingAction(
            action_type=ActionType.SELL_CALL,
            priority=Priority.HIGH,
            risk_level=RiskLevel.LOW,
            description="Sell calls on strong appreciation",
            rationale="Strong appreciation provides excellent call selling opportunity",
            parameters={"delta_range": (15, 25), "dte_range": (20, 35)},
            expected_outcome="High premium plus appreciation profit",
            success_probability=0.85,
            estimated_return=0.055,
            max_risk=0.08,
            time_horizon="2-5 weeks",
            prerequisites=["Long position", "High implied volatility"],
            alternatives=["Profit taking", "Protective puts"]
        )
        
        c_pno_action = TradingAction(
            action_type=ActionType.SELL_CALL,
            priority=Priority.MEDIUM,
            risk_level=RiskLevel.LOW,
            description="Sell calls for premium only",
            rationale="Flat market allows premium collection",
            parameters={"delta_range": (20, 30), "dte_range": (30, 45)},
            expected_outcome="Premium collection",
            success_probability=0.75,
            estimated_return=0.02,
            max_risk=0.05,
            time_horizon="1-4 weeks",
            prerequisites=["Neutral outlook", "Stable volatility"],
            alternatives=["Iron condors", "Strangles"]
        )
        
        c_ro_action = TradingAction(
            action_type=ActionType.ROLL_POSITION,
            priority=Priority.HIGH,
            risk_level=RiskLevel.MODERATE,
            description="Roll call positions to maintain exposure",
            rationale="Market decline requires call position adjustment",
            parameters={"roll_delta": -5, "extend_dte": 30},
            expected_outcome="Maintain position and collect premium",
            success_probability=0.70,
            estimated_return=0.012,
            max_risk=0.08,
            time_horizon="Immediate",
            prerequisites=["Liquid options", "Continued bullish outlook"],
            alternatives=["Close position", "Convert to spread"]
        )
        
        c_rec_action = TradingAction(
            action_type=ActionType.DECREASE_POSITION,
            priority=Priority.HIGH,
            risk_level=RiskLevel.HIGH,
            description="Reduce exposure in recovery mode",
            rationale="Significant decline requires risk reduction",
            parameters={"reduction_percentage": 0.5, "defensive_positioning": True},
            expected_outcome="Reduced risk exposure",
            success_probability=0.65,
            estimated_return=0.008,
            max_risk=0.15,
            time_horizon="Until recovery",
            prerequisites=["Risk management priority", "Capital preservation"],
            alternatives=["Full exit", "Hedging strategies"]
        )
        
        w_idl_action = TradingAction(
            action_type=ActionType.WAIT_SIGNAL,
            priority=Priority.LOW,
            risk_level=RiskLevel.VERY_LOW,
            description="Wait for market signals",
            rationale="Unclear market conditions suggest patience",
            parameters={"monitoring_frequency": "daily", "signal_threshold": 0.02},
            expected_outcome="Preserve capital and wait for opportunity",
            success_probability=0.90,
            estimated_return=0.0,
            max_risk=0.01,
            time_horizon="Until signal",
            prerequisites=["Patience", "Market monitoring"],
            alternatives=["Small test positions", "Paper trading"]
        )
        
        # Add actions to tree
        root.add_child("P-EW", DecisionNode("", p_ew_action))
        root.add_child("P-AWL", DecisionNode("", p_awl_action))
        root.add_child("P-RO", DecisionNode("", p_ro_action))
        root.add_child("P-AOL", DecisionNode("", p_aol_action))
        root.add_child("P-DD", DecisionNode("", p_dd_action))
        root.add_child("C-WAP", DecisionNode("", c_wap_action))
        root.add_child("C-WAP+", DecisionNode("", c_wap_plus_action))
        root.add_child("C-PNO", DecisionNode("", c_pno_action))
        root.add_child("C-RO", DecisionNode("", c_ro_action))
        root.add_child("C-REC", DecisionNode("", c_rec_action))
        root.add_child("W-IDL", DecisionNode("", w_idl_action))
        
        return root
    
    def _build_market_condition_tree(self) -> DecisionNode:
        """Build decision tree for market condition adjustments"""
        root = DecisionNode("market_condition")
        
        # Market condition adjustments would be added here
        # For now, return basic structure
        return root
    
    def _build_risk_management_tree(self) -> DecisionNode:
        """Build decision tree for risk management actions"""
        root = DecisionNode("risk_level")
        
        # Risk management actions would be added here
        # For now, return basic structure
        return root
    
    def _build_portfolio_optimization_tree(self) -> DecisionNode:
        """Build decision tree for portfolio optimization actions"""
        root = DecisionNode("portfolio_state")
        
        # Portfolio optimization actions would be added here
        # For now, return basic structure
        return root
    
    def _initialize_action_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize action templates for quick reference"""
        return {
            "sell_put": {
                "default_delta": 20,
                "default_dte": 35,
                "risk_multiplier": 1.0
            },
            "sell_call": {
                "default_delta": 25,
                "default_dte": 35,
                "risk_multiplier": 1.2
            },
            "roll_position": {
                "default_extension": 30,
                "default_delta_adjustment": 5,
                "risk_multiplier": 1.5
            }
        }
    
    def _get_primary_action(self, week_classification: Dict[str, Any], 
                          market_analysis: Dict[str, Any], account_type: str) -> TradingAction:
        """Get primary action based on week classification"""
        if not week_classification:
            return self._get_default_action()
        
        week_type = week_classification.get('week_type', 'W-IDL')
        
        # Use decision tree to get action
        context = {
            'week_type': week_type,
            'market_condition': market_analysis.get('condition', 'neutral') if market_analysis else 'neutral',
            'account_type': account_type
        }
        
        action = self.week_classification_tree.evaluate(context)
        
        if action:
            # Adjust action based on account type
            action = self._adjust_action_for_account_type(action, account_type)
            return action
        
        return self._get_default_action()
    
    def _get_secondary_actions(self, context: Dict[str, Any]) -> List[TradingAction]:
        """Get secondary supporting actions"""
        secondary_actions = []
        
        # Portfolio rebalancing action
        rebalance_action = TradingAction(
            action_type=ActionType.REBALANCE_PORTFOLIO,
            priority=Priority.MEDIUM,
            risk_level=RiskLevel.LOW,
            description="Rebalance portfolio allocations",
            rationale="Maintain optimal portfolio balance",
            parameters={"rebalance_threshold": 0.05},
            expected_outcome="Improved risk-adjusted returns",
            success_probability=0.75,
            estimated_return=0.005,
            max_risk=0.02,
            time_horizon="Weekly",
            prerequisites=["Portfolio analysis", "Transaction costs acceptable"],
            alternatives=["Maintain current allocation"]
        )
        
        secondary_actions.append(rebalance_action)
        
        return secondary_actions
    
    def _get_risk_management_actions(self, context: Dict[str, Any]) -> List[TradingAction]:
        """Get risk management actions"""
        risk_actions = []
        
        # Stop loss action
        stop_loss_action = TradingAction(
            action_type=ActionType.CLOSE_POSITION,
            priority=Priority.HIGH,
            risk_level=RiskLevel.MODERATE,
            description="Implement stop-loss protection",
            rationale="Limit downside risk",
            parameters={"stop_loss_level": 0.08},
            expected_outcome="Limited maximum loss",
            success_probability=0.85,
            estimated_return=-0.08,
            max_risk=0.08,
            time_horizon="Immediate when triggered",
            prerequisites=["Position monitoring", "Execution capability"],
            alternatives=["Hedging", "Position sizing reduction"]
        )
        
        risk_actions.append(stop_loss_action)
        
        return risk_actions
    
    def _get_contingency_actions(self, context: Dict[str, Any]) -> List[TradingAction]:
        """Get contingency actions for unexpected scenarios"""
        contingency_actions = []
        
        # Emergency exit action
        emergency_exit = TradingAction(
            action_type=ActionType.CLOSE_POSITION,
            priority=Priority.CRITICAL,
            risk_level=RiskLevel.HIGH,
            description="Emergency portfolio liquidation",
            rationale="Extreme market conditions require immediate exit",
            parameters={"liquidation_speed": "immediate"},
            expected_outcome="Capital preservation",
            success_probability=0.90,
            estimated_return=-0.05,
            max_risk=0.10,
            time_horizon="Immediate",
            prerequisites=["Market liquidity", "Execution capability"],
            alternatives=["Partial liquidation", "Hedging"]
        )
        
        contingency_actions.append(emergency_exit)
        
        return contingency_actions
    
    def _determine_overall_strategy(self, week_classification: Dict[str, Any], 
                                  market_analysis: Dict[str, Any], 
                                  primary_action: TradingAction) -> str:
        """Determine overall strategy description"""
        if not week_classification:
            return "Conservative approach with market monitoring"
        
        week_type = week_classification.get('week_type', 'W-IDL')
        action_type = primary_action.action_type.value
        
        strategy_map = {
            'P-EW': "Premium collection strategy with put selling",
            'P-AWL': "Controlled assignment strategy with put selling",
            'P-RO': "Defensive rolling strategy to avoid assignment",
            'P-AOL': "Assignment management and recovery strategy",
            'P-DD': "Capital preservation and protective strategy",
            'C-WAP': "Appreciation capture with call selling",
            'C-WAP+': "Strong appreciation monetization strategy",
            'C-PNO': "Premium collection with call selling",
            'C-RO': "Position maintenance with rolling strategy",
            'C-REC': "Risk reduction and recovery strategy",
            'W-IDL': "Market monitoring and opportunity identification"
        }
        
        return strategy_map.get(week_type, "Adaptive strategy based on market conditions")
    
    def _calculate_action_confidence(self, context: Dict[str, Any], 
                                   primary_action: TradingAction) -> float:
        """Calculate confidence in the action plan"""
        base_confidence = 0.7
        
        # Adjust based on market analysis confidence
        market_analysis = context.get('market_analysis', {})
        if market_analysis and 'confidence' in market_analysis:
            market_confidence = market_analysis['confidence']
            base_confidence = (base_confidence + market_confidence) / 2
        
        # Adjust based on week classification confidence
        week_classification = context.get('week_classification', {})
        if week_classification and 'confidence' in week_classification:
            week_confidence = week_classification['confidence']
            base_confidence = (base_confidence + week_confidence) / 2
        
        # Adjust based on action success probability
        action_confidence = primary_action.success_probability
        final_confidence = (base_confidence + action_confidence) / 2
        
        return min(0.95, final_confidence)
    
    def _estimate_portfolio_impact(self, primary_action: TradingAction, 
                                 secondary_actions: List[TradingAction],
                                 portfolio_state: Dict[str, Any]) -> Dict[str, float]:
        """Estimate impact on portfolio metrics"""
        return {
            'expected_return': primary_action.estimated_return,
            'max_risk': primary_action.max_risk,
            'sharpe_impact': 0.1,
            'volatility_impact': 0.02,
            'correlation_impact': 0.05
        }
    
    def _create_action_timeline(self, primary_action: TradingAction, 
                              secondary_actions: List[TradingAction]) -> Dict[str, str]:
        """Create timeline for action execution"""
        return {
            'immediate': primary_action.description,
            'daily': "Monitor positions and market conditions",
            'weekly': "Review and adjust strategy as needed",
            'monthly': "Comprehensive portfolio review"
        }
    
    def _define_monitoring_points(self, context: Dict[str, Any], 
                                primary_action: TradingAction) -> List[str]:
        """Define key monitoring points"""
        return [
            "Market condition changes",
            "Position P&L thresholds",
            "Volatility regime shifts",
            "Time decay progression",
            "Assignment risk levels"
        ]
    
    def _define_exit_criteria(self, context: Dict[str, Any], 
                            primary_action: TradingAction) -> List[str]:
        """Define exit criteria for positions"""
        return [
            f"Profit target: {primary_action.estimated_return:.1%}",
            f"Stop loss: {primary_action.max_risk:.1%}",
            "Time decay: 50% of maximum profit",
            "Market regime change",
            "Volatility expansion beyond thresholds"
        ]
    
    def _adjust_action_for_account_type(self, action: TradingAction, account_type: str) -> TradingAction:
        """Adjust action parameters based on account type"""
        # Account type adjustments
        risk_multipliers = {
            'GEN_ACC': 1.0,
            'REV_ACC': 0.8,  # More conservative
            'COM_ACC': 1.2   # More aggressive
        }
        
        multiplier = risk_multipliers.get(account_type, 1.0)
        
        # Adjust risk parameters
        action.max_risk *= multiplier
        action.estimated_return *= multiplier
        
        return action
    
    def _get_default_action(self) -> TradingAction:
        """Get default action when no specific action is determined"""
        return TradingAction(
            action_type=ActionType.WAIT_SIGNAL,
            priority=Priority.LOW,
            risk_level=RiskLevel.VERY_LOW,
            description="Monitor market conditions",
            rationale="Insufficient information for specific action",
            parameters={"monitoring_frequency": "daily"},
            expected_outcome="Preserve capital and identify opportunities",
            success_probability=0.85,
            estimated_return=0.0,
            max_risk=0.01,
            time_horizon="Until clear signal",
            prerequisites=["Market monitoring capability"],
            alternatives=["Small test positions"]
        )

def test_action_recommendation_system():
    """Test the action recommendation system"""
    print("Testing Action Recommendation System...")
    
    system = ActionRecommendationSystem()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'P-EW Scenario',
            'context': {
                'week_classification': {
                    'week_type': 'P-EW',
                    'confidence': 0.85,
                    'expected_return': 0.02
                },
                'market_analysis': {
                    'condition': 'bullish',
                    'confidence': 0.80,
                    'risk_level': 'low'
                },
                'portfolio_state': {
                    'total_value': 100000,
                    'cash_percentage': 0.2
                },
                'account_type': 'GEN_ACC'
            }
        },
        {
            'name': 'P-DD Scenario',
            'context': {
                'week_classification': {
                    'week_type': 'P-DD',
                    'confidence': 0.90,
                    'expected_return': -0.005
                },
                'market_analysis': {
                    'condition': 'extremely_bearish',
                    'confidence': 0.85,
                    'risk_level': 'extreme'
                },
                'portfolio_state': {
                    'total_value': 95000,
                    'cash_percentage': 0.1
                },
                'account_type': 'REV_ACC'
            }
        },
        {
            'name': 'C-WAP+ Scenario',
            'context': {
                'week_classification': {
                    'week_type': 'C-WAP+',
                    'confidence': 0.88,
                    'expected_return': 0.055
                },
                'market_analysis': {
                    'condition': 'extremely_bullish',
                    'confidence': 0.90,
                    'risk_level': 'low'
                },
                'portfolio_state': {
                    'total_value': 110000,
                    'cash_percentage': 0.15
                },
                'account_type': 'COM_ACC'
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        action_plan = system.generate_action_plan(scenario['context'])
        
        print(f"Primary Action: {action_plan.primary_action.action_type.value}")
        print(f"Priority: {action_plan.primary_action.priority.value}")
        print(f"Risk Level: {action_plan.primary_action.risk_level.value}")
        print(f"Description: {action_plan.primary_action.description}")
        print(f"Expected Return: {action_plan.primary_action.estimated_return:.1%}")
        print(f"Max Risk: {action_plan.primary_action.max_risk:.1%}")
        print(f"Success Probability: {action_plan.primary_action.success_probability:.1%}")
        print(f"Overall Strategy: {action_plan.overall_strategy}")
        print(f"Confidence Score: {action_plan.confidence_score:.1%}")
        print(f"Secondary Actions: {len(action_plan.secondary_actions)}")
        print(f"Risk Management Actions: {len(action_plan.risk_management_actions)}")
        print(f"Contingency Actions: {len(action_plan.contingency_actions)}")
    
    print("\n✅ Action Recommendation System test completed successfully!")

if __name__ == "__main__":
    test_action_recommendation_system()

