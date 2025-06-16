"""
ALL-USE Week Classification System

Core week classification engine implementing the 11 week types that drive
the ALL-USE trading strategy. This system classifies each trading week
based on market conditions, stock movement, and trading outcomes.

Week Types:
- Put Scenarios: P-EW, P-AWL, P-RO, P-AOL, P-DD
- Call Scenarios: C-WAP, C-WAP+, C-PNO, C-RO, C-REC  
- Special: W-IDL
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from protocol_engine.all_use_parameters import ALLUSEParameters

logger = logging.getLogger('week_classifier')


class WeekType(Enum):
    """Enumeration of all 11 week types in the ALL-USE system."""
    
    # Put Scenarios
    P_EW = "P-EW"    # Puts, Expired Worthless
    P_AWL = "P-AWL"  # Puts, Assigned Within Limit
    P_RO = "P-RO"    # Puts, Roll Over
    P_AOL = "P-AOL"  # Puts, Assigned Over Limit
    P_DD = "P-DD"    # Puts, Deep Drawdown
    
    # Call Scenarios
    C_WAP = "C-WAP"   # Calls, With Appreciation Profit
    C_WAP_PLUS = "C-WAP+"  # Calls, With Strong Appreciation Profit
    C_PNO = "C-PNO"   # Calls, Premium-Only
    C_RO = "C-RO"     # Calls, Roll Over
    C_REC = "C-REC"   # Calls, Recovery Mode
    
    # Special Scenarios
    W_IDL = "W-IDL"   # Week, Idle


class MarketMovement(Enum):
    """Market movement categories for week classification."""
    
    STRONG_UP = "strong_up"        # >10% up
    MODERATE_UP = "moderate_up"    # 5-10% up
    SLIGHT_UP = "slight_up"        # 0-5% up
    FLAT = "flat"                  # -1% to +1%
    SLIGHT_DOWN = "slight_down"    # 0-5% down
    MODERATE_DOWN = "moderate_down"  # 5-10% down
    STRONG_DOWN = "strong_down"    # 10-15% down
    EXTREME_DOWN = "extreme_down"  # >15% down


class TradingPosition(Enum):
    """Current trading position types."""
    
    CASH = "cash"                  # No active positions
    SHORT_PUT = "short_put"        # Short put position
    LONG_STOCK = "long_stock"      # Long stock position (from assignment)
    SHORT_CALL = "short_call"      # Short call position (covered call)


@dataclass
class WeekTypeDefinition:
    """Definition of a week type with all its characteristics."""
    
    week_type: WeekType
    name: str
    description: str
    frequency_annual: int
    frequency_percentage: float
    expected_return_min: float
    expected_return_max: float
    trigger_conditions: Dict[str, Any]
    actions: List[str]
    market_movement: List[MarketMovement]
    position_requirements: List[TradingPosition]


@dataclass
class MarketCondition:
    """Current market condition for week classification."""
    
    symbol: str
    current_price: float
    previous_close: float
    week_start_price: float
    movement_percentage: float
    movement_category: MarketMovement
    volatility: float
    volume_ratio: float
    timestamp: datetime


@dataclass
class WeekClassification:
    """Complete week classification result."""
    
    week_type: WeekType
    confidence: float
    reasoning: str
    market_condition: MarketCondition
    current_position: TradingPosition
    recommended_actions: List[str]
    expected_return: Tuple[float, float]
    risk_level: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class WeekClassifier:
    """
    Core week classification engine for the ALL-USE system.
    
    Classifies trading weeks into one of 11 distinct scenarios based on
    market conditions, stock movement, and current trading positions.
    """
    
    def __init__(self):
        self.parameters = ALLUSEParameters()
        self.week_definitions = self._initialize_week_definitions()
        self.classification_history = []
        
        logger.info("Week Classifier initialized with 11 week types")
    
    def _initialize_week_definitions(self) -> Dict[WeekType, WeekTypeDefinition]:
        """Initialize all week type definitions with their characteristics."""
        
        definitions = {}
        
        # P-EW: Puts, Expired Worthless
        definitions[WeekType.P_EW] = WeekTypeDefinition(
            week_type=WeekType.P_EW,
            name="Puts, Expired Worthless",
            description="Sell 1DTE Put → expires worthless",
            frequency_annual=16,
            frequency_percentage=31.0,
            expected_return_min=1.8,
            expected_return_max=2.2,
            trigger_conditions={
                "market_movement": ["slight_up", "flat"],
                "position": ["short_put"],
                "expiration": "1DTE",
                "outcome": "expired_worthless"
            },
            actions=["Collect premium", "Repeat put selling"],
            market_movement=[MarketMovement.SLIGHT_UP, MarketMovement.FLAT],
            position_requirements=[TradingPosition.SHORT_PUT]
        )
        
        # P-AWL: Puts, Assigned Within Limit
        definitions[WeekType.P_AWL] = WeekTypeDefinition(
            week_type=WeekType.P_AWL,
            name="Puts, Assigned Within Limit",
            description="Put assigned at strike",
            frequency_annual=0,  # Combined with call weeks
            frequency_percentage=0.0,
            expected_return_min=1.8,
            expected_return_max=2.2,
            trigger_conditions={
                "market_movement": ["slight_down"],
                "position": ["short_put"],
                "downside_percentage": [0, 5],
                "outcome": "assigned"
            },
            actions=["Accept assignment", "Prepare to sell calls"],
            market_movement=[MarketMovement.SLIGHT_DOWN],
            position_requirements=[TradingPosition.SHORT_PUT]
        )
        
        # P-RO: Puts, Roll Over
        definitions[WeekType.P_RO] = WeekTypeDefinition(
            week_type=WeekType.P_RO,
            name="Puts, Roll Over",
            description="Put at risk of assignment",
            frequency_annual=7,  # 6-8 weeks average
            frequency_percentage=13.5,
            expected_return_min=1.0,
            expected_return_max=1.5,
            trigger_conditions={
                "market_movement": ["moderate_down"],
                "position": ["short_put"],
                "downside_percentage": [5, 15],
                "action": "roll_out"
            },
            actions=["Roll put out (and possibly down)", "Avoid assignment"],
            market_movement=[MarketMovement.MODERATE_DOWN],
            position_requirements=[TradingPosition.SHORT_PUT]
        )
        
        # P-AOL: Puts, Assigned Over Limit
        definitions[WeekType.P_AOL] = WeekTypeDefinition(
            week_type=WeekType.P_AOL,
            name="Puts, Assigned Over Limit",
            description="Put assigned despite exceeding threshold",
            frequency_annual=0,  # Combined with call weeks
            frequency_percentage=0.0,
            expected_return_min=1.8,
            expected_return_max=2.2,
            trigger_conditions={
                "market_movement": ["moderate_down"],
                "position": ["short_put"],
                "downside_percentage": [5, 15],
                "outcome": "assigned",
                "rolling_failed": True
            },
            actions=["Accept assignment", "Rolling wasn't possible"],
            market_movement=[MarketMovement.MODERATE_DOWN],
            position_requirements=[TradingPosition.SHORT_PUT]
        )
        
        # P-DD: Puts, Deep Drawdown
        definitions[WeekType.P_DD] = WeekTypeDefinition(
            week_type=WeekType.P_DD,
            name="Puts, Deep Drawdown",
            description="Extreme market conditions",
            frequency_annual=2,
            frequency_percentage=4.0,
            expected_return_min=-0.5,
            expected_return_max=0.0,
            trigger_conditions={
                "market_movement": ["extreme_down"],
                "position": ["short_put", "long_stock"],
                "downside_percentage": [15, 100],
                "protocol": "GBR-4"
            },
            actions=["Implement GBR-4 protocol", "Capital preservation"],
            market_movement=[MarketMovement.EXTREME_DOWN],
            position_requirements=[TradingPosition.SHORT_PUT, TradingPosition.LONG_STOCK]
        )
        
        # C-WAP: Calls, With Appreciation Profit
        definitions[WeekType.C_WAP] = WeekTypeDefinition(
            week_type=WeekType.C_WAP,
            name="Calls, With Appreciation Profit",
            description="Assigned stock → ATM call sold",
            frequency_annual=14,
            frequency_percentage=27.0,
            expected_return_min=3.0,
            expected_return_max=4.0,
            trigger_conditions={
                "market_movement": ["slight_up"],
                "position": ["long_stock", "short_call"],
                "upside_percentage": [0, 5],
                "outcome": "call_assigned"
            },
            actions=["Accept call assignment", "Profit realization"],
            market_movement=[MarketMovement.SLIGHT_UP],
            position_requirements=[TradingPosition.LONG_STOCK, TradingPosition.SHORT_CALL]
        )
        
        # C-WAP+: Calls, With Strong Appreciation Profit
        definitions[WeekType.C_WAP_PLUS] = WeekTypeDefinition(
            week_type=WeekType.C_WAP_PLUS,
            name="Calls, With Strong Appreciation Profit",
            description="Assigned stock → ATM call sold",
            frequency_annual=6,
            frequency_percentage=11.0,
            expected_return_min=5.0,
            expected_return_max=6.0,
            trigger_conditions={
                "market_movement": ["moderate_up"],
                "position": ["long_stock", "short_call"],
                "upside_percentage": [5, 10],
                "outcome": "call_assigned"
            },
            actions=["Accept call assignment", "Greater profit realization"],
            market_movement=[MarketMovement.MODERATE_UP],
            position_requirements=[TradingPosition.LONG_STOCK, TradingPosition.SHORT_CALL]
        )
        
        # C-PNO: Calls, Premium-Only
        definitions[WeekType.C_PNO] = WeekTypeDefinition(
            week_type=WeekType.C_PNO,
            name="Calls, Premium-Only",
            description="Assigned stock → ATM call sold",
            frequency_annual=8,
            frequency_percentage=15.0,
            expected_return_min=1.8,
            expected_return_max=2.2,
            trigger_conditions={
                "market_movement": ["flat", "slight_down"],
                "position": ["long_stock", "short_call"],
                "movement_percentage": [-5, 5],
                "outcome": "premium_collected"
            },
            actions=["Collect premium", "Cost basis reduction"],
            market_movement=[MarketMovement.FLAT, MarketMovement.SLIGHT_DOWN],
            position_requirements=[TradingPosition.LONG_STOCK, TradingPosition.SHORT_CALL]
        )
        
        # C-RO: Calls, Roll Over
        definitions[WeekType.C_RO] = WeekTypeDefinition(
            week_type=WeekType.C_RO,
            name="Calls, Roll Over",
            description="Assigned stock → ATM call sold",
            frequency_annual=4,
            frequency_percentage=8.0,
            expected_return_min=0.8,
            expected_return_max=1.2,
            trigger_conditions={
                "market_movement": ["moderate_down"],
                "position": ["long_stock", "short_call"],
                "downside_percentage": [5, 10],
                "action": "roll_down"
            },
            actions=["Roll call down", "Match current price"],
            market_movement=[MarketMovement.MODERATE_DOWN],
            position_requirements=[TradingPosition.LONG_STOCK, TradingPosition.SHORT_CALL]
        )
        
        # C-REC: Calls, Recovery Mode
        definitions[WeekType.C_REC] = WeekTypeDefinition(
            week_type=WeekType.C_REC,
            name="Calls, Recovery Mode",
            description="Assigned stock → 20-30 delta call sold",
            frequency_annual=2,
            frequency_percentage=4.0,
            expected_return_min=0.5,
            expected_return_max=0.8,
            trigger_conditions={
                "market_movement": ["strong_down"],
                "position": ["long_stock"],
                "downside_percentage": [10, 15],
                "delta_range": [20, 30]
            },
            actions=["Sell lower delta calls", "Recovery room"],
            market_movement=[MarketMovement.STRONG_DOWN],
            position_requirements=[TradingPosition.LONG_STOCK]
        )
        
        # W-IDL: Week, Idle
        definitions[WeekType.W_IDL] = WeekTypeDefinition(
            week_type=WeekType.W_IDL,
            name="Week, Idle",
            description="No active trades",
            frequency_annual=1,  # 0-2 weeks average
            frequency_percentage=2.0,
            expected_return_min=0.0,
            expected_return_max=0.0,
            trigger_conditions={
                "market_movement": ["any"],
                "position": ["cash"],
                "reason": "strategic_pause"
            },
            actions=["Strategic pause", "Market condition assessment"],
            market_movement=[MarketMovement.FLAT],
            position_requirements=[TradingPosition.CASH]
        )
        
        return definitions
    
    def classify_week(
        self,
        market_condition: MarketCondition,
        current_position: TradingPosition,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> WeekClassification:
        """
        Classify the current week based on market conditions and position.
        
        Args:
            market_condition: Current market condition data
            current_position: Current trading position
            additional_context: Additional context for classification
            
        Returns:
            WeekClassification: Complete classification result
        """
        start_time = time.time()
        
        try:
            logger.info(f"Classifying week for {market_condition.symbol} with {current_position.value} position")
            
            # Analyze market movement
            movement_category = self._categorize_market_movement(market_condition.movement_percentage)
            market_condition.movement_category = movement_category
            
            # Find matching week types
            candidate_week_types = self._find_candidate_week_types(
                movement_category, current_position, additional_context or {}
            )
            
            # Select best week type
            best_week_type, confidence = self._select_best_week_type(
                candidate_week_types, market_condition, current_position, additional_context or {}
            )
            
            # Generate classification result
            classification = self._generate_classification_result(
                best_week_type, confidence, market_condition, current_position
            )
            
            # Store in history
            self.classification_history.append(classification)
            
            execution_time = time.time() - start_time
            logger.info(f"Week classified as {best_week_type.value} with {confidence:.1%} confidence in {execution_time*1000:.1f}ms")
            
            return classification
            
        except Exception as e:
            logger.error(f"Error in week classification: {e}")
            raise
    
    def _categorize_market_movement(self, movement_percentage: float) -> MarketMovement:
        """Categorize market movement based on percentage change."""
        
        if movement_percentage >= 10:
            return MarketMovement.STRONG_UP
        elif movement_percentage >= 5:
            return MarketMovement.MODERATE_UP
        elif movement_percentage >= 1:
            return MarketMovement.SLIGHT_UP
        elif movement_percentage >= -1:
            return MarketMovement.FLAT
        elif movement_percentage >= -5:
            return MarketMovement.SLIGHT_DOWN
        elif movement_percentage >= -10:
            return MarketMovement.MODERATE_DOWN
        elif movement_percentage >= -15:
            return MarketMovement.STRONG_DOWN
        else:
            return MarketMovement.EXTREME_DOWN
    
    def _find_candidate_week_types(
        self,
        movement_category: MarketMovement,
        current_position: TradingPosition,
        context: Dict[str, Any]
    ) -> List[WeekType]:
        """Find candidate week types based on market movement and position."""
        
        candidates = []
        
        for week_type, definition in self.week_definitions.items():
            # Check market movement compatibility
            if movement_category in definition.market_movement:
                # Check position compatibility
                if current_position in definition.position_requirements:
                    candidates.append(week_type)
        
        logger.debug(f"Found {len(candidates)} candidate week types: {[wt.value for wt in candidates]}")
        return candidates
    
    def _select_best_week_type(
        self,
        candidates: List[WeekType],
        market_condition: MarketCondition,
        current_position: TradingPosition,
        context: Dict[str, Any]
    ) -> Tuple[WeekType, float]:
        """Select the best week type from candidates with confidence score."""
        
        if not candidates:
            # Default to W-IDL if no candidates match
            return WeekType.W_IDL, 0.5
        
        if len(candidates) == 1:
            return candidates[0], 0.9
        
        # Score each candidate
        scores = {}
        for week_type in candidates:
            score = self._calculate_week_type_score(
                week_type, market_condition, current_position, context
            )
            scores[week_type] = score
        
        # Select highest scoring week type
        best_week_type = max(scores, key=scores.get)
        confidence = scores[best_week_type]
        
        logger.debug(f"Week type scores: {[(wt.value, score) for wt, score in scores.items()]}")
        
        return best_week_type, confidence
    
    def _calculate_week_type_score(
        self,
        week_type: WeekType,
        market_condition: MarketCondition,
        current_position: TradingPosition,
        context: Dict[str, Any]
    ) -> float:
        """Calculate a score for how well a week type matches current conditions."""
        
        definition = self.week_definitions[week_type]
        score = 0.0
        
        # Base score from frequency (more common scenarios get higher base score)
        score += definition.frequency_percentage / 100.0 * 0.3
        
        # Market movement alignment score
        movement_score = self._score_market_movement_alignment(
            market_condition.movement_category, definition.market_movement
        )
        score += movement_score * 0.4
        
        # Position alignment score
        position_score = 1.0 if current_position in definition.position_requirements else 0.0
        score += position_score * 0.2
        
        # Context-specific scoring
        context_score = self._score_context_alignment(week_type, context)
        score += context_score * 0.1
        
        return min(score, 1.0)
    
    def _score_market_movement_alignment(
        self,
        actual_movement: MarketMovement,
        expected_movements: List[MarketMovement]
    ) -> float:
        """Score how well actual market movement aligns with expected movements."""
        
        if actual_movement in expected_movements:
            return 1.0
        
        # Partial scoring for adjacent movement categories
        movement_order = [
            MarketMovement.EXTREME_DOWN,
            MarketMovement.STRONG_DOWN,
            MarketMovement.MODERATE_DOWN,
            MarketMovement.SLIGHT_DOWN,
            MarketMovement.FLAT,
            MarketMovement.SLIGHT_UP,
            MarketMovement.MODERATE_UP,
            MarketMovement.STRONG_UP
        ]
        
        actual_index = movement_order.index(actual_movement)
        
        min_distance = float('inf')
        for expected_movement in expected_movements:
            expected_index = movement_order.index(expected_movement)
            distance = abs(actual_index - expected_index)
            min_distance = min(min_distance, distance)
        
        # Score decreases with distance
        if min_distance == 1:
            return 0.7
        elif min_distance == 2:
            return 0.4
        elif min_distance == 3:
            return 0.2
        else:
            return 0.0
    
    def _score_context_alignment(self, week_type: WeekType, context: Dict[str, Any]) -> float:
        """Score context-specific factors for week type selection."""
        
        score = 0.0
        
        # Time to expiration factor
        if 'days_to_expiration' in context:
            dte = context['days_to_expiration']
            if week_type == WeekType.P_EW and dte <= 1:
                score += 0.5
        
        # Volatility factor
        if 'volatility' in context:
            vol = context['volatility']
            if week_type in [WeekType.P_DD, WeekType.C_REC] and vol > 0.3:
                score += 0.3
        
        # Assignment probability factor
        if 'assignment_probability' in context:
            prob = context['assignment_probability']
            if week_type in [WeekType.P_AWL, WeekType.P_AOL] and prob > 0.7:
                score += 0.4
        
        return min(score, 1.0)
    
    def _generate_classification_result(
        self,
        week_type: WeekType,
        confidence: float,
        market_condition: MarketCondition,
        current_position: TradingPosition
    ) -> WeekClassification:
        """Generate complete classification result."""
        
        definition = self.week_definitions[week_type]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(week_type, market_condition, current_position)
        
        # Determine risk level
        risk_level = self._determine_risk_level(week_type, market_condition)
        
        # Create classification
        classification = WeekClassification(
            week_type=week_type,
            confidence=confidence,
            reasoning=reasoning,
            market_condition=market_condition,
            current_position=current_position,
            recommended_actions=definition.actions.copy(),
            expected_return=(definition.expected_return_min, definition.expected_return_max),
            risk_level=risk_level,
            timestamp=datetime.now(),
            metadata={
                'definition': definition.name,
                'frequency_annual': definition.frequency_annual,
                'frequency_percentage': definition.frequency_percentage
            }
        )
        
        return classification
    
    def _generate_reasoning(
        self,
        week_type: WeekType,
        market_condition: MarketCondition,
        current_position: TradingPosition
    ) -> str:
        """Generate human-readable reasoning for the classification."""
        
        definition = self.week_definitions[week_type]
        movement_pct = market_condition.movement_percentage
        
        reasoning_parts = [
            f"Market movement: {movement_pct:+.1f}% ({market_condition.movement_category.value})",
            f"Current position: {current_position.value}",
            f"Week type: {definition.name}",
            f"Expected frequency: {definition.frequency_annual} weeks/year ({definition.frequency_percentage:.1f}%)"
        ]
        
        return " | ".join(reasoning_parts)
    
    def _determine_risk_level(self, week_type: WeekType, market_condition: MarketCondition) -> str:
        """Determine risk level for the classified week type."""
        
        if week_type == WeekType.P_DD:
            return "HIGH"
        elif week_type in [WeekType.P_RO, WeekType.C_REC]:
            return "MEDIUM"
        elif week_type == WeekType.W_IDL:
            return "LOW"
        elif market_condition.volatility > 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_week_type_statistics(self) -> Dict[str, Any]:
        """Get statistics about week type definitions and frequencies."""
        
        stats = {
            'total_week_types': len(self.week_definitions),
            'put_scenarios': len([wt for wt in self.week_definitions.keys() if wt.value.startswith('P-')]),
            'call_scenarios': len([wt for wt in self.week_definitions.keys() if wt.value.startswith('C-')]),
            'special_scenarios': len([wt for wt in self.week_definitions.keys() if wt.value.startswith('W-')]),
            'total_frequency': sum(defn.frequency_annual for defn in self.week_definitions.values()),
            'expected_annual_return_range': self._calculate_expected_annual_return(),
            'week_type_frequencies': {
                wt.value: {
                    'annual_frequency': defn.frequency_annual,
                    'percentage': defn.frequency_percentage,
                    'expected_return': (defn.expected_return_min, defn.expected_return_max)
                }
                for wt, defn in self.week_definitions.items()
            }
        }
        
        return stats
    
    def _calculate_expected_annual_return(self) -> Tuple[float, float]:
        """Calculate expected annual return range based on week type frequencies."""
        
        min_return = 0.0
        max_return = 0.0
        
        for definition in self.week_definitions.values():
            if definition.frequency_annual > 0:
                min_return += definition.frequency_annual * definition.expected_return_min
                max_return += definition.frequency_annual * definition.expected_return_max
        
        return (min_return, max_return)
    
    def get_classification_history(self, limit: Optional[int] = None) -> List[WeekClassification]:
        """Get classification history with optional limit."""
        
        if limit:
            return self.classification_history[-limit:]
        return self.classification_history.copy()
    
    def validate_classification(
        self,
        classification: WeekClassification,
        actual_outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a classification against actual trading outcome."""
        
        validation_result = {
            'classification_correct': False,
            'return_within_range': False,
            'confidence_appropriate': False,
            'validation_score': 0.0,
            'details': {}
        }
        
        # Check if classification was correct
        if 'actual_week_type' in actual_outcome:
            validation_result['classification_correct'] = (
                classification.week_type == actual_outcome['actual_week_type']
            )
        
        # Check if return was within expected range
        if 'actual_return' in actual_outcome:
            actual_return = actual_outcome['actual_return']
            min_expected, max_expected = classification.expected_return
            validation_result['return_within_range'] = (
                min_expected <= actual_return <= max_expected
            )
        
        # Check if confidence was appropriate
        validation_result['confidence_appropriate'] = (
            0.5 <= classification.confidence <= 1.0
        )
        
        # Calculate overall validation score
        score = 0.0
        if validation_result['classification_correct']:
            score += 0.5
        if validation_result['return_within_range']:
            score += 0.3
        if validation_result['confidence_appropriate']:
            score += 0.2
        
        validation_result['validation_score'] = score
        validation_result['details'] = {
            'classified_as': classification.week_type.value,
            'confidence': classification.confidence,
            'expected_return': classification.expected_return,
            'actual_outcome': actual_outcome
        }
        
        return validation_result


if __name__ == "__main__":
    # Test the week classification system
    classifier = WeekClassifier()
    
    print("="*80)
    print("ALL-USE WEEK CLASSIFICATION SYSTEM TEST")
    print("="*80)
    
    # Display week type statistics
    stats = classifier.get_week_type_statistics()
    print(f"\nWeek Type Statistics:")
    print(f"Total Week Types: {stats['total_week_types']}")
    print(f"Put Scenarios: {stats['put_scenarios']}")
    print(f"Call Scenarios: {stats['call_scenarios']}")
    print(f"Special Scenarios: {stats['special_scenarios']}")
    print(f"Expected Annual Return: {stats['expected_annual_return_range'][0]:.1f}% - {stats['expected_annual_return_range'][1]:.1f}%")
    
    # Test classification scenarios
    test_scenarios = [
        {
            'name': 'Bullish Week - Stock Up 3%',
            'market_condition': MarketCondition(
                symbol='SPY',
                current_price=450.0,
                previous_close=437.0,
                week_start_price=437.0,
                movement_percentage=3.0,
                movement_category=MarketMovement.SLIGHT_UP,
                volatility=0.15,
                volume_ratio=1.2,
                timestamp=datetime.now()
            ),
            'position': TradingPosition.SHORT_PUT
        },
        {
            'name': 'Bearish Week - Stock Down 7%',
            'market_condition': MarketCondition(
                symbol='SPY',
                current_price=420.0,
                previous_close=450.0,
                week_start_price=450.0,
                movement_percentage=-7.0,
                movement_category=MarketMovement.MODERATE_DOWN,
                volatility=0.25,
                volume_ratio=1.8,
                timestamp=datetime.now()
            ),
            'position': TradingPosition.SHORT_PUT
        },
        {
            'name': 'Extreme Drawdown - Stock Down 18%',
            'market_condition': MarketCondition(
                symbol='SPY',
                current_price=370.0,
                previous_close=450.0,
                week_start_price=450.0,
                movement_percentage=-18.0,
                movement_category=MarketMovement.EXTREME_DOWN,
                volatility=0.45,
                volume_ratio=3.2,
                timestamp=datetime.now()
            ),
            'position': TradingPosition.LONG_STOCK
        },
        {
            'name': 'Strong Rally - Stock Up 8%',
            'market_condition': MarketCondition(
                symbol='SPY',
                current_price=485.0,
                previous_close=450.0,
                week_start_price=450.0,
                movement_percentage=8.0,
                movement_category=MarketMovement.MODERATE_UP,
                volatility=0.20,
                volume_ratio=1.5,
                timestamp=datetime.now()
            ),
            'position': TradingPosition.SHORT_CALL
        }
    ]
    
    print(f"\nTesting {len(test_scenarios)} classification scenarios:")
    print("-" * 80)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        
        classification = classifier.classify_week(
            scenario['market_condition'],
            scenario['position']
        )
        
        print(f"   Week Type: {classification.week_type.value}")
        print(f"   Confidence: {classification.confidence:.1%}")
        print(f"   Expected Return: {classification.expected_return[0]:.1f}% - {classification.expected_return[1]:.1f}%")
        print(f"   Risk Level: {classification.risk_level}")
        print(f"   Actions: {', '.join(classification.recommended_actions)}")
        print(f"   Reasoning: {classification.reasoning}")
    
    print(f"\nClassification History: {len(classifier.get_classification_history())} classifications stored")
    
    print("\n" + "="*80)
    print("Week Classification System test completed successfully!")
    print("="*80)

