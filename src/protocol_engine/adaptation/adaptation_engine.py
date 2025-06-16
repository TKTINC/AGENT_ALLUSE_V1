"""
Real-time Protocol Adaptation and Learning Engine for ALL-USE Protocol
Implements dynamic protocol adaptation based on live market conditions and continuous learning

This module provides sophisticated real-time adaptation capabilities including:
- Dynamic protocol adjustment based on market regime changes
- Continuous learning from trade outcomes and market events
- Performance feedback loops for real-time optimization
- Adaptive parameter tuning based on current conditions
- Market regime detection and protocol switching
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS_MODE = "crisis_mode"
    RECOVERY_MODE = "recovery_mode"

class AdaptationTrigger(Enum):
    """Triggers for protocol adaptation"""
    MARKET_REGIME_CHANGE = "market_regime_change"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    RISK_THRESHOLD_BREACH = "risk_threshold_breach"
    LEARNING_MILESTONE = "learning_milestone"
    EXTERNAL_EVENT = "external_event"

class LearningType(Enum):
    """Types of learning mechanisms"""
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    SUPERVISED_LEARNING = "supervised_learning"
    UNSUPERVISED_LEARNING = "unsupervised_learning"
    ONLINE_LEARNING = "online_learning"
    TRANSFER_LEARNING = "transfer_learning"

@dataclass
class MarketState:
    """Current market state snapshot"""
    timestamp: datetime
    spy_price: float
    spy_return_1d: float
    spy_return_5d: float
    spy_return_20d: float
    vix_level: float
    vix_change: float
    volume_ratio: float
    put_call_ratio: float
    term_structure: Dict[str, float]
    sector_rotation: Dict[str, float]
    sentiment_indicators: Dict[str, float]
    economic_indicators: Dict[str, float]
    regime: MarketRegime
    regime_confidence: float

@dataclass
class AdaptationEvent:
    """Protocol adaptation event"""
    event_id: str
    timestamp: datetime
    trigger: AdaptationTrigger
    market_state: MarketState
    current_parameters: Dict[str, Any]
    adapted_parameters: Dict[str, Any]
    adaptation_rationale: str
    confidence: float
    expected_impact: str
    monitoring_period: timedelta

@dataclass
class LearningOutcome:
    """Learning outcome from trade or market event"""
    outcome_id: str
    timestamp: datetime
    event_type: str
    market_conditions: Dict[str, Any]
    strategy_parameters: Dict[str, Any]
    actual_outcome: Dict[str, Any]
    expected_outcome: Dict[str, Any]
    performance_delta: float
    lessons_learned: List[str]
    parameter_adjustments: Dict[str, Any]

@dataclass
class PerformanceFeedback:
    """Real-time performance feedback"""
    timestamp: datetime
    period: str  # '1d', '1w', '1m'
    total_return: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    trades_count: int
    performance_trend: str  # 'improving', 'stable', 'degrading'
    benchmark_comparison: float
    confidence_level: float

class RealTimeAdaptationEngine:
    """
    Real-time Protocol Adaptation and Learning Engine
    
    Provides sophisticated real-time adaptation capabilities including:
    - Dynamic protocol adjustment based on market conditions
    - Continuous learning from outcomes and market events
    - Performance feedback loops for optimization
    - Adaptive parameter tuning and regime switching
    - Market state monitoring and regime detection
    """
    
    def __init__(self):
        """Initialize the real-time adaptation engine"""
        self.logger = logging.getLogger(__name__)
        
        # Market state tracking
        self.current_market_state: Optional[MarketState] = None
        self.market_history: deque = deque(maxlen=1000)  # Last 1000 market states
        self.regime_history: deque = deque(maxlen=100)   # Last 100 regime changes
        
        # Adaptation tracking
        self.adaptation_events: List[AdaptationEvent] = []
        self.learning_outcomes: List[LearningOutcome] = []
        self.performance_feedback: deque = deque(maxlen=100)  # Last 100 feedback cycles
        
        # Current protocol parameters
        self.current_parameters = {
            'base_delta_range': {'GEN_ACC': (40, 50), 'REV_ACC': (30, 40), 'COM_ACC': (20, 30)},
            'position_sizing': {'base_size': 10, 'volatility_adjustment': True},
            'dte_preferences': {'min_dte': 20, 'max_dte': 50, 'target_dte': 35},
            'risk_management': {'max_portfolio_risk': 0.10, 'position_risk_limit': 0.05},
            'market_filters': {'min_vix': 12, 'max_vix': 40, 'volume_threshold': 0.8}
        }
        
        # Adaptation configuration
        self.adaptation_config = {
            'regime_detection_window': 20,  # Days for regime detection
            'adaptation_threshold': 0.05,   # 5% performance degradation triggers adaptation
            'learning_rate': 0.1,           # Learning rate for parameter updates
            'confidence_threshold': 0.7,    # Minimum confidence for adaptations
            'monitoring_period': timedelta(days=7),  # Period to monitor adaptations
            'max_adaptations_per_day': 3    # Limit adaptations to prevent over-fitting
        }
        
        # Learning mechanisms
        self.learning_models = {
            'parameter_optimizer': None,
            'regime_detector': None,
            'performance_predictor': None,
            'risk_assessor': None
        }
        
        # Performance tracking
        self.performance_metrics = {
            'daily_returns': deque(maxlen=252),    # 1 year of daily returns
            'weekly_returns': deque(maxlen=52),    # 1 year of weekly returns
            'monthly_returns': deque(maxlen=12),   # 1 year of monthly returns
            'trade_outcomes': deque(maxlen=1000),  # Last 1000 trades
            'regime_performance': {}               # Performance by regime
        }
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.adaptation_callbacks: List[Callable] = []
        
        # Initialize learning models
        self._initialize_learning_models()
        
        self.logger.info("Real-time Adaptation Engine initialized")
    
    def start_real_time_monitoring(self):
        """Start real-time market monitoring and adaptation"""
        if self.monitoring_active:
            self.logger.warning("Real-time monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Real-time monitoring started")
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Real-time monitoring stopped")
    
    def update_market_state(self, market_data: Dict[str, Any]) -> MarketState:
        """
        Update current market state and detect regime changes
        
        Args:
            market_data: Current market data
            
        Returns:
            Updated market state
        """
        try:
            # Create market state
            market_state = MarketState(
                timestamp=datetime.now(),
                spy_price=market_data.get('spy_price', 400.0),
                spy_return_1d=market_data.get('spy_return_1d', 0.0),
                spy_return_5d=market_data.get('spy_return_5d', 0.0),
                spy_return_20d=market_data.get('spy_return_20d', 0.0),
                vix_level=market_data.get('vix', 20.0),
                vix_change=market_data.get('vix_change', 0.0),
                volume_ratio=market_data.get('volume_ratio', 1.0),
                put_call_ratio=market_data.get('put_call_ratio', 1.0),
                term_structure=market_data.get('term_structure', {}),
                sector_rotation=market_data.get('sector_rotation', {}),
                sentiment_indicators=market_data.get('sentiment_indicators', {}),
                economic_indicators=market_data.get('economic_indicators', {}),
                regime=self._detect_market_regime(market_data),
                regime_confidence=0.0
            )
            
            # Calculate regime confidence
            market_state.regime_confidence = self._calculate_regime_confidence(market_state)
            
            # Update history
            self.market_history.append(market_state)
            
            # Check for regime change
            if self.current_market_state and market_state.regime != self.current_market_state.regime:
                self._handle_regime_change(self.current_market_state.regime, market_state.regime, market_state)
            
            # Update current state
            self.current_market_state = market_state
            
            # Check for adaptation triggers
            self._check_adaptation_triggers(market_state)
            
            return market_state
            
        except Exception as e:
            self.logger.error(f"Error updating market state: {str(e)}")
            raise
    
    def adapt_protocol(self, trigger: AdaptationTrigger, market_state: MarketState,
                      adaptation_context: Optional[Dict[str, Any]] = None) -> AdaptationEvent:
        """
        Adapt protocol parameters based on trigger and market conditions
        
        Args:
            trigger: What triggered the adaptation
            market_state: Current market state
            adaptation_context: Additional context for adaptation
            
        Returns:
            Adaptation event details
        """
        try:
            self.logger.info(f"Adapting protocol due to: {trigger.value}")
            
            # Generate adapted parameters
            adapted_parameters = self._generate_adapted_parameters(trigger, market_state, adaptation_context)
            
            # Calculate adaptation confidence
            confidence = self._calculate_adaptation_confidence(trigger, market_state, adapted_parameters)
            
            # Create adaptation event
            adaptation_event = AdaptationEvent(
                event_id=f"adapt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                trigger=trigger,
                market_state=market_state,
                current_parameters=self.current_parameters.copy(),
                adapted_parameters=adapted_parameters,
                adaptation_rationale=self._generate_adaptation_rationale(trigger, market_state, adapted_parameters),
                confidence=confidence,
                expected_impact=self._estimate_adaptation_impact(adapted_parameters),
                monitoring_period=self.adaptation_config['monitoring_period']
            )
            
            # Apply adaptation if confidence is sufficient
            if confidence >= self.adaptation_config['confidence_threshold']:
                self._apply_adaptation(adaptation_event)
                self.adaptation_events.append(adaptation_event)
                
                # Notify callbacks
                for callback in self.adaptation_callbacks:
                    try:
                        callback(adaptation_event)
                    except Exception as e:
                        self.logger.error(f"Error in adaptation callback: {str(e)}")
                
                self.logger.info(f"Protocol adapted with {confidence:.1%} confidence")
            else:
                self.logger.warning(f"Adaptation rejected due to low confidence: {confidence:.1%}")
            
            return adaptation_event
            
        except Exception as e:
            self.logger.error(f"Error in protocol adaptation: {str(e)}")
            raise
    
    def learn_from_outcome(self, trade_outcome: Dict[str, Any], market_conditions: Dict[str, Any]) -> LearningOutcome:
        """
        Learn from trade outcome and update models
        
        Args:
            trade_outcome: Actual trade outcome
            market_conditions: Market conditions during trade
            
        Returns:
            Learning outcome with lessons and adjustments
        """
        try:
            # Create learning outcome
            learning_outcome = LearningOutcome(
                outcome_id=f"learn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                event_type=trade_outcome.get('event_type', 'trade_completion'),
                market_conditions=market_conditions,
                strategy_parameters=self.current_parameters.copy(),
                actual_outcome=trade_outcome,
                expected_outcome=trade_outcome.get('expected_outcome', {}),
                performance_delta=self._calculate_performance_delta(trade_outcome),
                lessons_learned=[],
                parameter_adjustments={}
            )
            
            # Extract lessons learned
            learning_outcome.lessons_learned = self._extract_lessons_learned(learning_outcome)
            
            # Generate parameter adjustments
            learning_outcome.parameter_adjustments = self._generate_learning_adjustments(learning_outcome)
            
            # Update learning models
            self._update_learning_models(learning_outcome)
            
            # Apply learning adjustments if significant
            if self._should_apply_learning_adjustments(learning_outcome):
                self._apply_learning_adjustments(learning_outcome)
            
            self.learning_outcomes.append(learning_outcome)
            
            self.logger.info(f"Learning outcome processed: {len(learning_outcome.lessons_learned)} lessons learned")
            return learning_outcome
            
        except Exception as e:
            self.logger.error(f"Error in learning from outcome: {str(e)}")
            raise
    
    def update_performance_feedback(self, performance_data: Dict[str, Any]) -> PerformanceFeedback:
        """
        Update performance feedback and trigger adaptations if needed
        
        Args:
            performance_data: Current performance metrics
            
        Returns:
            Performance feedback analysis
        """
        try:
            # Create performance feedback
            feedback = PerformanceFeedback(
                timestamp=datetime.now(),
                period=performance_data.get('period', '1d'),
                total_return=performance_data.get('total_return', 0.0),
                sharpe_ratio=performance_data.get('sharpe_ratio', 0.0),
                win_rate=performance_data.get('win_rate', 0.0),
                max_drawdown=performance_data.get('max_drawdown', 0.0),
                trades_count=performance_data.get('trades_count', 0),
                performance_trend=self._analyze_performance_trend(performance_data),
                benchmark_comparison=performance_data.get('benchmark_comparison', 0.0),
                confidence_level=performance_data.get('confidence_level', 0.0)
            )
            
            # Update performance history
            self.performance_feedback.append(feedback)
            
            # Update performance metrics
            self._update_performance_metrics(performance_data)
            
            # Check for performance-based adaptations
            if feedback.performance_trend == 'degrading':
                self._trigger_performance_adaptation(feedback)
            
            self.logger.info(f"Performance feedback updated: {feedback.performance_trend} trend")
            return feedback
            
        except Exception as e:
            self.logger.error(f"Error updating performance feedback: {str(e)}")
            raise
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current protocol parameters"""
        return self.current_parameters.copy()
    
    def get_adaptation_history(self, days: int = 30) -> List[AdaptationEvent]:
        """Get adaptation history for specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [event for event in self.adaptation_events if event.timestamp >= cutoff_date]
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning outcomes"""
        if not self.learning_outcomes:
            return {'message': 'No learning outcomes available'}
        
        recent_outcomes = self.learning_outcomes[-50:]  # Last 50 outcomes
        
        # Analyze lessons learned
        all_lessons = []
        for outcome in recent_outcomes:
            all_lessons.extend(outcome.lessons_learned)
        
        # Count lesson frequency
        lesson_frequency = {}
        for lesson in all_lessons:
            lesson_frequency[lesson] = lesson_frequency.get(lesson, 0) + 1
        
        # Analyze performance improvements
        performance_deltas = [outcome.performance_delta for outcome in recent_outcomes]
        
        insights = {
            'total_learning_outcomes': len(self.learning_outcomes),
            'recent_outcomes_analyzed': len(recent_outcomes),
            'top_lessons_learned': sorted(lesson_frequency.items(), key=lambda x: x[1], reverse=True)[:5],
            'average_performance_delta': np.mean(performance_deltas) if performance_deltas else 0,
            'learning_effectiveness': self._calculate_learning_effectiveness(),
            'parameter_stability': self._analyze_parameter_stability(),
            'adaptation_frequency': len(self.adaptation_events) / max(1, len(self.learning_outcomes)),
            'recommendations': self._generate_learning_recommendations()
        }
        
        return insights
    
    def register_adaptation_callback(self, callback: Callable[[AdaptationEvent], None]):
        """Register callback for adaptation events"""
        self.adaptation_callbacks.append(callback)
    
    # Helper methods for core functionality
    def _initialize_learning_models(self):
        """Initialize learning models for adaptation"""
        # Simplified model initialization
        self.learning_models = {
            'parameter_optimizer': {'type': 'gradient_descent', 'learning_rate': 0.01},
            'regime_detector': {'type': 'clustering', 'n_clusters': 6},
            'performance_predictor': {'type': 'regression', 'window': 20},
            'risk_assessor': {'type': 'classification', 'threshold': 0.1}
        }
        
        self.logger.info("Learning models initialized")
    
    def _monitoring_loop(self):
        """Main monitoring loop for real-time adaptation"""
        while self.monitoring_active:
            try:
                # Simulate market data update (in real implementation, this would come from data feeds)
                if self.current_market_state:
                    # Check for time-based adaptations
                    self._check_time_based_adaptations()
                    
                    # Monitor adaptation effectiveness
                    self._monitor_adaptation_effectiveness()
                    
                    # Update learning models
                    self._periodic_model_updates()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _detect_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """Detect current market regime"""
        vix = market_data.get('vix', 20)
        spy_return_20d = market_data.get('spy_return_20d', 0)
        
        # Simplified regime detection logic
        if vix > 35:
            return MarketRegime.CRISIS_MODE
        elif vix > 25:
            return MarketRegime.HIGH_VOLATILITY
        elif vix < 15:
            return MarketRegime.LOW_VOLATILITY
        elif spy_return_20d > 0.05:
            return MarketRegime.BULL_MARKET
        elif spy_return_20d < -0.05:
            return MarketRegime.BEAR_MARKET
        else:
            return MarketRegime.SIDEWAYS_MARKET
    
    def _calculate_regime_confidence(self, market_state: MarketState) -> float:
        """Calculate confidence in regime classification"""
        # Simplified confidence calculation
        vix_stability = 1.0 - min(1.0, abs(market_state.vix_change) / 10.0)
        return min(0.95, 0.6 + vix_stability * 0.35)
    
    def _handle_regime_change(self, old_regime: MarketRegime, new_regime: MarketRegime, market_state: MarketState):
        """Handle market regime change"""
        self.logger.info(f"Market regime change detected: {old_regime.value} -> {new_regime.value}")
        
        # Record regime change
        self.regime_history.append({
            'timestamp': datetime.now(),
            'old_regime': old_regime,
            'new_regime': new_regime,
            'confidence': market_state.regime_confidence
        })
        
        # Trigger regime-based adaptation
        self.adapt_protocol(AdaptationTrigger.MARKET_REGIME_CHANGE, market_state, {
            'old_regime': old_regime,
            'new_regime': new_regime
        })
    
    def _check_adaptation_triggers(self, market_state: MarketState):
        """Check for various adaptation triggers"""
        # Volatility spike check
        if market_state.vix_change > 5:  # VIX spike > 5 points
            self.adapt_protocol(AdaptationTrigger.VOLATILITY_SPIKE, market_state)
        
        # Risk threshold breach check
        if hasattr(self, 'current_portfolio_risk') and self.current_portfolio_risk > 0.15:
            self.adapt_protocol(AdaptationTrigger.RISK_THRESHOLD_BREACH, market_state)
    
    def _generate_adapted_parameters(self, trigger: AdaptationTrigger, market_state: MarketState,
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate adapted parameters based on trigger and market state"""
        adapted_params = self.current_parameters.copy()
        
        # Regime-based adaptations
        if trigger == AdaptationTrigger.MARKET_REGIME_CHANGE:
            if market_state.regime == MarketRegime.HIGH_VOLATILITY:
                # Reduce position sizes in high volatility
                adapted_params['position_sizing']['base_size'] = max(5, adapted_params['position_sizing']['base_size'] * 0.8)
                adapted_params['risk_management']['max_portfolio_risk'] = 0.08
            elif market_state.regime == MarketRegime.LOW_VOLATILITY:
                # Increase position sizes in low volatility
                adapted_params['position_sizing']['base_size'] = min(15, adapted_params['position_sizing']['base_size'] * 1.2)
                adapted_params['risk_management']['max_portfolio_risk'] = 0.12
        
        # Volatility spike adaptations
        elif trigger == AdaptationTrigger.VOLATILITY_SPIKE:
            # Reduce risk exposure
            adapted_params['position_sizing']['base_size'] *= 0.7
            adapted_params['risk_management']['max_portfolio_risk'] = 0.06
            adapted_params['market_filters']['max_vix'] = 35
        
        return adapted_params
    
    def _calculate_adaptation_confidence(self, trigger: AdaptationTrigger, market_state: MarketState,
                                       adapted_parameters: Dict[str, Any]) -> float:
        """Calculate confidence in adaptation"""
        base_confidence = 0.7
        
        # Adjust based on regime confidence
        regime_adjustment = market_state.regime_confidence * 0.2
        
        # Adjust based on trigger type
        trigger_adjustments = {
            AdaptationTrigger.MARKET_REGIME_CHANGE: 0.1,
            AdaptationTrigger.VOLATILITY_SPIKE: 0.15,
            AdaptationTrigger.PERFORMANCE_DEGRADATION: 0.05,
            AdaptationTrigger.RISK_THRESHOLD_BREACH: 0.2
        }
        
        trigger_adjustment = trigger_adjustments.get(trigger, 0.0)
        
        return min(0.95, base_confidence + regime_adjustment + trigger_adjustment)
    
    def _generate_adaptation_rationale(self, trigger: AdaptationTrigger, market_state: MarketState,
                                     adapted_parameters: Dict[str, Any]) -> str:
        """Generate human-readable rationale for adaptation"""
        rationales = {
            AdaptationTrigger.MARKET_REGIME_CHANGE: f"Market regime changed to {market_state.regime.value}, adjusting parameters for new conditions",
            AdaptationTrigger.VOLATILITY_SPIKE: f"VIX spike detected ({market_state.vix_level:.1f}), reducing risk exposure",
            AdaptationTrigger.PERFORMANCE_DEGRADATION: "Performance degradation detected, optimizing parameters",
            AdaptationTrigger.RISK_THRESHOLD_BREACH: "Risk threshold breached, implementing defensive measures"
        }
        
        return rationales.get(trigger, f"Adaptation triggered by {trigger.value}")
    
    def _estimate_adaptation_impact(self, adapted_parameters: Dict[str, Any]) -> str:
        """Estimate impact of parameter adaptation"""
        # Simplified impact estimation
        current_size = self.current_parameters['position_sizing']['base_size']
        new_size = adapted_parameters['position_sizing']['base_size']
        
        size_change = (new_size - current_size) / current_size
        
        if abs(size_change) > 0.2:
            return "Significant impact expected"
        elif abs(size_change) > 0.1:
            return "Moderate impact expected"
        else:
            return "Minor impact expected"
    
    def _apply_adaptation(self, adaptation_event: AdaptationEvent):
        """Apply adaptation to current parameters"""
        self.current_parameters = adaptation_event.adapted_parameters.copy()
        self.logger.info(f"Adaptation applied: {adaptation_event.event_id}")
    
    def _calculate_performance_delta(self, trade_outcome: Dict[str, Any]) -> float:
        """Calculate performance delta from trade outcome"""
        actual_return = trade_outcome.get('actual_return', 0.0)
        expected_return = trade_outcome.get('expected_return', 0.0)
        return actual_return - expected_return
    
    def _extract_lessons_learned(self, learning_outcome: LearningOutcome) -> List[str]:
        """Extract lessons learned from outcome"""
        lessons = []
        
        if learning_outcome.performance_delta > 0.02:  # Outperformed by 2%
            lessons.append("Strategy performed better than expected in current market conditions")
        elif learning_outcome.performance_delta < -0.02:  # Underperformed by 2%
            lessons.append("Strategy underperformed, consider parameter adjustments")
        
        # Market condition specific lessons
        vix = learning_outcome.market_conditions.get('vix', 20)
        if vix > 30 and learning_outcome.performance_delta > 0:
            lessons.append("High volatility environment favorable for current strategy")
        
        return lessons
    
    def _generate_learning_adjustments(self, learning_outcome: LearningOutcome) -> Dict[str, Any]:
        """Generate parameter adjustments based on learning"""
        adjustments = {}
        
        # Adjust based on performance delta
        if learning_outcome.performance_delta < -0.05:  # Significant underperformance
            adjustments['position_sizing'] = {'base_size': -1}  # Reduce position size
        elif learning_outcome.performance_delta > 0.05:  # Significant outperformance
            adjustments['position_sizing'] = {'base_size': 1}   # Increase position size
        
        return adjustments
    
    def _update_learning_models(self, learning_outcome: LearningOutcome):
        """Update learning models with new outcome"""
        # Simplified model update
        self.logger.debug(f"Updating learning models with outcome: {learning_outcome.outcome_id}")
    
    def _should_apply_learning_adjustments(self, learning_outcome: LearningOutcome) -> bool:
        """Determine if learning adjustments should be applied"""
        return abs(learning_outcome.performance_delta) > 0.03  # 3% threshold
    
    def _apply_learning_adjustments(self, learning_outcome: LearningOutcome):
        """Apply learning-based parameter adjustments"""
        for category, adjustments in learning_outcome.parameter_adjustments.items():
            if category in self.current_parameters:
                for param, adjustment in adjustments.items():
                    if param in self.current_parameters[category]:
                        current_value = self.current_parameters[category][param]
                        if isinstance(current_value, (int, float)):
                            self.current_parameters[category][param] = max(1, current_value + adjustment)
        
        self.logger.info(f"Learning adjustments applied from outcome: {learning_outcome.outcome_id}")
    
    def _analyze_performance_trend(self, performance_data: Dict[str, Any]) -> str:
        """Analyze performance trend"""
        if len(self.performance_feedback) < 3:
            return 'stable'
        
        recent_returns = [fb.total_return for fb in list(self.performance_feedback)[-3:]]
        
        if all(recent_returns[i] > recent_returns[i-1] for i in range(1, len(recent_returns))):
            return 'improving'
        elif all(recent_returns[i] < recent_returns[i-1] for i in range(1, len(recent_returns))):
            return 'degrading'
        else:
            return 'stable'
    
    def _update_performance_metrics(self, performance_data: Dict[str, Any]):
        """Update internal performance metrics"""
        period = performance_data.get('period', '1d')
        total_return = performance_data.get('total_return', 0.0)
        
        if period == '1d':
            self.performance_metrics['daily_returns'].append(total_return)
        elif period == '1w':
            self.performance_metrics['weekly_returns'].append(total_return)
        elif period == '1m':
            self.performance_metrics['monthly_returns'].append(total_return)
    
    def _trigger_performance_adaptation(self, feedback: PerformanceFeedback):
        """Trigger adaptation based on performance degradation"""
        if self.current_market_state:
            self.adapt_protocol(AdaptationTrigger.PERFORMANCE_DEGRADATION, self.current_market_state, {
                'performance_feedback': feedback
            })
    
    def _check_time_based_adaptations(self):
        """Check for time-based adaptation needs"""
        # Check if any adaptations need monitoring updates
        current_time = datetime.now()
        
        for event in self.adaptation_events[-10:]:  # Check last 10 adaptations
            if event.timestamp + event.monitoring_period < current_time:
                self._evaluate_adaptation_effectiveness(event)
    
    def _monitor_adaptation_effectiveness(self):
        """Monitor effectiveness of recent adaptations"""
        # Simplified monitoring
        if len(self.adaptation_events) > 0:
            recent_adaptations = [e for e in self.adaptation_events if 
                                (datetime.now() - e.timestamp).days <= 7]
            
            if recent_adaptations:
                self.logger.debug(f"Monitoring {len(recent_adaptations)} recent adaptations")
    
    def _periodic_model_updates(self):
        """Perform periodic updates to learning models"""
        # Simplified periodic updates
        current_hour = datetime.now().hour
        if current_hour == 0:  # Daily update at midnight
            self.logger.debug("Performing daily model updates")
    
    def _evaluate_adaptation_effectiveness(self, adaptation_event: AdaptationEvent):
        """Evaluate effectiveness of a completed adaptation"""
        self.logger.debug(f"Evaluating adaptation effectiveness: {adaptation_event.event_id}")
    
    def _calculate_learning_effectiveness(self) -> float:
        """Calculate overall learning effectiveness"""
        if not self.learning_outcomes:
            return 0.0
        
        positive_outcomes = len([o for o in self.learning_outcomes if o.performance_delta > 0])
        return positive_outcomes / len(self.learning_outcomes)
    
    def _analyze_parameter_stability(self) -> Dict[str, Any]:
        """Analyze stability of parameter adaptations"""
        if len(self.adaptation_events) < 5:
            return {'stability': 'insufficient_data'}
        
        recent_events = self.adaptation_events[-10:]
        adaptation_frequency = len(recent_events) / 10
        
        return {
            'stability': 'stable' if adaptation_frequency < 0.3 else 'volatile',
            'adaptation_frequency': adaptation_frequency,
            'recent_adaptations': len(recent_events)
        }
    
    def _generate_learning_recommendations(self) -> List[str]:
        """Generate recommendations based on learning analysis"""
        recommendations = []
        
        if self._calculate_learning_effectiveness() < 0.6:
            recommendations.append("Consider reviewing learning parameters - effectiveness below 60%")
        
        stability = self._analyze_parameter_stability()
        if stability.get('stability') == 'volatile':
            recommendations.append("Parameter adaptations are frequent - consider increasing confidence thresholds")
        
        return recommendations

def test_real_time_adaptation_engine():
    """Test the real-time adaptation engine"""
    print("Testing Real-time Adaptation Engine...")
    
    engine = RealTimeAdaptationEngine()
    
    # Test market state update
    print("\n--- Testing Market State Update ---")
    market_data = {
        'spy_price': 420.0,
        'spy_return_1d': 0.015,
        'spy_return_5d': 0.025,
        'spy_return_20d': 0.08,
        'vix': 22.0,
        'vix_change': 2.0,
        'volume_ratio': 1.2,
        'put_call_ratio': 0.9
    }
    
    market_state = engine.update_market_state(market_data)
    print(f"Market Regime: {market_state.regime.value}")
    print(f"Regime Confidence: {market_state.regime_confidence:.1%}")
    print(f"VIX Level: {market_state.vix_level}")
    
    # Test protocol adaptation
    print("\n--- Testing Protocol Adaptation ---")
    adaptation_event = engine.adapt_protocol(
        AdaptationTrigger.VOLATILITY_SPIKE, 
        market_state
    )
    
    print(f"Adaptation Event: {adaptation_event.event_id}")
    print(f"Trigger: {adaptation_event.trigger.value}")
    print(f"Confidence: {adaptation_event.confidence:.1%}")
    print(f"Rationale: {adaptation_event.adaptation_rationale}")
    print(f"Expected Impact: {adaptation_event.expected_impact}")
    
    # Test learning from outcome
    print("\n--- Testing Learning from Outcome ---")
    trade_outcome = {
        'event_type': 'trade_completion',
        'actual_return': 0.025,
        'expected_return': 0.020,
        'trade_id': 'TRADE_001'
    }
    
    learning_outcome = engine.learn_from_outcome(trade_outcome, market_data)
    print(f"Learning Outcome: {learning_outcome.outcome_id}")
    print(f"Performance Delta: {learning_outcome.performance_delta:.1%}")
    print(f"Lessons Learned: {len(learning_outcome.lessons_learned)}")
    for lesson in learning_outcome.lessons_learned:
        print(f"  - {lesson}")
    
    # Test performance feedback
    print("\n--- Testing Performance Feedback ---")
    performance_data = {
        'period': '1d',
        'total_return': 0.018,
        'sharpe_ratio': 2.1,
        'win_rate': 0.75,
        'max_drawdown': 0.03,
        'trades_count': 5
    }
    
    feedback = engine.update_performance_feedback(performance_data)
    print(f"Performance Trend: {feedback.performance_trend}")
    print(f"Total Return: {feedback.total_return:.1%}")
    print(f"Sharpe Ratio: {feedback.sharpe_ratio:.2f}")
    print(f"Win Rate: {feedback.win_rate:.1%}")
    
    # Test current parameters
    print("\n--- Testing Current Parameters ---")
    current_params = engine.get_current_parameters()
    print(f"Base Position Size: {current_params['position_sizing']['base_size']}")
    print(f"Max Portfolio Risk: {current_params['risk_management']['max_portfolio_risk']:.1%}")
    print(f"Target DTE: {current_params['dte_preferences']['target_dte']}")
    
    # Test learning insights
    print("\n--- Testing Learning Insights ---")
    insights = engine.get_learning_insights()
    print(f"Total Learning Outcomes: {insights['total_learning_outcomes']}")
    print(f"Learning Effectiveness: {insights['learning_effectiveness']:.1%}")
    print(f"Parameter Stability: {insights['parameter_stability']['stability']}")
    
    # Test regime change
    print("\n--- Testing Regime Change ---")
    high_vol_data = market_data.copy()
    high_vol_data['vix'] = 35.0
    high_vol_data['vix_change'] = 8.0
    
    new_market_state = engine.update_market_state(high_vol_data)
    print(f"New Regime: {new_market_state.regime.value}")
    print(f"Adaptation Events: {len(engine.adaptation_events)}")
    
    print("\n✅ Real-time Adaptation Engine test completed successfully!")

if __name__ == "__main__":
    test_real_time_adaptation_engine()

