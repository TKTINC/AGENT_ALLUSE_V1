"""
HITL Integration and Trust Building System for ALL-USE Protocol
Integrates all advanced capabilities with human oversight and builds trust through demonstrated performance

This module provides comprehensive HITL integration including:
- Trust scoring and confidence tracking across all system components
- Gradual automation based on demonstrated AI performance
- Human override analysis and learning integration
- Comprehensive decision audit trails
- Progressive trust building through validated outcomes
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

class TrustLevel(Enum):
    """Trust levels for automation"""
    NO_TRUST = "no_trust"           # 0-20% - Full human control
    LOW_TRUST = "low_trust"         # 20-40% - AI suggests, human decides
    MEDIUM_TRUST = "medium_trust"   # 40-70% - AI decides with human oversight
    HIGH_TRUST = "high_trust"       # 70-90% - AI decides, human monitors
    FULL_TRUST = "full_trust"       # 90-100% - Full automation

class DecisionType(Enum):
    """Types of decisions in the system"""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    POSITION_ADJUSTMENT = "position_adjustment"
    RISK_MANAGEMENT = "risk_management"
    PARAMETER_OPTIMIZATION = "parameter_optimization"
    PROTOCOL_ADAPTATION = "protocol_adaptation"
    EMERGENCY_ACTION = "emergency_action"

class TrustComponent(Enum):
    """Components that contribute to trust scoring"""
    WEEK_CLASSIFICATION = "week_classification"
    ML_OPTIMIZATION = "ml_optimization"
    REAL_TIME_ADAPTATION = "real_time_adaptation"
    RISK_MANAGEMENT = "risk_management"
    PERFORMANCE_PREDICTION = "performance_prediction"
    MARKET_ANALYSIS = "market_analysis"

@dataclass
class TrustMetrics:
    """Trust metrics for a specific component"""
    component: TrustComponent
    accuracy_score: float           # Historical accuracy (0-1)
    confidence_score: float         # AI confidence in decisions (0-1)
    consistency_score: float        # Consistency over time (0-1)
    performance_impact: float       # Impact on overall performance (0-1)
    human_agreement_rate: float     # Rate of human agreement with AI (0-1)
    override_learn_rate: float      # How well AI learns from overrides (0-1)
    trust_score: float             # Overall trust score (0-1)
    last_updated: datetime
    sample_size: int               # Number of decisions evaluated

@dataclass
class DecisionRecord:
    """Record of a decision made in the system"""
    decision_id: str
    timestamp: datetime
    decision_type: DecisionType
    ai_recommendation: Dict[str, Any]
    ai_confidence: float
    human_decision: Optional[Dict[str, Any]]
    human_override: bool
    override_reason: Optional[str]
    final_decision: Dict[str, Any]
    outcome: Optional[Dict[str, Any]]
    trust_level_at_decision: TrustLevel
    components_involved: List[TrustComponent]
    decision_latency: timedelta
    performance_impact: Optional[float]

@dataclass
class TrustBuildingEvent:
    """Event that contributes to trust building"""
    event_id: str
    timestamp: datetime
    event_type: str
    component: TrustComponent
    ai_prediction: Any
    actual_outcome: Any
    accuracy: float
    confidence: float
    trust_impact: float
    description: str

@dataclass
class AutomationLevel:
    """Current automation level for different decision types"""
    decision_type: DecisionType
    trust_level: TrustLevel
    automation_percentage: float
    human_oversight_required: bool
    approval_timeout: Optional[timedelta]
    escalation_threshold: float
    last_updated: datetime

class HITLTrustSystem:
    """
    HITL Integration and Trust Building System
    
    Provides comprehensive integration of all advanced capabilities with human oversight:
    - Trust scoring across all system components
    - Gradual automation based on demonstrated performance
    - Human override analysis and learning
    - Decision audit trails and performance tracking
    - Progressive trust building through validated outcomes
    """
    
    def __init__(self):
        """Initialize the HITL trust system"""
        self.logger = logging.getLogger(__name__)
        
        # Trust metrics for each component
        self.trust_metrics: Dict[TrustComponent, TrustMetrics] = {}
        
        # Decision tracking
        self.decision_history: List[DecisionRecord] = []
        self.pending_decisions: Dict[str, DecisionRecord] = {}
        
        # Trust building events
        self.trust_events: List[TrustBuildingEvent] = []
        
        # Automation levels
        self.automation_levels: Dict[DecisionType, AutomationLevel] = {}
        
        # Trust building configuration
        self.trust_config = {
            'initial_trust_score': 0.3,        # Start with 30% trust
            'trust_decay_rate': 0.02,          # 2% decay per week without validation
            'trust_growth_rate': 0.05,         # 5% growth per successful validation
            'min_sample_size': 10,             # Minimum decisions for trust calculation
            'trust_update_frequency': timedelta(hours=1),  # Update trust hourly
            'automation_thresholds': {
                TrustLevel.NO_TRUST: 0.2,
                TrustLevel.LOW_TRUST: 0.4,
                TrustLevel.MEDIUM_TRUST: 0.7,
                TrustLevel.HIGH_TRUST: 0.9,
                TrustLevel.FULL_TRUST: 0.95
            }
        }
        
        # Human interaction tracking
        self.human_interactions = {
            'total_decisions': 0,
            'human_overrides': 0,
            'override_success_rate': 0.0,
            'ai_success_rate': 0.0,
            'average_decision_time': timedelta(minutes=5),
            'trust_building_rate': 0.0
        }
        
        # Callbacks for different events
        self.decision_callbacks: List[Callable] = []
        self.trust_update_callbacks: List[Callable] = []
        self.automation_change_callbacks: List[Callable] = []
        
        # Initialize trust metrics and automation levels
        self._initialize_trust_metrics()
        self._initialize_automation_levels()
        
        # Start trust monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.logger.info("HITL Trust System initialized")
    
    def start_trust_monitoring(self):
        """Start continuous trust monitoring and updates"""
        if self.monitoring_active:
            self.logger.warning("Trust monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._trust_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Trust monitoring started")
    
    def stop_trust_monitoring(self):
        """Stop trust monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Trust monitoring stopped")
    
    def process_decision(self, decision_type: DecisionType, ai_recommendation: Dict[str, Any],
                        ai_confidence: float, components_involved: List[TrustComponent],
                        context: Optional[Dict[str, Any]] = None) -> DecisionRecord:
        """
        Process a decision through the HITL system
        
        Args:
            decision_type: Type of decision being made
            ai_recommendation: AI's recommended action
            ai_confidence: AI's confidence in the recommendation
            components_involved: Which trust components contributed
            context: Additional context for the decision
            
        Returns:
            Decision record with human interaction results
        """
        try:
            # Create decision record
            decision_record = DecisionRecord(
                decision_id=f"dec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.decision_history)}",
                timestamp=datetime.now(),
                decision_type=decision_type,
                ai_recommendation=ai_recommendation,
                ai_confidence=ai_confidence,
                human_decision=None,
                human_override=False,
                override_reason=None,
                final_decision=ai_recommendation.copy(),
                outcome=None,
                trust_level_at_decision=self._get_current_trust_level(decision_type),
                components_involved=components_involved,
                decision_latency=timedelta(0),
                performance_impact=None
            )
            
            # Determine if human oversight is required
            automation_level = self.automation_levels[decision_type]
            requires_human = self._requires_human_oversight(decision_record, automation_level)
            
            if requires_human:
                # Request human input
                decision_record = self._request_human_decision(decision_record, automation_level)
            else:
                # Proceed with AI decision
                decision_record.final_decision = ai_recommendation
                self.logger.info(f"AI decision approved automatically: {decision_record.decision_id}")
            
            # Record decision
            self.decision_history.append(decision_record)
            self.human_interactions['total_decisions'] += 1
            
            # Update trust metrics based on decision
            self._update_trust_from_decision(decision_record)
            
            # Notify callbacks
            for callback in self.decision_callbacks:
                try:
                    callback(decision_record)
                except Exception as e:
                    self.logger.error(f"Error in decision callback: {str(e)}")
            
            self.logger.info(f"Decision processed: {decision_record.decision_id} (Override: {decision_record.human_override})")
            return decision_record
            
        except Exception as e:
            self.logger.error(f"Error processing decision: {str(e)}")
            raise
    
    def record_decision_outcome(self, decision_id: str, outcome: Dict[str, Any]):
        """
        Record the outcome of a decision for trust building
        
        Args:
            decision_id: ID of the decision
            outcome: Actual outcome of the decision
        """
        try:
            # Find decision record
            decision_record = None
            for record in self.decision_history:
                if record.decision_id == decision_id:
                    decision_record = record
                    break
            
            if not decision_record:
                self.logger.warning(f"Decision record not found: {decision_id}")
                return
            
            # Update outcome
            decision_record.outcome = outcome
            decision_record.performance_impact = outcome.get('performance_impact', 0.0)
            
            # Create trust building event
            trust_event = self._create_trust_event_from_outcome(decision_record)
            self.trust_events.append(trust_event)
            
            # Update trust metrics
            self._update_trust_from_outcome(decision_record, trust_event)
            
            # Update human interaction statistics
            self._update_human_interaction_stats(decision_record)
            
            self.logger.info(f"Decision outcome recorded: {decision_id} (Impact: {decision_record.performance_impact:.1%})")
            
        except Exception as e:
            self.logger.error(f"Error recording decision outcome: {str(e)}")
    
    def get_trust_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive trust dashboard"""
        try:
            # Calculate overall trust score
            overall_trust = self._calculate_overall_trust_score()
            
            # Get component trust scores
            component_scores = {
                component.value: metrics.trust_score 
                for component, metrics in self.trust_metrics.items()
            }
            
            # Get automation status
            automation_status = {
                decision_type.value: {
                    'trust_level': level.trust_level.value,
                    'automation_percentage': level.automation_percentage,
                    'human_oversight_required': level.human_oversight_required
                }
                for decision_type, level in self.automation_levels.items()
            }
            
            # Recent performance
            recent_decisions = self.decision_history[-50:] if self.decision_history else []
            recent_override_rate = len([d for d in recent_decisions if d.human_override]) / len(recent_decisions) if recent_decisions else 0
            
            dashboard = {
                'overall_trust_score': overall_trust,
                'trust_level': self._trust_score_to_level(overall_trust).value,
                'component_trust_scores': component_scores,
                'automation_status': automation_status,
                'human_interaction_stats': self.human_interactions.copy(),
                'recent_performance': {
                    'total_decisions_last_50': len(recent_decisions),
                    'override_rate_last_50': recent_override_rate,
                    'avg_ai_confidence': np.mean([d.ai_confidence for d in recent_decisions]) if recent_decisions else 0,
                    'avg_decision_latency': str(np.mean([d.decision_latency.total_seconds() for d in recent_decisions])) + " seconds" if recent_decisions else "0 seconds"
                },
                'trust_building_progress': {
                    'total_trust_events': len(self.trust_events),
                    'trust_growth_rate': self.human_interactions['trust_building_rate'],
                    'components_ready_for_automation': len([c for c, m in self.trust_metrics.items() if m.trust_score > 0.7]),
                    'next_automation_milestone': self._get_next_automation_milestone()
                },
                'recommendations': self._generate_trust_recommendations()
            }
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error generating trust dashboard: {str(e)}")
            return {'error': str(e)}
    
    def get_component_trust_details(self, component: TrustComponent) -> Dict[str, Any]:
        """Get detailed trust information for a specific component"""
        if component not in self.trust_metrics:
            return {'error': f'Component {component.value} not found'}
        
        metrics = self.trust_metrics[component]
        
        # Get recent decisions involving this component
        recent_decisions = [
            d for d in self.decision_history[-100:] 
            if component in d.components_involved
        ]
        
        # Calculate component-specific statistics
        component_stats = {
            'trust_metrics': {
                'accuracy_score': metrics.accuracy_score,
                'confidence_score': metrics.confidence_score,
                'consistency_score': metrics.consistency_score,
                'performance_impact': metrics.performance_impact,
                'human_agreement_rate': metrics.human_agreement_rate,
                'override_learn_rate': metrics.override_learn_rate,
                'overall_trust_score': metrics.trust_score,
                'sample_size': metrics.sample_size,
                'last_updated': metrics.last_updated.isoformat()
            },
            'recent_performance': {
                'decisions_involving_component': len(recent_decisions),
                'override_rate': len([d for d in recent_decisions if d.human_override]) / len(recent_decisions) if recent_decisions else 0,
                'avg_confidence': np.mean([d.ai_confidence for d in recent_decisions]) if recent_decisions else 0,
                'success_rate': self._calculate_component_success_rate(component, recent_decisions)
            },
            'trust_trajectory': self._get_component_trust_trajectory(component),
            'automation_readiness': self._assess_automation_readiness(component),
            'improvement_recommendations': self._generate_component_recommendations(component)
        }
        
        return component_stats
    
    def simulate_trust_building(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulate trust building under different scenarios
        
        Args:
            scenarios: List of scenario configurations
            
        Returns:
            Simulation results showing trust progression
        """
        try:
            simulation_results = {}
            
            for i, scenario in enumerate(scenarios):
                scenario_name = scenario.get('name', f'Scenario_{i+1}')
                
                # Create simulation copy of trust metrics
                sim_trust_metrics = {
                    component: TrustMetrics(
                        component=component,
                        accuracy_score=metrics.accuracy_score,
                        confidence_score=metrics.confidence_score,
                        consistency_score=metrics.consistency_score,
                        performance_impact=metrics.performance_impact,
                        human_agreement_rate=metrics.human_agreement_rate,
                        override_learn_rate=metrics.override_learn_rate,
                        trust_score=metrics.trust_score,
                        last_updated=metrics.last_updated,
                        sample_size=metrics.sample_size
                    )
                    for component, metrics in self.trust_metrics.items()
                }
                
                # Simulate scenario
                sim_results = self._run_trust_simulation(scenario, sim_trust_metrics)
                simulation_results[scenario_name] = sim_results
            
            return {
                'simulation_results': simulation_results,
                'summary': self._summarize_simulation_results(simulation_results),
                'recommendations': self._generate_simulation_recommendations(simulation_results)
            }
            
        except Exception as e:
            self.logger.error(f"Error in trust building simulation: {str(e)}")
            return {'error': str(e)}
    
    def register_decision_callback(self, callback: Callable[[DecisionRecord], None]):
        """Register callback for decision events"""
        self.decision_callbacks.append(callback)
    
    def register_trust_update_callback(self, callback: Callable[[TrustComponent, TrustMetrics], None]):
        """Register callback for trust updates"""
        self.trust_update_callbacks.append(callback)
    
    def register_automation_change_callback(self, callback: Callable[[DecisionType, AutomationLevel], None]):
        """Register callback for automation level changes"""
        self.automation_change_callbacks.append(callback)
    
    # Helper methods for core functionality
    def _initialize_trust_metrics(self):
        """Initialize trust metrics for all components"""
        for component in TrustComponent:
            self.trust_metrics[component] = TrustMetrics(
                component=component,
                accuracy_score=self.trust_config['initial_trust_score'],
                confidence_score=self.trust_config['initial_trust_score'],
                consistency_score=self.trust_config['initial_trust_score'],
                performance_impact=self.trust_config['initial_trust_score'],
                human_agreement_rate=self.trust_config['initial_trust_score'],
                override_learn_rate=self.trust_config['initial_trust_score'],
                trust_score=self.trust_config['initial_trust_score'],
                last_updated=datetime.now(),
                sample_size=0
            )
        
        self.logger.info("Trust metrics initialized for all components")
    
    def _initialize_automation_levels(self):
        """Initialize automation levels for all decision types"""
        for decision_type in DecisionType:
            # Start with low trust for all decision types
            initial_trust_level = TrustLevel.LOW_TRUST
            
            self.automation_levels[decision_type] = AutomationLevel(
                decision_type=decision_type,
                trust_level=initial_trust_level,
                automation_percentage=0.2,  # 20% automation initially
                human_oversight_required=True,
                approval_timeout=self._get_approval_timeout(decision_type),
                escalation_threshold=0.1,
                last_updated=datetime.now()
            )
        
        self.logger.info("Automation levels initialized for all decision types")
    
    def _get_approval_timeout(self, decision_type: DecisionType) -> timedelta:
        """Get approval timeout for decision type"""
        timeouts = {
            DecisionType.EMERGENCY_ACTION: timedelta(minutes=2),
            DecisionType.RISK_MANAGEMENT: timedelta(minutes=5),
            DecisionType.TRADE_ENTRY: timedelta(minutes=15),
            DecisionType.TRADE_EXIT: timedelta(minutes=10),
            DecisionType.POSITION_ADJUSTMENT: timedelta(minutes=30),
            DecisionType.PARAMETER_OPTIMIZATION: timedelta(hours=2),
            DecisionType.PROTOCOL_ADAPTATION: timedelta(hours=4)
        }
        return timeouts.get(decision_type, timedelta(minutes=15))
    
    def _get_current_trust_level(self, decision_type: DecisionType) -> TrustLevel:
        """Get current trust level for decision type"""
        return self.automation_levels[decision_type].trust_level
    
    def _requires_human_oversight(self, decision_record: DecisionRecord, 
                                automation_level: AutomationLevel) -> bool:
        """Determine if human oversight is required"""
        # Always require human oversight for emergency actions
        if decision_record.decision_type == DecisionType.EMERGENCY_ACTION:
            return True
        
        # Check automation level
        if automation_level.trust_level in [TrustLevel.NO_TRUST, TrustLevel.LOW_TRUST]:
            return True
        
        # Check AI confidence
        if decision_record.ai_confidence < automation_level.escalation_threshold:
            return True
        
        # Check if any involved components have low trust
        for component in decision_record.components_involved:
            if self.trust_metrics[component].trust_score < 0.5:
                return True
        
        return False
    
    def _request_human_decision(self, decision_record: DecisionRecord, 
                              automation_level: AutomationLevel) -> DecisionRecord:
        """Request human decision (simulated for testing)"""
        start_time = datetime.now()
        
        # Simulate human decision-making process
        # In real implementation, this would interface with UI/notification system
        
        # Simulate human response time
        response_time = np.random.normal(300, 120)  # 5 minutes average, 2 minute std dev
        response_time = max(30, min(1800, response_time))  # Between 30 seconds and 30 minutes
        
        # Simulate human decision
        # 80% of the time, human agrees with AI if confidence > 0.7
        # 60% of the time, human agrees with AI if confidence < 0.7
        agreement_probability = 0.8 if decision_record.ai_confidence > 0.7 else 0.6
        human_agrees = np.random.random() < agreement_probability
        
        if human_agrees:
            decision_record.human_decision = decision_record.ai_recommendation.copy()
            decision_record.final_decision = decision_record.ai_recommendation.copy()
            decision_record.human_override = False
        else:
            # Simulate human override
            decision_record.human_decision = self._generate_human_override_decision(decision_record)
            decision_record.final_decision = decision_record.human_decision.copy()
            decision_record.human_override = True
            decision_record.override_reason = "Human judgment based on market conditions"
            
            self.human_interactions['human_overrides'] += 1
        
        decision_record.decision_latency = timedelta(seconds=response_time)
        
        self.logger.info(f"Human decision received: {decision_record.decision_id} (Override: {decision_record.human_override})")
        return decision_record
    
    def _generate_human_override_decision(self, decision_record: DecisionRecord) -> Dict[str, Any]:
        """Generate a simulated human override decision"""
        # Simplified override logic - in reality this would be actual human input
        override_decision = decision_record.ai_recommendation.copy()
        
        # Modify some parameters to simulate human judgment
        if 'position_size' in override_decision:
            override_decision['position_size'] = max(1, override_decision['position_size'] * 0.8)
        
        if 'risk_level' in override_decision:
            override_decision['risk_level'] = min(0.1, override_decision.get('risk_level', 0.05) * 0.9)
        
        return override_decision
    
    def _update_trust_from_decision(self, decision_record: DecisionRecord):
        """Update trust metrics based on decision"""
        # Update trust for involved components
        for component in decision_record.components_involved:
            metrics = self.trust_metrics[component]
            
            # Update human agreement rate
            if decision_record.human_decision is not None:
                agreement = not decision_record.human_override
                metrics.human_agreement_rate = self._update_running_average(
                    metrics.human_agreement_rate, agreement, metrics.sample_size
                )
            
            # Update confidence score
            metrics.confidence_score = self._update_running_average(
                metrics.confidence_score, decision_record.ai_confidence, metrics.sample_size
            )
            
            metrics.sample_size += 1
            metrics.last_updated = datetime.now()
            
            # Recalculate overall trust score
            metrics.trust_score = self._calculate_component_trust_score(metrics)
    
    def _create_trust_event_from_outcome(self, decision_record: DecisionRecord) -> TrustBuildingEvent:
        """Create trust building event from decision outcome"""
        # Determine accuracy based on outcome
        performance_impact = decision_record.performance_impact or 0.0
        accuracy = 1.0 if performance_impact > 0 else 0.0 if performance_impact < -0.02 else 0.5
        
        # Calculate trust impact
        trust_impact = self._calculate_trust_impact(decision_record, accuracy)
        
        return TrustBuildingEvent(
            event_id=f"trust_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            event_type="decision_outcome",
            component=decision_record.components_involved[0] if decision_record.components_involved else TrustComponent.WEEK_CLASSIFICATION,
            ai_prediction=decision_record.ai_recommendation,
            actual_outcome=decision_record.outcome,
            accuracy=accuracy,
            confidence=decision_record.ai_confidence,
            trust_impact=trust_impact,
            description=f"Decision outcome for {decision_record.decision_type.value}"
        )
    
    def _update_trust_from_outcome(self, decision_record: DecisionRecord, trust_event: TrustBuildingEvent):
        """Update trust metrics based on decision outcome"""
        for component in decision_record.components_involved:
            metrics = self.trust_metrics[component]
            
            # Update accuracy score
            metrics.accuracy_score = self._update_running_average(
                metrics.accuracy_score, trust_event.accuracy, metrics.sample_size
            )
            
            # Update performance impact
            if decision_record.performance_impact is not None:
                impact_score = 1.0 if decision_record.performance_impact > 0 else 0.0
                metrics.performance_impact = self._update_running_average(
                    metrics.performance_impact, impact_score, metrics.sample_size
                )
            
            # Update consistency score (simplified)
            recent_accuracy = self._get_recent_accuracy(component)
            consistency = 1.0 - abs(metrics.accuracy_score - recent_accuracy)
            metrics.consistency_score = self._update_running_average(
                metrics.consistency_score, consistency, metrics.sample_size
            )
            
            # Update override learning rate
            if decision_record.human_override:
                override_success = 1.0 if decision_record.performance_impact > 0 else 0.0
                metrics.override_learn_rate = self._update_running_average(
                    metrics.override_learn_rate, override_success, metrics.sample_size
                )
            
            # Recalculate overall trust score
            metrics.trust_score = self._calculate_component_trust_score(metrics)
            metrics.last_updated = datetime.now()
            
            # Check for automation level updates
            self._update_automation_level(decision_record.decision_type)
            
            # Notify trust update callbacks
            for callback in self.trust_update_callbacks:
                try:
                    callback(component, metrics)
                except Exception as e:
                    self.logger.error(f"Error in trust update callback: {str(e)}")
    
    def _calculate_component_trust_score(self, metrics: TrustMetrics) -> float:
        """Calculate overall trust score for a component"""
        if metrics.sample_size < self.trust_config['min_sample_size']:
            return self.trust_config['initial_trust_score']
        
        # Weighted average of all metrics
        weights = {
            'accuracy_score': 0.3,
            'confidence_score': 0.2,
            'consistency_score': 0.2,
            'performance_impact': 0.15,
            'human_agreement_rate': 0.1,
            'override_learn_rate': 0.05
        }
        
        trust_score = (
            weights['accuracy_score'] * metrics.accuracy_score +
            weights['confidence_score'] * metrics.confidence_score +
            weights['consistency_score'] * metrics.consistency_score +
            weights['performance_impact'] * metrics.performance_impact +
            weights['human_agreement_rate'] * metrics.human_agreement_rate +
            weights['override_learn_rate'] * metrics.override_learn_rate
        )
        
        return min(1.0, max(0.0, trust_score))
    
    def _calculate_overall_trust_score(self) -> float:
        """Calculate overall system trust score"""
        if not self.trust_metrics:
            return self.trust_config['initial_trust_score']
        
        # Weight components by their importance
        component_weights = {
            TrustComponent.WEEK_CLASSIFICATION: 0.25,
            TrustComponent.ML_OPTIMIZATION: 0.20,
            TrustComponent.REAL_TIME_ADAPTATION: 0.20,
            TrustComponent.RISK_MANAGEMENT: 0.15,
            TrustComponent.PERFORMANCE_PREDICTION: 0.10,
            TrustComponent.MARKET_ANALYSIS: 0.10
        }
        
        weighted_score = sum(
            component_weights.get(component, 0.1) * metrics.trust_score
            for component, metrics in self.trust_metrics.items()
        )
        
        return min(1.0, max(0.0, weighted_score))
    
    def _trust_score_to_level(self, trust_score: float) -> TrustLevel:
        """Convert trust score to trust level"""
        thresholds = self.trust_config['automation_thresholds']
        
        if trust_score >= thresholds[TrustLevel.FULL_TRUST]:
            return TrustLevel.FULL_TRUST
        elif trust_score >= thresholds[TrustLevel.HIGH_TRUST]:
            return TrustLevel.HIGH_TRUST
        elif trust_score >= thresholds[TrustLevel.MEDIUM_TRUST]:
            return TrustLevel.MEDIUM_TRUST
        elif trust_score >= thresholds[TrustLevel.LOW_TRUST]:
            return TrustLevel.LOW_TRUST
        else:
            return TrustLevel.NO_TRUST
    
    def _update_automation_level(self, decision_type: DecisionType):
        """Update automation level based on current trust"""
        current_level = self.automation_levels[decision_type]
        
        # Calculate trust score for this decision type
        relevant_components = self._get_relevant_components(decision_type)
        avg_trust = np.mean([self.trust_metrics[comp].trust_score for comp in relevant_components])
        
        # Determine new trust level
        new_trust_level = self._trust_score_to_level(avg_trust)
        
        if new_trust_level != current_level.trust_level:
            # Update automation level
            old_level = current_level.trust_level
            current_level.trust_level = new_trust_level
            current_level.automation_percentage = self._get_automation_percentage(new_trust_level)
            current_level.human_oversight_required = new_trust_level in [TrustLevel.NO_TRUST, TrustLevel.LOW_TRUST]
            current_level.last_updated = datetime.now()
            
            self.logger.info(f"Automation level updated for {decision_type.value}: {old_level.value} -> {new_trust_level.value}")
            
            # Notify automation change callbacks
            for callback in self.automation_change_callbacks:
                try:
                    callback(decision_type, current_level)
                except Exception as e:
                    self.logger.error(f"Error in automation change callback: {str(e)}")
    
    def _get_relevant_components(self, decision_type: DecisionType) -> List[TrustComponent]:
        """Get components relevant to a decision type"""
        component_mapping = {
            DecisionType.TRADE_ENTRY: [TrustComponent.WEEK_CLASSIFICATION, TrustComponent.MARKET_ANALYSIS],
            DecisionType.TRADE_EXIT: [TrustComponent.PERFORMANCE_PREDICTION, TrustComponent.RISK_MANAGEMENT],
            DecisionType.POSITION_ADJUSTMENT: [TrustComponent.REAL_TIME_ADAPTATION, TrustComponent.RISK_MANAGEMENT],
            DecisionType.RISK_MANAGEMENT: [TrustComponent.RISK_MANAGEMENT],
            DecisionType.PARAMETER_OPTIMIZATION: [TrustComponent.ML_OPTIMIZATION],
            DecisionType.PROTOCOL_ADAPTATION: [TrustComponent.REAL_TIME_ADAPTATION],
            DecisionType.EMERGENCY_ACTION: [TrustComponent.RISK_MANAGEMENT, TrustComponent.REAL_TIME_ADAPTATION]
        }
        
        return component_mapping.get(decision_type, list(TrustComponent))
    
    def _get_automation_percentage(self, trust_level: TrustLevel) -> float:
        """Get automation percentage for trust level"""
        percentages = {
            TrustLevel.NO_TRUST: 0.0,
            TrustLevel.LOW_TRUST: 0.2,
            TrustLevel.MEDIUM_TRUST: 0.5,
            TrustLevel.HIGH_TRUST: 0.8,
            TrustLevel.FULL_TRUST: 0.95
        }
        return percentages.get(trust_level, 0.2)
    
    def _update_running_average(self, current_avg: float, new_value: float, sample_size: int) -> float:
        """Update running average with new value"""
        if sample_size == 0:
            return new_value
        return (current_avg * sample_size + new_value) / (sample_size + 1)
    
    def _calculate_trust_impact(self, decision_record: DecisionRecord, accuracy: float) -> float:
        """Calculate trust impact of a decision outcome"""
        base_impact = 0.02 if accuracy > 0.5 else -0.01
        confidence_multiplier = decision_record.ai_confidence
        return base_impact * confidence_multiplier
    
    def _get_recent_accuracy(self, component: TrustComponent) -> float:
        """Get recent accuracy for a component"""
        recent_events = [
            event for event in self.trust_events[-20:]
            if event.component == component
        ]
        
        if not recent_events:
            return self.trust_metrics[component].accuracy_score
        
        return np.mean([event.accuracy for event in recent_events])
    
    def _update_human_interaction_stats(self, decision_record: DecisionRecord):
        """Update human interaction statistics"""
        if decision_record.outcome:
            performance_impact = decision_record.performance_impact or 0.0
            
            if decision_record.human_override:
                # Update override success rate
                override_success = performance_impact > 0
                total_overrides = self.human_interactions['human_overrides']
                current_rate = self.human_interactions['override_success_rate']
                self.human_interactions['override_success_rate'] = (
                    (current_rate * (total_overrides - 1) + override_success) / total_overrides
                )
            else:
                # Update AI success rate
                ai_success = performance_impact > 0
                ai_decisions = self.human_interactions['total_decisions'] - self.human_interactions['human_overrides']
                if ai_decisions > 0:
                    current_rate = self.human_interactions['ai_success_rate']
                    self.human_interactions['ai_success_rate'] = (
                        (current_rate * (ai_decisions - 1) + ai_success) / ai_decisions
                    )
        
        # Update average decision time
        if decision_record.decision_latency:
            total_decisions = self.human_interactions['total_decisions']
            current_avg = self.human_interactions['average_decision_time']
            new_avg_seconds = (
                (current_avg.total_seconds() * (total_decisions - 1) + decision_record.decision_latency.total_seconds()) 
                / total_decisions
            )
            self.human_interactions['average_decision_time'] = timedelta(seconds=new_avg_seconds)
    
    def _trust_monitoring_loop(self):
        """Main trust monitoring loop"""
        while self.monitoring_active:
            try:
                # Apply trust decay
                self._apply_trust_decay()
                
                # Update automation levels
                for decision_type in DecisionType:
                    self._update_automation_level(decision_type)
                
                # Calculate trust building rate
                self._calculate_trust_building_rate()
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Error in trust monitoring loop: {str(e)}")
                time.sleep(1800)  # Wait 30 minutes on error
    
    def _apply_trust_decay(self):
        """Apply trust decay over time"""
        decay_rate = self.trust_config['trust_decay_rate']
        
        for component, metrics in self.trust_metrics.items():
            time_since_update = datetime.now() - metrics.last_updated
            weeks_since_update = time_since_update.days / 7
            
            if weeks_since_update > 1:  # Apply decay after 1 week
                decay_factor = (1 - decay_rate) ** weeks_since_update
                metrics.trust_score *= decay_factor
                metrics.trust_score = max(0.1, metrics.trust_score)  # Minimum trust floor
    
    def _calculate_trust_building_rate(self):
        """Calculate current trust building rate"""
        if len(self.trust_events) < 10:
            self.human_interactions['trust_building_rate'] = 0.0
            return
        
        recent_events = self.trust_events[-50:]  # Last 50 events
        positive_events = len([e for e in recent_events if e.trust_impact > 0])
        
        self.human_interactions['trust_building_rate'] = positive_events / len(recent_events)
    
    def _get_next_automation_milestone(self) -> str:
        """Get next automation milestone"""
        overall_trust = self._calculate_overall_trust_score()
        current_level = self._trust_score_to_level(overall_trust)
        
        next_levels = {
            TrustLevel.NO_TRUST: TrustLevel.LOW_TRUST,
            TrustLevel.LOW_TRUST: TrustLevel.MEDIUM_TRUST,
            TrustLevel.MEDIUM_TRUST: TrustLevel.HIGH_TRUST,
            TrustLevel.HIGH_TRUST: TrustLevel.FULL_TRUST,
            TrustLevel.FULL_TRUST: None
        }
        
        next_level = next_levels.get(current_level)
        if next_level:
            required_score = self.trust_config['automation_thresholds'][next_level]
            return f"Reach {required_score:.0%} trust for {next_level.value} ({overall_trust:.0%} current)"
        else:
            return "Maximum automation level achieved"
    
    def _generate_trust_recommendations(self) -> List[str]:
        """Generate trust building recommendations"""
        recommendations = []
        
        overall_trust = self._calculate_overall_trust_score()
        
        if overall_trust < 0.5:
            recommendations.append("Focus on building basic trust through consistent AI performance")
        
        # Component-specific recommendations
        for component, metrics in self.trust_metrics.items():
            if metrics.trust_score < 0.4:
                recommendations.append(f"Improve {component.value} accuracy and consistency")
        
        # Human interaction recommendations
        override_rate = self.human_interactions['human_overrides'] / max(1, self.human_interactions['total_decisions'])
        if override_rate > 0.3:
            recommendations.append("High override rate - review AI confidence calibration")
        
        return recommendations
    
    # Additional helper methods for simulation and analysis
    def _run_trust_simulation(self, scenario: Dict[str, Any], 
                            sim_trust_metrics: Dict[TrustComponent, TrustMetrics]) -> Dict[str, Any]:
        """Run trust building simulation"""
        # Simplified simulation
        simulation_days = scenario.get('days', 30)
        ai_accuracy = scenario.get('ai_accuracy', 0.75)
        decision_frequency = scenario.get('decisions_per_day', 5)
        
        trust_progression = []
        
        for day in range(simulation_days):
            # Simulate daily decisions
            for _ in range(decision_frequency):
                # Simulate decision outcome
                success = np.random.random() < ai_accuracy
                
                # Update trust metrics
                for component, metrics in sim_trust_metrics.items():
                    if success:
                        metrics.accuracy_score = min(1.0, metrics.accuracy_score + 0.01)
                    else:
                        metrics.accuracy_score = max(0.0, metrics.accuracy_score - 0.005)
                    
                    metrics.trust_score = self._calculate_component_trust_score(metrics)
            
            # Record daily trust score
            overall_trust = np.mean([m.trust_score for m in sim_trust_metrics.values()])
            trust_progression.append(overall_trust)
        
        return {
            'trust_progression': trust_progression,
            'final_trust_score': trust_progression[-1] if trust_progression else 0,
            'trust_growth': trust_progression[-1] - trust_progression[0] if len(trust_progression) > 1 else 0,
            'automation_level_achieved': self._trust_score_to_level(trust_progression[-1]).value if trust_progression else 'no_trust'
        }
    
    def _summarize_simulation_results(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize simulation results"""
        all_final_scores = [result['final_trust_score'] for result in simulation_results.values()]
        
        return {
            'average_final_trust': np.mean(all_final_scores),
            'best_scenario': max(simulation_results.items(), key=lambda x: x[1]['final_trust_score'])[0],
            'worst_scenario': min(simulation_results.items(), key=lambda x: x[1]['final_trust_score'])[0],
            'scenarios_reaching_high_trust': len([s for s in all_final_scores if s > 0.8])
        }
    
    def _generate_simulation_recommendations(self, simulation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on simulation results"""
        recommendations = []
        
        summary = self._summarize_simulation_results(simulation_results)
        
        if summary['average_final_trust'] < 0.6:
            recommendations.append("Focus on improving AI accuracy to build trust faster")
        
        if summary['scenarios_reaching_high_trust'] < len(simulation_results) / 2:
            recommendations.append("Consider adjusting trust building parameters for faster progression")
        
        return recommendations
    
    def _calculate_component_success_rate(self, component: TrustComponent, 
                                        recent_decisions: List[DecisionRecord]) -> float:
        """Calculate success rate for a component"""
        component_decisions = [d for d in recent_decisions if component in d.components_involved and d.outcome]
        
        if not component_decisions:
            return 0.0
        
        successful_decisions = len([d for d in component_decisions if d.performance_impact and d.performance_impact > 0])
        return successful_decisions / len(component_decisions)
    
    def _get_component_trust_trajectory(self, component: TrustComponent) -> List[float]:
        """Get trust trajectory for a component"""
        # Simplified trajectory - in real implementation, this would track historical trust scores
        current_trust = self.trust_metrics[component].trust_score
        return [max(0.1, current_trust - 0.1 + i * 0.02) for i in range(10)]
    
    def _assess_automation_readiness(self, component: TrustComponent) -> Dict[str, Any]:
        """Assess automation readiness for a component"""
        metrics = self.trust_metrics[component]
        
        readiness_score = (
            metrics.accuracy_score * 0.4 +
            metrics.consistency_score * 0.3 +
            metrics.human_agreement_rate * 0.3
        )
        
        return {
            'readiness_score': readiness_score,
            'ready_for_automation': readiness_score > 0.7,
            'estimated_time_to_readiness': max(0, (0.7 - readiness_score) * 30) if readiness_score < 0.7 else 0,
            'blocking_factors': self._identify_blocking_factors(metrics)
        }
    
    def _identify_blocking_factors(self, metrics: TrustMetrics) -> List[str]:
        """Identify factors blocking automation readiness"""
        factors = []
        
        if metrics.accuracy_score < 0.7:
            factors.append("Low accuracy score")
        if metrics.consistency_score < 0.6:
            factors.append("Inconsistent performance")
        if metrics.human_agreement_rate < 0.6:
            factors.append("Low human agreement rate")
        if metrics.sample_size < self.trust_config['min_sample_size']:
            factors.append("Insufficient sample size")
        
        return factors
    
    def _generate_component_recommendations(self, component: TrustComponent) -> List[str]:
        """Generate improvement recommendations for a component"""
        metrics = self.trust_metrics[component]
        recommendations = []
        
        if metrics.accuracy_score < 0.7:
            recommendations.append(f"Improve {component.value} model accuracy through better training data")
        
        if metrics.consistency_score < 0.6:
            recommendations.append(f"Focus on {component.value} consistency across different market conditions")
        
        if metrics.human_agreement_rate < 0.6:
            recommendations.append(f"Calibrate {component.value} confidence scores to better match human judgment")
        
        return recommendations

def test_hitl_trust_system():
    """Test the HITL trust system"""
    print("Testing HITL Trust System...")
    
    system = HITLTrustSystem()
    
    # Test decision processing
    print("\n--- Testing Decision Processing ---")
    ai_recommendation = {
        'action': 'sell_put',
        'position_size': 10,
        'delta': 30,
        'confidence': 0.75
    }
    
    decision_record = system.process_decision(
        DecisionType.TRADE_ENTRY,
        ai_recommendation,
        0.75,
        [TrustComponent.WEEK_CLASSIFICATION, TrustComponent.MARKET_ANALYSIS]
    )
    
    print(f"Decision ID: {decision_record.decision_id}")
    print(f"Human Override: {decision_record.human_override}")
    print(f"Trust Level: {decision_record.trust_level_at_decision.value}")
    print(f"Decision Latency: {decision_record.decision_latency.total_seconds():.1f} seconds")
    
    # Test decision outcome
    print("\n--- Testing Decision Outcome ---")
    outcome = {
        'performance_impact': 0.025,  # 2.5% positive impact
        'success': True,
        'actual_return': 0.025
    }
    
    system.record_decision_outcome(decision_record.decision_id, outcome)
    print(f"Outcome recorded with {outcome['performance_impact']:.1%} performance impact")
    
    # Test trust dashboard
    print("\n--- Testing Trust Dashboard ---")
    dashboard = system.get_trust_dashboard()
    
    print(f"Overall Trust Score: {dashboard['overall_trust_score']:.1%}")
    print(f"Trust Level: {dashboard['trust_level']}")
    print(f"Total Decisions: {dashboard['human_interaction_stats']['total_decisions']}")
    print(f"Override Rate: {dashboard['recent_performance']['override_rate_last_50']:.1%}")
    
    # Test component trust details
    print("\n--- Testing Component Trust Details ---")
    component_details = system.get_component_trust_details(TrustComponent.WEEK_CLASSIFICATION)
    
    print(f"Component Trust Score: {component_details['trust_metrics']['overall_trust_score']:.1%}")
    print(f"Accuracy Score: {component_details['trust_metrics']['accuracy_score']:.1%}")
    print(f"Human Agreement Rate: {component_details['trust_metrics']['human_agreement_rate']:.1%}")
    print(f"Automation Ready: {component_details['automation_readiness']['ready_for_automation']}")
    
    # Test multiple decisions to build trust
    print("\n--- Testing Trust Building ---")
    for i in range(5):
        ai_rec = {
            'action': f'action_{i}',
            'confidence': 0.8 + i * 0.02
        }
        
        decision = system.process_decision(
            DecisionType.TRADE_ENTRY,
            ai_rec,
            0.8 + i * 0.02,
            [TrustComponent.WEEK_CLASSIFICATION]
        )
        
        # Simulate positive outcome
        outcome = {
            'performance_impact': 0.02 + i * 0.005,
            'success': True
        }
        
        system.record_decision_outcome(decision.decision_id, outcome)
    
    # Check updated trust
    updated_dashboard = system.get_trust_dashboard()
    print(f"Updated Overall Trust: {updated_dashboard['overall_trust_score']:.1%}")
    print(f"Trust Building Rate: {updated_dashboard['trust_building_progress']['trust_growth_rate']:.1%}")
    
    # Test trust simulation
    print("\n--- Testing Trust Simulation ---")
    scenarios = [
        {'name': 'High Accuracy', 'ai_accuracy': 0.85, 'days': 30, 'decisions_per_day': 3},
        {'name': 'Medium Accuracy', 'ai_accuracy': 0.70, 'days': 30, 'decisions_per_day': 3}
    ]
    
    simulation_results = system.simulate_trust_building(scenarios)
    
    for scenario_name, results in simulation_results['simulation_results'].items():
        print(f"{scenario_name}: Final Trust {results['final_trust_score']:.1%}, Growth {results['trust_growth']:.1%}")
    
    print(f"Best Scenario: {simulation_results['summary']['best_scenario']}")
    
    print("\n✅ HITL Trust System test completed successfully!")

if __name__ == "__main__":
    test_hitl_trust_system()

