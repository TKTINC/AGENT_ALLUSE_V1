"""
Human-in-the-Loop Decision Gateway for ALL-USE Protocol
Implements comprehensive human oversight and approval systems for building trust and ensuring control

This module provides sophisticated human-in-the-loop decision making capabilities
including approval gates, confidence-based automation, override mechanisms,
and trust building systems for gradual automation adoption.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of decisions requiring human oversight"""
    TRADE_ENTRY = "trade_entry"
    POSITION_ADJUSTMENT = "position_adjustment"
    RECOVERY_STRATEGY = "recovery_strategy"
    RISK_BREACH = "risk_breach"
    RULE_VIOLATION = "rule_violation"
    EMERGENCY_EXIT = "emergency_exit"

class ApprovalStatus(Enum):
    """Status of approval requests"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    OVERRIDDEN = "overridden"
    AUTO_APPROVED = "auto_approved"

class AutomationLevel(Enum):
    """Levels of automation based on confidence"""
    FULL_AUTOMATION = "full_automation"          # >90% confidence
    SUPERVISED_AUTOMATION = "supervised_automation"  # 80-90% confidence
    APPROVAL_REQUIRED = "approval_required"      # 70-80% confidence
    HUMAN_DECISION = "human_decision"            # <70% confidence

class TrustLevel(Enum):
    """Trust levels for gradual automation"""
    INITIAL = "initial"        # All decisions require approval
    LEARNING = "learning"      # High confidence decisions automated
    MATURE = "mature"         # Most decisions automated
    EXPERT = "expert"         # Full automation with monitoring

@dataclass
class DecisionRequest:
    """Request for human decision or approval"""
    request_id: str
    decision_type: DecisionType
    automation_level: AutomationLevel
    confidence: float
    priority: str
    title: str
    description: str
    recommendation: Dict[str, Any]
    analysis: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    expected_outcome: str
    timeline_requirement: str
    created_at: datetime
    expires_at: datetime
    context: Dict[str, Any]

@dataclass
class ApprovalResponse:
    """Response to approval request"""
    request_id: str
    status: ApprovalStatus
    approved_action: Optional[Dict[str, Any]]
    human_reasoning: str
    override_details: Optional[Dict[str, Any]]
    response_time_seconds: float
    responded_at: datetime
    human_id: str

@dataclass
class TrustMetrics:
    """Trust metrics for automation decisions"""
    total_decisions: int
    ai_recommendations_followed: int
    human_overrides: int
    successful_outcomes: int
    failed_outcomes: int
    average_confidence: float
    trust_score: float
    automation_eligibility: AutomationLevel
    last_updated: datetime

class HumanDecisionGateway:
    """
    Human-in-the-Loop Decision Gateway for ALL-USE Protocol
    
    Provides comprehensive human oversight capabilities including:
    - Decision approval workflow management
    - Confidence-based automation framework
    - Override mechanism and reasoning capture
    - Trust metrics and gradual automation progression
    - Real-time decision tracking and analytics
    """
    
    def __init__(self, initial_trust_level: TrustLevel = TrustLevel.INITIAL):
        """Initialize the human decision gateway"""
        self.logger = logging.getLogger(__name__)
        
        # Decision tracking
        self.pending_decisions: Dict[str, DecisionRequest] = {}
        self.decision_history: List[Tuple[DecisionRequest, ApprovalResponse]] = []
        
        # Trust and automation settings
        self.current_trust_level = initial_trust_level
        self.trust_metrics = TrustMetrics(
            total_decisions=0,
            ai_recommendations_followed=0,
            human_overrides=0,
            successful_outcomes=0,
            failed_outcomes=0,
            average_confidence=0.0,
            trust_score=0.0,
            automation_eligibility=AutomationLevel.HUMAN_DECISION,
            last_updated=datetime.now()
        )
        
        # Automation thresholds by trust level
        self.automation_thresholds = {
            TrustLevel.INITIAL: {
                AutomationLevel.FULL_AUTOMATION: 0.98,      # Nearly impossible
                AutomationLevel.SUPERVISED_AUTOMATION: 0.95, # Very high bar
                AutomationLevel.APPROVAL_REQUIRED: 0.80,    # Most decisions
                AutomationLevel.HUMAN_DECISION: 0.0         # Everything else
            },
            TrustLevel.LEARNING: {
                AutomationLevel.FULL_AUTOMATION: 0.92,
                AutomationLevel.SUPERVISED_AUTOMATION: 0.85,
                AutomationLevel.APPROVAL_REQUIRED: 0.70,
                AutomationLevel.HUMAN_DECISION: 0.0
            },
            TrustLevel.MATURE: {
                AutomationLevel.FULL_AUTOMATION: 0.88,
                AutomationLevel.SUPERVISED_AUTOMATION: 0.80,
                AutomationLevel.APPROVAL_REQUIRED: 0.65,
                AutomationLevel.HUMAN_DECISION: 0.0
            },
            TrustLevel.EXPERT: {
                AutomationLevel.FULL_AUTOMATION: 0.85,
                AutomationLevel.SUPERVISED_AUTOMATION: 0.75,
                AutomationLevel.APPROVAL_REQUIRED: 0.60,
                AutomationLevel.HUMAN_DECISION: 0.0
            }
        }
        
        # Decision timeouts by type
        self.decision_timeouts = {
            DecisionType.EMERGENCY_EXIT: timedelta(minutes=2),
            DecisionType.RISK_BREACH: timedelta(minutes=5),
            DecisionType.RECOVERY_STRATEGY: timedelta(minutes=15),
            DecisionType.POSITION_ADJUSTMENT: timedelta(minutes=30),
            DecisionType.TRADE_ENTRY: timedelta(hours=2),
            DecisionType.RULE_VIOLATION: timedelta(hours=1)
        }
        
        # Callback functions for different decision outcomes
        self.approval_callbacks: Dict[str, Callable] = {}
        self.rejection_callbacks: Dict[str, Callable] = {}
        
        # Background thread for monitoring expired decisions
        self._start_monitoring_thread()
        
        self.logger.info(f"Human Decision Gateway initialized with trust level: {initial_trust_level.value}")
    
    def request_decision(self, decision_type: DecisionType, recommendation: Dict[str, Any],
                        analysis: Dict[str, Any], confidence: float,
                        context: Optional[Dict[str, Any]] = None) -> str:
        """
        Request human decision or approval for an AI recommendation
        
        Args:
            decision_type: Type of decision being requested
            recommendation: AI recommendation details
            analysis: Analysis supporting the recommendation
            confidence: Confidence level in the recommendation (0.0-1.0)
            context: Additional context for the decision
            
        Returns:
            str: Request ID for tracking the decision
        """
        try:
            # Generate unique request ID
            request_id = str(uuid.uuid4())
            
            # Determine automation level based on confidence and trust
            automation_level = self._determine_automation_level(confidence, decision_type)
            
            # Calculate priority
            priority = self._calculate_priority(decision_type, confidence, analysis)
            
            # Generate decision request
            request = DecisionRequest(
                request_id=request_id,
                decision_type=decision_type,
                automation_level=automation_level,
                confidence=confidence,
                priority=priority,
                title=self._generate_decision_title(decision_type, recommendation),
                description=self._generate_decision_description(decision_type, recommendation, analysis),
                recommendation=recommendation,
                analysis=analysis,
                alternatives=self._generate_alternatives(recommendation, analysis),
                risk_assessment=self._extract_risk_assessment(analysis),
                expected_outcome=analysis.get('expected_outcome', 'Positive outcome expected'),
                timeline_requirement=self._get_timeline_requirement(decision_type),
                created_at=datetime.now(),
                expires_at=datetime.now() + self.decision_timeouts[decision_type],
                context=context or {}
            )
            
            # Handle based on automation level
            if automation_level == AutomationLevel.FULL_AUTOMATION:
                # Execute automatically and notify
                self._execute_automatic_decision(request)
                return request_id
            
            elif automation_level == AutomationLevel.SUPERVISED_AUTOMATION:
                # Execute with override window
                self._execute_supervised_decision(request)
                return request_id
            
            else:
                # Require human approval
                self.pending_decisions[request_id] = request
                self._notify_human_decision_required(request)
                
                self.logger.info(f"Decision requested: {request_id} ({decision_type.value}, {automation_level.value})")
                return request_id
                
        except Exception as e:
            self.logger.error(f"Error requesting decision: {str(e)}")
            raise
    
    def provide_approval(self, request_id: str, approved: bool, 
                        human_reasoning: str, human_id: str = "user",
                        override_details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Provide human approval or rejection for a pending decision
        
        Args:
            request_id: ID of the decision request
            approved: Whether the decision is approved
            human_reasoning: Human reasoning for the decision
            human_id: ID of the human making the decision
            override_details: Details if overriding the AI recommendation
            
        Returns:
            bool: Success of approval processing
        """
        try:
            if request_id not in self.pending_decisions:
                self.logger.warning(f"Decision request {request_id} not found or already processed")
                return False
            
            request = self.pending_decisions[request_id]
            
            # Create approval response
            response = ApprovalResponse(
                request_id=request_id,
                status=ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED,
                approved_action=request.recommendation if approved else override_details,
                human_reasoning=human_reasoning,
                override_details=override_details,
                response_time_seconds=(datetime.now() - request.created_at).total_seconds(),
                responded_at=datetime.now(),
                human_id=human_id
            )
            
            # Process the decision
            self._process_approval_response(request, response)
            
            # Remove from pending
            del self.pending_decisions[request_id]
            
            # Update trust metrics
            self._update_trust_metrics(request, response)
            
            self.logger.info(f"Decision processed: {request_id} ({'approved' if approved else 'rejected'})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing approval: {str(e)}")
            return False
    
    def get_pending_decisions(self, decision_type: Optional[DecisionType] = None) -> List[DecisionRequest]:
        """Get list of pending decisions, optionally filtered by type"""
        if decision_type:
            return [req for req in self.pending_decisions.values() if req.decision_type == decision_type]
        return list(self.pending_decisions.values())
    
    def get_trust_metrics(self) -> TrustMetrics:
        """Get current trust metrics"""
        return self.trust_metrics
    
    def update_outcome(self, request_id: str, successful: bool, outcome_details: Dict[str, Any]):
        """Update the outcome of a previously processed decision"""
        try:
            # Find the decision in history
            for request, response in self.decision_history:
                if request.request_id == request_id:
                    # Update trust metrics based on outcome
                    if successful:
                        self.trust_metrics.successful_outcomes += 1
                    else:
                        self.trust_metrics.failed_outcomes += 1
                    
                    # Recalculate trust score
                    self._recalculate_trust_score()
                    
                    self.logger.info(f"Outcome updated for {request_id}: {'successful' if successful else 'failed'}")
                    return
            
            self.logger.warning(f"Decision {request_id} not found in history")
            
        except Exception as e:
            self.logger.error(f"Error updating outcome: {str(e)}")
    
    def _determine_automation_level(self, confidence: float, decision_type: DecisionType) -> AutomationLevel:
        """Determine automation level based on confidence and trust"""
        thresholds = self.automation_thresholds[self.current_trust_level]
        
        # Emergency decisions always require human oversight initially
        if decision_type in [DecisionType.EMERGENCY_EXIT, DecisionType.RISK_BREACH]:
            if self.current_trust_level in [TrustLevel.INITIAL, TrustLevel.LEARNING]:
                return AutomationLevel.HUMAN_DECISION
        
        # Check confidence against thresholds
        if confidence >= thresholds[AutomationLevel.FULL_AUTOMATION]:
            return AutomationLevel.FULL_AUTOMATION
        elif confidence >= thresholds[AutomationLevel.SUPERVISED_AUTOMATION]:
            return AutomationLevel.SUPERVISED_AUTOMATION
        elif confidence >= thresholds[AutomationLevel.APPROVAL_REQUIRED]:
            return AutomationLevel.APPROVAL_REQUIRED
        else:
            return AutomationLevel.HUMAN_DECISION
    
    def _calculate_priority(self, decision_type: DecisionType, confidence: float, 
                          analysis: Dict[str, Any]) -> str:
        """Calculate priority level for the decision"""
        # Emergency decisions are always critical
        if decision_type in [DecisionType.EMERGENCY_EXIT, DecisionType.RISK_BREACH]:
            return "critical"
        
        # Recovery strategies are high priority
        if decision_type == DecisionType.RECOVERY_STRATEGY:
            return "high"
        
        # Low confidence decisions get higher priority
        if confidence < 0.7:
            return "high"
        
        # Check for high-impact decisions
        impact = analysis.get('financial_impact', 0)
        if impact > 5000:  # $5000+ impact
            return "high"
        
        return "medium"
    
    def _generate_decision_title(self, decision_type: DecisionType, recommendation: Dict[str, Any]) -> str:
        """Generate human-readable title for the decision"""
        titles = {
            DecisionType.TRADE_ENTRY: f"New Trade Entry: {recommendation.get('action', 'Unknown')} {recommendation.get('symbol', 'Unknown')}",
            DecisionType.POSITION_ADJUSTMENT: f"Position Adjustment: {recommendation.get('action', 'Unknown')} {recommendation.get('symbol', 'Unknown')}",
            DecisionType.RECOVERY_STRATEGY: f"Recovery Strategy: {recommendation.get('strategy', 'Unknown')} for {recommendation.get('symbol', 'Unknown')}",
            DecisionType.RISK_BREACH: f"Risk Breach Alert: {recommendation.get('breach_type', 'Unknown')}",
            DecisionType.RULE_VIOLATION: f"Rule Violation: {recommendation.get('rule_type', 'Unknown')}",
            DecisionType.EMERGENCY_EXIT: f"Emergency Exit: {recommendation.get('symbol', 'Unknown')}"
        }
        
        return titles.get(decision_type, f"Decision Required: {decision_type.value}")
    
    def _generate_decision_description(self, decision_type: DecisionType, 
                                     recommendation: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate detailed description for the decision"""
        base_description = f"AI recommends: {recommendation.get('action', 'Unknown action')}"
        
        # Add confidence and rationale
        confidence = analysis.get('confidence', 0)
        rationale = analysis.get('rationale', 'No rationale provided')
        
        description = f"{base_description}\n\nConfidence: {confidence:.1%}\nRationale: {rationale}"
        
        # Add financial impact if available
        impact = analysis.get('financial_impact', 0)
        if impact:
            description += f"\nEstimated Financial Impact: ${impact:,.2f}"
        
        # Add risk assessment
        risk = analysis.get('risk_level', 'Unknown')
        description += f"\nRisk Level: {risk}"
        
        return description
    
    def _generate_alternatives(self, recommendation: Dict[str, Any], 
                             analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative actions for human consideration"""
        alternatives = analysis.get('alternatives', [])
        
        # Always include "Do Nothing" as an alternative
        alternatives.append({
            'action': 'do_nothing',
            'description': 'Take no action and continue monitoring',
            'risk_level': 'low',
            'expected_outcome': 'Maintain current position'
        })
        
        # Add "Emergency Exit" for risky situations
        if analysis.get('risk_level') in ['high', 'critical']:
            alternatives.append({
                'action': 'emergency_exit',
                'description': 'Close position immediately to limit losses',
                'risk_level': 'low',
                'expected_outcome': 'Crystallize current loss but prevent further losses'
            })
        
        return alternatives
    
    def _extract_risk_assessment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format risk assessment"""
        return {
            'risk_level': analysis.get('risk_level', 'medium'),
            'max_loss': analysis.get('max_loss', 0),
            'probability_of_loss': analysis.get('probability_of_loss', 0.5),
            'risk_factors': analysis.get('risk_factors', []),
            'mitigation_strategies': analysis.get('mitigation_strategies', [])
        }
    
    def _get_timeline_requirement(self, decision_type: DecisionType) -> str:
        """Get human-readable timeline requirement"""
        timeout = self.decision_timeouts[decision_type]
        
        if timeout.total_seconds() < 300:  # Less than 5 minutes
            return "Immediate response required"
        elif timeout.total_seconds() < 3600:  # Less than 1 hour
            return f"Response required within {int(timeout.total_seconds() / 60)} minutes"
        else:
            return f"Response required within {int(timeout.total_seconds() / 3600)} hours"
    
    def _execute_automatic_decision(self, request: DecisionRequest):
        """Execute decision automatically and notify human"""
        response = ApprovalResponse(
            request_id=request.request_id,
            status=ApprovalStatus.AUTO_APPROVED,
            approved_action=request.recommendation,
            human_reasoning="Automatically approved based on high confidence",
            override_details=None,
            response_time_seconds=0.0,
            responded_at=datetime.now(),
            human_id="system"
        )
        
        self._process_approval_response(request, response)
        self._update_trust_metrics(request, response)
        
        # Notify human of automatic execution
        self._notify_automatic_execution(request)
    
    def _execute_supervised_decision(self, request: DecisionRequest):
        """Execute decision with human override window"""
        # Add to pending with shorter timeout for override
        self.pending_decisions[request.request_id] = request
        
        # Execute the decision
        self._process_approval_response(request, ApprovalResponse(
            request_id=request.request_id,
            status=ApprovalStatus.AUTO_APPROVED,
            approved_action=request.recommendation,
            human_reasoning="Supervised automation - executed with override window",
            override_details=None,
            response_time_seconds=0.0,
            responded_at=datetime.now(),
            human_id="system"
        ))
        
        # Notify human with override option
        self._notify_supervised_execution(request)
        
        # Set shorter timeout for override
        request.expires_at = datetime.now() + timedelta(minutes=5)
    
    def _process_approval_response(self, request: DecisionRequest, response: ApprovalResponse):
        """Process the approval response and execute callbacks"""
        # Add to history
        self.decision_history.append((request, response))
        
        # Execute appropriate callback
        if response.status == ApprovalStatus.APPROVED or response.status == ApprovalStatus.AUTO_APPROVED:
            callback = self.approval_callbacks.get(request.request_id)
            if callback:
                try:
                    callback(request, response)
                except Exception as e:
                    self.logger.error(f"Error executing approval callback: {str(e)}")
        else:
            callback = self.rejection_callbacks.get(request.request_id)
            if callback:
                try:
                    callback(request, response)
                except Exception as e:
                    self.logger.error(f"Error executing rejection callback: {str(e)}")
    
    def _update_trust_metrics(self, request: DecisionRequest, response: ApprovalResponse):
        """Update trust metrics based on decision outcome"""
        self.trust_metrics.total_decisions += 1
        
        if response.status in [ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED]:
            if response.override_details is None:
                self.trust_metrics.ai_recommendations_followed += 1
            else:
                self.trust_metrics.human_overrides += 1
        else:
            self.trust_metrics.human_overrides += 1
        
        # Update average confidence
        total_confidence = self.trust_metrics.average_confidence * (self.trust_metrics.total_decisions - 1)
        self.trust_metrics.average_confidence = (total_confidence + request.confidence) / self.trust_metrics.total_decisions
        
        # Recalculate trust score
        self._recalculate_trust_score()
        
        # Update automation eligibility
        self._update_automation_eligibility()
        
        self.trust_metrics.last_updated = datetime.now()
    
    def _recalculate_trust_score(self):
        """Recalculate overall trust score"""
        if self.trust_metrics.total_decisions == 0:
            self.trust_metrics.trust_score = 0.0
            return
        
        # Base trust on AI recommendation follow rate
        follow_rate = self.trust_metrics.ai_recommendations_followed / self.trust_metrics.total_decisions
        
        # Adjust for success rate
        total_outcomes = self.trust_metrics.successful_outcomes + self.trust_metrics.failed_outcomes
        if total_outcomes > 0:
            success_rate = self.trust_metrics.successful_outcomes / total_outcomes
            trust_score = (follow_rate * 0.6) + (success_rate * 0.4)
        else:
            trust_score = follow_rate * 0.8  # Conservative without outcome data
        
        # Adjust for average confidence
        confidence_factor = min(1.0, self.trust_metrics.average_confidence / 0.8)
        trust_score *= confidence_factor
        
        self.trust_metrics.trust_score = min(1.0, trust_score)
    
    def _update_automation_eligibility(self):
        """Update automation eligibility based on trust metrics"""
        trust_score = self.trust_metrics.trust_score
        total_decisions = self.trust_metrics.total_decisions
        
        # Require minimum number of decisions for automation
        if total_decisions < 10:
            self.trust_metrics.automation_eligibility = AutomationLevel.HUMAN_DECISION
        elif total_decisions < 50:
            if trust_score > 0.8:
                self.trust_metrics.automation_eligibility = AutomationLevel.APPROVAL_REQUIRED
            else:
                self.trust_metrics.automation_eligibility = AutomationLevel.HUMAN_DECISION
        else:
            if trust_score > 0.9:
                self.trust_metrics.automation_eligibility = AutomationLevel.FULL_AUTOMATION
            elif trust_score > 0.8:
                self.trust_metrics.automation_eligibility = AutomationLevel.SUPERVISED_AUTOMATION
            elif trust_score > 0.7:
                self.trust_metrics.automation_eligibility = AutomationLevel.APPROVAL_REQUIRED
            else:
                self.trust_metrics.automation_eligibility = AutomationLevel.HUMAN_DECISION
    
    def _notify_human_decision_required(self, request: DecisionRequest):
        """Notify human that a decision is required"""
        # In a real implementation, this would send notifications via email, SMS, UI, etc.
        self.logger.info(f"HUMAN DECISION REQUIRED: {request.title}")
        self.logger.info(f"Priority: {request.priority}, Expires: {request.expires_at}")
        self.logger.info(f"Description: {request.description}")
    
    def _notify_automatic_execution(self, request: DecisionRequest):
        """Notify human of automatic execution"""
        self.logger.info(f"AUTOMATIC EXECUTION: {request.title}")
        self.logger.info(f"Confidence: {request.confidence:.1%}, Action: {request.recommendation}")
    
    def _notify_supervised_execution(self, request: DecisionRequest):
        """Notify human of supervised execution with override option"""
        self.logger.info(f"SUPERVISED EXECUTION: {request.title}")
        self.logger.info(f"Action executed with 5-minute override window")
        self.logger.info(f"Confidence: {request.confidence:.1%}")
    
    def _start_monitoring_thread(self):
        """Start background thread to monitor expired decisions"""
        def monitor_expired_decisions():
            while True:
                try:
                    current_time = datetime.now()
                    expired_requests = []
                    
                    for request_id, request in self.pending_decisions.items():
                        if current_time > request.expires_at:
                            expired_requests.append(request_id)
                    
                    for request_id in expired_requests:
                        request = self.pending_decisions[request_id]
                        
                        # Create expired response
                        response = ApprovalResponse(
                            request_id=request_id,
                            status=ApprovalStatus.EXPIRED,
                            approved_action=None,
                            human_reasoning="Decision expired without human response",
                            override_details=None,
                            response_time_seconds=(current_time - request.created_at).total_seconds(),
                            responded_at=current_time,
                            human_id="system"
                        )
                        
                        # Process expired decision
                        self._process_approval_response(request, response)
                        del self.pending_decisions[request_id]
                        
                        self.logger.warning(f"Decision expired: {request_id}")
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring thread: {str(e)}")
                    time.sleep(60)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitor_expired_decisions, daemon=True)
        monitor_thread.start()

def test_human_decision_gateway():
    """Test the human decision gateway"""
    print("Testing Human Decision Gateway...")
    
    gateway = HumanDecisionGateway(TrustLevel.INITIAL)
    
    # Test decision requests
    test_decisions = [
        {
            'type': DecisionType.TRADE_ENTRY,
            'recommendation': {
                'action': 'sell_put',
                'symbol': 'SPY',
                'quantity': 10,
                'strike': 450.0,
                'expiration': '2025-02-21'
            },
            'analysis': {
                'confidence': 0.85,
                'rationale': 'High probability trade based on week classification P-EW',
                'expected_outcome': '2.5% return over 35 days',
                'risk_level': 'medium',
                'financial_impact': 2500,
                'alternatives': [
                    {'action': 'sell_call', 'description': 'Sell call instead of put'}
                ]
            },
            'confidence': 0.85
        },
        {
            'type': DecisionType.RECOVERY_STRATEGY,
            'recommendation': {
                'strategy': 'roll_out',
                'symbol': 'TSLA',
                'current_loss': -2000,
                'recovery_target': -500
            },
            'analysis': {
                'confidence': 0.65,
                'rationale': 'Position showing 25% loss, rolling out may recover 75% of loss',
                'expected_outcome': 'Recover to -5% loss over 30 days',
                'risk_level': 'high',
                'financial_impact': -2000,
                'max_loss': -3000,
                'probability_of_loss': 0.3
            },
            'confidence': 0.65
        },
        {
            'type': DecisionType.RISK_BREACH,
            'recommendation': {
                'action': 'reduce_position_size',
                'breach_type': 'portfolio_risk_exceeded',
                'current_risk': 0.15,
                'target_risk': 0.10
            },
            'analysis': {
                'confidence': 0.95,
                'rationale': 'Portfolio risk exceeded 15% threshold, immediate action required',
                'expected_outcome': 'Reduce risk to acceptable levels',
                'risk_level': 'critical',
                'financial_impact': 0
            },
            'confidence': 0.95
        }
    ]
    
    # Submit decision requests
    request_ids = []
    for decision in test_decisions:
        request_id = gateway.request_decision(
            decision['type'],
            decision['recommendation'],
            decision['analysis'],
            decision['confidence']
        )
        request_ids.append(request_id)
        print(f"Submitted decision: {request_id} ({decision['type'].value})")
    
    # Check pending decisions
    print(f"\nPending decisions: {len(gateway.get_pending_decisions())}")
    
    # Simulate human approvals
    for i, request_id in enumerate(request_ids):
        if i == 0:  # Approve first decision
            success = gateway.provide_approval(
                request_id, 
                approved=True,
                human_reasoning="Agree with AI recommendation, good risk/reward ratio",
                human_id="trader_1"
            )
            print(f"Approved decision {request_id}: {success}")
        
        elif i == 1:  # Reject second decision with override
            success = gateway.provide_approval(
                request_id,
                approved=False,
                human_reasoning="Prefer to close position instead of rolling",
                human_id="trader_1",
                override_details={'action': 'close_position', 'reason': 'Cut losses early'}
            )
            print(f"Rejected decision {request_id} with override: {success}")
        
        # Leave third decision pending to test automation
    
    # Check trust metrics
    print(f"\n--- Trust Metrics ---")
    metrics = gateway.get_trust_metrics()
    print(f"Total Decisions: {metrics.total_decisions}")
    print(f"AI Recommendations Followed: {metrics.ai_recommendations_followed}")
    print(f"Human Overrides: {metrics.human_overrides}")
    print(f"Trust Score: {metrics.trust_score:.2f}")
    print(f"Automation Eligibility: {metrics.automation_eligibility.value}")
    
    # Simulate successful outcomes
    for request_id in request_ids[:2]:
        gateway.update_outcome(request_id, successful=True, outcome_details={'profit': 500})
    
    # Check updated trust metrics
    print(f"\n--- Updated Trust Metrics ---")
    metrics = gateway.get_trust_metrics()
    print(f"Successful Outcomes: {metrics.successful_outcomes}")
    print(f"Trust Score: {metrics.trust_score:.2f}")
    print(f"Automation Eligibility: {metrics.automation_eligibility.value}")
    
    print("\n✅ Human Decision Gateway test completed successfully!")

if __name__ == "__main__":
    test_human_decision_gateway()

