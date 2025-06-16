"""
Advanced Trading Protocol Rules Engine for ALL-USE Protocol
Implements sophisticated rule-based trading logic with account-specific constraints and intelligent decision making

This module provides comprehensive rule-based trading logic that enforces
account-specific constraints, validates trading decisions, and ensures
compliance with ALL-USE protocol requirements.
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

class AccountType(Enum):
    """Account types with specific trading rules"""
    GEN_ACC = "GEN_ACC"  # General Account
    REV_ACC = "REV_ACC"  # Revenue Account  
    COM_ACC = "COM_ACC"  # Commercial Account

class RuleType(Enum):
    """Types of trading rules"""
    DELTA_RANGE = "delta_range"
    POSITION_SIZE = "position_size"
    TIME_CONSTRAINT = "time_constraint"
    MARKET_CONDITION = "market_condition"
    RISK_LIMIT = "risk_limit"
    CORRELATION_LIMIT = "correlation_limit"
    VOLATILITY_CONSTRAINT = "volatility_constraint"
    CONCENTRATION_LIMIT = "concentration_limit"

class RulePriority(Enum):
    """Rule priority levels"""
    CRITICAL = "critical"      # Must be enforced
    HIGH = "high"             # Should be enforced
    MEDIUM = "medium"         # Preferred enforcement
    LOW = "low"               # Optional enforcement
    ADVISORY = "advisory"     # Information only

class RuleStatus(Enum):
    """Rule validation status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

@dataclass
class RuleViolation:
    """Rule violation details"""
    rule_id: str
    rule_type: RuleType
    priority: RulePriority
    description: str
    current_value: Any
    required_value: Any
    severity: str
    recommendation: str
    can_override: bool

@dataclass
class RuleValidationResult:
    """Result of rule validation"""
    rule_id: str
    status: RuleStatus
    passed: bool
    message: str
    violation: Optional[RuleViolation] = None
    execution_time_ms: float = 0.0

@dataclass
class TradingDecision:
    """Trading decision with rule validation"""
    action: str
    symbol: str
    quantity: int
    delta: float
    expiration: datetime
    strike: float
    account_type: AccountType
    market_conditions: Dict[str, Any]
    week_classification: str
    confidence: float
    expected_return: float
    max_risk: float

class TradingRule(ABC):
    """Abstract base class for trading rules"""
    
    def __init__(self, rule_id: str, rule_type: RuleType, priority: RulePriority, 
                 description: str, enabled: bool = True):
        self.rule_id = rule_id
        self.rule_type = rule_type
        self.priority = priority
        self.description = description
        self.enabled = enabled
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.execution_count = 0
        self.violation_count = 0
    
    @abstractmethod
    def validate(self, decision: TradingDecision, context: Dict[str, Any]) -> RuleValidationResult:
        """Validate a trading decision against this rule"""
        pass
    
    def update_stats(self, result: RuleValidationResult):
        """Update rule execution statistics"""
        self.execution_count += 1
        if result.violation:
            self.violation_count += 1
        self.last_updated = datetime.now()

class DeltaRangeRule(TradingRule):
    """Rule for enforcing account-specific delta ranges"""
    
    def __init__(self, account_type: AccountType):
        # Account-specific delta ranges
        delta_ranges = {
            AccountType.GEN_ACC: (40, 50),  # 40-50 delta
            AccountType.REV_ACC: (30, 40),  # 30-40 delta
            AccountType.COM_ACC: (20, 30)   # 20-30 delta
        }
        
        self.min_delta, self.max_delta = delta_ranges[account_type]
        
        super().__init__(
            rule_id=f"delta_range_{account_type.value}",
            rule_type=RuleType.DELTA_RANGE,
            priority=RulePriority.CRITICAL,
            description=f"Delta must be between {self.min_delta}-{self.max_delta} for {account_type.value}"
        )
        self.account_type = account_type
    
    def validate(self, decision: TradingDecision, context: Dict[str, Any]) -> RuleValidationResult:
        """Validate delta range for the account type"""
        start_time = datetime.now()
        
        try:
            # Check if rule applies to this account type
            if decision.account_type != self.account_type:
                return RuleValidationResult(
                    rule_id=self.rule_id,
                    status=RuleStatus.SKIPPED,
                    passed=True,
                    message=f"Rule skipped - different account type ({decision.account_type.value})",
                    execution_time_ms=0.0
                )
            
            # Validate delta range
            delta = abs(decision.delta)  # Use absolute delta
            
            if self.min_delta <= delta <= self.max_delta:
                result = RuleValidationResult(
                    rule_id=self.rule_id,
                    status=RuleStatus.PASSED,
                    passed=True,
                    message=f"Delta {delta} is within allowed range {self.min_delta}-{self.max_delta}"
                )
            else:
                violation = RuleViolation(
                    rule_id=self.rule_id,
                    rule_type=self.rule_type,
                    priority=self.priority,
                    description=f"Delta {delta} outside allowed range {self.min_delta}-{self.max_delta}",
                    current_value=delta,
                    required_value=f"{self.min_delta}-{self.max_delta}",
                    severity="critical",
                    recommendation=f"Adjust delta to be within {self.min_delta}-{self.max_delta} range",
                    can_override=False
                )
                
                result = RuleValidationResult(
                    rule_id=self.rule_id,
                    status=RuleStatus.FAILED,
                    passed=False,
                    message=f"Delta {delta} violates range {self.min_delta}-{self.max_delta}",
                    violation=violation
                )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            self.update_stats(result)
            return result
            
        except Exception as e:
            logger.error(f"Error validating delta range rule: {str(e)}")
            return RuleValidationResult(
                rule_id=self.rule_id,
                status=RuleStatus.FAILED,
                passed=False,
                message=f"Rule validation error: {str(e)}"
            )

class PositionSizeRule(TradingRule):
    """Rule for enforcing position size limits"""
    
    def __init__(self, max_position_percentage: float = 0.1, max_single_position: float = 50000):
        super().__init__(
            rule_id="position_size_limit",
            rule_type=RuleType.POSITION_SIZE,
            priority=RulePriority.HIGH,
            description=f"Position size must not exceed {max_position_percentage:.1%} of portfolio or ${max_single_position:,.0f}"
        )
        self.max_position_percentage = max_position_percentage
        self.max_single_position = max_single_position
    
    def validate(self, decision: TradingDecision, context: Dict[str, Any]) -> RuleValidationResult:
        """Validate position size limits"""
        start_time = datetime.now()
        
        try:
            portfolio_value = context.get('portfolio_value', 100000)
            position_value = decision.quantity * decision.strike
            
            max_allowed_by_percentage = portfolio_value * self.max_position_percentage
            max_allowed = min(max_allowed_by_percentage, self.max_single_position)
            
            if position_value <= max_allowed:
                result = RuleValidationResult(
                    rule_id=self.rule_id,
                    status=RuleStatus.PASSED,
                    passed=True,
                    message=f"Position size ${position_value:,.0f} is within limit ${max_allowed:,.0f}"
                )
            else:
                violation = RuleViolation(
                    rule_id=self.rule_id,
                    rule_type=self.rule_type,
                    priority=self.priority,
                    description=f"Position size ${position_value:,.0f} exceeds limit ${max_allowed:,.0f}",
                    current_value=position_value,
                    required_value=max_allowed,
                    severity="high",
                    recommendation=f"Reduce position size to ${max_allowed:,.0f} or less",
                    can_override=True
                )
                
                result = RuleValidationResult(
                    rule_id=self.rule_id,
                    status=RuleStatus.FAILED,
                    passed=False,
                    message=f"Position size ${position_value:,.0f} exceeds limit ${max_allowed:,.0f}",
                    violation=violation
                )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            self.update_stats(result)
            return result
            
        except Exception as e:
            logger.error(f"Error validating position size rule: {str(e)}")
            return RuleValidationResult(
                rule_id=self.rule_id,
                status=RuleStatus.FAILED,
                passed=False,
                message=f"Rule validation error: {str(e)}"
            )

class TimeConstraintRule(TradingRule):
    """Rule for enforcing time-based constraints"""
    
    def __init__(self, min_dte: int = 20, max_dte: int = 60):
        super().__init__(
            rule_id="time_constraint",
            rule_type=RuleType.TIME_CONSTRAINT,
            priority=RulePriority.MEDIUM,
            description=f"Days to expiration must be between {min_dte}-{max_dte} days"
        )
        self.min_dte = min_dte
        self.max_dte = max_dte
    
    def validate(self, decision: TradingDecision, context: Dict[str, Any]) -> RuleValidationResult:
        """Validate time constraints"""
        start_time = datetime.now()
        
        try:
            current_date = datetime.now().date()
            expiration_date = decision.expiration.date()
            dte = (expiration_date - current_date).days
            
            if self.min_dte <= dte <= self.max_dte:
                result = RuleValidationResult(
                    rule_id=self.rule_id,
                    status=RuleStatus.PASSED,
                    passed=True,
                    message=f"DTE {dte} is within allowed range {self.min_dte}-{self.max_dte}"
                )
            else:
                severity = "medium" if dte < self.min_dte else "low"
                can_override = dte >= 10  # Can override if at least 10 days
                
                violation = RuleViolation(
                    rule_id=self.rule_id,
                    rule_type=self.rule_type,
                    priority=self.priority,
                    description=f"DTE {dte} outside preferred range {self.min_dte}-{self.max_dte}",
                    current_value=dte,
                    required_value=f"{self.min_dte}-{self.max_dte}",
                    severity=severity,
                    recommendation=f"Consider expiration between {self.min_dte}-{self.max_dte} days",
                    can_override=can_override
                )
                
                result = RuleValidationResult(
                    rule_id=self.rule_id,
                    status=RuleStatus.WARNING if can_override else RuleStatus.FAILED,
                    passed=can_override,
                    message=f"DTE {dte} outside preferred range {self.min_dte}-{self.max_dte}",
                    violation=violation
                )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            self.update_stats(result)
            return result
            
        except Exception as e:
            logger.error(f"Error validating time constraint rule: {str(e)}")
            return RuleValidationResult(
                rule_id=self.rule_id,
                status=RuleStatus.FAILED,
                passed=False,
                message=f"Rule validation error: {str(e)}"
            )

class MarketConditionRule(TradingRule):
    """Rule for market condition constraints"""
    
    def __init__(self):
        super().__init__(
            rule_id="market_condition_constraint",
            rule_type=RuleType.MARKET_CONDITION,
            priority=RulePriority.MEDIUM,
            description="Trading decisions must align with market conditions"
        )
        
        # Define allowed actions by market condition
        self.allowed_actions = {
            'extremely_bullish': ['sell_put', 'sell_call'],
            'bullish': ['sell_put', 'sell_call'],
            'neutral_bullish': ['sell_put', 'sell_call'],
            'neutral': ['sell_put', 'sell_call'],
            'neutral_bearish': ['sell_put', 'roll_position'],
            'bearish': ['roll_position', 'close_position'],
            'extremely_bearish': ['close_position', 'enter_protective']
        }
    
    def validate(self, decision: TradingDecision, context: Dict[str, Any]) -> RuleValidationResult:
        """Validate market condition alignment"""
        start_time = datetime.now()
        
        try:
            market_condition = decision.market_conditions.get('condition', 'neutral')
            allowed_actions = self.allowed_actions.get(market_condition, ['sell_put', 'sell_call'])
            
            if decision.action in allowed_actions:
                result = RuleValidationResult(
                    rule_id=self.rule_id,
                    status=RuleStatus.PASSED,
                    passed=True,
                    message=f"Action '{decision.action}' is appropriate for {market_condition} market"
                )
            else:
                violation = RuleViolation(
                    rule_id=self.rule_id,
                    rule_type=self.rule_type,
                    priority=self.priority,
                    description=f"Action '{decision.action}' may not be optimal for {market_condition} market",
                    current_value=decision.action,
                    required_value=allowed_actions,
                    severity="medium",
                    recommendation=f"Consider actions: {', '.join(allowed_actions)}",
                    can_override=True
                )
                
                result = RuleValidationResult(
                    rule_id=self.rule_id,
                    status=RuleStatus.WARNING,
                    passed=True,  # Warning, not failure
                    message=f"Action '{decision.action}' may not be optimal for {market_condition} market",
                    violation=violation
                )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            self.update_stats(result)
            return result
            
        except Exception as e:
            logger.error(f"Error validating market condition rule: {str(e)}")
            return RuleValidationResult(
                rule_id=self.rule_id,
                status=RuleStatus.FAILED,
                passed=False,
                message=f"Rule validation error: {str(e)}"
            )

class RiskLimitRule(TradingRule):
    """Rule for enforcing risk limits"""
    
    def __init__(self, max_portfolio_risk: float = 0.15, max_position_risk: float = 0.05):
        super().__init__(
            rule_id="risk_limit",
            rule_type=RuleType.RISK_LIMIT,
            priority=RulePriority.CRITICAL,
            description=f"Portfolio risk must not exceed {max_portfolio_risk:.1%}, position risk {max_position_risk:.1%}"
        )
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
    
    def validate(self, decision: TradingDecision, context: Dict[str, Any]) -> RuleValidationResult:
        """Validate risk limits"""
        start_time = datetime.now()
        
        try:
            position_risk = decision.max_risk
            portfolio_risk = context.get('portfolio_risk', 0.0)
            
            # Check position risk
            if position_risk > self.max_position_risk:
                violation = RuleViolation(
                    rule_id=self.rule_id,
                    rule_type=self.rule_type,
                    priority=self.priority,
                    description=f"Position risk {position_risk:.1%} exceeds limit {self.max_position_risk:.1%}",
                    current_value=position_risk,
                    required_value=self.max_position_risk,
                    severity="critical",
                    recommendation=f"Reduce position risk to {self.max_position_risk:.1%} or less",
                    can_override=False
                )
                
                result = RuleValidationResult(
                    rule_id=self.rule_id,
                    status=RuleStatus.FAILED,
                    passed=False,
                    message=f"Position risk {position_risk:.1%} exceeds limit {self.max_position_risk:.1%}",
                    violation=violation
                )
            
            # Check portfolio risk
            elif portfolio_risk > self.max_portfolio_risk:
                violation = RuleViolation(
                    rule_id=self.rule_id,
                    rule_type=self.rule_type,
                    priority=self.priority,
                    description=f"Portfolio risk {portfolio_risk:.1%} exceeds limit {self.max_portfolio_risk:.1%}",
                    current_value=portfolio_risk,
                    required_value=self.max_portfolio_risk,
                    severity="critical",
                    recommendation=f"Reduce portfolio risk to {self.max_portfolio_risk:.1%} or less",
                    can_override=False
                )
                
                result = RuleValidationResult(
                    rule_id=self.rule_id,
                    status=RuleStatus.FAILED,
                    passed=False,
                    message=f"Portfolio risk {portfolio_risk:.1%} exceeds limit {self.max_portfolio_risk:.1%}",
                    violation=violation
                )
            else:
                result = RuleValidationResult(
                    rule_id=self.rule_id,
                    status=RuleStatus.PASSED,
                    passed=True,
                    message=f"Risk levels within limits: position {position_risk:.1%}, portfolio {portfolio_risk:.1%}"
                )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            self.update_stats(result)
            return result
            
        except Exception as e:
            logger.error(f"Error validating risk limit rule: {str(e)}")
            return RuleValidationResult(
                rule_id=self.rule_id,
                status=RuleStatus.FAILED,
                passed=False,
                message=f"Rule validation error: {str(e)}"
            )

class TradingProtocolRulesEngine:
    """
    Advanced Trading Protocol Rules Engine for ALL-USE Protocol
    
    Provides sophisticated rule-based trading logic with:
    - Account-specific delta range enforcement
    - Position size and risk limit validation
    - Market condition alignment checking
    - Time constraint enforcement
    - Rule hierarchy and conflict resolution
    """
    
    def __init__(self):
        """Initialize the trading protocol rules engine"""
        self.logger = logging.getLogger(__name__)
        
        # Rule registry
        self.rules: Dict[str, TradingRule] = {}
        self.rule_groups: Dict[str, List[str]] = {}
        
        # Rule execution statistics
        self.total_validations = 0
        self.total_violations = 0
        self.rule_performance = {}
        
        # Initialize default rules
        self._initialize_default_rules()
        
        self.logger.info("Trading Protocol Rules Engine initialized")
    
    def _initialize_default_rules(self):
        """Initialize default trading rules"""
        # Delta range rules for each account type
        for account_type in AccountType:
            rule = DeltaRangeRule(account_type)
            self.add_rule(rule)
        
        # Position size rule
        position_size_rule = PositionSizeRule()
        self.add_rule(position_size_rule)
        
        # Time constraint rule
        time_constraint_rule = TimeConstraintRule()
        self.add_rule(time_constraint_rule)
        
        # Market condition rule
        market_condition_rule = MarketConditionRule()
        self.add_rule(market_condition_rule)
        
        # Risk limit rule
        risk_limit_rule = RiskLimitRule()
        self.add_rule(risk_limit_rule)
        
        # Create rule groups
        self.rule_groups['critical'] = [rule_id for rule_id, rule in self.rules.items() 
                                       if rule.priority == RulePriority.CRITICAL]
        self.rule_groups['high'] = [rule_id for rule_id, rule in self.rules.items() 
                                   if rule.priority == RulePriority.HIGH]
        self.rule_groups['medium'] = [rule_id for rule_id, rule in self.rules.items() 
                                     if rule.priority == RulePriority.MEDIUM]
    
    def add_rule(self, rule: TradingRule):
        """Add a new trading rule"""
        self.rules[rule.rule_id] = rule
        self.logger.info(f"Added rule: {rule.rule_id} ({rule.priority.value})")
    
    def remove_rule(self, rule_id: str):
        """Remove a trading rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.info(f"Removed rule: {rule_id}")
    
    def validate_decision(self, decision: TradingDecision, 
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate a trading decision against all applicable rules
        
        Args:
            decision: Trading decision to validate
            context: Additional context for rule validation
            
        Returns:
            Dictionary containing validation results
        """
        try:
            start_time = datetime.now()
            
            if context is None:
                context = {}
            
            # Validation results
            results = []
            violations = []
            critical_failures = []
            warnings = []
            
            # Validate against all enabled rules
            for rule_id, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                result = rule.validate(decision, context)
                results.append(result)
                
                if result.violation:
                    violations.append(result.violation)
                    
                    if result.violation.priority == RulePriority.CRITICAL:
                        critical_failures.append(result.violation)
                    elif result.status == RuleStatus.WARNING:
                        warnings.append(result.violation)
            
            # Determine overall validation status
            has_critical_failures = len(critical_failures) > 0
            has_violations = len(violations) > 0
            
            if has_critical_failures:
                overall_status = "REJECTED"
                can_proceed = False
            elif has_violations:
                overall_status = "WARNING"
                can_proceed = True  # Can proceed with warnings
            else:
                overall_status = "APPROVED"
                can_proceed = True
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update statistics
            self.total_validations += 1
            if has_violations:
                self.total_violations += 1
            
            validation_result = {
                'overall_status': overall_status,
                'can_proceed': can_proceed,
                'total_rules_checked': len([r for r in self.rules.values() if r.enabled]),
                'rules_passed': len([r for r in results if r.passed]),
                'rules_failed': len([r for r in results if not r.passed]),
                'critical_failures': len(critical_failures),
                'warnings': len(warnings),
                'violations': violations,
                'detailed_results': results,
                'execution_time_ms': execution_time,
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"Decision validation: {overall_status} ({len(violations)} violations)")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating trading decision: {str(e)}")
            return {
                'overall_status': 'ERROR',
                'can_proceed': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def get_applicable_rules(self, decision: TradingDecision) -> List[TradingRule]:
        """Get rules applicable to a specific trading decision"""
        applicable_rules = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Check if rule applies to this decision
            if isinstance(rule, DeltaRangeRule):
                if rule.account_type == decision.account_type:
                    applicable_rules.append(rule)
            else:
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get rule execution statistics"""
        rule_stats = {}
        
        for rule_id, rule in self.rules.items():
            rule_stats[rule_id] = {
                'execution_count': rule.execution_count,
                'violation_count': rule.violation_count,
                'violation_rate': rule.violation_count / rule.execution_count if rule.execution_count > 0 else 0,
                'priority': rule.priority.value,
                'enabled': rule.enabled,
                'last_updated': rule.last_updated
            }
        
        return {
            'total_validations': self.total_validations,
            'total_violations': self.total_violations,
            'overall_violation_rate': self.total_violations / self.total_validations if self.total_validations > 0 else 0,
            'rule_statistics': rule_stats
        }
    
    def update_rule_parameters(self, rule_id: str, parameters: Dict[str, Any]):
        """Update rule parameters dynamically"""
        if rule_id not in self.rules:
            raise ValueError(f"Rule {rule_id} not found")
        
        rule = self.rules[rule_id]
        
        # Update parameters based on rule type
        for param, value in parameters.items():
            if hasattr(rule, param):
                setattr(rule, param, value)
                self.logger.info(f"Updated {rule_id}.{param} = {value}")
        
        rule.last_updated = datetime.now()

def test_trading_protocol_rules_engine():
    """Test the trading protocol rules engine"""
    print("Testing Trading Protocol Rules Engine...")
    
    engine = TradingProtocolRulesEngine()
    
    # Test scenarios
    test_decisions = [
        TradingDecision(
            action="sell_put",
            symbol="SPY",
            quantity=10,
            delta=45,
            expiration=datetime.now() + timedelta(days=35),
            strike=450.0,
            account_type=AccountType.GEN_ACC,
            market_conditions={'condition': 'bullish', 'volatility': 0.18},
            week_classification='P-EW',
            confidence=0.85,
            expected_return=0.02,
            max_risk=0.04
        ),
        TradingDecision(
            action="sell_put",
            symbol="TSLA",
            quantity=5,
            delta=25,
            expiration=datetime.now() + timedelta(days=30),
            strike=250.0,
            account_type=AccountType.REV_ACC,
            market_conditions={'condition': 'neutral', 'volatility': 0.25},
            week_classification='P-AWL',
            confidence=0.80,
            expected_return=0.018,
            max_risk=0.03
        ),
        TradingDecision(
            action="sell_call",
            symbol="NVDA",
            quantity=20,
            delta=60,  # Violates GEN_ACC delta range
            expiration=datetime.now() + timedelta(days=15),  # Violates time constraint
            strike=500.0,
            account_type=AccountType.GEN_ACC,
            market_conditions={'condition': 'extremely_bearish', 'volatility': 0.40},
            week_classification='C-WAP',
            confidence=0.70,
            expected_return=0.025,
            max_risk=0.08  # Violates risk limit
        )
    ]
    
    context = {
        'portfolio_value': 100000,
        'portfolio_risk': 0.12
    }
    
    for i, decision in enumerate(test_decisions, 1):
        print(f"\n--- Test Decision {i}: {decision.action} {decision.symbol} ---")
        
        validation_result = engine.validate_decision(decision, context)
        
        print(f"Overall Status: {validation_result['overall_status']}")
        print(f"Can Proceed: {validation_result['can_proceed']}")
        print(f"Rules Checked: {validation_result['total_rules_checked']}")
        print(f"Rules Passed: {validation_result['rules_passed']}")
        print(f"Rules Failed: {validation_result['rules_failed']}")
        print(f"Critical Failures: {validation_result['critical_failures']}")
        print(f"Warnings: {validation_result['warnings']}")
        print(f"Execution Time: {validation_result['execution_time_ms']:.1f}ms")
        
        if validation_result['violations']:
            print("Violations:")
            for violation in validation_result['violations']:
                print(f"  - {violation.description} ({violation.severity})")
                print(f"    Recommendation: {violation.recommendation}")
    
    # Test rule statistics
    print("\n--- Rule Statistics ---")
    stats = engine.get_rule_statistics()
    print(f"Total Validations: {stats['total_validations']}")
    print(f"Total Violations: {stats['total_violations']}")
    print(f"Overall Violation Rate: {stats['overall_violation_rate']:.1%}")
    
    print("\nRule Performance:")
    for rule_id, rule_stats in stats['rule_statistics'].items():
        print(f"  {rule_id}: {rule_stats['execution_count']} executions, "
              f"{rule_stats['violation_count']} violations ({rule_stats['violation_rate']:.1%})")
    
    print("\n✅ Trading Protocol Rules Engine test completed successfully!")

if __name__ == "__main__":
    test_trading_protocol_rules_engine()

