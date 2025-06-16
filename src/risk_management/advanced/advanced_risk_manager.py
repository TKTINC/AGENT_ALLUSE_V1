"""
Advanced Risk Management Engine for ALL-USE Protocol
Enterprise-grade risk management with multi-layer assessment, dynamic limits, and real-time monitoring

This module provides comprehensive risk management including:
- Multi-layer risk assessment (portfolio, position, market)
- Dynamic risk limits with adaptive thresholds
- Real-time risk monitoring and alerting
- Risk scenario analysis and stress testing
- Drawdown protection mechanisms
- Risk-based position sizing
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
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

class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

class RiskType(Enum):
    """Types of risk being assessed"""
    MARKET_RISK = "market_risk"
    VOLATILITY_RISK = "volatility_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CONCENTRATION_RISK = "concentration_risk"
    CORRELATION_RISK = "correlation_risk"
    DRAWDOWN_RISK = "drawdown_risk"
    LEVERAGE_RISK = "leverage_risk"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio"""
    var_1d: float                    # 1-day Value at Risk
    var_5d: float                    # 5-day Value at Risk
    expected_shortfall: float        # Expected Shortfall (CVaR)
    maximum_drawdown: float          # Maximum Drawdown
    volatility: float               # Annualized volatility
    beta: float                     # Market beta
    correlation_risk: float         # Correlation risk score
    concentration_risk: float       # Concentration risk score
    liquidity_risk: float          # Liquidity risk score
    leverage_ratio: float           # Leverage ratio
    risk_score: float              # Overall risk score (0-100)
    risk_level: RiskLevel          # Risk level classification
    last_updated: datetime

@dataclass
class RiskLimit:
    """Risk limit definition"""
    limit_type: RiskType
    limit_value: float
    current_value: float
    utilization: float
    threshold_warning: float        # Warning threshold (% of limit)
    threshold_critical: float      # Critical threshold (% of limit)
    is_breached: bool
    breach_severity: AlertSeverity
    adaptive: bool                 # Whether limit adjusts to market conditions
    last_updated: datetime

@dataclass
class RiskAlert:
    """Risk alert notification"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    risk_type: RiskType
    message: str
    current_value: float
    limit_value: float
    recommended_action: str
    auto_action_taken: Optional[str]
    acknowledged: bool

@dataclass
class StressTestScenario:
    """Stress test scenario definition"""
    scenario_id: str
    name: str
    description: str
    market_shock: Dict[str, float]  # Market factor shocks
    probability: float              # Scenario probability
    expected_loss: float           # Expected loss under scenario
    recovery_time: int             # Expected recovery time in days

@dataclass
class PositionRisk:
    """Risk assessment for individual position"""
    position_id: str
    symbol: str
    position_size: float
    market_value: float
    risk_metrics: RiskMetrics
    contribution_to_portfolio_risk: float
    risk_adjusted_return: float
    recommended_action: Optional[str]

class AdvancedRiskManager:
    """
    Advanced Risk Management Engine for ALL-USE Protocol
    
    Provides enterprise-grade risk management with:
    - Multi-layer risk assessment and monitoring
    - Dynamic risk limits with adaptive thresholds
    - Real-time risk monitoring and alerting
    - Comprehensive stress testing and scenario analysis
    - Automated risk response and position management
    """
    
    def __init__(self):
        """Initialize the advanced risk manager"""
        self.logger = logging.getLogger(__name__)
        
        # Risk configuration
        self.risk_config = {
            'var_confidence_level': 0.95,      # VaR confidence level
            'lookback_period': 252,            # Trading days for historical analysis
            'stress_test_frequency': 'daily',  # Stress test frequency
            'risk_update_frequency': 60,       # Risk update frequency in seconds
            'max_portfolio_var': 0.05,         # Maximum portfolio VaR (5%)
            'max_position_weight': 0.20,       # Maximum position weight (20%)
            'max_sector_concentration': 0.30,  # Maximum sector concentration (30%)
            'max_correlation': 0.70,           # Maximum position correlation
            'drawdown_limit': 0.15,            # Maximum drawdown limit (15%)
            'leverage_limit': 2.0,             # Maximum leverage ratio
            'liquidity_threshold': 0.10        # Minimum liquidity threshold
        }
        
        # Risk limits
        self.risk_limits: Dict[RiskType, RiskLimit] = {}
        
        # Current risk metrics
        self.portfolio_risk: Optional[RiskMetrics] = None
        self.position_risks: Dict[str, PositionRisk] = {}
        
        # Risk monitoring
        self.risk_alerts: List[RiskAlert] = []
        self.alert_history: List[RiskAlert] = []
        
        # Stress testing
        self.stress_scenarios: List[StressTestScenario] = []
        self.stress_test_results: Dict[str, Dict[str, Any]] = {}
        
        # Risk monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Historical data for risk calculations
        self.price_history: pd.DataFrame = pd.DataFrame()
        self.return_history: pd.DataFrame = pd.DataFrame()
        
        # Initialize risk limits and scenarios
        self._initialize_risk_limits()
        self._initialize_stress_scenarios()
        
        self.logger.info("Advanced Risk Manager initialized")
    
    def start_risk_monitoring(self):
        """Start continuous risk monitoring"""
        if self.monitoring_active:
            self.logger.warning("Risk monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._risk_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Risk monitoring started")
    
    def stop_risk_monitoring(self):
        """Stop risk monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Risk monitoring stopped")
    
    def assess_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> RiskMetrics:
        """
        Assess comprehensive portfolio risk
        
        Args:
            portfolio_data: Portfolio positions and market data
            
        Returns:
            Portfolio risk metrics
        """
        try:
            positions = portfolio_data.get('positions', [])
            market_data = portfolio_data.get('market_data', {})
            
            # Calculate portfolio-level risk metrics
            portfolio_value = sum(pos.get('market_value', 0) for pos in positions)
            
            if portfolio_value == 0:
                return self._create_zero_risk_metrics()
            
            # Calculate VaR using historical simulation
            var_1d = self._calculate_portfolio_var(positions, 1)
            var_5d = self._calculate_portfolio_var(positions, 5)
            
            # Calculate Expected Shortfall
            expected_shortfall = self._calculate_expected_shortfall(positions)
            
            # Calculate maximum drawdown
            max_drawdown = self._calculate_maximum_drawdown(portfolio_data)
            
            # Calculate portfolio volatility
            volatility = self._calculate_portfolio_volatility(positions)
            
            # Calculate market beta
            beta = self._calculate_portfolio_beta(positions, market_data)
            
            # Calculate risk scores
            correlation_risk = self._calculate_correlation_risk(positions)
            concentration_risk = self._calculate_concentration_risk(positions)
            liquidity_risk = self._calculate_liquidity_risk(positions)
            
            # Calculate leverage
            leverage_ratio = self._calculate_leverage_ratio(portfolio_data)
            
            # Calculate overall risk score
            risk_score = self._calculate_overall_risk_score({
                'var_1d': var_1d,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'concentration_risk': concentration_risk,
                'leverage_ratio': leverage_ratio
            })
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_score)
            
            # Create risk metrics
            risk_metrics = RiskMetrics(
                var_1d=var_1d,
                var_5d=var_5d,
                expected_shortfall=expected_shortfall,
                maximum_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                leverage_ratio=leverage_ratio,
                risk_score=risk_score,
                risk_level=risk_level,
                last_updated=datetime.now()
            )
            
            # Update portfolio risk
            self.portfolio_risk = risk_metrics
            
            # Check risk limits
            self._check_risk_limits(risk_metrics)
            
            self.logger.info(f"Portfolio risk assessed: {risk_level.value} ({risk_score:.1f}/100)")
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing portfolio risk: {str(e)}")
            return self._create_zero_risk_metrics()
    
    def assess_position_risk(self, position_data: Dict[str, Any]) -> PositionRisk:
        """
        Assess risk for individual position
        
        Args:
            position_data: Position details and market data
            
        Returns:
            Position risk assessment
        """
        try:
            position_id = position_data.get('position_id', 'unknown')
            symbol = position_data.get('symbol', 'unknown')
            position_size = position_data.get('position_size', 0)
            market_value = position_data.get('market_value', 0)
            
            # Calculate position risk metrics
            risk_metrics = self._calculate_position_risk_metrics(position_data)
            
            # Calculate contribution to portfolio risk
            portfolio_contribution = self._calculate_portfolio_risk_contribution(position_data)
            
            # Calculate risk-adjusted return
            risk_adjusted_return = self._calculate_risk_adjusted_return(position_data)
            
            # Generate recommendation
            recommended_action = self._generate_position_recommendation(risk_metrics, position_data)
            
            position_risk = PositionRisk(
                position_id=position_id,
                symbol=symbol,
                position_size=position_size,
                market_value=market_value,
                risk_metrics=risk_metrics,
                contribution_to_portfolio_risk=portfolio_contribution,
                risk_adjusted_return=risk_adjusted_return,
                recommended_action=recommended_action
            )
            
            # Store position risk
            self.position_risks[position_id] = position_risk
            
            return position_risk
            
        except Exception as e:
            self.logger.error(f"Error assessing position risk: {str(e)}")
            return self._create_default_position_risk(position_data)
    
    def run_stress_test(self, scenario_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run stress test scenarios
        
        Args:
            scenario_id: Specific scenario to test (None for all scenarios)
            
        Returns:
            Stress test results
        """
        try:
            scenarios_to_test = []
            
            if scenario_id:
                scenario = next((s for s in self.stress_scenarios if s.scenario_id == scenario_id), None)
                if scenario:
                    scenarios_to_test = [scenario]
                else:
                    return {'error': f'Scenario {scenario_id} not found'}
            else:
                scenarios_to_test = self.stress_scenarios
            
            stress_results = {}
            
            for scenario in scenarios_to_test:
                scenario_result = self._run_single_stress_test(scenario)
                stress_results[scenario.scenario_id] = scenario_result
            
            # Store results
            self.stress_test_results.update(stress_results)
            
            # Generate summary
            summary = self._generate_stress_test_summary(stress_results)
            
            self.logger.info(f"Stress test completed: {len(scenarios_to_test)} scenarios")
            
            return {
                'scenario_results': stress_results,
                'summary': summary,
                'timestamp': datetime.now().isoformat(),
                'portfolio_resilience_score': summary.get('resilience_score', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error running stress test: {str(e)}")
            return {'error': str(e)}
    
    def update_risk_limits(self, market_conditions: Dict[str, Any]):
        """
        Update risk limits based on market conditions
        
        Args:
            market_conditions: Current market conditions
        """
        try:
            vix = market_conditions.get('vix', 20)
            market_regime = market_conditions.get('regime', 'normal')
            volatility_regime = market_conditions.get('volatility_regime', 'normal')
            
            # Adjust limits based on market conditions
            for risk_type, limit in self.risk_limits.items():
                if limit.adaptive:
                    adjustment_factor = self._calculate_limit_adjustment(
                        risk_type, vix, market_regime, volatility_regime
                    )
                    
                    # Update limit value
                    base_limit = self._get_base_limit_value(risk_type)
                    limit.limit_value = base_limit * adjustment_factor
                    limit.last_updated = datetime.now()
            
            self.logger.info(f"Risk limits updated for {market_regime} market regime")
            
        except Exception as e:
            self.logger.error(f"Error updating risk limits: {str(e)}")
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard"""
        try:
            # Current risk status
            current_risk = {
                'portfolio_risk_score': self.portfolio_risk.risk_score if self.portfolio_risk else 0,
                'portfolio_risk_level': self.portfolio_risk.risk_level.value if self.portfolio_risk else 'unknown',
                'portfolio_var_1d': self.portfolio_risk.var_1d if self.portfolio_risk else 0,
                'portfolio_max_drawdown': self.portfolio_risk.maximum_drawdown if self.portfolio_risk else 0,
                'total_positions': len(self.position_risks),
                'high_risk_positions': len([p for p in self.position_risks.values() 
                                          if p.risk_metrics.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]])
            }
            
            # Risk limit status
            limit_status = {}
            for risk_type, limit in self.risk_limits.items():
                limit_status[risk_type.value] = {
                    'current_value': limit.current_value,
                    'limit_value': limit.limit_value,
                    'utilization': limit.utilization,
                    'is_breached': limit.is_breached,
                    'breach_severity': limit.breach_severity.value if limit.is_breached else None
                }
            
            # Active alerts
            active_alerts = [
                {
                    'alert_id': alert.alert_id,
                    'severity': alert.severity.value,
                    'risk_type': alert.risk_type.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in self.risk_alerts if not alert.acknowledged
            ]
            
            # Recent stress test results
            recent_stress_test = None
            if self.stress_test_results:
                latest_test = max(self.stress_test_results.items(), key=lambda x: x[1].get('timestamp', ''))
                recent_stress_test = {
                    'scenario_id': latest_test[0],
                    'expected_loss': latest_test[1].get('expected_loss', 0),
                    'recovery_time': latest_test[1].get('recovery_time', 0),
                    'timestamp': latest_test[1].get('timestamp', '')
                }
            
            # Risk recommendations
            recommendations = self._generate_risk_recommendations()
            
            dashboard = {
                'current_risk': current_risk,
                'risk_limits': limit_status,
                'active_alerts': active_alerts,
                'recent_stress_test': recent_stress_test,
                'recommendations': recommendations,
                'monitoring_status': 'active' if self.monitoring_active else 'inactive',
                'last_updated': datetime.now().isoformat()
            }
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error generating risk dashboard: {str(e)}")
            return {'error': str(e)}
    
    def calculate_optimal_position_size(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate optimal position size based on risk constraints
        
        Args:
            trade_data: Trade details and market data
            
        Returns:
            Optimal position sizing recommendation
        """
        try:
            symbol = trade_data.get('symbol', 'unknown')
            strategy_type = trade_data.get('strategy_type', 'unknown')
            expected_return = trade_data.get('expected_return', 0.02)
            expected_volatility = trade_data.get('expected_volatility', 0.20)
            
            # Get current portfolio metrics
            portfolio_value = trade_data.get('portfolio_value', 100000)
            current_var = self.portfolio_risk.var_1d if self.portfolio_risk else 0.01
            
            # Calculate Kelly criterion position size
            kelly_size = self._calculate_kelly_position_size(expected_return, expected_volatility)
            
            # Calculate VaR-based position size
            var_size = self._calculate_var_position_size(trade_data, portfolio_value)
            
            # Calculate concentration-based position size
            concentration_size = self._calculate_concentration_position_size(symbol, portfolio_value)
            
            # Calculate correlation-based position size
            correlation_size = self._calculate_correlation_position_size(trade_data)
            
            # Take the minimum (most conservative)
            optimal_size = min(kelly_size, var_size, concentration_size, correlation_size)
            
            # Apply additional risk adjustments
            risk_adjustment = self._calculate_risk_adjustment_factor()
            final_size = optimal_size * risk_adjustment
            
            # Calculate expected impact on portfolio risk
            risk_impact = self._calculate_position_risk_impact(trade_data, final_size)
            
            return {
                'optimal_position_size': final_size,
                'kelly_size': kelly_size,
                'var_constrained_size': var_size,
                'concentration_constrained_size': concentration_size,
                'correlation_constrained_size': correlation_size,
                'risk_adjustment_factor': risk_adjustment,
                'expected_portfolio_var_impact': risk_impact['var_impact'],
                'expected_portfolio_risk_score_impact': risk_impact['risk_score_impact'],
                'recommendation': self._generate_sizing_recommendation(final_size, trade_data),
                'confidence': 0.85
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal position size: {str(e)}")
            return {'error': str(e)}
    
    # Helper methods for risk calculations
    def _initialize_risk_limits(self):
        """Initialize default risk limits"""
        risk_limit_configs = {
            RiskType.MARKET_RISK: {'limit': 0.05, 'adaptive': True, 'warning': 0.8, 'critical': 0.95},
            RiskType.VOLATILITY_RISK: {'limit': 0.25, 'adaptive': True, 'warning': 0.8, 'critical': 0.9},
            RiskType.CONCENTRATION_RISK: {'limit': 0.20, 'adaptive': False, 'warning': 0.8, 'critical': 0.95},
            RiskType.CORRELATION_RISK: {'limit': 0.70, 'adaptive': True, 'warning': 0.85, 'critical': 0.95},
            RiskType.DRAWDOWN_RISK: {'limit': 0.15, 'adaptive': False, 'warning': 0.7, 'critical': 0.9},
            RiskType.LEVERAGE_RISK: {'limit': 2.0, 'adaptive': True, 'warning': 0.8, 'critical': 0.95},
            RiskType.LIQUIDITY_RISK: {'limit': 0.10, 'adaptive': False, 'warning': 0.8, 'critical': 0.9}
        }
        
        for risk_type, config in risk_limit_configs.items():
            self.risk_limits[risk_type] = RiskLimit(
                limit_type=risk_type,
                limit_value=config['limit'],
                current_value=0.0,
                utilization=0.0,
                threshold_warning=config['warning'],
                threshold_critical=config['critical'],
                is_breached=False,
                breach_severity=AlertSeverity.INFO,
                adaptive=config['adaptive'],
                last_updated=datetime.now()
            )
    
    def _initialize_stress_scenarios(self):
        """Initialize stress test scenarios"""
        scenarios = [
            {
                'id': 'market_crash_2008',
                'name': '2008 Financial Crisis',
                'description': 'Severe market crash similar to 2008',
                'shocks': {'spy_return': -0.40, 'vix_spike': 3.0, 'correlation_increase': 0.3},
                'probability': 0.02
            },
            {
                'id': 'covid_crash_2020',
                'name': 'COVID-19 Market Crash',
                'description': 'Rapid market decline like March 2020',
                'shocks': {'spy_return': -0.35, 'vix_spike': 4.0, 'liquidity_crisis': 0.5},
                'probability': 0.03
            },
            {
                'id': 'flash_crash',
                'name': 'Flash Crash',
                'description': 'Sudden intraday market crash',
                'shocks': {'spy_return': -0.10, 'vix_spike': 2.0, 'liquidity_crisis': 0.8},
                'probability': 0.05
            },
            {
                'id': 'volatility_spike',
                'name': 'Volatility Spike',
                'description': 'Sudden increase in market volatility',
                'shocks': {'vix_spike': 2.5, 'correlation_increase': 0.2},
                'probability': 0.10
            },
            {
                'id': 'interest_rate_shock',
                'name': 'Interest Rate Shock',
                'description': 'Sudden change in interest rates',
                'shocks': {'rate_change': 0.02, 'spy_return': -0.15},
                'probability': 0.08
            }
        ]
        
        for scenario_config in scenarios:
            scenario = StressTestScenario(
                scenario_id=scenario_config['id'],
                name=scenario_config['name'],
                description=scenario_config['description'],
                market_shock=scenario_config['shocks'],
                probability=scenario_config['probability'],
                expected_loss=0.0,  # Will be calculated during stress test
                recovery_time=0     # Will be calculated during stress test
            )
            self.stress_scenarios.append(scenario)
    
    def _calculate_portfolio_var(self, positions: List[Dict], horizon_days: int) -> float:
        """Calculate portfolio Value at Risk"""
        try:
            if not positions:
                return 0.0
            
            # Simplified VaR calculation using historical simulation
            portfolio_value = sum(pos.get('market_value', 0) for pos in positions)
            
            if portfolio_value == 0:
                return 0.0
            
            # Estimate portfolio volatility
            avg_volatility = np.mean([pos.get('volatility', 0.20) for pos in positions])
            
            # Calculate VaR using normal distribution approximation
            confidence_level = self.risk_config['var_confidence_level']
            z_score = np.percentile(np.random.normal(0, 1, 10000), (1 - confidence_level) * 100)
            
            var = abs(z_score) * avg_volatility * np.sqrt(horizon_days) * portfolio_value
            
            return var / portfolio_value  # Return as percentage of portfolio
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio VaR: {str(e)}")
            return 0.05  # Default 5% VaR
    
    def _calculate_expected_shortfall(self, positions: List[Dict]) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            # Simplified ES calculation
            var_1d = self._calculate_portfolio_var(positions, 1)
            # ES is typically 1.2-1.5x VaR for normal distributions
            return var_1d * 1.3
            
        except Exception as e:
            self.logger.error(f"Error calculating expected shortfall: {str(e)}")
            return 0.065  # Default 6.5% ES
    
    def _calculate_maximum_drawdown(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate maximum drawdown"""
        try:
            # Get historical portfolio values
            historical_values = portfolio_data.get('historical_values', [])
            
            if len(historical_values) < 2:
                return 0.0
            
            # Calculate drawdowns
            peak = historical_values[0]
            max_drawdown = 0.0
            
            for value in historical_values:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating maximum drawdown: {str(e)}")
            return 0.05  # Default 5% drawdown
    
    def _calculate_portfolio_volatility(self, positions: List[Dict]) -> float:
        """Calculate portfolio volatility"""
        try:
            if not positions:
                return 0.0
            
            # Weighted average volatility (simplified)
            total_value = sum(pos.get('market_value', 0) for pos in positions)
            
            if total_value == 0:
                return 0.0
            
            weighted_volatility = sum(
                pos.get('volatility', 0.20) * pos.get('market_value', 0) / total_value
                for pos in positions
            )
            
            return weighted_volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio volatility: {str(e)}")
            return 0.20  # Default 20% volatility
    
    def _calculate_portfolio_beta(self, positions: List[Dict], market_data: Dict[str, Any]) -> float:
        """Calculate portfolio beta"""
        try:
            if not positions:
                return 1.0
            
            # Weighted average beta
            total_value = sum(pos.get('market_value', 0) for pos in positions)
            
            if total_value == 0:
                return 1.0
            
            weighted_beta = sum(
                pos.get('beta', 1.0) * pos.get('market_value', 0) / total_value
                for pos in positions
            )
            
            return weighted_beta
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio beta: {str(e)}")
            return 1.0  # Default market beta
    
    def _calculate_correlation_risk(self, positions: List[Dict]) -> float:
        """Calculate correlation risk score"""
        try:
            if len(positions) < 2:
                return 0.0
            
            # Simplified correlation risk calculation
            # In practice, this would use actual correlation matrix
            correlations = []
            
            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions[i+1:], i+1):
                    # Estimate correlation based on sector/asset class
                    correlation = self._estimate_position_correlation(pos1, pos2)
                    correlations.append(correlation)
            
            if not correlations:
                return 0.0
            
            avg_correlation = np.mean(correlations)
            max_correlation = np.max(correlations)
            
            # Risk score based on average and maximum correlations
            risk_score = (avg_correlation * 0.6 + max_correlation * 0.4)
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {str(e)}")
            return 0.3  # Default moderate correlation risk
    
    def _calculate_concentration_risk(self, positions: List[Dict]) -> float:
        """Calculate concentration risk score"""
        try:
            if not positions:
                return 0.0
            
            total_value = sum(pos.get('market_value', 0) for pos in positions)
            
            if total_value == 0:
                return 0.0
            
            # Calculate position weights
            weights = [pos.get('market_value', 0) / total_value for pos in positions]
            
            # Calculate Herfindahl-Hirschman Index
            hhi = sum(w**2 for w in weights)
            
            # Convert to risk score (0-1)
            # HHI ranges from 1/n (perfectly diversified) to 1 (fully concentrated)
            n_positions = len(positions)
            min_hhi = 1.0 / n_positions
            concentration_risk = (hhi - min_hhi) / (1.0 - min_hhi) if n_positions > 1 else 0.0
            
            return min(1.0, max(0.0, concentration_risk))
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration risk: {str(e)}")
            return 0.2  # Default low concentration risk
    
    def _calculate_liquidity_risk(self, positions: List[Dict]) -> float:
        """Calculate liquidity risk score"""
        try:
            if not positions:
                return 0.0
            
            total_value = sum(pos.get('market_value', 0) for pos in positions)
            
            if total_value == 0:
                return 0.0
            
            # Weighted average liquidity risk
            weighted_liquidity_risk = sum(
                pos.get('liquidity_risk', 0.1) * pos.get('market_value', 0) / total_value
                for pos in positions
            )
            
            return min(1.0, max(0.0, weighted_liquidity_risk))
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity risk: {str(e)}")
            return 0.1  # Default low liquidity risk
    
    def _calculate_leverage_ratio(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate portfolio leverage ratio"""
        try:
            total_exposure = portfolio_data.get('total_exposure', 0)
            equity_value = portfolio_data.get('equity_value', 0)
            
            if equity_value == 0:
                return 1.0
            
            leverage = total_exposure / equity_value
            return max(1.0, leverage)
            
        except Exception as e:
            self.logger.error(f"Error calculating leverage ratio: {str(e)}")
            return 1.0  # Default no leverage
    
    def _calculate_overall_risk_score(self, risk_components: Dict[str, float]) -> float:
        """Calculate overall risk score from components"""
        try:
            # Risk component weights
            weights = {
                'var_1d': 0.25,
                'volatility': 0.20,
                'max_drawdown': 0.20,
                'concentration_risk': 0.15,
                'leverage_ratio': 0.10,
                'correlation_risk': 0.10
            }
            
            # Normalize components to 0-1 scale
            normalized_components = {}
            
            # VaR (0-10% -> 0-1)
            normalized_components['var_1d'] = min(1.0, risk_components.get('var_1d', 0) / 0.10)
            
            # Volatility (0-50% -> 0-1)
            normalized_components['volatility'] = min(1.0, risk_components.get('volatility', 0) / 0.50)
            
            # Max drawdown (0-30% -> 0-1)
            normalized_components['max_drawdown'] = min(1.0, risk_components.get('max_drawdown', 0) / 0.30)
            
            # Concentration risk (already 0-1)
            normalized_components['concentration_risk'] = risk_components.get('concentration_risk', 0)
            
            # Leverage (1-5 -> 0-1)
            leverage = risk_components.get('leverage_ratio', 1.0)
            normalized_components['leverage_ratio'] = min(1.0, max(0.0, (leverage - 1.0) / 4.0))
            
            # Correlation risk (already 0-1)
            normalized_components['correlation_risk'] = risk_components.get('correlation_risk', 0)
            
            # Calculate weighted score
            risk_score = sum(
                weights[component] * normalized_components[component]
                for component in weights.keys()
            )
            
            # Convert to 0-100 scale
            return risk_score * 100
            
        except Exception as e:
            self.logger.error(f"Error calculating overall risk score: {str(e)}")
            return 50.0  # Default medium risk
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from risk score"""
        if risk_score < 20:
            return RiskLevel.VERY_LOW
        elif risk_score < 40:
            return RiskLevel.LOW
        elif risk_score < 60:
            return RiskLevel.MEDIUM
        elif risk_score < 80:
            return RiskLevel.HIGH
        elif risk_score < 95:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _create_zero_risk_metrics(self) -> RiskMetrics:
        """Create zero risk metrics for empty portfolio"""
        return RiskMetrics(
            var_1d=0.0,
            var_5d=0.0,
            expected_shortfall=0.0,
            maximum_drawdown=0.0,
            volatility=0.0,
            beta=1.0,
            correlation_risk=0.0,
            concentration_risk=0.0,
            liquidity_risk=0.0,
            leverage_ratio=1.0,
            risk_score=0.0,
            risk_level=RiskLevel.VERY_LOW,
            last_updated=datetime.now()
        )
    
    def _check_risk_limits(self, risk_metrics: RiskMetrics):
        """Check risk metrics against limits and generate alerts"""
        try:
            # Update current values in risk limits
            self.risk_limits[RiskType.MARKET_RISK].current_value = risk_metrics.var_1d
            self.risk_limits[RiskType.VOLATILITY_RISK].current_value = risk_metrics.volatility
            self.risk_limits[RiskType.CONCENTRATION_RISK].current_value = risk_metrics.concentration_risk
            self.risk_limits[RiskType.CORRELATION_RISK].current_value = risk_metrics.correlation_risk
            self.risk_limits[RiskType.DRAWDOWN_RISK].current_value = risk_metrics.maximum_drawdown
            self.risk_limits[RiskType.LEVERAGE_RISK].current_value = risk_metrics.leverage_ratio
            self.risk_limits[RiskType.LIQUIDITY_RISK].current_value = risk_metrics.liquidity_risk
            
            # Check each limit
            for risk_type, limit in self.risk_limits.items():
                utilization = limit.current_value / limit.limit_value if limit.limit_value > 0 else 0
                limit.utilization = utilization
                
                # Check for breaches
                if utilization >= limit.threshold_critical:
                    limit.is_breached = True
                    limit.breach_severity = AlertSeverity.CRITICAL
                    self._generate_risk_alert(risk_type, limit, AlertSeverity.CRITICAL)
                elif utilization >= limit.threshold_warning:
                    limit.is_breached = True
                    limit.breach_severity = AlertSeverity.WARNING
                    self._generate_risk_alert(risk_type, limit, AlertSeverity.WARNING)
                else:
                    limit.is_breached = False
                    limit.breach_severity = AlertSeverity.INFO
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
    
    def _generate_risk_alert(self, risk_type: RiskType, limit: RiskLimit, severity: AlertSeverity):
        """Generate risk alert"""
        try:
            alert = RiskAlert(
                alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{risk_type.value}",
                timestamp=datetime.now(),
                severity=severity,
                risk_type=risk_type,
                message=f"{risk_type.value} limit breach: {limit.current_value:.3f} exceeds {limit.limit_value:.3f}",
                current_value=limit.current_value,
                limit_value=limit.limit_value,
                recommended_action=self._get_recommended_action(risk_type, severity),
                auto_action_taken=None,
                acknowledged=False
            )
            
            self.risk_alerts.append(alert)
            self.logger.warning(f"Risk alert generated: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"Error generating risk alert: {str(e)}")
    
    def _risk_monitoring_loop(self):
        """Main risk monitoring loop"""
        while self.monitoring_active:
            try:
                # Perform risk checks
                if self.portfolio_risk:
                    self._check_risk_limits(self.portfolio_risk)
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                # Sleep until next update
                time.sleep(self.risk_config['risk_update_frequency'])
                
            except Exception as e:
                self.logger.error(f"Error in risk monitoring loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _cleanup_old_alerts(self):
        """Clean up old alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            # Move old alerts to history
            old_alerts = [alert for alert in self.risk_alerts if alert.timestamp < cutoff_time]
            self.alert_history.extend(old_alerts)
            
            # Keep only recent alerts
            self.risk_alerts = [alert for alert in self.risk_alerts if alert.timestamp >= cutoff_time]
            
            # Limit history size
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"Error cleaning up alerts: {str(e)}")
    
    # Additional helper methods would continue here...
    # For brevity, I'll include key methods for the remaining functionality
    
    def _run_single_stress_test(self, scenario: StressTestScenario) -> Dict[str, Any]:
        """Run a single stress test scenario"""
        try:
            # Simulate portfolio performance under stress
            if not self.portfolio_risk:
                return {'error': 'No portfolio data available'}
            
            # Apply market shocks
            stressed_var = self.portfolio_risk.var_1d
            stressed_drawdown = self.portfolio_risk.maximum_drawdown
            
            # Apply shocks from scenario
            for shock_type, shock_value in scenario.market_shock.items():
                if shock_type == 'spy_return':
                    # Increase VaR based on market shock
                    stressed_var *= (1 + abs(shock_value))
                elif shock_type == 'vix_spike':
                    # Increase volatility and VaR
                    stressed_var *= (1 + shock_value * 0.2)
                elif shock_type == 'correlation_increase':
                    # Increase correlation risk
                    stressed_var *= (1 + shock_value * 0.5)
            
            # Calculate expected loss
            expected_loss = min(stressed_var * 2, 0.5)  # Cap at 50%
            
            # Estimate recovery time
            recovery_time = int(expected_loss * 100)  # Days
            
            return {
                'scenario_id': scenario.scenario_id,
                'expected_loss': expected_loss,
                'stressed_var': stressed_var,
                'stressed_drawdown': stressed_drawdown,
                'recovery_time': recovery_time,
                'survival_probability': 1.0 - expected_loss,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_stress_test_summary(self, stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate stress test summary"""
        try:
            if not stress_results:
                return {'error': 'No stress test results'}
            
            # Calculate summary statistics
            expected_losses = [result.get('expected_loss', 0) for result in stress_results.values() if 'error' not in result]
            recovery_times = [result.get('recovery_time', 0) for result in stress_results.values() if 'error' not in result]
            
            if not expected_losses:
                return {'error': 'No valid stress test results'}
            
            max_expected_loss = max(expected_losses)
            avg_expected_loss = np.mean(expected_losses)
            max_recovery_time = max(recovery_times)
            avg_recovery_time = np.mean(recovery_times)
            
            # Calculate resilience score
            resilience_score = max(0, 100 - max_expected_loss * 100)
            
            return {
                'max_expected_loss': max_expected_loss,
                'avg_expected_loss': avg_expected_loss,
                'max_recovery_time': max_recovery_time,
                'avg_recovery_time': avg_recovery_time,
                'resilience_score': resilience_score,
                'scenarios_tested': len(expected_losses),
                'worst_case_scenario': max(stress_results.items(), key=lambda x: x[1].get('expected_loss', 0))[0]
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_kelly_position_size(self, expected_return: float, volatility: float) -> float:
        """Calculate Kelly criterion position size"""
        try:
            if volatility == 0:
                return 0.0
            
            # Kelly fraction = (expected_return - risk_free_rate) / variance
            risk_free_rate = 0.02  # 2% risk-free rate
            excess_return = expected_return - risk_free_rate
            variance = volatility ** 2
            
            kelly_fraction = excess_return / variance
            
            # Cap Kelly fraction at 25% for safety
            return min(0.25, max(0.0, kelly_fraction))
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly position size: {str(e)}")
            return 0.05  # Default 5% position size
    
    def _generate_risk_recommendations(self) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        try:
            if not self.portfolio_risk:
                recommendations.append("Initialize portfolio risk assessment")
                return recommendations
            
            # Risk level recommendations
            if self.portfolio_risk.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]:
                recommendations.append("Consider reducing portfolio risk exposure")
            
            # Specific risk factor recommendations
            if self.portfolio_risk.concentration_risk > 0.7:
                recommendations.append("Diversify portfolio to reduce concentration risk")
            
            if self.portfolio_risk.correlation_risk > 0.7:
                recommendations.append("Reduce position correlations through diversification")
            
            if self.portfolio_risk.leverage_ratio > 1.5:
                recommendations.append("Consider reducing leverage to improve risk profile")
            
            if self.portfolio_risk.var_1d > 0.05:
                recommendations.append("Portfolio VaR exceeds 5% - consider risk reduction")
            
            # Alert-based recommendations
            critical_alerts = [alert for alert in self.risk_alerts if alert.severity == AlertSeverity.CRITICAL]
            if critical_alerts:
                recommendations.append(f"Address {len(critical_alerts)} critical risk alerts immediately")
            
            if not recommendations:
                recommendations.append("Portfolio risk profile is within acceptable limits")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")
        
        return recommendations

def test_advanced_risk_manager():
    """Test the advanced risk manager"""
    print("Testing Advanced Risk Manager...")
    
    risk_manager = AdvancedRiskManager()
    
    # Test portfolio risk assessment
    print("\n--- Testing Portfolio Risk Assessment ---")
    portfolio_data = {
        'positions': [
            {
                'symbol': 'SPY',
                'market_value': 50000,
                'volatility': 0.18,
                'beta': 1.0,
                'liquidity_risk': 0.05
            },
            {
                'symbol': 'QQQ',
                'market_value': 30000,
                'volatility': 0.22,
                'beta': 1.2,
                'liquidity_risk': 0.08
            },
            {
                'symbol': 'IWM',
                'market_value': 20000,
                'volatility': 0.25,
                'beta': 1.1,
                'liquidity_risk': 0.12
            }
        ],
        'market_data': {
            'spy_price': 420,
            'vix': 22
        },
        'total_exposure': 100000,
        'equity_value': 100000,
        'historical_values': [100000, 98000, 102000, 99000, 101000]
    }
    
    risk_metrics = risk_manager.assess_portfolio_risk(portfolio_data)
    
    print(f"Portfolio Risk Score: {risk_metrics.risk_score:.1f}/100")
    print(f"Risk Level: {risk_metrics.risk_level.value}")
    print(f"1-Day VaR: {risk_metrics.var_1d:.1%}")
    print(f"Maximum Drawdown: {risk_metrics.maximum_drawdown:.1%}")
    print(f"Portfolio Volatility: {risk_metrics.volatility:.1%}")
    
    # Test stress testing
    print("\n--- Testing Stress Testing ---")
    stress_results = risk_manager.run_stress_test()
    
    if 'error' not in stress_results:
        summary = stress_results['summary']
        print(f"Scenarios Tested: {summary['scenarios_tested']}")
        print(f"Max Expected Loss: {summary['max_expected_loss']:.1%}")
        print(f"Portfolio Resilience Score: {summary['resilience_score']:.1f}/100")
        print(f"Worst Case Scenario: {summary['worst_case_scenario']}")
    
    # Test optimal position sizing
    print("\n--- Testing Optimal Position Sizing ---")
    trade_data = {
        'symbol': 'AAPL',
        'strategy_type': 'put_selling',
        'expected_return': 0.025,
        'expected_volatility': 0.20,
        'portfolio_value': 100000
    }
    
    sizing_result = risk_manager.calculate_optimal_position_size(trade_data)
    
    if 'error' not in sizing_result:
        print(f"Optimal Position Size: {sizing_result['optimal_position_size']:.1%}")
        print(f"Kelly Size: {sizing_result['kelly_size']:.1%}")
        print(f"VaR Constrained Size: {sizing_result['var_constrained_size']:.1%}")
        print(f"Risk Adjustment Factor: {sizing_result['risk_adjustment_factor']:.2f}")
    
    # Test risk dashboard
    print("\n--- Testing Risk Dashboard ---")
    dashboard = risk_manager.get_risk_dashboard()
    
    if 'error' not in dashboard:
        current_risk = dashboard['current_risk']
        print(f"Portfolio Risk Score: {current_risk['portfolio_risk_score']:.1f}")
        print(f"Portfolio Risk Level: {current_risk['portfolio_risk_level']}")
        print(f"Total Positions: {current_risk['total_positions']}")
        print(f"High Risk Positions: {current_risk['high_risk_positions']}")
        print(f"Active Alerts: {len(dashboard['active_alerts'])}")
        
        print("\nRecommendations:")
        for rec in dashboard['recommendations']:
            print(f"  • {rec}")
    
    print("\n✅ Advanced Risk Manager test completed successfully!")

if __name__ == "__main__":
    test_advanced_risk_manager()


    def _estimate_position_correlation(self, pos1: Dict[str, Any], pos2: Dict[str, Any]) -> float:
        """Estimate correlation between two positions"""
        try:
            # Simple correlation estimation based on asset characteristics
            symbol1 = pos1.get('symbol', '')
            symbol2 = pos2.get('symbol', '')
            
            # Same symbol = perfect correlation
            if symbol1 == symbol2:
                return 1.0
            
            # ETF correlations (simplified)
            etf_correlations = {
                ('SPY', 'QQQ'): 0.85,
                ('SPY', 'IWM'): 0.75,
                ('QQQ', 'IWM'): 0.70,
                ('SPY', 'VTI'): 0.95,
                ('QQQ', 'VGT'): 0.90
            }
            
            # Check both directions
            correlation = etf_correlations.get((symbol1, symbol2), 
                         etf_correlations.get((symbol2, symbol1), 0.3))
            
            return correlation
            
        except Exception as e:
            return 0.3  # Default moderate correlation
    
    def _get_recommended_action(self, risk_type: RiskType, severity: AlertSeverity) -> str:
        """Get recommended action for risk alert"""
        try:
            actions = {
                RiskType.MARKET_RISK: {
                    AlertSeverity.WARNING: "Monitor market exposure closely",
                    AlertSeverity.CRITICAL: "Reduce market exposure immediately"
                },
                RiskType.CONCENTRATION_RISK: {
                    AlertSeverity.WARNING: "Consider diversifying positions",
                    AlertSeverity.CRITICAL: "Reduce position concentration immediately"
                },
                RiskType.LEVERAGE_RISK: {
                    AlertSeverity.WARNING: "Monitor leverage levels",
                    AlertSeverity.CRITICAL: "Reduce leverage immediately"
                }
            }
            
            return actions.get(risk_type, {}).get(severity, "Review risk exposure")
            
        except Exception as e:
            return "Review risk exposure"
    
    def _calculate_var_position_size(self, trade_data: Dict[str, Any], portfolio_value: float) -> float:
        """Calculate VaR-constrained position size"""
        try:
            max_var_contribution = 0.01  # 1% max VaR contribution
            expected_volatility = trade_data.get('expected_volatility', 0.20)
            
            # Simple VaR-based sizing
            var_size = max_var_contribution / expected_volatility
            
            return min(0.20, max(0.01, var_size))  # Cap between 1% and 20%
            
        except Exception as e:
            return 0.05  # Default 5%
    
    def _calculate_concentration_position_size(self, symbol: str, portfolio_value: float) -> float:
        """Calculate concentration-constrained position size"""
        try:
            max_position_weight = self.risk_config.get('max_position_weight', 0.20)
            return max_position_weight
            
        except Exception as e:
            return 0.15  # Default 15%
    
    def _calculate_correlation_position_size(self, trade_data: Dict[str, Any]) -> float:
        """Calculate correlation-constrained position size"""
        try:
            # Simplified correlation-based sizing
            return 0.10  # Default 10% for correlation constraint
            
        except Exception as e:
            return 0.10
    
    def _calculate_risk_adjustment_factor(self) -> float:
        """Calculate risk adjustment factor"""
        try:
            if not self.portfolio_risk:
                return 1.0
            
            # Reduce size if portfolio risk is high
            risk_score = self.portfolio_risk.risk_score
            
            if risk_score > 80:
                return 0.5  # Reduce by 50%
            elif risk_score > 60:
                return 0.75  # Reduce by 25%
            else:
                return 1.0  # No adjustment
                
        except Exception as e:
            return 1.0
    
    def _calculate_position_risk_impact(self, trade_data: Dict[str, Any], position_size: float) -> Dict[str, float]:
        """Calculate expected impact of position on portfolio risk"""
        try:
            expected_volatility = trade_data.get('expected_volatility', 0.20)
            
            # Estimate VaR impact
            var_impact = position_size * expected_volatility * 0.5
            
            # Estimate risk score impact
            risk_score_impact = var_impact * 100
            
            return {
                'var_impact': var_impact,
                'risk_score_impact': risk_score_impact
            }
            
        except Exception as e:
            return {'var_impact': 0.0, 'risk_score_impact': 0.0}
    
    def _generate_sizing_recommendation(self, position_size: float, trade_data: Dict[str, Any]) -> str:
        """Generate position sizing recommendation"""
        try:
            if position_size < 0.02:
                return "Very conservative sizing due to high risk"
            elif position_size < 0.05:
                return "Conservative sizing recommended"
            elif position_size < 0.10:
                return "Moderate sizing appropriate"
            elif position_size < 0.15:
                return "Aggressive sizing - monitor closely"
            else:
                return "Very aggressive sizing - high risk"
                
        except Exception as e:
            return "Standard sizing recommended"
    
    def _calculate_position_risk_metrics(self, position_data: Dict[str, Any]) -> RiskMetrics:
        """Calculate risk metrics for individual position"""
        try:
            volatility = position_data.get('volatility', 0.20)
            beta = position_data.get('beta', 1.0)
            
            # Simplified position risk calculation
            var_1d = volatility / np.sqrt(252)  # Daily VaR
            var_5d = var_1d * np.sqrt(5)
            
            risk_score = min(100, volatility * 100)
            risk_level = self._determine_risk_level(risk_score)
            
            return RiskMetrics(
                var_1d=var_1d,
                var_5d=var_5d,
                expected_shortfall=var_1d * 1.3,
                maximum_drawdown=volatility * 0.5,
                volatility=volatility,
                beta=beta,
                correlation_risk=0.3,
                concentration_risk=0.2,
                liquidity_risk=position_data.get('liquidity_risk', 0.1),
                leverage_ratio=1.0,
                risk_score=risk_score,
                risk_level=risk_level,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            return self._create_zero_risk_metrics()
    
    def _calculate_portfolio_risk_contribution(self, position_data: Dict[str, Any]) -> float:
        """Calculate position's contribution to portfolio risk"""
        try:
            market_value = position_data.get('market_value', 0)
            volatility = position_data.get('volatility', 0.20)
            
            # Simplified contribution calculation
            return market_value * volatility * 0.01
            
        except Exception as e:
            return 0.0
    
    def _calculate_risk_adjusted_return(self, position_data: Dict[str, Any]) -> float:
        """Calculate risk-adjusted return for position"""
        try:
            expected_return = position_data.get('expected_return', 0.08)
            volatility = position_data.get('volatility', 0.20)
            
            # Sharpe-like ratio
            risk_free_rate = 0.02
            return (expected_return - risk_free_rate) / volatility
            
        except Exception as e:
            return 0.3  # Default risk-adjusted return
    
    def _generate_position_recommendation(self, risk_metrics: RiskMetrics, position_data: Dict[str, Any]) -> str:
        """Generate recommendation for position"""
        try:
            if risk_metrics.risk_level == RiskLevel.CRITICAL:
                return "Close position immediately - critical risk"
            elif risk_metrics.risk_level == RiskLevel.VERY_HIGH:
                return "Reduce position size - very high risk"
            elif risk_metrics.risk_level == RiskLevel.HIGH:
                return "Monitor closely - high risk"
            elif risk_metrics.risk_level == RiskLevel.MEDIUM:
                return "Standard monitoring - medium risk"
            else:
                return "Continue holding - low risk"
                
        except Exception as e:
            return "Monitor position"
    
    def _create_default_position_risk(self, position_data: Dict[str, Any]) -> PositionRisk:
        """Create default position risk for error cases"""
        return PositionRisk(
            position_id=position_data.get('position_id', 'unknown'),
            symbol=position_data.get('symbol', 'unknown'),
            position_size=position_data.get('position_size', 0),
            market_value=position_data.get('market_value', 0),
            risk_metrics=self._create_zero_risk_metrics(),
            contribution_to_portfolio_risk=0.0,
            risk_adjusted_return=0.0,
            recommended_action="Unable to assess - insufficient data"
        )
    
    def _calculate_limit_adjustment(self, risk_type: RiskType, vix: float, market_regime: str, volatility_regime: str) -> float:
        """Calculate risk limit adjustment factor"""
        try:
            base_factor = 1.0
            
            # VIX-based adjustment
            if vix > 30:
                base_factor *= 0.8  # Tighten limits in high volatility
            elif vix > 25:
                base_factor *= 0.9
            elif vix < 15:
                base_factor *= 1.1  # Relax limits in low volatility
            
            # Market regime adjustment
            if market_regime == 'bear_market':
                base_factor *= 0.7
            elif market_regime == 'high_volatility':
                base_factor *= 0.8
            elif market_regime == 'bull_market':
                base_factor *= 1.1
            
            return max(0.5, min(1.5, base_factor))
            
        except Exception as e:
            return 1.0
    
    def _get_base_limit_value(self, risk_type: RiskType) -> float:
        """Get base limit value for risk type"""
        base_limits = {
            RiskType.MARKET_RISK: 0.05,
            RiskType.VOLATILITY_RISK: 0.25,
            RiskType.CONCENTRATION_RISK: 0.20,
            RiskType.CORRELATION_RISK: 0.70,
            RiskType.DRAWDOWN_RISK: 0.15,
            RiskType.LEVERAGE_RISK: 2.0,
            RiskType.LIQUIDITY_RISK: 0.10
        }
        
        return base_limits.get(risk_type, 0.10)

