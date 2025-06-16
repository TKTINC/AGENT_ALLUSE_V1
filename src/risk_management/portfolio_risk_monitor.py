"""
ALL-USE Risk Management: Portfolio Risk Monitor Module

This module implements real-time portfolio risk monitoring for the ALL-USE trading system.
It provides comprehensive risk assessment, monitoring, and alerting capabilities.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
import sys
import os
from dataclasses import dataclass
import asyncio
import threading
import time

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocol_engine.all_use_parameters import ALLUSEParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('all_use_risk_monitor.log')
    ]
)

logger = logging.getLogger('all_use_risk_monitor')

class RiskLevel(Enum):
    """Enumeration of risk levels."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class AlertType(Enum):
    """Enumeration of alert types."""
    CONCENTRATION = "Concentration"
    DRAWDOWN = "Drawdown"
    VAR_BREACH = "VaR_Breach"
    CORRELATION = "Correlation"
    LIQUIDITY = "Liquidity"
    VOLATILITY = "Volatility"

@dataclass
class RiskAlert:
    """Risk alert data structure."""
    alert_type: AlertType
    risk_level: RiskLevel
    message: str
    timestamp: datetime
    portfolio_id: str
    metric_value: float
    threshold_value: float
    recommended_action: str

@dataclass
class PortfolioRiskMetrics:
    """Portfolio risk metrics data structure."""
    portfolio_id: str
    timestamp: datetime
    total_value: float
    var_1day_95: float
    var_1day_99: float
    cvar_1day_95: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    overall_risk_score: float
    risk_level: RiskLevel

class PortfolioRiskMonitor:
    """
    Advanced portfolio risk monitoring system for the ALL-USE trading strategy.
    
    This class provides real-time risk monitoring including:
    - Value at Risk (VaR) and Conditional VaR calculations
    - Drawdown monitoring and analysis
    - Concentration and correlation risk assessment
    - Liquidity risk evaluation
    - Real-time alerting and notification
    """
    
    def __init__(self, alert_callback: Optional[Callable] = None):
        """Initialize the portfolio risk monitor."""
        self.parameters = ALLUSEParameters
        self.alert_callback = alert_callback
        
        # Risk monitoring configuration
        self.monitoring_config = {
            'update_interval': 30,  # seconds
            'var_confidence_levels': [0.95, 0.99],
            'lookback_periods': {
                'var': 252,      # 1 year for VaR calculation
                'correlation': 63,  # 3 months for correlation
                'volatility': 21    # 1 month for volatility
            },
            'risk_thresholds': {
                'var_1day_95': 0.02,      # 2% daily VaR
                'var_1day_99': 0.03,      # 3% daily VaR
                'max_drawdown': 0.10,     # 10% maximum drawdown
                'concentration': 0.25,     # 25% max position concentration
                'correlation': 0.80,       # 80% max correlation
                'volatility': 0.30         # 30% max portfolio volatility
            }
        }
        
        # Risk monitoring state
        self.portfolios = {}
        self.risk_history = {}
        self.active_alerts = {}
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance tracking
        self.calculation_times = []
        self.last_update = None
        
        logger.info("Portfolio risk monitor initialized")
    
    def add_portfolio(self, portfolio_id: str, initial_positions: Dict[str, Any]) -> None:
        """
        Add a portfolio to risk monitoring.
        
        Args:
            portfolio_id: Unique portfolio identifier
            initial_positions: Initial portfolio positions
        """
        logger.info(f"Adding portfolio {portfolio_id} to risk monitoring")
        
        self.portfolios[portfolio_id] = {
            'positions': initial_positions,
            'last_update': datetime.now(),
            'risk_metrics': None,
            'price_history': {},
            'pnl_history': []
        }
        
        self.risk_history[portfolio_id] = []
        self.active_alerts[portfolio_id] = []
    
    def update_portfolio_positions(self, portfolio_id: str, positions: Dict[str, Any]) -> None:
        """
        Update portfolio positions for risk monitoring.
        
        Args:
            portfolio_id: Portfolio identifier
            positions: Updated portfolio positions
        """
        if portfolio_id not in self.portfolios:
            logger.warning(f"Portfolio {portfolio_id} not found in monitoring")
            return
        
        self.portfolios[portfolio_id]['positions'] = positions
        self.portfolios[portfolio_id]['last_update'] = datetime.now()
        
        # Trigger immediate risk calculation
        self._calculate_portfolio_risk(portfolio_id)
    
    def start_monitoring(self) -> None:
        """Start real-time risk monitoring."""
        if self.monitoring_active:
            logger.warning("Risk monitoring already active")
            return
        
        logger.info("Starting real-time risk monitoring")
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop real-time risk monitoring."""
        logger.info("Stopping real-time risk monitoring")
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for real-time risk assessment."""
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Update risk metrics for all portfolios
                for portfolio_id in self.portfolios.keys():
                    self._calculate_portfolio_risk(portfolio_id)
                
                # Track calculation performance
                calculation_time = time.time() - start_time
                self.calculation_times.append(calculation_time)
                
                # Keep only recent calculation times
                if len(self.calculation_times) > 100:
                    self.calculation_times = self.calculation_times[-100:]
                
                self.last_update = datetime.now()
                
                # Sleep until next update
                time.sleep(self.monitoring_config['update_interval'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)  # Brief pause before retrying
    
    def _calculate_portfolio_risk(self, portfolio_id: str) -> PortfolioRiskMetrics:
        """
        Calculate comprehensive risk metrics for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            PortfolioRiskMetrics object
        """
        try:
            portfolio = self.portfolios[portfolio_id]
            positions = portfolio['positions']
            
            # Calculate portfolio value
            total_value = self._calculate_portfolio_value(positions)
            
            # Calculate VaR metrics
            var_metrics = self._calculate_var_metrics(portfolio_id, positions)
            
            # Calculate drawdown metrics
            drawdown_metrics = self._calculate_drawdown_metrics(portfolio_id)
            
            # Calculate correlation and concentration risks
            correlation_risk = self._calculate_correlation_risk(positions)
            concentration_risk = self._calculate_concentration_risk(positions)
            
            # Calculate liquidity risk
            liquidity_risk = self._calculate_liquidity_risk(positions)
            
            # Calculate portfolio volatility
            volatility = self._calculate_portfolio_volatility(positions)
            
            # Calculate beta (simplified)
            beta = self._calculate_portfolio_beta(positions)
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(
                var_metrics, drawdown_metrics, correlation_risk, 
                concentration_risk, liquidity_risk, volatility
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(overall_risk_score)
            
            # Create risk metrics object
            risk_metrics = PortfolioRiskMetrics(
                portfolio_id=portfolio_id,
                timestamp=datetime.now(),
                total_value=total_value,
                var_1day_95=var_metrics['var_95'],
                var_1day_99=var_metrics['var_99'],
                cvar_1day_95=var_metrics['cvar_95'],
                max_drawdown=drawdown_metrics['max_drawdown'],
                current_drawdown=drawdown_metrics['current_drawdown'],
                volatility=volatility,
                beta=beta,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                overall_risk_score=overall_risk_score,
                risk_level=risk_level
            )
            
            # Store risk metrics
            portfolio['risk_metrics'] = risk_metrics
            self.risk_history[portfolio_id].append(risk_metrics)
            
            # Keep only recent history
            if len(self.risk_history[portfolio_id]) > 1000:
                self.risk_history[portfolio_id] = self.risk_history[portfolio_id][-1000:]
            
            # Check for risk alerts
            self._check_risk_alerts(portfolio_id, risk_metrics)
            
            logger.debug(f"Risk metrics calculated for {portfolio_id}: {risk_level.value} risk")
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics for {portfolio_id}: {str(e)}")
            return self._get_default_risk_metrics(portfolio_id)
    
    def _calculate_portfolio_value(self, positions: Dict[str, Any]) -> float:
        """Calculate total portfolio value."""
        total_value = 0.0
        
        for symbol, position in positions.items():
            if isinstance(position, dict):
                market_value = position.get('market_value', 0)
                quantity = position.get('quantity', 0)
                price = position.get('current_price', 0)
                
                if market_value > 0:
                    total_value += market_value
                elif quantity > 0 and price > 0:
                    total_value += quantity * price
        
        return total_value
    
    def _calculate_var_metrics(self, portfolio_id: str, positions: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate Value at Risk metrics.
        
        Args:
            portfolio_id: Portfolio identifier
            positions: Portfolio positions
            
        Returns:
            Dict containing VaR metrics
        """
        # Simplified VaR calculation using historical simulation
        portfolio = self.portfolios[portfolio_id]
        pnl_history = portfolio.get('pnl_history', [])
        
        if len(pnl_history) < 30:
            # Insufficient history, use parametric VaR
            portfolio_value = self._calculate_portfolio_value(positions)
            volatility = self._calculate_portfolio_volatility(positions)
            
            # Parametric VaR calculation (as percentage of portfolio value)
            var_95_pct = volatility * 1.645  # 95% confidence
            var_99_pct = volatility * 2.326  # 99% confidence
            
            var_95 = portfolio_value * var_95_pct
            var_99 = portfolio_value * var_99_pct
            cvar_95 = var_95 * 1.3  # Simplified CVaR approximation
        else:
            # Historical simulation VaR
            pnl_returns = np.array(pnl_history[-252:])  # Last year of data
            
            var_95 = np.percentile(pnl_returns, 5)  # 5th percentile
            var_99 = np.percentile(pnl_returns, 1)  # 1st percentile
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = np.mean(pnl_returns[pnl_returns <= var_95])
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95
        }
    
    def _calculate_drawdown_metrics(self, portfolio_id: str) -> Dict[str, float]:
        """
        Calculate drawdown metrics.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Dict containing drawdown metrics
        """
        portfolio = self.portfolios[portfolio_id]
        pnl_history = portfolio.get('pnl_history', [])
        
        if len(pnl_history) < 2:
            return {'max_drawdown': 0.0, 'current_drawdown': 0.0}
        
        # Calculate cumulative returns
        cumulative_returns = np.cumsum(pnl_history)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdowns
        drawdowns = (cumulative_returns - running_max) / np.maximum(running_max, 1)
        
        max_drawdown = abs(np.min(drawdowns))
        current_drawdown = abs(drawdowns[-1])
        
        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown
        }
    
    def _calculate_correlation_risk(self, positions: Dict[str, Any]) -> float:
        """
        Calculate portfolio correlation risk.
        
        Args:
            positions: Portfolio positions
            
        Returns:
            Correlation risk score (0-1)
        """
        # Simplified correlation risk calculation
        symbols = list(positions.keys())
        
        if len(symbols) < 2:
            return 0.0
        
        # Sector concentration as proxy for correlation risk
        sector_exposure = {}
        total_value = self._calculate_portfolio_value(positions)
        
        for symbol, position in positions.items():
            sector = self._get_symbol_sector(symbol)
            market_value = position.get('market_value', 0)
            
            if sector not in sector_exposure:
                sector_exposure[sector] = 0
            sector_exposure[sector] += market_value
        
        # Calculate sector concentration
        max_sector_concentration = 0.0
        if total_value > 0:
            for sector_value in sector_exposure.values():
                concentration = sector_value / total_value
                max_sector_concentration = max(max_sector_concentration, concentration)
        
        # Convert to risk score (higher concentration = higher risk)
        correlation_risk = min(1.0, max_sector_concentration * 2)
        
        return correlation_risk
    
    def _calculate_concentration_risk(self, positions: Dict[str, Any]) -> float:
        """
        Calculate portfolio concentration risk.
        
        Args:
            positions: Portfolio positions
            
        Returns:
            Concentration risk score (0-1)
        """
        total_value = self._calculate_portfolio_value(positions)
        
        if total_value == 0:
            return 0.0
        
        # Calculate position concentrations
        concentrations = []
        for position in positions.values():
            market_value = position.get('market_value', 0)
            concentration = market_value / total_value
            concentrations.append(concentration)
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        hhi = sum(c**2 for c in concentrations)
        
        # Convert to risk score (higher HHI = higher concentration risk)
        # HHI ranges from 1/n (perfectly diversified) to 1 (fully concentrated)
        n_positions = len(positions)
        min_hhi = 1.0 / n_positions if n_positions > 0 else 1.0
        
        concentration_risk = (hhi - min_hhi) / (1.0 - min_hhi) if min_hhi < 1.0 else 0.0
        
        return min(1.0, concentration_risk)
    
    def _calculate_liquidity_risk(self, positions: Dict[str, Any]) -> float:
        """
        Calculate portfolio liquidity risk.
        
        Args:
            positions: Portfolio positions
            
        Returns:
            Liquidity risk score (0-1)
        """
        # Simplified liquidity risk based on position sizes and symbols
        total_value = self._calculate_portfolio_value(positions)
        liquidity_risk = 0.0
        
        for symbol, position in positions.items():
            market_value = position.get('market_value', 0)
            weight = market_value / total_value if total_value > 0 else 0
            
            # Assign liquidity scores based on symbol (simplified)
            symbol_liquidity = self._get_symbol_liquidity_score(symbol)
            
            # Weight by position size
            liquidity_risk += weight * (1.0 - symbol_liquidity)
        
        return min(1.0, liquidity_risk)
    
    def _calculate_portfolio_volatility(self, positions: Dict[str, Any]) -> float:
        """
        Calculate portfolio volatility.
        
        Args:
            positions: Portfolio positions
            
        Returns:
            Portfolio volatility (annualized)
        """
        # Simplified volatility calculation
        total_value = self._calculate_portfolio_value(positions)
        weighted_volatility = 0.0
        
        for symbol, position in positions.items():
            market_value = position.get('market_value', 0)
            weight = market_value / total_value if total_value > 0 else 0
            
            # Get symbol volatility (simplified)
            symbol_volatility = self._get_symbol_volatility(symbol)
            
            weighted_volatility += weight * symbol_volatility
        
        return weighted_volatility
    
    def _calculate_portfolio_beta(self, positions: Dict[str, Any]) -> float:
        """
        Calculate portfolio beta.
        
        Args:
            positions: Portfolio positions
            
        Returns:
            Portfolio beta
        """
        # Simplified beta calculation
        total_value = self._calculate_portfolio_value(positions)
        weighted_beta = 0.0
        
        for symbol, position in positions.items():
            market_value = position.get('market_value', 0)
            weight = market_value / total_value if total_value > 0 else 0
            
            # Get symbol beta (simplified)
            symbol_beta = self._get_symbol_beta(symbol)
            
            weighted_beta += weight * symbol_beta
        
        return weighted_beta
    
    def _calculate_overall_risk_score(self, var_metrics: Dict[str, float], 
                                    drawdown_metrics: Dict[str, float],
                                    correlation_risk: float, concentration_risk: float,
                                    liquidity_risk: float, volatility: float) -> float:
        """
        Calculate overall portfolio risk score.
        
        Args:
            var_metrics: VaR metrics
            drawdown_metrics: Drawdown metrics
            correlation_risk: Correlation risk score
            concentration_risk: Concentration risk score
            liquidity_risk: Liquidity risk score
            volatility: Portfolio volatility
            
        Returns:
            Overall risk score (0-1)
        """
        # Normalize individual risk components
        var_score = min(1.0, var_metrics['var_95'] / (self.monitoring_config['risk_thresholds']['var_1day_95'] * 100000))  # Adjust for dollar amounts
        drawdown_score = min(1.0, drawdown_metrics['current_drawdown'] / self.monitoring_config['risk_thresholds']['max_drawdown'])
        volatility_score = min(1.0, volatility / self.monitoring_config['risk_thresholds']['volatility'])
        
        # Weight the risk components
        weights = {
            'var': 0.25,
            'drawdown': 0.25,
            'correlation': 0.15,
            'concentration': 0.15,
            'liquidity': 0.10,
            'volatility': 0.10
        }
        
        overall_score = (
            weights['var'] * var_score +
            weights['drawdown'] * drawdown_score +
            weights['correlation'] * correlation_risk +
            weights['concentration'] * concentration_risk +
            weights['liquidity'] * liquidity_risk +
            weights['volatility'] * volatility_score
        )
        
        return min(1.0, overall_score)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """
        Determine risk level based on overall risk score.
        
        Args:
            risk_score: Overall risk score (0-1)
            
        Returns:
            RiskLevel enum
        """
        if risk_score < 0.25:
            return RiskLevel.LOW
        elif risk_score < 0.50:
            return RiskLevel.MEDIUM
        elif risk_score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _check_risk_alerts(self, portfolio_id: str, risk_metrics: PortfolioRiskMetrics) -> None:
        """
        Check for risk threshold breaches and generate alerts.
        
        Args:
            portfolio_id: Portfolio identifier
            risk_metrics: Current risk metrics
        """
        alerts = []
        thresholds = self.monitoring_config['risk_thresholds']
        
        # VaR breach check (convert to percentage for comparison)
        portfolio_value = self._calculate_portfolio_value(self.portfolios[portfolio_id]['positions'])
        var_percentage = risk_metrics.var_1day_95 / portfolio_value if portfolio_value > 0 else 0
        
        if var_percentage > thresholds['var_1day_95']:
            alert = RiskAlert(
                alert_type=AlertType.VAR_BREACH,
                risk_level=RiskLevel.HIGH,
                message=f"VaR 95% breach: {var_percentage:.2%} > {thresholds['var_1day_95']:.2%}",
                timestamp=datetime.now(),
                portfolio_id=portfolio_id,
                metric_value=var_percentage,
                threshold_value=thresholds['var_1day_95'],
                recommended_action="Reduce position sizes or hedge exposure"
            )
            alerts.append(alert)
        
        # Drawdown check
        if risk_metrics.current_drawdown > thresholds['max_drawdown']:
            alert = RiskAlert(
                alert_type=AlertType.DRAWDOWN,
                risk_level=RiskLevel.CRITICAL,
                message=f"Drawdown breach: {risk_metrics.current_drawdown:.2%} > {thresholds['max_drawdown']:.2%}",
                timestamp=datetime.now(),
                portfolio_id=portfolio_id,
                metric_value=risk_metrics.current_drawdown,
                threshold_value=thresholds['max_drawdown'],
                recommended_action="Implement emergency risk reduction measures"
            )
            alerts.append(alert)
        
        # Concentration risk check
        if risk_metrics.concentration_risk > thresholds['concentration']:
            alert = RiskAlert(
                alert_type=AlertType.CONCENTRATION,
                risk_level=RiskLevel.MEDIUM,
                message=f"High concentration risk: {risk_metrics.concentration_risk:.2%} > {thresholds['concentration']:.2%}",
                timestamp=datetime.now(),
                portfolio_id=portfolio_id,
                metric_value=risk_metrics.concentration_risk,
                threshold_value=thresholds['concentration'],
                recommended_action="Diversify portfolio positions"
            )
            alerts.append(alert)
        
        # Correlation risk check
        if risk_metrics.correlation_risk > thresholds['correlation']:
            alert = RiskAlert(
                alert_type=AlertType.CORRELATION,
                risk_level=RiskLevel.MEDIUM,
                message=f"High correlation risk: {risk_metrics.correlation_risk:.2%} > {thresholds['correlation']:.2%}",
                timestamp=datetime.now(),
                portfolio_id=portfolio_id,
                metric_value=risk_metrics.correlation_risk,
                threshold_value=thresholds['correlation'],
                recommended_action="Reduce correlated positions"
            )
            alerts.append(alert)
        
        # Store and notify alerts
        for alert in alerts:
            self.active_alerts[portfolio_id].append(alert)
            
            # Keep only recent alerts
            if len(self.active_alerts[portfolio_id]) > 100:
                self.active_alerts[portfolio_id] = self.active_alerts[portfolio_id][-100:]
            
            # Trigger callback if provided
            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {str(e)}")
            
            logger.warning(f"Risk alert for {portfolio_id}: {alert.message}")
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol (simplified mapping)."""
        sector_mapping = {
            'TSLA': 'Automotive',
            'NVDA': 'Technology',
            'AAPL': 'Technology',
            'AMZN': 'Consumer',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'META': 'Technology',
            'NFLX': 'Communication'
        }
        return sector_mapping.get(symbol, 'Unknown')
    
    def _get_symbol_liquidity_score(self, symbol: str) -> float:
        """Get liquidity score for a symbol (0-1, higher = more liquid)."""
        liquidity_scores = {
            'AAPL': 0.95, 'MSFT': 0.95, 'AMZN': 0.90,
            'GOOGL': 0.90, 'TSLA': 0.85, 'NVDA': 0.85,
            'META': 0.85, 'NFLX': 0.80
        }
        return liquidity_scores.get(symbol, 0.70)  # Default for unknown symbols
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """Get volatility for a symbol (annualized)."""
        volatility_estimates = {
            'AAPL': 0.25, 'MSFT': 0.22, 'AMZN': 0.30,
            'GOOGL': 0.28, 'TSLA': 0.45, 'NVDA': 0.40,
            'META': 0.35, 'NFLX': 0.38
        }
        return volatility_estimates.get(symbol, 0.30)  # Default volatility
    
    def _get_symbol_beta(self, symbol: str) -> float:
        """Get beta for a symbol."""
        beta_estimates = {
            'AAPL': 1.2, 'MSFT': 0.9, 'AMZN': 1.3,
            'GOOGL': 1.1, 'TSLA': 2.0, 'NVDA': 1.8,
            'META': 1.4, 'NFLX': 1.2
        }
        return beta_estimates.get(symbol, 1.0)  # Default beta
    
    def _get_default_risk_metrics(self, portfolio_id: str) -> PortfolioRiskMetrics:
        """Get default risk metrics when calculation fails."""
        return PortfolioRiskMetrics(
            portfolio_id=portfolio_id,
            timestamp=datetime.now(),
            total_value=0.0,
            var_1day_95=0.0,
            var_1day_99=0.0,
            cvar_1day_95=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            volatility=0.0,
            beta=1.0,
            correlation_risk=0.0,
            concentration_risk=0.0,
            liquidity_risk=0.0,
            overall_risk_score=0.0,
            risk_level=RiskLevel.LOW
        )
    
    def get_portfolio_risk_summary(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Get comprehensive risk summary for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Dict containing risk summary
        """
        if portfolio_id not in self.portfolios:
            return {'error': f'Portfolio {portfolio_id} not found'}
        
        portfolio = self.portfolios[portfolio_id]
        risk_metrics = portfolio.get('risk_metrics')
        
        if not risk_metrics:
            return {'error': f'No risk metrics available for {portfolio_id}'}
        
        # Get recent alerts
        recent_alerts = [
            alert for alert in self.active_alerts.get(portfolio_id, [])
            if (datetime.now() - alert.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        # Calculate performance metrics
        avg_calculation_time = np.mean(self.calculation_times) if self.calculation_times else 0
        
        return {
            'portfolio_id': portfolio_id,
            'timestamp': risk_metrics.timestamp,
            'risk_level': risk_metrics.risk_level.value,
            'overall_risk_score': risk_metrics.overall_risk_score,
            'key_metrics': {
                'total_value': risk_metrics.total_value,
                'var_1day_95': risk_metrics.var_1day_95,
                'current_drawdown': risk_metrics.current_drawdown,
                'volatility': risk_metrics.volatility,
                'concentration_risk': risk_metrics.concentration_risk
            },
            'recent_alerts': len(recent_alerts),
            'alert_details': [
                {
                    'type': alert.alert_type.value,
                    'level': alert.risk_level.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp
                }
                for alert in recent_alerts[-5:]  # Last 5 alerts
            ],
            'monitoring_status': {
                'active': self.monitoring_active,
                'last_update': self.last_update,
                'avg_calculation_time': avg_calculation_time
            }
        }
    
    def get_all_portfolios_summary(self) -> Dict[str, Any]:
        """
        Get risk summary for all monitored portfolios.
        
        Returns:
            Dict containing summary for all portfolios
        """
        summaries = {}
        
        for portfolio_id in self.portfolios.keys():
            summaries[portfolio_id] = self.get_portfolio_risk_summary(portfolio_id)
        
        # Overall system metrics
        total_alerts = sum(len(alerts) for alerts in self.active_alerts.values())
        avg_risk_score = np.mean([
            portfolio['risk_metrics'].overall_risk_score 
            for portfolio in self.portfolios.values() 
            if portfolio.get('risk_metrics')
        ]) if self.portfolios else 0.0
        
        return {
            'timestamp': datetime.now(),
            'total_portfolios': len(self.portfolios),
            'monitoring_active': self.monitoring_active,
            'total_active_alerts': total_alerts,
            'average_risk_score': avg_risk_score,
            'portfolios': summaries
        }


# Example usage and testing
if __name__ == "__main__":
    def alert_handler(alert: RiskAlert):
        """Example alert handler."""
        print(f"ALERT: {alert.alert_type.value} - {alert.message}")
    
    # Create risk monitor
    monitor = PortfolioRiskMonitor(alert_callback=alert_handler)
    
    # Sample portfolio positions
    sample_positions = {
        'TSLA': {
            'quantity': 100,
            'current_price': 250.0,
            'market_value': 25000.0
        },
        'AAPL': {
            'quantity': 200,
            'current_price': 180.0,
            'market_value': 36000.0
        },
        'NVDA': {
            'quantity': 50,
            'current_price': 800.0,
            'market_value': 40000.0
        }
    }
    
    # Add portfolio to monitoring
    monitor.add_portfolio('test_portfolio', sample_positions)
    
    # Calculate initial risk metrics
    risk_metrics = monitor._calculate_portfolio_risk('test_portfolio')
    
    print("=== Portfolio Risk Monitoring Test ===")
    print(f"Portfolio Value: ${risk_metrics.total_value:,.2f}")
    print(f"Risk Level: {risk_metrics.risk_level.value}")
    print(f"Overall Risk Score: {risk_metrics.overall_risk_score:.3f}")
    print(f"VaR (95%): ${risk_metrics.var_1day_95:,.2f}")
    print(f"Current Drawdown: {risk_metrics.current_drawdown:.2%}")
    print(f"Concentration Risk: {risk_metrics.concentration_risk:.2%}")
    print(f"Correlation Risk: {risk_metrics.correlation_risk:.2%}")
    print(f"Portfolio Volatility: {risk_metrics.volatility:.2%}")
    
    # Get risk summary
    summary = monitor.get_portfolio_risk_summary('test_portfolio')
    print(f"\nRecent Alerts: {summary['recent_alerts']}")
    print(f"Monitoring Active: {summary['monitoring_status']['active']}")
    
    # Test with high-risk portfolio (concentrated position)
    high_risk_positions = {
        'TSLA': {
            'quantity': 1000,
            'current_price': 250.0,
            'market_value': 250000.0
        }
    }
    
    monitor.add_portfolio('high_risk_portfolio', high_risk_positions)
    high_risk_metrics = monitor._calculate_portfolio_risk('high_risk_portfolio')
    
    print(f"\n=== High Risk Portfolio Test ===")
    print(f"Risk Level: {high_risk_metrics.risk_level.value}")
    print(f"Concentration Risk: {high_risk_metrics.concentration_risk:.2%}")
    
    # Get all portfolios summary
    all_summary = monitor.get_all_portfolios_summary()
    print(f"\nTotal Portfolios: {all_summary['total_portfolios']}")
    print(f"Average Risk Score: {all_summary['average_risk_score']:.3f}")
    print(f"Total Active Alerts: {all_summary['total_active_alerts']}")

