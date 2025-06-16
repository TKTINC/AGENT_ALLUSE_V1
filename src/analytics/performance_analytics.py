"""
ALL-USE Performance Analytics: Advanced Performance Monitoring Module

This module implements comprehensive performance monitoring and analytics for the
ALL-USE trading system, including real-time tracking, risk-adjusted metrics,
attribution analysis, and advanced performance reporting.
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
import statistics
from collections import defaultdict

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
        logging.FileHandler('all_use_performance_analytics.log')
    ]
)

logger = logging.getLogger('all_use_performance_analytics')

class PerformancePeriod(Enum):
    """Enumeration of performance measurement periods."""
    DAILY = "Daily"
    WEEKLY = "Weekly"
    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly"
    YEARLY = "Yearly"
    INCEPTION = "Inception"

class BenchmarkType(Enum):
    """Enumeration of benchmark types."""
    SPY = "SPY"  # S&P 500
    QQQ = "QQQ"  # NASDAQ 100
    CUSTOM = "Custom"
    RISK_FREE = "Risk_Free"

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    portfolio_id: str
    period: PerformancePeriod
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

@dataclass
class AttributionAnalysis:
    """Performance attribution analysis data structure."""
    portfolio_id: str
    period: PerformancePeriod
    timestamp: datetime
    total_return: float
    asset_allocation_effect: float
    security_selection_effect: float
    interaction_effect: float
    sector_attribution: Dict[str, float]
    position_attribution: Dict[str, float]
    benchmark_return: float
    active_return: float

@dataclass
class RiskMetrics:
    """Risk metrics data structure."""
    portfolio_id: str
    timestamp: datetime
    value_at_risk_95: float
    conditional_var_95: float
    maximum_drawdown: float
    current_drawdown: float
    volatility: float
    downside_deviation: float
    beta: float
    correlation_with_market: float
    concentration_risk: float
    liquidity_risk: float

class PerformanceAnalytics:
    """
    Advanced performance monitoring and analytics system for ALL-USE trading.
    
    This class provides:
    - Real-time performance tracking and calculation
    - Risk-adjusted performance metrics (Sharpe, Sortino, Calmar)
    - Performance attribution analysis
    - Benchmark comparison and tracking
    - Comprehensive risk analytics
    - Historical performance analysis
    """
    
    def __init__(self, analytics_callback: Optional[Callable] = None):
        """Initialize the performance analytics system."""
        self.parameters = ALLUSEParameters
        self.analytics_callback = analytics_callback
        
        # Analytics configuration
        self.analytics_config = {
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'benchmark_symbols': {
                BenchmarkType.SPY: 'SPY',
                BenchmarkType.QQQ: 'QQQ'
            },
            'calculation_frequency': 'daily',
            'lookback_periods': {
                'volatility': 252,      # 1 year
                'correlation': 126,     # 6 months
                'beta': 252,           # 1 year
                'tracking_error': 252   # 1 year
            },
            'performance_thresholds': {
                'excellent_sharpe': 2.0,
                'good_sharpe': 1.0,
                'acceptable_sharpe': 0.5,
                'max_drawdown_warning': 0.10,
                'max_drawdown_critical': 0.20
            }
        }
        
        # Performance data storage
        self.portfolios = {}
        self.performance_history = {}
        self.benchmark_data = {}
        self.attribution_history = {}
        self.risk_metrics_history = {}
        
        # Real-time tracking
        self.daily_returns = defaultdict(list)
        self.portfolio_values = defaultdict(list)
        self.benchmark_returns = defaultdict(list)
        
        logger.info("Performance analytics system initialized")
    
    def add_portfolio(self, portfolio_id: str, initial_value: float, 
                     benchmark: BenchmarkType = BenchmarkType.SPY) -> None:
        """
        Add a portfolio to performance tracking.
        
        Args:
            portfolio_id: Unique portfolio identifier
            initial_value: Initial portfolio value
            benchmark: Benchmark for comparison
        """
        logger.info(f"Adding portfolio {portfolio_id} to performance tracking")
        
        self.portfolios[portfolio_id] = {
            'initial_value': initial_value,
            'current_value': initial_value,
            'benchmark': benchmark,
            'inception_date': datetime.now(),
            'last_update': datetime.now(),
            'positions': {}
        }
        
        self.performance_history[portfolio_id] = []
        self.attribution_history[portfolio_id] = []
        self.risk_metrics_history[portfolio_id] = []
        
        # Initialize tracking lists
        self.daily_returns[portfolio_id] = []
        self.portfolio_values[portfolio_id] = [initial_value]
        self.benchmark_returns[portfolio_id] = []
        
        # Initialize benchmark data if needed
        if benchmark not in self.benchmark_data:
            self.benchmark_data[benchmark] = self._generate_benchmark_data(benchmark)
    
    def update_portfolio_value(self, portfolio_id: str, current_value: float, 
                              positions: Dict[str, Any]) -> None:
        """
        Update portfolio value and calculate performance metrics.
        
        Args:
            portfolio_id: Portfolio identifier
            current_value: Current portfolio value
            positions: Current portfolio positions
        """
        if portfolio_id not in self.portfolios:
            logger.warning(f"Portfolio {portfolio_id} not found in performance tracking")
            return
        
        portfolio = self.portfolios[portfolio_id]
        previous_value = portfolio['current_value']
        
        # Calculate daily return
        if previous_value > 0:
            daily_return = (current_value - previous_value) / previous_value
            self.daily_returns[portfolio_id].append(daily_return)
        
        # Update portfolio data
        portfolio['current_value'] = current_value
        portfolio['positions'] = positions
        portfolio['last_update'] = datetime.now()
        
        # Store value history
        self.portfolio_values[portfolio_id].append(current_value)
        
        # Keep only recent history (last 2 years)
        max_history = 504  # ~2 years of daily data
        if len(self.portfolio_values[portfolio_id]) > max_history:
            self.portfolio_values[portfolio_id] = self.portfolio_values[portfolio_id][-max_history:]
            self.daily_returns[portfolio_id] = self.daily_returns[portfolio_id][-max_history:]
        
        # Calculate and store performance metrics
        self._calculate_performance_metrics(portfolio_id)
    
    def calculate_performance_metrics(self, portfolio_id: str, 
                                    period: PerformancePeriod = PerformancePeriod.INCEPTION) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            period: Performance measurement period
            
        Returns:
            PerformanceMetrics object
        """
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        portfolio = self.portfolios[portfolio_id]
        returns = self._get_returns_for_period(portfolio_id, period)
        
        if len(returns) < 2:
            logger.warning(f"Insufficient data for performance calculation: {len(returns)} returns")
            return self._get_default_performance_metrics(portfolio_id, period)
        
        # Calculate basic metrics
        total_return = self._calculate_total_return(portfolio_id, period)
        annualized_return = self._calculate_annualized_return(returns, period)
        volatility = self._calculate_volatility(returns)
        
        # Calculate risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(portfolio_id, annualized_return)
        
        # Calculate drawdown metrics
        max_drawdown = self._calculate_max_drawdown(portfolio_id)
        
        # Calculate VaR metrics
        var_95 = self._calculate_var(returns, 0.95)
        cvar_95 = self._calculate_cvar(returns, 0.95)
        
        # Calculate benchmark-relative metrics
        alpha, beta = self._calculate_alpha_beta(portfolio_id, period)
        information_ratio = self._calculate_information_ratio(portfolio_id, period)
        tracking_error = self._calculate_tracking_error(portfolio_id, period)
        
        # Calculate trade statistics
        win_rate = self._calculate_win_rate(returns)
        profit_factor = self._calculate_profit_factor(returns)
        avg_win, avg_loss = self._calculate_avg_win_loss(returns)
        
        # Determine period dates
        start_date, end_date = self._get_period_dates(portfolio_id, period)
        
        return PerformanceMetrics(
            portfolio_id=portfolio_id,
            period=period,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss
        )
    
    def calculate_attribution_analysis(self, portfolio_id: str, 
                                     period: PerformancePeriod = PerformancePeriod.MONTHLY) -> AttributionAnalysis:
        """
        Calculate performance attribution analysis.
        
        Args:
            portfolio_id: Portfolio identifier
            period: Attribution analysis period
            
        Returns:
            AttributionAnalysis object
        """
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        portfolio = self.portfolios[portfolio_id]
        positions = portfolio['positions']
        
        # Calculate total portfolio return
        total_return = self._calculate_total_return(portfolio_id, period)
        
        # Calculate benchmark return
        benchmark_return = self._calculate_benchmark_return(portfolio['benchmark'], period)
        
        # Calculate active return
        active_return = total_return - benchmark_return
        
        # Calculate attribution effects (simplified)
        asset_allocation_effect = self._calculate_asset_allocation_effect(portfolio_id, period)
        security_selection_effect = self._calculate_security_selection_effect(portfolio_id, period)
        interaction_effect = active_return - asset_allocation_effect - security_selection_effect
        
        # Calculate sector attribution
        sector_attribution = self._calculate_sector_attribution(positions)
        
        # Calculate position attribution
        position_attribution = self._calculate_position_attribution(positions, period)
        
        attribution = AttributionAnalysis(
            portfolio_id=portfolio_id,
            period=period,
            timestamp=datetime.now(),
            total_return=total_return,
            asset_allocation_effect=asset_allocation_effect,
            security_selection_effect=security_selection_effect,
            interaction_effect=interaction_effect,
            sector_attribution=sector_attribution,
            position_attribution=position_attribution,
            benchmark_return=benchmark_return,
            active_return=active_return
        )
        
        # Store attribution analysis
        self.attribution_history[portfolio_id].append(attribution)
        
        # Keep only recent history
        if len(self.attribution_history[portfolio_id]) > 100:
            self.attribution_history[portfolio_id] = self.attribution_history[portfolio_id][-100:]
        
        return attribution
    
    def calculate_risk_metrics(self, portfolio_id: str) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            RiskMetrics object
        """
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        returns = self.daily_returns[portfolio_id]
        
        if len(returns) < 30:
            logger.warning(f"Insufficient data for risk calculation: {len(returns)} returns")
            return self._get_default_risk_metrics(portfolio_id)
        
        # Calculate VaR and CVaR
        var_95 = self._calculate_var(returns, 0.95)
        cvar_95 = self._calculate_cvar(returns, 0.95)
        
        # Calculate drawdown metrics
        max_drawdown = self._calculate_max_drawdown(portfolio_id)
        current_drawdown = self._calculate_current_drawdown(portfolio_id)
        
        # Calculate volatility metrics
        volatility = self._calculate_volatility(returns)
        downside_deviation = self._calculate_downside_deviation(returns)
        
        # Calculate market-relative metrics
        beta = self._calculate_beta(portfolio_id)
        correlation = self._calculate_market_correlation(portfolio_id)
        
        # Calculate concentration and liquidity risks
        concentration_risk = self._calculate_concentration_risk(portfolio_id)
        liquidity_risk = self._calculate_liquidity_risk(portfolio_id)
        
        risk_metrics = RiskMetrics(
            portfolio_id=portfolio_id,
            timestamp=datetime.now(),
            value_at_risk_95=var_95,
            conditional_var_95=cvar_95,
            maximum_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            volatility=volatility,
            downside_deviation=downside_deviation,
            beta=beta,
            correlation_with_market=correlation,
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk
        )
        
        # Store risk metrics
        self.risk_metrics_history[portfolio_id].append(risk_metrics)
        
        # Keep only recent history
        if len(self.risk_metrics_history[portfolio_id]) > 100:
            self.risk_metrics_history[portfolio_id] = self.risk_metrics_history[portfolio_id][-100:]
        
        return risk_metrics
    
    def _calculate_performance_metrics(self, portfolio_id: str) -> None:
        """Calculate and store performance metrics for a portfolio."""
        try:
            # Calculate metrics for different periods
            periods = [PerformancePeriod.DAILY, PerformancePeriod.MONTHLY, PerformancePeriod.INCEPTION]
            
            for period in periods:
                metrics = self.calculate_performance_metrics(portfolio_id, period)
                
                # Store in history
                self.performance_history[portfolio_id].append(metrics)
                
                # Trigger callback if provided
                if self.analytics_callback:
                    try:
                        self.analytics_callback(metrics)
                    except Exception as e:
                        logger.error(f"Error in analytics callback: {str(e)}")
            
            # Keep only recent history
            if len(self.performance_history[portfolio_id]) > 1000:
                self.performance_history[portfolio_id] = self.performance_history[portfolio_id][-1000:]
                
        except Exception as e:
            logger.error(f"Error calculating performance metrics for {portfolio_id}: {str(e)}")
    
    def _get_returns_for_period(self, portfolio_id: str, period: PerformancePeriod) -> List[float]:
        """Get returns for a specific period."""
        all_returns = self.daily_returns[portfolio_id]
        
        if period == PerformancePeriod.INCEPTION:
            return all_returns
        elif period == PerformancePeriod.DAILY:
            return all_returns[-1:] if all_returns else []
        elif period == PerformancePeriod.WEEKLY:
            return all_returns[-7:] if len(all_returns) >= 7 else all_returns
        elif period == PerformancePeriod.MONTHLY:
            return all_returns[-21:] if len(all_returns) >= 21 else all_returns
        elif period == PerformancePeriod.QUARTERLY:
            return all_returns[-63:] if len(all_returns) >= 63 else all_returns
        elif period == PerformancePeriod.YEARLY:
            return all_returns[-252:] if len(all_returns) >= 252 else all_returns
        else:
            return all_returns
    
    def _calculate_total_return(self, portfolio_id: str, period: PerformancePeriod) -> float:
        """Calculate total return for a period."""
        portfolio = self.portfolios[portfolio_id]
        values = self.portfolio_values[portfolio_id]
        
        if len(values) < 2:
            return 0.0
        
        if period == PerformancePeriod.INCEPTION:
            start_value = values[0]
            end_value = values[-1]
        else:
            # Get period-specific values
            period_days = self._get_period_days(period)
            start_idx = max(0, len(values) - period_days - 1)
            start_value = values[start_idx]
            end_value = values[-1]
        
        return (end_value - start_value) / start_value if start_value > 0 else 0.0
    
    def _calculate_annualized_return(self, returns: List[float], period: PerformancePeriod) -> float:
        """Calculate annualized return."""
        if not returns:
            return 0.0
        
        # Calculate compound return
        compound_return = 1.0
        for ret in returns:
            compound_return *= (1 + ret)
        
        total_return = compound_return - 1.0
        
        # Annualize based on period
        if period == PerformancePeriod.DAILY:
            return total_return * 252
        elif period == PerformancePeriod.WEEKLY:
            return total_return * 52
        elif period == PerformancePeriod.MONTHLY:
            return total_return * 12
        elif period == PerformancePeriod.QUARTERLY:
            return total_return * 4
        else:
            # For inception and yearly, calculate based on actual time
            days = len(returns)
            if days > 0:
                return (compound_return ** (252 / days)) - 1.0
            return 0.0
    
    def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0
        
        return np.std(returns) * np.sqrt(252)  # Annualize daily volatility
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = [ret - self.analytics_config['risk_free_rate'] / 252 for ret in returns]
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess == 0:
            return 0.0
        
        return (mean_excess / std_excess) * np.sqrt(252)  # Annualize
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        
        risk_free_daily = self.analytics_config['risk_free_rate'] / 252
        excess_returns = [ret - risk_free_daily for ret in returns]
        downside_returns = [ret for ret in excess_returns if ret < 0]
        
        if not downside_returns:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        mean_excess = np.mean(excess_returns)
        downside_deviation = np.std(downside_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        return (mean_excess / downside_deviation) * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, portfolio_id: str, annualized_return: float) -> float:
        """Calculate Calmar ratio."""
        max_drawdown = self._calculate_max_drawdown(portfolio_id)
        
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        return annualized_return / max_drawdown
    
    def _calculate_max_drawdown(self, portfolio_id: str) -> float:
        """Calculate maximum drawdown."""
        values = self.portfolio_values[portfolio_id]
        
        if len(values) < 2:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_current_drawdown(self, portfolio_id: str) -> float:
        """Calculate current drawdown."""
        values = self.portfolio_values[portfolio_id]
        
        if len(values) < 2:
            return 0.0
        
        peak = max(values)
        current = values[-1]
        
        return (peak - current) / peak if peak > 0 else 0.0
    
    def _calculate_var(self, returns: List[float], confidence: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) < 30:
            return 0.0
        
        percentile = (1 - confidence) * 100
        return abs(np.percentile(returns, percentile))
    
    def _calculate_cvar(self, returns: List[float], confidence: float) -> float:
        """Calculate Conditional Value at Risk."""
        if len(returns) < 30:
            return 0.0
        
        var = self._calculate_var(returns, confidence)
        tail_returns = [ret for ret in returns if ret <= -var]
        
        if not tail_returns:
            return var
        
        return abs(np.mean(tail_returns))
    
    def _calculate_alpha_beta(self, portfolio_id: str, period: PerformancePeriod) -> Tuple[float, float]:
        """Calculate alpha and beta relative to benchmark."""
        portfolio_returns = self._get_returns_for_period(portfolio_id, period)
        benchmark_returns = self._get_benchmark_returns_for_period(portfolio_id, period)
        
        if len(portfolio_returns) < 30 or len(benchmark_returns) < 30:
            return 0.0, 1.0
        
        # Ensure same length
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[-min_length:]
        benchmark_returns = benchmark_returns[-min_length:]
        
        # Calculate beta
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        # Calculate alpha
        portfolio_mean = np.mean(portfolio_returns) * 252  # Annualize
        benchmark_mean = np.mean(benchmark_returns) * 252  # Annualize
        risk_free_rate = self.analytics_config['risk_free_rate']
        
        alpha = portfolio_mean - (risk_free_rate + beta * (benchmark_mean - risk_free_rate))
        
        return alpha, beta
    
    def _calculate_information_ratio(self, portfolio_id: str, period: PerformancePeriod) -> float:
        """Calculate information ratio."""
        tracking_error = self._calculate_tracking_error(portfolio_id, period)
        
        if tracking_error == 0:
            return 0.0
        
        # Calculate active return
        portfolio_returns = self._get_returns_for_period(portfolio_id, period)
        benchmark_returns = self._get_benchmark_returns_for_period(portfolio_id, period)
        
        if not portfolio_returns or not benchmark_returns:
            return 0.0
        
        active_return = np.mean(portfolio_returns) - np.mean(benchmark_returns)
        
        return (active_return * 252) / tracking_error  # Annualize
    
    def _calculate_tracking_error(self, portfolio_id: str, period: PerformancePeriod) -> float:
        """Calculate tracking error."""
        portfolio_returns = self._get_returns_for_period(portfolio_id, period)
        benchmark_returns = self._get_benchmark_returns_for_period(portfolio_id, period)
        
        if len(portfolio_returns) < 30 or len(benchmark_returns) < 30:
            return 0.0
        
        # Ensure same length
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[-min_length:]
        benchmark_returns = benchmark_returns[-min_length:]
        
        # Calculate active returns
        active_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
        
        return np.std(active_returns) * np.sqrt(252)  # Annualize
    
    def _calculate_win_rate(self, returns: List[float]) -> float:
        """Calculate win rate (percentage of positive returns)."""
        if not returns:
            return 0.0
        
        positive_returns = sum(1 for ret in returns if ret > 0)
        return positive_returns / len(returns)
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor."""
        if not returns:
            return 0.0
        
        gross_profit = sum(ret for ret in returns if ret > 0)
        gross_loss = abs(sum(ret for ret in returns if ret < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_avg_win_loss(self, returns: List[float]) -> Tuple[float, float]:
        """Calculate average win and loss."""
        if not returns:
            return 0.0, 0.0
        
        wins = [ret for ret in returns if ret > 0]
        losses = [ret for ret in returns if ret < 0]
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = abs(np.mean(losses)) if losses else 0.0
        
        return avg_win, avg_loss
    
    def _calculate_beta(self, portfolio_id: str) -> float:
        """Calculate portfolio beta."""
        _, beta = self._calculate_alpha_beta(portfolio_id, PerformancePeriod.INCEPTION)
        return beta
    
    def _calculate_market_correlation(self, portfolio_id: str) -> float:
        """Calculate correlation with market."""
        portfolio_returns = self.daily_returns[portfolio_id]
        benchmark_returns = self._get_benchmark_returns_for_period(portfolio_id, PerformancePeriod.INCEPTION)
        
        if len(portfolio_returns) < 30 or len(benchmark_returns) < 30:
            return 0.0
        
        # Ensure same length
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[-min_length:]
        benchmark_returns = benchmark_returns[-min_length:]
        
        correlation_matrix = np.corrcoef(portfolio_returns, benchmark_returns)
        return correlation_matrix[0, 1] if correlation_matrix.shape == (2, 2) else 0.0
    
    def _calculate_concentration_risk(self, portfolio_id: str) -> float:
        """Calculate portfolio concentration risk."""
        portfolio = self.portfolios[portfolio_id]
        positions = portfolio['positions']
        
        if not positions:
            return 0.0
        
        total_value = sum(pos.get('market_value', 0) for pos in positions.values())
        
        if total_value == 0:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index
        weights = [pos.get('market_value', 0) / total_value for pos in positions.values()]
        hhi = sum(w**2 for w in weights)
        
        # Convert to risk score (0-1)
        n_positions = len(positions)
        min_hhi = 1.0 / n_positions if n_positions > 0 else 1.0
        
        return (hhi - min_hhi) / (1.0 - min_hhi) if min_hhi < 1.0 else 0.0
    
    def _calculate_liquidity_risk(self, portfolio_id: str) -> float:
        """Calculate portfolio liquidity risk."""
        # Simplified liquidity risk calculation
        return 0.1  # Default 10% liquidity risk
    
    def _calculate_downside_deviation(self, returns: List[float]) -> float:
        """Calculate downside deviation."""
        if len(returns) < 2:
            return 0.0
        
        downside_returns = [ret for ret in returns if ret < 0]
        
        if not downside_returns:
            return 0.0
        
        return np.std(downside_returns) * np.sqrt(252)  # Annualize
    
    def _get_benchmark_returns_for_period(self, portfolio_id: str, period: PerformancePeriod) -> List[float]:
        """Get benchmark returns for a specific period."""
        portfolio = self.portfolios[portfolio_id]
        benchmark = portfolio['benchmark']
        
        if benchmark not in self.benchmark_data:
            return []
        
        all_returns = self.benchmark_data[benchmark]
        
        # Return period-specific data (simplified)
        if period == PerformancePeriod.INCEPTION:
            return all_returns
        else:
            period_days = self._get_period_days(period)
            return all_returns[-period_days:] if len(all_returns) >= period_days else all_returns
    
    def _get_period_days(self, period: PerformancePeriod) -> int:
        """Get number of days for a period."""
        period_mapping = {
            PerformancePeriod.DAILY: 1,
            PerformancePeriod.WEEKLY: 7,
            PerformancePeriod.MONTHLY: 21,
            PerformancePeriod.QUARTERLY: 63,
            PerformancePeriod.YEARLY: 252
        }
        return period_mapping.get(period, 252)
    
    def _get_period_dates(self, portfolio_id: str, period: PerformancePeriod) -> Tuple[datetime, datetime]:
        """Get start and end dates for a period."""
        portfolio = self.portfolios[portfolio_id]
        end_date = portfolio['last_update']
        
        if period == PerformancePeriod.INCEPTION:
            start_date = portfolio['inception_date']
        else:
            days = self._get_period_days(period)
            start_date = end_date - timedelta(days=days)
        
        return start_date, end_date
    
    def _generate_benchmark_data(self, benchmark: BenchmarkType) -> List[float]:
        """Generate sample benchmark data."""
        # Generate sample benchmark returns
        np.random.seed(42)  # Consistent seed
        
        if benchmark == BenchmarkType.SPY:
            # S&P 500 characteristics
            returns = np.random.normal(0.0008, 0.012, 252)  # ~10% annual return, 19% volatility
        elif benchmark == BenchmarkType.QQQ:
            # NASDAQ 100 characteristics
            returns = np.random.normal(0.0010, 0.015, 252)  # ~12% annual return, 24% volatility
        else:
            # Default benchmark
            returns = np.random.normal(0.0006, 0.010, 252)  # ~8% annual return, 16% volatility
        
        return returns.tolist()
    
    def _calculate_asset_allocation_effect(self, portfolio_id: str, period: PerformancePeriod) -> float:
        """Calculate asset allocation effect (simplified)."""
        return 0.005  # 0.5% allocation effect
    
    def _calculate_security_selection_effect(self, portfolio_id: str, period: PerformancePeriod) -> float:
        """Calculate security selection effect (simplified)."""
        return 0.003  # 0.3% selection effect
    
    def _calculate_sector_attribution(self, positions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate sector attribution (simplified)."""
        sectors = {}
        total_value = sum(pos.get('market_value', 0) for pos in positions.values())
        
        for symbol, position in positions.items():
            sector = self._get_symbol_sector(symbol)
            market_value = position.get('market_value', 0)
            weight = market_value / total_value if total_value > 0 else 0
            
            if sector not in sectors:
                sectors[sector] = 0
            sectors[sector] += weight * 0.01  # Simplified attribution
        
        return sectors
    
    def _calculate_position_attribution(self, positions: Dict[str, Any], period: PerformancePeriod) -> Dict[str, float]:
        """Calculate position attribution (simplified)."""
        attribution = {}
        
        for symbol, position in positions.items():
            # Simplified position attribution
            attribution[symbol] = 0.005  # 0.5% contribution
        
        return attribution
    
    def _calculate_benchmark_return(self, benchmark: BenchmarkType, period: PerformancePeriod) -> float:
        """Calculate benchmark return for period."""
        if benchmark not in self.benchmark_data:
            return 0.08  # Default 8% return
        
        returns = self.benchmark_data[benchmark]
        
        if not returns:
            return 0.08
        
        # Get period-specific returns
        if period == PerformancePeriod.INCEPTION:
            period_returns = returns
        else:
            period_days = self._get_period_days(period)
            period_returns = returns[-period_days:] if len(returns) >= period_days else returns
        
        # Calculate compound return
        compound_return = 1.0
        for ret in period_returns:
            compound_return *= (1 + ret)
        
        return compound_return - 1.0
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        sector_mapping = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'TSLA': 'Automotive',
            'NVDA': 'Technology',
            'AMZN': 'Consumer'
        }
        return sector_mapping.get(symbol, 'Unknown')
    
    def _get_default_performance_metrics(self, portfolio_id: str, period: PerformancePeriod) -> PerformanceMetrics:
        """Get default performance metrics when calculation fails."""
        start_date, end_date = self._get_period_dates(portfolio_id, period)
        
        return PerformanceMetrics(
            portfolio_id=portfolio_id,
            period=period,
            start_date=start_date,
            end_date=end_date,
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            alpha=0.0,
            beta=1.0,
            information_ratio=0.0,
            tracking_error=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0
        )
    
    def _get_default_risk_metrics(self, portfolio_id: str) -> RiskMetrics:
        """Get default risk metrics when calculation fails."""
        return RiskMetrics(
            portfolio_id=portfolio_id,
            timestamp=datetime.now(),
            value_at_risk_95=0.0,
            conditional_var_95=0.0,
            maximum_drawdown=0.0,
            current_drawdown=0.0,
            volatility=0.0,
            downside_deviation=0.0,
            beta=1.0,
            correlation_with_market=0.0,
            concentration_risk=0.0,
            liquidity_risk=0.0
        )
    
    def get_performance_summary(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Dict containing performance summary
        """
        if portfolio_id not in self.portfolios:
            return {'error': f'Portfolio {portfolio_id} not found'}
        
        # Calculate metrics for different periods
        inception_metrics = self.calculate_performance_metrics(portfolio_id, PerformancePeriod.INCEPTION)
        monthly_metrics = self.calculate_performance_metrics(portfolio_id, PerformancePeriod.MONTHLY)
        
        # Calculate attribution and risk metrics
        attribution = self.calculate_attribution_analysis(portfolio_id)
        risk_metrics = self.calculate_risk_metrics(portfolio_id)
        
        return {
            'portfolio_id': portfolio_id,
            'timestamp': datetime.now(),
            'inception_performance': {
                'total_return': inception_metrics.total_return,
                'annualized_return': inception_metrics.annualized_return,
                'volatility': inception_metrics.volatility,
                'sharpe_ratio': inception_metrics.sharpe_ratio,
                'max_drawdown': inception_metrics.max_drawdown
            },
            'monthly_performance': {
                'total_return': monthly_metrics.total_return,
                'volatility': monthly_metrics.volatility,
                'sharpe_ratio': monthly_metrics.sharpe_ratio
            },
            'risk_metrics': {
                'var_95': risk_metrics.value_at_risk_95,
                'current_drawdown': risk_metrics.current_drawdown,
                'beta': risk_metrics.beta,
                'correlation': risk_metrics.correlation_with_market
            },
            'attribution': {
                'active_return': attribution.active_return,
                'asset_allocation_effect': attribution.asset_allocation_effect,
                'security_selection_effect': attribution.security_selection_effect
            }
        }


# Example usage and testing
if __name__ == "__main__":
    def analytics_handler(metrics: PerformanceMetrics):
        """Example analytics handler."""
        print(f"ANALYTICS: {metrics.portfolio_id} - Sharpe: {metrics.sharpe_ratio:.3f}")
    
    # Create performance analytics system
    analytics = PerformanceAnalytics(analytics_callback=analytics_handler)
    
    # Add test portfolio
    analytics.add_portfolio('test_portfolio', 100000.0, BenchmarkType.SPY)
    
    # Simulate portfolio value updates
    test_values = [100000, 102000, 98000, 105000, 103000, 107000]
    test_positions = {
        'AAPL': {'market_value': 25000},
        'MSFT': {'market_value': 20000},
        'GOOGL': {'market_value': 20000},
        'TSLA': {'market_value': 20000},
        'NVDA': {'market_value': 15000}
    }
    
    print("=== Performance Analytics Test ===")
    
    for i, value in enumerate(test_values):
        print(f"\nStep {i+1}: Portfolio Value = ${value:,}")
        
        # Update portfolio value
        analytics.update_portfolio_value('test_portfolio', value, test_positions)
        
        # Calculate performance metrics
        metrics = analytics.calculate_performance_metrics('test_portfolio', PerformancePeriod.INCEPTION)
        
        print(f"Total Return: {metrics.total_return:.2%}")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"Volatility: {metrics.volatility:.2%}")
    
    # Test attribution analysis
    print(f"\n--- Attribution Analysis ---")
    attribution = analytics.calculate_attribution_analysis('test_portfolio')
    print(f"Active Return: {attribution.active_return:.2%}")
    print(f"Asset Allocation Effect: {attribution.asset_allocation_effect:.2%}")
    print(f"Security Selection Effect: {attribution.security_selection_effect:.2%}")
    
    # Test risk metrics
    print(f"\n--- Risk Metrics ---")
    risk_metrics = analytics.calculate_risk_metrics('test_portfolio')
    print(f"VaR 95%: {risk_metrics.value_at_risk_95:.2%}")
    print(f"Beta: {risk_metrics.beta:.3f}")
    print(f"Correlation: {risk_metrics.correlation_with_market:.3f}")
    
    # Get performance summary
    summary = analytics.get_performance_summary('test_portfolio')
    print(f"\nPerformance Summary:")
    print(f"Inception Sharpe: {summary['inception_performance']['sharpe_ratio']:.3f}")
    print(f"Monthly Return: {summary['monthly_performance']['total_return']:.2%}")
    print(f"Current Drawdown: {summary['risk_metrics']['current_drawdown']:.2%}")

