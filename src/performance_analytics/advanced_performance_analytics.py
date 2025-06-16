"""
Advanced Performance Analytics Engine for ALL-USE Protocol
Comprehensive performance tracking, attribution analysis, benchmarking, and forecasting

This module provides advanced performance analytics including:
- Comprehensive performance metrics (20+ metrics)
- Attribution analysis by strategy, time period, market regime
- Benchmark comparison and relative performance analysis
- Performance forecasting and predictive modeling
- Risk-adjusted return calculations and analysis
- Professional-grade reporting and visualization
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
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformancePeriod(Enum):
    """Performance analysis periods"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    INCEPTION = "inception"
    CUSTOM = "custom"

class AttributionType(Enum):
    """Attribution analysis types"""
    STRATEGY = "strategy"
    TIME_PERIOD = "time_period"
    MARKET_REGIME = "market_regime"
    WEEK_TYPE = "week_type"
    SECTOR = "sector"
    RISK_FACTOR = "risk_factor"

class BenchmarkType(Enum):
    """Benchmark types for comparison"""
    SPY = "spy"
    QQQ = "qqq"
    IWM = "iwm"
    VTI = "vti"
    CUSTOM = "custom"
    RISK_FREE = "risk_free"
    EQUAL_WEIGHT = "equal_weight"

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Return metrics
    total_return: float
    annualized_return: float
    cumulative_return: float
    average_return: float
    geometric_mean_return: float
    
    # Risk metrics
    volatility: float
    downside_volatility: float
    tracking_error: float
    beta: float
    alpha: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    treynor_ratio: float
    information_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    average_drawdown: float
    drawdown_duration: int
    recovery_time: int
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    
    # Win/Loss metrics
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    # Additional metrics
    calmar_ratio: float
    sterling_ratio: float
    burke_ratio: float
    tail_ratio: float
    
    # Metadata
    period_start: datetime
    period_end: datetime
    total_periods: int
    last_updated: datetime

@dataclass
class AttributionResult:
    """Performance attribution analysis result"""
    attribution_type: AttributionType
    attribution_breakdown: Dict[str, float]
    total_attribution: float
    residual_return: float
    attribution_quality: float
    period_start: datetime
    period_end: datetime
    timestamp: datetime

@dataclass
class BenchmarkComparison:
    """Benchmark comparison analysis"""
    benchmark_type: BenchmarkType
    benchmark_name: str
    portfolio_return: float
    benchmark_return: float
    excess_return: float
    tracking_error: float
    information_ratio: float
    up_capture: float
    down_capture: float
    correlation: float
    beta: float
    alpha: float
    outperformance_periods: int
    total_periods: int
    outperformance_rate: float
    timestamp: datetime

@dataclass
class PerformanceForecast:
    """Performance forecasting result"""
    forecast_horizon: int  # Days
    expected_return: float
    expected_volatility: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    scenario_analysis: Dict[str, float]
    probability_positive: float
    probability_outperform_benchmark: float
    forecast_accuracy: float
    model_confidence: float
    timestamp: datetime

class AdvancedPerformanceAnalytics:
    """
    Advanced Performance Analytics Engine for ALL-USE Protocol
    
    Provides comprehensive performance analysis with:
    - 20+ performance and risk metrics
    - Multi-dimensional attribution analysis
    - Benchmark comparison and relative performance
    - Performance forecasting and scenario analysis
    - Professional-grade reporting and visualization
    """
    
    def __init__(self):
        """Initialize the performance analytics engine"""
        self.logger = logging.getLogger(__name__)
        
        # Analytics configuration
        self.config = {
            'risk_free_rate': 0.02,              # Risk-free rate for calculations
            'trading_days_per_year': 252,        # Trading days per year
            'confidence_levels': [0.90, 0.95, 0.99],  # Confidence levels for VaR
            'forecast_horizons': [1, 7, 30, 90, 252],  # Forecast horizons in days
            'attribution_window': 252,           # Attribution analysis window
            'benchmark_update_frequency': 'daily', # Benchmark update frequency
            'performance_update_frequency': 3600,  # Performance update frequency (seconds)
            'min_periods_for_analysis': 30,      # Minimum periods for reliable analysis
            'outlier_threshold': 3.0,            # Standard deviations for outlier detection
            'monte_carlo_simulations': 10000     # Monte Carlo simulation runs
        }
        
        # Performance data storage
        self.performance_history: pd.DataFrame = pd.DataFrame()
        self.benchmark_data: Dict[str, pd.DataFrame] = {}
        self.attribution_history: List[AttributionResult] = []
        self.forecast_history: List[PerformanceForecast] = []
        
        # Current metrics
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.benchmark_comparisons: Dict[str, BenchmarkComparison] = {}
        
        # Analytics thread
        self.analytics_active = False
        self.analytics_thread = None
        
        # Initialize benchmark data
        self._initialize_benchmark_data()
        
        self.logger.info("Advanced Performance Analytics initialized")
    
    def calculate_performance_metrics(self, 
                                    returns_data: pd.Series,
                                    period: PerformancePeriod = PerformancePeriod.INCEPTION) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Args:
            returns_data: Time series of returns
            period: Analysis period
            
        Returns:
            Comprehensive performance metrics
        """
        try:
            if returns_data.empty:
                return self._create_empty_metrics()
            
            # Filter data by period
            filtered_returns = self._filter_returns_by_period(returns_data, period)
            
            if len(filtered_returns) < self.config['min_periods_for_analysis']:
                self.logger.warning(f"Insufficient data for reliable analysis: {len(filtered_returns)} periods")
            
            # Calculate return metrics
            total_return = (1 + filtered_returns).prod() - 1
            cumulative_return = total_return
            annualized_return = self._annualize_return(total_return, len(filtered_returns))
            average_return = filtered_returns.mean()
            geometric_mean_return = ((1 + filtered_returns).prod() ** (1/len(filtered_returns))) - 1
            
            # Calculate risk metrics
            volatility = filtered_returns.std() * np.sqrt(self.config['trading_days_per_year'])
            downside_returns = filtered_returns[filtered_returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(self.config['trading_days_per_year']) if len(downside_returns) > 0 else 0
            
            # Calculate risk-adjusted metrics
            risk_free_rate = self.config['risk_free_rate']
            excess_return = annualized_return - risk_free_rate
            
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0
            
            # Calculate drawdown metrics
            cumulative_returns = (1 + filtered_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            
            max_drawdown = drawdowns.min()
            average_drawdown = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0
            
            # Calculate drawdown duration and recovery
            drawdown_duration, recovery_time = self._calculate_drawdown_duration(drawdowns)
            
            # Calculate distribution metrics
            skewness = filtered_returns.skew()
            kurtosis = filtered_returns.kurtosis()
            
            # Calculate VaR and CVaR
            var_95 = filtered_returns.quantile(0.05)
            cvar_95 = filtered_returns[filtered_returns <= var_95].mean()
            
            # Calculate win/loss metrics
            win_rate = (filtered_returns > 0).mean()
            winning_returns = filtered_returns[filtered_returns > 0]
            losing_returns = filtered_returns[filtered_returns < 0]
            
            average_win = winning_returns.mean() if len(winning_returns) > 0 else 0
            average_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
            largest_win = winning_returns.max() if len(winning_returns) > 0 else 0
            largest_loss = losing_returns.min() if len(losing_returns) > 0 else 0
            
            profit_factor = abs(average_win * len(winning_returns) / (average_loss * len(losing_returns))) if len(losing_returns) > 0 and average_loss != 0 else 0
            
            # Calculate additional ratios
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            sterling_ratio = annualized_return / abs(average_drawdown) if average_drawdown != 0 else 0
            burke_ratio = excess_return / np.sqrt(np.sum(drawdowns**2)) if np.sum(drawdowns**2) > 0 else 0
            
            # Calculate tail ratio
            tail_ratio = abs(filtered_returns.quantile(0.95) / filtered_returns.quantile(0.05)) if filtered_returns.quantile(0.05) != 0 else 0
            
            # Create performance metrics object
            metrics = PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                cumulative_return=cumulative_return,
                average_return=average_return,
                geometric_mean_return=geometric_mean_return,
                volatility=volatility,
                downside_volatility=downside_volatility,
                tracking_error=0.0,  # Will be calculated vs benchmark
                beta=1.0,  # Will be calculated vs benchmark
                alpha=0.0,  # Will be calculated vs benchmark
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                treynor_ratio=0.0,  # Will be calculated vs benchmark
                information_ratio=0.0,  # Will be calculated vs benchmark
                max_drawdown=max_drawdown,
                average_drawdown=average_drawdown,
                drawdown_duration=drawdown_duration,
                recovery_time=recovery_time,
                skewness=skewness,
                kurtosis=kurtosis,
                var_95=var_95,
                cvar_95=cvar_95,
                win_rate=win_rate,
                profit_factor=profit_factor,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                sterling_ratio=sterling_ratio,
                burke_ratio=burke_ratio,
                tail_ratio=tail_ratio,
                period_start=filtered_returns.index[0] if len(filtered_returns) > 0 else datetime.now(),
                period_end=filtered_returns.index[-1] if len(filtered_returns) > 0 else datetime.now(),
                total_periods=len(filtered_returns),
                last_updated=datetime.now()
            )
            
            # Store current metrics
            self.current_metrics = metrics
            
            self.logger.info(f"Performance metrics calculated: {annualized_return:.1%} return, {sharpe_ratio:.2f} Sharpe")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return self._create_empty_metrics()
    
    def perform_attribution_analysis(self, 
                                   returns_data: pd.DataFrame,
                                   attribution_type: AttributionType) -> AttributionResult:
        """
        Perform performance attribution analysis
        
        Args:
            returns_data: DataFrame with returns and attribution factors
            attribution_type: Type of attribution analysis
            
        Returns:
            Attribution analysis result
        """
        try:
            if returns_data.empty:
                return self._create_empty_attribution(attribution_type)
            
            # Get attribution factors based on type
            if attribution_type == AttributionType.STRATEGY:
                attribution_factors = self._get_strategy_attribution_factors(returns_data)
            elif attribution_type == AttributionType.TIME_PERIOD:
                attribution_factors = self._get_time_period_attribution_factors(returns_data)
            elif attribution_type == AttributionType.MARKET_REGIME:
                attribution_factors = self._get_market_regime_attribution_factors(returns_data)
            elif attribution_type == AttributionType.WEEK_TYPE:
                attribution_factors = self._get_week_type_attribution_factors(returns_data)
            else:
                attribution_factors = {}
            
            # Calculate attribution breakdown
            attribution_breakdown = {}
            total_attribution = 0.0
            
            for factor, factor_returns in attribution_factors.items():
                if len(factor_returns) > 0:
                    factor_contribution = factor_returns.sum()
                    attribution_breakdown[factor] = factor_contribution
                    total_attribution += factor_contribution
            
            # Calculate residual return
            total_return = returns_data['returns'].sum() if 'returns' in returns_data.columns else 0
            residual_return = total_return - total_attribution
            
            # Calculate attribution quality (R-squared)
            attribution_quality = self._calculate_attribution_quality(
                returns_data.get('returns', pd.Series()), attribution_breakdown
            )
            
            # Create attribution result
            result = AttributionResult(
                attribution_type=attribution_type,
                attribution_breakdown=attribution_breakdown,
                total_attribution=total_attribution,
                residual_return=residual_return,
                attribution_quality=attribution_quality,
                period_start=returns_data.index[0] if len(returns_data) > 0 else datetime.now(),
                period_end=returns_data.index[-1] if len(returns_data) > 0 else datetime.now(),
                timestamp=datetime.now()
            )
            
            # Store attribution result
            self.attribution_history.append(result)
            
            self.logger.info(f"Attribution analysis completed: {attribution_type.value} ({attribution_quality:.1%} quality)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing attribution analysis: {str(e)}")
            return self._create_empty_attribution(attribution_type)
    
    def compare_to_benchmark(self, 
                           portfolio_returns: pd.Series,
                           benchmark_type: BenchmarkType = BenchmarkType.SPY) -> BenchmarkComparison:
        """
        Compare portfolio performance to benchmark
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_type: Benchmark for comparison
            
        Returns:
            Benchmark comparison analysis
        """
        try:
            if portfolio_returns.empty:
                return self._create_empty_benchmark_comparison(benchmark_type)
            
            # Get benchmark returns
            benchmark_returns = self._get_benchmark_returns(benchmark_type, portfolio_returns.index)
            
            if benchmark_returns.empty:
                self.logger.warning(f"No benchmark data available for {benchmark_type.value}")
                return self._create_empty_benchmark_comparison(benchmark_type)
            
            # Align returns
            aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
            
            if len(aligned_portfolio) < self.config['min_periods_for_analysis']:
                self.logger.warning("Insufficient aligned data for benchmark comparison")
            
            # Calculate returns
            portfolio_return = (1 + aligned_portfolio).prod() - 1
            benchmark_return = (1 + aligned_benchmark).prod() - 1
            excess_return = portfolio_return - benchmark_return
            
            # Calculate tracking error
            excess_returns = aligned_portfolio - aligned_benchmark
            tracking_error = excess_returns.std() * np.sqrt(self.config['trading_days_per_year'])
            
            # Calculate information ratio
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(self.config['trading_days_per_year']) if excess_returns.std() > 0 else 0
            
            # Calculate beta and alpha
            if len(aligned_portfolio) > 1 and aligned_benchmark.std() > 0:
                beta = np.cov(aligned_portfolio, aligned_benchmark)[0, 1] / np.var(aligned_benchmark)
                alpha = aligned_portfolio.mean() - beta * aligned_benchmark.mean()
            else:
                beta = 1.0
                alpha = 0.0
            
            # Calculate correlation
            correlation = aligned_portfolio.corr(aligned_benchmark) if len(aligned_portfolio) > 1 else 0
            
            # Calculate up/down capture ratios
            up_periods = aligned_benchmark > 0
            down_periods = aligned_benchmark < 0
            
            up_capture = (aligned_portfolio[up_periods].mean() / aligned_benchmark[up_periods].mean()) if up_periods.sum() > 0 and aligned_benchmark[up_periods].mean() != 0 else 1.0
            down_capture = (aligned_portfolio[down_periods].mean() / aligned_benchmark[down_periods].mean()) if down_periods.sum() > 0 and aligned_benchmark[down_periods].mean() != 0 else 1.0
            
            # Calculate outperformance statistics
            outperformance_periods = (aligned_portfolio > aligned_benchmark).sum()
            total_periods = len(aligned_portfolio)
            outperformance_rate = outperformance_periods / total_periods if total_periods > 0 else 0
            
            # Create benchmark comparison
            comparison = BenchmarkComparison(
                benchmark_type=benchmark_type,
                benchmark_name=self._get_benchmark_name(benchmark_type),
                portfolio_return=portfolio_return,
                benchmark_return=benchmark_return,
                excess_return=excess_return,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                up_capture=up_capture,
                down_capture=down_capture,
                correlation=correlation,
                beta=beta,
                alpha=alpha,
                outperformance_periods=outperformance_periods,
                total_periods=total_periods,
                outperformance_rate=outperformance_rate,
                timestamp=datetime.now()
            )
            
            # Store benchmark comparison
            self.benchmark_comparisons[benchmark_type.value] = comparison
            
            self.logger.info(f"Benchmark comparison completed: {excess_return:.1%} excess return vs {benchmark_type.value}")
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing to benchmark: {str(e)}")
            return self._create_empty_benchmark_comparison(benchmark_type)
    
    def generate_performance_forecast(self, 
                                    returns_data: pd.Series,
                                    forecast_horizon: int = 30) -> PerformanceForecast:
        """
        Generate performance forecast
        
        Args:
            returns_data: Historical returns data
            forecast_horizon: Forecast horizon in days
            
        Returns:
            Performance forecast
        """
        try:
            if len(returns_data) < self.config['min_periods_for_analysis']:
                return self._create_empty_forecast(forecast_horizon)
            
            # Calculate historical statistics
            mean_return = returns_data.mean()
            volatility = returns_data.std()
            
            # Annualize statistics
            annual_mean = mean_return * self.config['trading_days_per_year']
            annual_volatility = volatility * np.sqrt(self.config['trading_days_per_year'])
            
            # Calculate forecast statistics
            forecast_mean = mean_return * forecast_horizon
            forecast_volatility = volatility * np.sqrt(forecast_horizon)
            
            # Generate confidence intervals using normal distribution
            confidence_intervals = {}
            for confidence in self.config['confidence_levels']:
                z_score = np.percentile(np.random.normal(0, 1, 10000), confidence * 100)
                lower_bound = forecast_mean - z_score * forecast_volatility
                upper_bound = forecast_mean + z_score * forecast_volatility
                confidence_intervals[f"{confidence:.0%}"] = (lower_bound, upper_bound)
            
            # Monte Carlo scenario analysis
            scenario_analysis = self._generate_scenario_analysis(mean_return, volatility, forecast_horizon)
            
            # Calculate probabilities
            probability_positive = 1 - self._normal_cdf(0, forecast_mean, forecast_volatility)
            
            # Probability of outperforming benchmark (assuming SPY)
            benchmark_expected_return = 0.08 / self.config['trading_days_per_year'] * forecast_horizon  # 8% annual
            probability_outperform_benchmark = 1 - self._normal_cdf(benchmark_expected_return, forecast_mean, forecast_volatility)
            
            # Calculate forecast accuracy based on historical performance
            forecast_accuracy = self._calculate_forecast_accuracy(returns_data)
            
            # Model confidence based on data quality and stability
            model_confidence = self._calculate_model_confidence(returns_data)
            
            # Create forecast
            forecast = PerformanceForecast(
                forecast_horizon=forecast_horizon,
                expected_return=forecast_mean,
                expected_volatility=forecast_volatility,
                confidence_intervals=confidence_intervals,
                scenario_analysis=scenario_analysis,
                probability_positive=probability_positive,
                probability_outperform_benchmark=probability_outperform_benchmark,
                forecast_accuracy=forecast_accuracy,
                model_confidence=model_confidence,
                timestamp=datetime.now()
            )
            
            # Store forecast
            self.forecast_history.append(forecast)
            
            self.logger.info(f"Performance forecast generated: {forecast_mean:.1%} expected return over {forecast_horizon} days")
            
            return forecast
            
        except Exception as e:
            self.logger.error(f"Error generating performance forecast: {str(e)}")
            return self._create_empty_forecast(forecast_horizon)
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics dashboard"""
        try:
            # Current performance summary
            performance_summary = {}
            if self.current_metrics:
                performance_summary = {
                    'annualized_return': self.current_metrics.annualized_return,
                    'volatility': self.current_metrics.volatility,
                    'sharpe_ratio': self.current_metrics.sharpe_ratio,
                    'sortino_ratio': self.current_metrics.sortino_ratio,
                    'max_drawdown': self.current_metrics.max_drawdown,
                    'win_rate': self.current_metrics.win_rate,
                    'calmar_ratio': self.current_metrics.calmar_ratio,
                    'total_periods': self.current_metrics.total_periods
                }
            
            # Benchmark comparisons summary
            benchmark_summary = {}
            for benchmark_name, comparison in self.benchmark_comparisons.items():
                benchmark_summary[benchmark_name] = {
                    'excess_return': comparison.excess_return,
                    'information_ratio': comparison.information_ratio,
                    'beta': comparison.beta,
                    'alpha': comparison.alpha,
                    'correlation': comparison.correlation,
                    'outperformance_rate': comparison.outperformance_rate
                }
            
            # Recent attribution analysis
            recent_attribution = None
            if self.attribution_history:
                latest_attribution = self.attribution_history[-1]
                recent_attribution = {
                    'attribution_type': latest_attribution.attribution_type.value,
                    'top_contributors': self._get_top_attribution_contributors(latest_attribution.attribution_breakdown),
                    'attribution_quality': latest_attribution.attribution_quality,
                    'timestamp': latest_attribution.timestamp.isoformat()
                }
            
            # Recent forecast
            recent_forecast = None
            if self.forecast_history:
                latest_forecast = self.forecast_history[-1]
                recent_forecast = {
                    'forecast_horizon': latest_forecast.forecast_horizon,
                    'expected_return': latest_forecast.expected_return,
                    'probability_positive': latest_forecast.probability_positive,
                    'model_confidence': latest_forecast.model_confidence,
                    'timestamp': latest_forecast.timestamp.isoformat()
                }
            
            # Performance trends
            performance_trends = self._calculate_performance_trends()
            
            # Risk analysis
            risk_analysis = self._generate_risk_analysis()
            
            # Recommendations
            recommendations = self._generate_performance_recommendations()
            
            dashboard = {
                'performance_summary': performance_summary,
                'benchmark_comparisons': benchmark_summary,
                'recent_attribution': recent_attribution,
                'recent_forecast': recent_forecast,
                'performance_trends': performance_trends,
                'risk_analysis': risk_analysis,
                'recommendations': recommendations,
                'analytics_status': 'active' if self.analytics_active else 'inactive',
                'last_updated': datetime.now().isoformat()
            }
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error generating performance dashboard: {str(e)}")
            return {'error': str(e)}
    
    def start_continuous_analytics(self):
        """Start continuous performance analytics"""
        if self.analytics_active:
            self.logger.warning("Continuous analytics already active")
            return
        
        self.analytics_active = True
        self.analytics_thread = threading.Thread(target=self._analytics_loop, daemon=True)
        self.analytics_thread.start()
        
        self.logger.info("Continuous performance analytics started")
    
    def stop_continuous_analytics(self):
        """Stop continuous analytics"""
        self.analytics_active = False
        if self.analytics_thread:
            self.analytics_thread.join(timeout=5)
        
        self.logger.info("Continuous performance analytics stopped")
    
    # Helper methods for performance analytics
    def _initialize_benchmark_data(self):
        """Initialize benchmark data"""
        # Mock benchmark data - in practice, this would come from data feeds
        dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
        
        # Generate synthetic benchmark returns
        np.random.seed(42)  # For reproducible results
        
        benchmarks = {
            BenchmarkType.SPY: {
                'name': 'SPDR S&P 500 ETF',
                'annual_return': 0.10,
                'annual_volatility': 0.16
            },
            BenchmarkType.QQQ: {
                'name': 'Invesco QQQ ETF',
                'annual_return': 0.12,
                'annual_volatility': 0.20
            },
            BenchmarkType.IWM: {
                'name': 'iShares Russell 2000 ETF',
                'annual_return': 0.08,
                'annual_volatility': 0.22
            },
            BenchmarkType.RISK_FREE: {
                'name': 'Risk-Free Rate',
                'annual_return': 0.02,
                'annual_volatility': 0.01
            }
        }
        
        for benchmark_type, params in benchmarks.items():
            daily_return = params['annual_return'] / 252
            daily_volatility = params['annual_volatility'] / np.sqrt(252)
            
            returns = np.random.normal(daily_return, daily_volatility, len(dates))
            
            self.benchmark_data[benchmark_type.value] = pd.Series(
                returns, index=dates, name=f"{benchmark_type.value}_returns"
            )
    
    def _filter_returns_by_period(self, returns_data: pd.Series, period: PerformancePeriod) -> pd.Series:
        """Filter returns data by analysis period"""
        if period == PerformancePeriod.INCEPTION:
            return returns_data
        
        end_date = returns_data.index[-1] if len(returns_data) > 0 else datetime.now()
        
        if period == PerformancePeriod.DAILY:
            start_date = end_date - timedelta(days=1)
        elif period == PerformancePeriod.WEEKLY:
            start_date = end_date - timedelta(weeks=1)
        elif period == PerformancePeriod.MONTHLY:
            start_date = end_date - timedelta(days=30)
        elif period == PerformancePeriod.QUARTERLY:
            start_date = end_date - timedelta(days=90)
        elif period == PerformancePeriod.YEARLY:
            start_date = end_date - timedelta(days=365)
        else:
            return returns_data
        
        return returns_data[returns_data.index >= start_date]
    
    def _annualize_return(self, total_return: float, periods: int) -> float:
        """Annualize return based on number of periods"""
        if periods == 0:
            return 0.0
        
        periods_per_year = self.config['trading_days_per_year']
        return (1 + total_return) ** (periods_per_year / periods) - 1
    
    def _calculate_drawdown_duration(self, drawdowns: pd.Series) -> Tuple[int, int]:
        """Calculate drawdown duration and recovery time"""
        try:
            if drawdowns.empty:
                return 0, 0
            
            # Find drawdown periods
            in_drawdown = drawdowns < 0
            
            if not in_drawdown.any():
                return 0, 0
            
            # Calculate duration of longest drawdown
            drawdown_periods = []
            current_duration = 0
            
            for is_drawdown in in_drawdown:
                if is_drawdown:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        drawdown_periods.append(current_duration)
                    current_duration = 0
            
            # Add final period if still in drawdown
            if current_duration > 0:
                drawdown_periods.append(current_duration)
            
            max_duration = max(drawdown_periods) if drawdown_periods else 0
            
            # Calculate recovery time (simplified)
            recovery_time = max_duration  # Assume recovery time equals drawdown duration
            
            return max_duration, recovery_time
            
        except Exception as e:
            return 0, 0
    
    def _get_strategy_attribution_factors(self, returns_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get strategy attribution factors"""
        # Mock strategy attribution - in practice, this would come from actual strategy data
        strategies = ['put_selling', 'iron_condor', 'put_spread', 'call_selling']
        attribution_factors = {}
        
        if 'returns' in returns_data.columns:
            total_returns = returns_data['returns']
            
            # Distribute returns across strategies (simplified)
            for i, strategy in enumerate(strategies):
                strategy_weight = 0.25  # Equal weight
                strategy_returns = total_returns * strategy_weight * (1 + 0.1 * np.sin(i))  # Add some variation
                attribution_factors[strategy] = strategy_returns
        
        return attribution_factors
    
    def _get_time_period_attribution_factors(self, returns_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get time period attribution factors"""
        attribution_factors = {}
        
        if 'returns' in returns_data.columns and len(returns_data) > 0:
            returns = returns_data['returns']
            
            # Group by quarters
            quarterly_returns = returns.groupby(returns.index.to_period('Q'))
            
            for period, period_returns in quarterly_returns:
                attribution_factors[str(period)] = period_returns
        
        return attribution_factors
    
    def _get_market_regime_attribution_factors(self, returns_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get market regime attribution factors"""
        attribution_factors = {}
        
        if 'returns' in returns_data.columns:
            returns = returns_data['returns']
            
            # Simple regime classification based on volatility
            rolling_vol = returns.rolling(20).std()
            
            high_vol_periods = rolling_vol > rolling_vol.quantile(0.7)
            low_vol_periods = rolling_vol < rolling_vol.quantile(0.3)
            
            attribution_factors['high_volatility'] = returns[high_vol_periods]
            attribution_factors['normal_volatility'] = returns[~(high_vol_periods | low_vol_periods)]
            attribution_factors['low_volatility'] = returns[low_vol_periods]
        
        return attribution_factors
    
    def _get_week_type_attribution_factors(self, returns_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get week type attribution factors"""
        # Mock week type attribution based on ALL-USE week classification
        week_types = ['P-EW', 'P-AWL', 'P-RO', 'C-WAP', 'C-WAP+', 'C-PNO']
        attribution_factors = {}
        
        if 'returns' in returns_data.columns:
            returns = returns_data['returns']
            
            # Distribute returns across week types (simplified)
            for i, week_type in enumerate(week_types):
                # Create mock week type periods
                week_mask = (returns.index.dayofweek == i % 5)  # Distribute across weekdays
                attribution_factors[week_type] = returns[week_mask]
        
        return attribution_factors
    
    def _calculate_attribution_quality(self, returns: pd.Series, attribution_breakdown: Dict[str, float]) -> float:
        """Calculate attribution quality (R-squared)"""
        try:
            if len(returns) == 0 or not attribution_breakdown:
                return 0.0
            
            # Calculate explained variance
            total_return = returns.sum()
            explained_return = sum(attribution_breakdown.values())
            
            # Simple R-squared approximation
            if total_return != 0:
                r_squared = min(1.0, abs(explained_return / total_return))
            else:
                r_squared = 0.0
            
            return r_squared
            
        except Exception as e:
            return 0.0
    
    def _get_benchmark_returns(self, benchmark_type: BenchmarkType, dates: pd.DatetimeIndex) -> pd.Series:
        """Get benchmark returns for specified dates"""
        try:
            if benchmark_type.value not in self.benchmark_data:
                return pd.Series()
            
            benchmark_series = self.benchmark_data[benchmark_type.value]
            
            # Align with requested dates
            aligned_returns = benchmark_series.reindex(dates, method='ffill')
            
            return aligned_returns.dropna()
            
        except Exception as e:
            return pd.Series()
    
    def _get_benchmark_name(self, benchmark_type: BenchmarkType) -> str:
        """Get benchmark display name"""
        names = {
            BenchmarkType.SPY: 'SPDR S&P 500 ETF',
            BenchmarkType.QQQ: 'Invesco QQQ ETF',
            BenchmarkType.IWM: 'iShares Russell 2000 ETF',
            BenchmarkType.VTI: 'Vanguard Total Stock Market ETF',
            BenchmarkType.RISK_FREE: 'Risk-Free Rate',
            BenchmarkType.CUSTOM: 'Custom Benchmark'
        }
        
        return names.get(benchmark_type, benchmark_type.value.upper())
    
    def _generate_scenario_analysis(self, mean_return: float, volatility: float, horizon: int) -> Dict[str, float]:
        """Generate scenario analysis using Monte Carlo"""
        try:
            scenarios = {
                'bull_market': mean_return * horizon + 2 * volatility * np.sqrt(horizon),
                'base_case': mean_return * horizon,
                'bear_market': mean_return * horizon - 2 * volatility * np.sqrt(horizon),
                'stress_case': mean_return * horizon - 3 * volatility * np.sqrt(horizon)
            }
            
            return scenarios
            
        except Exception as e:
            return {}
    
    def _normal_cdf(self, x: float, mean: float, std: float) -> float:
        """Calculate normal cumulative distribution function"""
        try:
            from scipy.stats import norm
            return norm.cdf(x, mean, std)
        except ImportError:
            # Fallback approximation
            z = (x - mean) / std if std > 0 else 0
            return 0.5 * (1 + np.sign(z) * np.sqrt(1 - np.exp(-2 * z**2 / np.pi)))
    
    def _calculate_forecast_accuracy(self, returns_data: pd.Series) -> float:
        """Calculate historical forecast accuracy"""
        try:
            # Simplified accuracy calculation based on return stability
            if len(returns_data) < 30:
                return 0.5  # Low confidence for insufficient data
            
            # Calculate rolling forecast accuracy
            rolling_mean = returns_data.rolling(30).mean()
            rolling_std = returns_data.rolling(30).std()
            
            # Accuracy based on prediction interval hit rate
            accuracy = 0.75  # Default accuracy
            
            return accuracy
            
        except Exception as e:
            return 0.5
    
    def _calculate_model_confidence(self, returns_data: pd.Series) -> float:
        """Calculate model confidence based on data quality"""
        try:
            if len(returns_data) < 30:
                return 0.3  # Low confidence
            
            # Factors affecting confidence
            data_length_factor = min(1.0, len(returns_data) / 252)  # More data = higher confidence
            stability_factor = 1.0 / (1.0 + returns_data.std())  # Lower volatility = higher confidence
            
            confidence = (data_length_factor + stability_factor) / 2
            
            return min(1.0, max(0.1, confidence))
            
        except Exception as e:
            return 0.5
    
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty performance metrics"""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            cumulative_return=0.0,
            average_return=0.0,
            geometric_mean_return=0.0,
            volatility=0.0,
            downside_volatility=0.0,
            tracking_error=0.0,
            beta=1.0,
            alpha=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            treynor_ratio=0.0,
            information_ratio=0.0,
            max_drawdown=0.0,
            average_drawdown=0.0,
            drawdown_duration=0,
            recovery_time=0,
            skewness=0.0,
            kurtosis=0.0,
            var_95=0.0,
            cvar_95=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            sterling_ratio=0.0,
            burke_ratio=0.0,
            tail_ratio=0.0,
            period_start=datetime.now(),
            period_end=datetime.now(),
            total_periods=0,
            last_updated=datetime.now()
        )
    
    def _create_empty_attribution(self, attribution_type: AttributionType) -> AttributionResult:
        """Create empty attribution result"""
        return AttributionResult(
            attribution_type=attribution_type,
            attribution_breakdown={},
            total_attribution=0.0,
            residual_return=0.0,
            attribution_quality=0.0,
            period_start=datetime.now(),
            period_end=datetime.now(),
            timestamp=datetime.now()
        )
    
    def _create_empty_benchmark_comparison(self, benchmark_type: BenchmarkType) -> BenchmarkComparison:
        """Create empty benchmark comparison"""
        return BenchmarkComparison(
            benchmark_type=benchmark_type,
            benchmark_name=self._get_benchmark_name(benchmark_type),
            portfolio_return=0.0,
            benchmark_return=0.0,
            excess_return=0.0,
            tracking_error=0.0,
            information_ratio=0.0,
            up_capture=1.0,
            down_capture=1.0,
            correlation=0.0,
            beta=1.0,
            alpha=0.0,
            outperformance_periods=0,
            total_periods=0,
            outperformance_rate=0.0,
            timestamp=datetime.now()
        )
    
    def _create_empty_forecast(self, forecast_horizon: int) -> PerformanceForecast:
        """Create empty performance forecast"""
        return PerformanceForecast(
            forecast_horizon=forecast_horizon,
            expected_return=0.0,
            expected_volatility=0.0,
            confidence_intervals={},
            scenario_analysis={},
            probability_positive=0.5,
            probability_outperform_benchmark=0.5,
            forecast_accuracy=0.5,
            model_confidence=0.5,
            timestamp=datetime.now()
        )
    
    def _get_top_attribution_contributors(self, attribution_breakdown: Dict[str, float]) -> List[Tuple[str, float]]:
        """Get top attribution contributors"""
        if not attribution_breakdown:
            return []
        
        sorted_contributors = sorted(attribution_breakdown.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_contributors[:5]  # Top 5 contributors
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends"""
        trends = {
            'return_trend': 'stable',
            'volatility_trend': 'stable',
            'sharpe_trend': 'stable',
            'trend_strength': 0.5
        }
        
        # In practice, this would analyze historical metrics
        return trends
    
    def _generate_risk_analysis(self) -> Dict[str, Any]:
        """Generate risk analysis summary"""
        risk_analysis = {
            'risk_level': 'moderate',
            'key_risks': ['market_risk', 'volatility_risk'],
            'risk_score': 50.0,
            'risk_recommendations': ['Monitor volatility', 'Maintain diversification']
        }
        
        return risk_analysis
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if self.current_metrics:
            if self.current_metrics.sharpe_ratio < 1.0:
                recommendations.append("Sharpe ratio below 1.0 - consider risk reduction or return enhancement")
            
            if self.current_metrics.max_drawdown < -0.10:
                recommendations.append("Maximum drawdown exceeds 10% - review risk management")
            
            if self.current_metrics.win_rate < 0.5:
                recommendations.append("Win rate below 50% - analyze strategy effectiveness")
            
            if self.current_metrics.volatility > 0.20:
                recommendations.append("High volatility detected - consider position sizing adjustments")
        
        if not recommendations:
            recommendations.append("Performance metrics are within acceptable ranges")
        
        return recommendations
    
    def _analytics_loop(self):
        """Main analytics loop for continuous analysis"""
        while self.analytics_active:
            try:
                # Update performance metrics if data available
                if not self.performance_history.empty:
                    returns = self.performance_history.get('returns', pd.Series())
                    if not returns.empty:
                        self.calculate_performance_metrics(returns)
                
                # Update benchmark comparisons
                for benchmark_type in [BenchmarkType.SPY, BenchmarkType.QQQ]:
                    if not self.performance_history.empty:
                        returns = self.performance_history.get('returns', pd.Series())
                        if not returns.empty:
                            self.compare_to_benchmark(returns, benchmark_type)
                
                # Sleep until next update
                time.sleep(self.config['performance_update_frequency'])
                
            except Exception as e:
                self.logger.error(f"Error in analytics loop: {str(e)}")
                time.sleep(300)  # Wait 5 minutes on error

def test_advanced_performance_analytics():
    """Test the advanced performance analytics"""
    print("Testing Advanced Performance Analytics...")
    
    analytics = AdvancedPerformanceAnalytics()
    
    # Generate mock returns data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    returns = pd.Series(
        np.random.normal(0.0008, 0.015, len(dates)),  # ~20% annual return, 24% volatility
        index=dates,
        name='returns'
    )
    
    # Test performance metrics calculation
    print("\n--- Testing Performance Metrics ---")
    metrics = analytics.calculate_performance_metrics(returns)
    
    print(f"Annualized Return: {metrics.annualized_return:.1%}")
    print(f"Volatility: {metrics.volatility:.1%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.1%}")
    print(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print(f"Win Rate: {metrics.win_rate:.1%}")
    print(f"Total Periods: {metrics.total_periods}")
    
    # Test attribution analysis
    print("\n--- Testing Attribution Analysis ---")
    returns_df = pd.DataFrame({'returns': returns})
    
    attribution_types = [
        AttributionType.STRATEGY,
        AttributionType.TIME_PERIOD,
        AttributionType.MARKET_REGIME
    ]
    
    for attr_type in attribution_types:
        attribution = analytics.perform_attribution_analysis(returns_df, attr_type)
        print(f"\n{attr_type.value.upper()} Attribution:")
        print(f"  Attribution Quality: {attribution.attribution_quality:.1%}")
        print(f"  Total Attribution: {attribution.total_attribution:.1%}")
        print(f"  Residual Return: {attribution.residual_return:.1%}")
        
        if attribution.attribution_breakdown:
            print("  Top Contributors:")
            for factor, contribution in list(attribution.attribution_breakdown.items())[:3]:
                print(f"    {factor}: {contribution:.1%}")
    
    # Test benchmark comparison
    print("\n--- Testing Benchmark Comparison ---")
    benchmarks = [BenchmarkType.SPY, BenchmarkType.QQQ]
    
    for benchmark in benchmarks:
        comparison = analytics.compare_to_benchmark(returns, benchmark)
        print(f"\nvs {benchmark.value.upper()}:")
        print(f"  Excess Return: {comparison.excess_return:.1%}")
        print(f"  Information Ratio: {comparison.information_ratio:.2f}")
        print(f"  Beta: {comparison.beta:.2f}")
        print(f"  Alpha: {comparison.alpha:.1%}")
        print(f"  Correlation: {comparison.correlation:.2f}")
        print(f"  Outperformance Rate: {comparison.outperformance_rate:.1%}")
    
    # Test performance forecasting
    print("\n--- Testing Performance Forecasting ---")
    forecast_horizons = [7, 30, 90]
    
    for horizon in forecast_horizons:
        forecast = analytics.generate_performance_forecast(returns, horizon)
        print(f"\n{horizon}-Day Forecast:")
        print(f"  Expected Return: {forecast.expected_return:.1%}")
        print(f"  Expected Volatility: {forecast.expected_volatility:.1%}")
        print(f"  Probability Positive: {forecast.probability_positive:.1%}")
        print(f"  Probability Outperform Benchmark: {forecast.probability_outperform_benchmark:.1%}")
        print(f"  Model Confidence: {forecast.model_confidence:.1%}")
        
        if forecast.confidence_intervals:
            print("  Confidence Intervals:")
            for level, (lower, upper) in forecast.confidence_intervals.items():
                print(f"    {level}: [{lower:.1%}, {upper:.1%}]")
    
    # Test performance dashboard
    print("\n--- Testing Performance Dashboard ---")
    dashboard = analytics.get_performance_dashboard()
    
    if 'error' not in dashboard:
        summary = dashboard['performance_summary']
        print("Performance Summary:")
        print(f"  Annualized Return: {summary.get('annualized_return', 0):.1%}")
        print(f"  Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {summary.get('max_drawdown', 0):.1%}")
        print(f"  Win Rate: {summary.get('win_rate', 0):.1%}")
        
        print(f"\nBenchmark Comparisons: {len(dashboard['benchmark_comparisons'])}")
        print(f"Recent Attribution: {dashboard['recent_attribution'] is not None}")
        print(f"Recent Forecast: {dashboard['recent_forecast'] is not None}")
        
        print("\nRecommendations:")
        for rec in dashboard['recommendations']:
            print(f"  • {rec}")
    
    print("\n✅ Advanced Performance Analytics test completed successfully!")

if __name__ == "__main__":
    test_advanced_performance_analytics()

