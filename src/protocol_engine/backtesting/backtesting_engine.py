"""
Advanced Backtesting and Validation System for ALL-USE Protocol
Implements comprehensive backtesting capabilities for strategy validation and performance analysis

This module provides sophisticated backtesting capabilities including:
- Historical strategy validation across all 11 week types
- Comprehensive performance metrics and risk analysis
- Stress testing and scenario analysis
- ML optimization validation
- Portfolio-level backtesting with realistic execution modeling
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestType(Enum):
    """Types of backtesting strategies"""
    SINGLE_STRATEGY = "single_strategy"
    MULTI_STRATEGY = "multi_strategy"
    WEEK_TYPE_ANALYSIS = "week_type_analysis"
    PARAMETER_OPTIMIZATION = "parameter_optimization"
    STRESS_TEST = "stress_test"
    MONTE_CARLO = "monte_carlo"

class PerformanceMetric(Enum):
    """Performance metrics for backtesting"""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    VAR_95 = "var_95"
    EXPECTED_SHORTFALL = "expected_shortfall"

@dataclass
class BacktestPosition:
    """Individual position in backtest"""
    position_id: str
    symbol: str
    strategy_type: str
    week_type: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    delta: float
    dte: int
    premium_collected: float
    realized_pnl: Optional[float]
    unrealized_pnl: float
    max_profit: float
    max_loss: float
    status: str  # 'open', 'closed', 'expired', 'assigned'

@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    backtest_id: str
    strategy_name: str
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    var_95: float
    expected_shortfall: float
    positions: List[BacktestPosition]
    daily_returns: List[float]
    equity_curve: List[float]
    drawdown_curve: List[float]
    performance_by_week_type: Dict[str, Dict[str, float]]
    risk_metrics: Dict[str, float]
    execution_stats: Dict[str, Any]

@dataclass
class StressTestScenario:
    """Stress test scenario definition"""
    scenario_name: str
    market_shock: float  # Market return shock (e.g., -0.20 for 20% crash)
    volatility_shock: float  # VIX shock multiplier (e.g., 2.0 for doubling)
    duration_days: int  # Duration of stress scenario
    recovery_days: int  # Recovery period
    description: str

class AdvancedBacktestingEngine:
    """
    Advanced Backtesting and Validation System for ALL-USE Protocol
    
    Provides comprehensive backtesting capabilities including:
    - Historical strategy validation across all week types
    - Performance metrics calculation and risk analysis
    - Stress testing and scenario analysis
    - ML optimization validation
    - Portfolio-level backtesting with realistic execution
    """
    
    def __init__(self):
        """Initialize the backtesting engine"""
        self.logger = logging.getLogger(__name__)
        
        # Backtesting configuration
        self.config = {
            'initial_capital': 100000,  # $100k starting capital
            'commission_per_contract': 1.0,  # $1 per contract
            'slippage_bps': 2,  # 2 basis points slippage
            'margin_requirement': 0.20,  # 20% margin requirement
            'risk_free_rate': 0.02,  # 2% risk-free rate
            'trading_days_per_year': 252
        }
        
        # Week type characteristics for backtesting
        self.week_type_characteristics = {
            'P-EW': {'frequency': 31/52, 'avg_return': 0.025, 'success_rate': 0.85, 'avg_dte': 35},
            'P-AWL': {'frequency': 6/52, 'avg_return': 0.020, 'success_rate': 0.80, 'avg_dte': 30},
            'P-RO': {'frequency': 4/52, 'avg_return': -0.005, 'success_rate': 0.60, 'avg_dte': 25},
            'P-AOL': {'frequency': 2/52, 'avg_return': -0.010, 'success_rate': 0.50, 'avg_dte': 20},
            'P-DD': {'frequency': 1/52, 'avg_return': -0.050, 'success_rate': 0.30, 'avg_dte': 15},
            'C-WAP': {'frequency': 14/52, 'avg_return': 0.018, 'success_rate': 0.75, 'avg_dte': 35},
            'C-WAP+': {'frequency': 6/52, 'avg_return': 0.030, 'success_rate': 0.70, 'avg_dte': 30},
            'C-PNO': {'frequency': 8/52, 'avg_return': 0.015, 'success_rate': 0.65, 'avg_dte': 25},
            'C-RO': {'frequency': 4/52, 'avg_return': 0.010, 'success_rate': 0.60, 'avg_dte': 20},
            'C-REC': {'frequency': 2/52, 'avg_return': 0.005, 'success_rate': 0.55, 'avg_dte': 15},
            'W-IDL': {'frequency': 2/52, 'avg_return': 0.000, 'success_rate': 1.00, 'avg_dte': 0}
        }
        
        # Stress test scenarios
        self.stress_scenarios = [
            StressTestScenario("2008 Financial Crisis", -0.35, 3.0, 60, 120, "Severe market crash with high volatility"),
            StressTestScenario("COVID-19 Crash", -0.25, 2.5, 30, 90, "Pandemic-induced market crash"),
            StressTestScenario("Flash Crash", -0.10, 2.0, 1, 5, "Sudden market drop with quick recovery"),
            StressTestScenario("Volatility Spike", 0.0, 2.5, 14, 30, "High volatility without major market move"),
            StressTestScenario("Bear Market", -0.20, 1.5, 180, 360, "Extended bear market conditions")
        ]
        
        # Performance tracking
        self.backtest_results: List[BacktestResult] = []
        self.benchmark_results: Dict[str, BacktestResult] = {}
        
        self.logger.info("Advanced Backtesting Engine initialized")
    
    def run_comprehensive_backtest(self, start_date: datetime, end_date: datetime,
                                 strategy_parameters: Dict[str, Any]) -> BacktestResult:
        """
        Run comprehensive backtest of ALL-USE protocol
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            strategy_parameters: Strategy configuration parameters
            
        Returns:
            Comprehensive backtest results
        """
        try:
            self.logger.info(f"Starting comprehensive backtest from {start_date} to {end_date}")
            
            # Generate historical market data
            market_data = self._generate_historical_market_data(start_date, end_date)
            
            # Initialize portfolio
            portfolio = self._initialize_portfolio()
            
            # Run day-by-day simulation
            positions = []
            daily_returns = []
            equity_curve = [self.config['initial_capital']]
            
            current_date = start_date
            while current_date <= end_date:
                # Get market data for current date
                daily_market_data = market_data.get(current_date, {})
                
                # Classify week type
                week_type = self._classify_week_type(daily_market_data, current_date)
                
                # Generate trading signals
                signals = self._generate_trading_signals(week_type, daily_market_data, strategy_parameters)
                
                # Execute trades
                new_positions = self._execute_trades(signals, current_date, daily_market_data, portfolio)
                positions.extend(new_positions)
                
                # Update existing positions
                self._update_positions(positions, current_date, daily_market_data)
                
                # Calculate daily P&L
                daily_pnl = self._calculate_daily_pnl(positions, portfolio)
                daily_return = daily_pnl / portfolio['equity']
                daily_returns.append(daily_return)
                
                # Update portfolio equity
                portfolio['equity'] += daily_pnl
                equity_curve.append(portfolio['equity'])
                
                current_date += timedelta(days=1)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(daily_returns, equity_curve)
            
            # Analyze performance by week type
            week_type_performance = self._analyze_week_type_performance(positions)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(daily_returns, equity_curve)
            
            # Create backtest result
            result = BacktestResult(
                backtest_id=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                strategy_name="ALL-USE Protocol",
                start_date=start_date,
                end_date=end_date,
                total_trades=len(positions),
                winning_trades=len([p for p in positions if p.realized_pnl and p.realized_pnl > 0]),
                losing_trades=len([p for p in positions if p.realized_pnl and p.realized_pnl < 0]),
                total_return=performance_metrics['total_return'],
                annualized_return=performance_metrics['annualized_return'],
                sharpe_ratio=performance_metrics['sharpe_ratio'],
                sortino_ratio=performance_metrics['sortino_ratio'],
                max_drawdown=performance_metrics['max_drawdown'],
                max_drawdown_duration=performance_metrics['max_drawdown_duration'],
                win_rate=performance_metrics['win_rate'],
                profit_factor=performance_metrics['profit_factor'],
                calmar_ratio=performance_metrics['calmar_ratio'],
                var_95=risk_metrics['var_95'],
                expected_shortfall=risk_metrics['expected_shortfall'],
                positions=positions,
                daily_returns=daily_returns,
                equity_curve=equity_curve,
                drawdown_curve=self._calculate_drawdown_curve(equity_curve),
                performance_by_week_type=week_type_performance,
                risk_metrics=risk_metrics,
                execution_stats=self._calculate_execution_stats(positions)
            )
            
            self.backtest_results.append(result)
            self.logger.info(f"Backtest completed: {result.annualized_return:.1%} annual return, {result.sharpe_ratio:.2f} Sharpe ratio")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive backtest: {str(e)}")
            raise
    
    def run_stress_test(self, base_result: BacktestResult, 
                       scenario: StressTestScenario) -> Dict[str, Any]:
        """
        Run stress test on strategy
        
        Args:
            base_result: Base backtest result
            scenario: Stress test scenario
            
        Returns:
            Stress test results
        """
        try:
            self.logger.info(f"Running stress test: {scenario.scenario_name}")
            
            # Apply stress scenario to historical data
            stressed_returns = self._apply_stress_scenario(base_result.daily_returns, scenario)
            
            # Recalculate performance under stress
            stressed_equity_curve = self._calculate_stressed_equity_curve(
                base_result.equity_curve, stressed_returns
            )
            
            # Calculate stressed performance metrics
            stressed_metrics = self._calculate_performance_metrics(
                stressed_returns, stressed_equity_curve
            )
            
            # Calculate impact metrics
            impact_analysis = {
                'scenario': scenario.scenario_name,
                'description': scenario.description,
                'original_return': base_result.annualized_return,
                'stressed_return': stressed_metrics['annualized_return'],
                'return_impact': stressed_metrics['annualized_return'] - base_result.annualized_return,
                'original_sharpe': base_result.sharpe_ratio,
                'stressed_sharpe': stressed_metrics['sharpe_ratio'],
                'sharpe_impact': stressed_metrics['sharpe_ratio'] - base_result.sharpe_ratio,
                'original_max_dd': base_result.max_drawdown,
                'stressed_max_dd': stressed_metrics['max_drawdown'],
                'max_dd_impact': stressed_metrics['max_drawdown'] - base_result.max_drawdown,
                'survival_probability': self._calculate_survival_probability(stressed_equity_curve),
                'recovery_time': self._estimate_recovery_time(stressed_equity_curve),
                'risk_assessment': self._assess_stress_risk(stressed_metrics)
            }
            
            self.logger.info(f"Stress test completed: {impact_analysis['return_impact']:.1%} return impact")
            return impact_analysis
            
        except Exception as e:
            self.logger.error(f"Error in stress test: {str(e)}")
            return {'error': str(e)}
    
    def validate_ml_optimization(self, original_parameters: Dict[str, Any],
                               optimized_parameters: Dict[str, Any],
                               validation_period: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """
        Validate ML optimization results through backtesting
        
        Args:
            original_parameters: Original strategy parameters
            optimized_parameters: ML-optimized parameters
            validation_period: Period for validation testing
            
        Returns:
            Validation results comparing original vs optimized
        """
        try:
            start_date, end_date = validation_period
            self.logger.info(f"Validating ML optimization from {start_date} to {end_date}")
            
            # Run backtest with original parameters
            original_result = self.run_comprehensive_backtest(
                start_date, end_date, original_parameters
            )
            
            # Run backtest with optimized parameters
            optimized_result = self.run_comprehensive_backtest(
                start_date, end_date, optimized_parameters
            )
            
            # Compare results
            validation_analysis = {
                'validation_period': f"{start_date} to {end_date}",
                'original_performance': {
                    'annual_return': original_result.annualized_return,
                    'sharpe_ratio': original_result.sharpe_ratio,
                    'max_drawdown': original_result.max_drawdown,
                    'win_rate': original_result.win_rate
                },
                'optimized_performance': {
                    'annual_return': optimized_result.annualized_return,
                    'sharpe_ratio': optimized_result.sharpe_ratio,
                    'max_drawdown': optimized_result.max_drawdown,
                    'win_rate': optimized_result.win_rate
                },
                'improvement_metrics': {
                    'return_improvement': optimized_result.annualized_return - original_result.annualized_return,
                    'sharpe_improvement': optimized_result.sharpe_ratio - original_result.sharpe_ratio,
                    'drawdown_improvement': original_result.max_drawdown - optimized_result.max_drawdown,
                    'win_rate_improvement': optimized_result.win_rate - original_result.win_rate
                },
                'statistical_significance': self._test_statistical_significance(
                    original_result.daily_returns, optimized_result.daily_returns
                ),
                'optimization_validation': self._validate_optimization_effectiveness(
                    original_result, optimized_result
                ),
                'recommendation': self._generate_optimization_recommendation(
                    original_result, optimized_result
                )
            }
            
            self.logger.info(f"ML optimization validation completed: {validation_analysis['improvement_metrics']['return_improvement']:.1%} return improvement")
            return validation_analysis
            
        except Exception as e:
            self.logger.error(f"Error in ML optimization validation: {str(e)}")
            return {'error': str(e)}
    
    def run_monte_carlo_simulation(self, base_parameters: Dict[str, Any],
                                 num_simulations: int = 1000,
                                 simulation_years: int = 3) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for strategy robustness testing
        
        Args:
            base_parameters: Base strategy parameters
            num_simulations: Number of simulation runs
            simulation_years: Years to simulate
            
        Returns:
            Monte Carlo simulation results
        """
        try:
            self.logger.info(f"Running Monte Carlo simulation: {num_simulations} runs over {simulation_years} years")
            
            simulation_results = []
            
            for sim_run in range(num_simulations):
                # Generate random market scenario
                start_date = datetime(2020, 1, 1)
                end_date = start_date + timedelta(days=simulation_years * 365)
                
                # Add parameter noise for robustness testing
                noisy_parameters = self._add_parameter_noise(base_parameters)
                
                # Run simulation
                sim_result = self.run_comprehensive_backtest(start_date, end_date, noisy_parameters)
                
                simulation_results.append({
                    'run': sim_run + 1,
                    'annual_return': sim_result.annualized_return,
                    'sharpe_ratio': sim_result.sharpe_ratio,
                    'max_drawdown': sim_result.max_drawdown,
                    'win_rate': sim_result.win_rate,
                    'total_trades': sim_result.total_trades
                })
                
                if (sim_run + 1) % 100 == 0:
                    self.logger.info(f"Completed {sim_run + 1}/{num_simulations} simulations")
            
            # Analyze simulation results
            returns = [r['annual_return'] for r in simulation_results]
            sharpe_ratios = [r['sharpe_ratio'] for r in simulation_results]
            max_drawdowns = [r['max_drawdown'] for r in simulation_results]
            
            monte_carlo_analysis = {
                'num_simulations': num_simulations,
                'simulation_years': simulation_years,
                'return_statistics': {
                    'mean': np.mean(returns),
                    'median': np.median(returns),
                    'std': np.std(returns),
                    'min': np.min(returns),
                    'max': np.max(returns),
                    'percentile_5': np.percentile(returns, 5),
                    'percentile_95': np.percentile(returns, 95)
                },
                'sharpe_statistics': {
                    'mean': np.mean(sharpe_ratios),
                    'median': np.median(sharpe_ratios),
                    'std': np.std(sharpe_ratios),
                    'percentile_5': np.percentile(sharpe_ratios, 5),
                    'percentile_95': np.percentile(sharpe_ratios, 95)
                },
                'drawdown_statistics': {
                    'mean': np.mean(max_drawdowns),
                    'median': np.median(max_drawdowns),
                    'worst_case': np.max(max_drawdowns),
                    'percentile_95': np.percentile(max_drawdowns, 95)
                },
                'probability_analysis': {
                    'prob_positive_return': len([r for r in returns if r > 0]) / len(returns),
                    'prob_beat_benchmark': len([r for r in returns if r > 0.08]) / len(returns),  # Beat 8% benchmark
                    'prob_large_drawdown': len([d for d in max_drawdowns if d > 0.20]) / len(max_drawdowns)  # >20% drawdown
                },
                'robustness_score': self._calculate_robustness_score(simulation_results),
                'risk_assessment': self._assess_monte_carlo_risk(simulation_results)
            }
            
            self.logger.info(f"Monte Carlo simulation completed: {monte_carlo_analysis['return_statistics']['mean']:.1%} mean return")
            return monte_carlo_analysis
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {'error': str(e)}
    
    def generate_performance_report(self, backtest_result: BacktestResult) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            report = {
                'executive_summary': {
                    'strategy': backtest_result.strategy_name,
                    'period': f"{backtest_result.start_date.strftime('%Y-%m-%d')} to {backtest_result.end_date.strftime('%Y-%m-%d')}",
                    'total_return': backtest_result.total_return,
                    'annualized_return': backtest_result.annualized_return,
                    'sharpe_ratio': backtest_result.sharpe_ratio,
                    'max_drawdown': backtest_result.max_drawdown,
                    'win_rate': backtest_result.win_rate,
                    'total_trades': backtest_result.total_trades
                },
                'performance_metrics': {
                    'return_metrics': {
                        'total_return': backtest_result.total_return,
                        'annualized_return': backtest_result.annualized_return,
                        'monthly_return': backtest_result.annualized_return / 12,
                        'weekly_return': backtest_result.annualized_return / 52
                    },
                    'risk_metrics': {
                        'sharpe_ratio': backtest_result.sharpe_ratio,
                        'sortino_ratio': backtest_result.sortino_ratio,
                        'calmar_ratio': backtest_result.calmar_ratio,
                        'max_drawdown': backtest_result.max_drawdown,
                        'var_95': backtest_result.var_95,
                        'expected_shortfall': backtest_result.expected_shortfall
                    },
                    'trading_metrics': {
                        'total_trades': backtest_result.total_trades,
                        'winning_trades': backtest_result.winning_trades,
                        'losing_trades': backtest_result.losing_trades,
                        'win_rate': backtest_result.win_rate,
                        'profit_factor': backtest_result.profit_factor,
                        'avg_trade_return': backtest_result.total_return / backtest_result.total_trades if backtest_result.total_trades > 0 else 0
                    }
                },
                'week_type_analysis': backtest_result.performance_by_week_type,
                'risk_analysis': backtest_result.risk_metrics,
                'execution_analysis': backtest_result.execution_stats,
                'benchmark_comparison': self._compare_to_benchmarks(backtest_result),
                'recommendations': self._generate_performance_recommendations(backtest_result)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            return {'error': str(e)}
    
    # Helper methods (implementing key functionality)
    def _generate_historical_market_data(self, start_date: datetime, end_date: datetime) -> Dict[datetime, Dict[str, Any]]:
        """Generate realistic historical market data for backtesting"""
        market_data = {}
        current_date = start_date
        
        # Initialize market state
        spy_price = 400.0
        vix_level = 20.0
        
        while current_date <= end_date:
            # Generate realistic market movements
            daily_return = np.random.normal(0.0005, 0.015)  # ~12% annual return, 24% volatility
            spy_price *= (1 + daily_return)
            
            # VIX mean reversion
            vix_level = vix_level * 0.95 + 20 * 0.05 + np.random.normal(0, 2)
            vix_level = max(10, min(80, vix_level))
            
            market_data[current_date] = {
                'spy_price': spy_price,
                'spy_return': daily_return,
                'vix': vix_level,
                'vix_change': vix_level - 20,
                'volume_ratio': np.random.normal(1.0, 0.2),
                'put_call_ratio': np.random.normal(1.0, 0.3),
                'rsi': np.random.normal(50, 15),
                'macd': np.random.normal(0, 0.5),
                'bollinger_position': np.random.uniform(0, 1),
                'trend_strength': np.random.normal(0, 0.3),
                'momentum': np.random.normal(0, 0.2)
            }
            
            current_date += timedelta(days=1)
        
        return market_data
    
    def _classify_week_type(self, market_data: Dict[str, Any], date: datetime) -> str:
        """Classify week type based on market data"""
        # Simplified week type classification
        spy_return = market_data.get('spy_return', 0)
        vix_level = market_data.get('vix', 20)
        
        if spy_return > 0.02:  # Strong positive week
            return 'C-WAP+' if vix_level < 25 else 'C-WAP'
        elif spy_return > 0.005:  # Moderate positive week
            return 'P-EW' if vix_level < 20 else 'C-WAP'
        elif spy_return > -0.005:  # Flat week
            return 'P-EW'
        elif spy_return > -0.02:  # Moderate negative week
            return 'P-RO' if vix_level > 25 else 'P-AWL'
        else:  # Strong negative week
            return 'P-DD' if vix_level > 30 else 'P-RO'
    
    def _initialize_portfolio(self) -> Dict[str, Any]:
        """Initialize portfolio for backtesting"""
        return {
            'equity': self.config['initial_capital'],
            'cash': self.config['initial_capital'],
            'positions': [],
            'margin_used': 0
        }
    
    def _generate_trading_signals(self, week_type: str, market_data: Dict[str, Any], 
                                parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on week type and parameters"""
        signals = []
        
        # Get week type characteristics
        week_char = self.week_type_characteristics.get(week_type, {})
        
        # Generate signal based on week type
        if week_type in ['P-EW', 'P-AWL']:
            signals.append({
                'action': 'sell_put',
                'symbol': 'SPY',
                'quantity': parameters.get('position_size', 10),
                'delta': parameters.get('target_delta', 30),
                'dte': week_char.get('avg_dte', 35),
                'week_type': week_type
            })
        elif week_type in ['C-WAP', 'C-WAP+']:
            signals.append({
                'action': 'sell_call',
                'symbol': 'SPY',
                'quantity': parameters.get('position_size', 10),
                'delta': parameters.get('target_delta', 30),
                'dte': week_char.get('avg_dte', 35),
                'week_type': week_type
            })
        
        return signals
    
    def _execute_trades(self, signals: List[Dict[str, Any]], date: datetime,
                       market_data: Dict[str, Any], portfolio: Dict[str, Any]) -> List[BacktestPosition]:
        """Execute trades based on signals"""
        new_positions = []
        
        for signal in signals:
            # Calculate option price (simplified)
            underlying_price = market_data.get('spy_price', 400)
            vix = market_data.get('vix', 20)
            
            # Simplified option pricing
            option_price = self._calculate_option_price(
                underlying_price, signal['delta'], signal['dte'], vix
            )
            
            # Create position
            position = BacktestPosition(
                position_id=f"pos_{date.strftime('%Y%m%d')}_{len(new_positions)}",
                symbol=signal['symbol'],
                strategy_type=signal['action'],
                week_type=signal['week_type'],
                entry_date=date,
                exit_date=None,
                entry_price=option_price,
                exit_price=None,
                quantity=signal['quantity'],
                delta=signal['delta'],
                dte=signal['dte'],
                premium_collected=option_price * signal['quantity'] * 100,  # $100 per contract
                realized_pnl=None,
                unrealized_pnl=0,
                max_profit=option_price * signal['quantity'] * 100,
                max_loss=0,
                status='open'
            )
            
            new_positions.append(position)
            
            # Update portfolio
            portfolio['cash'] += position.premium_collected
            portfolio['margin_used'] += position.premium_collected * self.config['margin_requirement']
        
        return new_positions
    
    def _calculate_option_price(self, underlying_price: float, delta: float, 
                              dte: int, vix: float) -> float:
        """Calculate simplified option price"""
        # Simplified option pricing based on delta and volatility
        time_value = (vix / 100) * np.sqrt(dte / 365) * underlying_price * 0.1
        intrinsic_value = max(0, underlying_price * (delta / 100) * 0.01)
        return time_value + intrinsic_value
    
    def _update_positions(self, positions: List[BacktestPosition], date: datetime,
                         market_data: Dict[str, Any]):
        """Update existing positions"""
        for position in positions:
            if position.status == 'open':
                # Update DTE
                days_passed = (date - position.entry_date).days
                current_dte = position.dte - days_passed
                
                # Check for expiration
                if current_dte <= 0:
                    position.status = 'expired'
                    position.exit_date = date
                    position.exit_price = 0  # Expired worthless
                    position.realized_pnl = position.premium_collected
                else:
                    # Update unrealized P&L (simplified)
                    current_price = self._calculate_option_price(
                        market_data.get('spy_price', 400),
                        position.delta,
                        current_dte,
                        market_data.get('vix', 20)
                    )
                    position.unrealized_pnl = position.premium_collected - (current_price * position.quantity * 100)
    
    def _calculate_daily_pnl(self, positions: List[BacktestPosition], 
                           portfolio: Dict[str, Any]) -> float:
        """Calculate daily P&L"""
        total_unrealized = sum(pos.unrealized_pnl for pos in positions if pos.status == 'open')
        total_realized = sum(pos.realized_pnl for pos in positions if pos.realized_pnl is not None)
        return total_unrealized + total_realized
    
    def _calculate_performance_metrics(self, daily_returns: List[float], 
                                     equity_curve: List[float]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        returns_array = np.array(daily_returns)
        
        # Basic return metrics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        
        # Risk metrics
        volatility = np.std(returns_array) * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.config['risk_free_rate']) / volatility if volatility > 0 else 0
        
        # Downside metrics
        negative_returns = returns_array[returns_array < 0]
        downside_volatility = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.config['risk_free_rate']) / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown metrics
        drawdown_curve = self._calculate_drawdown_curve(equity_curve)
        max_drawdown = max(drawdown_curve)
        max_dd_duration = self._calculate_max_drawdown_duration(drawdown_curve)
        
        # Trading metrics
        positive_returns = len([r for r in daily_returns if r > 0])
        win_rate = positive_returns / len(daily_returns) if daily_returns else 0
        
        # Profit factor
        gross_profit = sum([r for r in daily_returns if r > 0])
        gross_loss = abs(sum([r for r in daily_returns if r < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio
        }
    
    def _calculate_drawdown_curve(self, equity_curve: List[float]) -> List[float]:
        """Calculate drawdown curve"""
        peak = equity_curve[0]
        drawdowns = []
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            drawdowns.append(drawdown)
        
        return drawdowns
    
    def _calculate_max_drawdown_duration(self, drawdown_curve: List[float]) -> int:
        """Calculate maximum drawdown duration in days"""
        max_duration = 0
        current_duration = 0
        
        for dd in drawdown_curve:
            if dd > 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _analyze_week_type_performance(self, positions: List[BacktestPosition]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by week type"""
        week_type_stats = {}
        
        for week_type in self.week_type_characteristics.keys():
            week_positions = [p for p in positions if p.week_type == week_type]
            
            if week_positions:
                total_pnl = sum(p.realized_pnl for p in week_positions if p.realized_pnl is not None)
                winning_trades = len([p for p in week_positions if p.realized_pnl and p.realized_pnl > 0])
                total_trades = len([p for p in week_positions if p.realized_pnl is not None])
                
                week_type_stats[week_type] = {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                    'total_pnl': total_pnl,
                    'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0
                }
        
        return week_type_stats
    
    def _calculate_risk_metrics(self, daily_returns: List[float], 
                              equity_curve: List[float]) -> Dict[str, float]:
        """Calculate advanced risk metrics"""
        returns_array = np.array(daily_returns)
        
        # VaR and Expected Shortfall
        var_95 = np.percentile(returns_array, 5)
        tail_returns = returns_array[returns_array <= var_95]
        expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else 0
        
        return {
            'var_95': var_95,
            'expected_shortfall': expected_shortfall,
            'volatility': np.std(returns_array) * np.sqrt(252),
            'skewness': self._calculate_skewness(returns_array),
            'kurtosis': self._calculate_kurtosis(returns_array)
        }
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0
        return np.mean(((returns - mean_return) / std_return) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0
        return np.mean(((returns - mean_return) / std_return) ** 4) - 3
    
    def _calculate_execution_stats(self, positions: List[BacktestPosition]) -> Dict[str, Any]:
        """Calculate execution statistics"""
        closed_positions = [p for p in positions if p.realized_pnl is not None]
        
        if not closed_positions:
            return {}
        
        holding_periods = [(p.exit_date - p.entry_date).days for p in closed_positions if p.exit_date]
        
        return {
            'total_positions': len(positions),
            'closed_positions': len(closed_positions),
            'avg_holding_period': np.mean(holding_periods) if holding_periods else 0,
            'max_holding_period': max(holding_periods) if holding_periods else 0,
            'min_holding_period': min(holding_periods) if holding_periods else 0,
            'avg_premium_collected': np.mean([p.premium_collected for p in closed_positions]),
            'total_premium_collected': sum(p.premium_collected for p in closed_positions)
        }
    
    # Additional helper methods for stress testing and validation
    def _apply_stress_scenario(self, daily_returns: List[float], 
                             scenario: StressTestScenario) -> List[float]:
        """Apply stress scenario to returns"""
        stressed_returns = daily_returns.copy()
        
        # Apply market shock over scenario duration
        shock_per_day = scenario.market_shock / scenario.duration_days
        
        for i in range(min(scenario.duration_days, len(stressed_returns))):
            stressed_returns[i] += shock_per_day
        
        return stressed_returns
    
    def _calculate_stressed_equity_curve(self, original_curve: List[float], 
                                       stressed_returns: List[float]) -> List[float]:
        """Calculate equity curve under stress"""
        stressed_curve = [original_curve[0]]
        
        for i, ret in enumerate(stressed_returns):
            new_equity = stressed_curve[-1] * (1 + ret)
            stressed_curve.append(new_equity)
        
        return stressed_curve
    
    def _calculate_survival_probability(self, equity_curve: List[float]) -> float:
        """Calculate probability of strategy survival"""
        min_equity = min(equity_curve)
        initial_equity = equity_curve[0]
        return 1.0 if min_equity > 0 else 0.5  # Simplified calculation
    
    def _estimate_recovery_time(self, equity_curve: List[float]) -> int:
        """Estimate recovery time from maximum drawdown"""
        peak = max(equity_curve)
        peak_index = equity_curve.index(peak)
        
        # Find recovery point
        for i in range(peak_index, len(equity_curve)):
            if equity_curve[i] >= peak:
                return i - peak_index
        
        return len(equity_curve) - peak_index  # Still recovering
    
    def _assess_stress_risk(self, stressed_metrics: Dict[str, float]) -> str:
        """Assess risk level under stress"""
        if stressed_metrics['max_drawdown'] > 0.5:
            return "High Risk"
        elif stressed_metrics['max_drawdown'] > 0.3:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def _test_statistical_significance(self, returns1: List[float], 
                                     returns2: List[float]) -> Dict[str, Any]:
        """Test statistical significance of performance difference"""
        from scipy import stats
        
        # T-test for difference in means
        t_stat, p_value = stats.ttest_ind(returns1, returns2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'confidence_level': 0.95
        }
    
    def _validate_optimization_effectiveness(self, original: BacktestResult, 
                                           optimized: BacktestResult) -> str:
        """Validate effectiveness of optimization"""
        improvements = 0
        
        if optimized.annualized_return > original.annualized_return:
            improvements += 1
        if optimized.sharpe_ratio > original.sharpe_ratio:
            improvements += 1
        if optimized.max_drawdown < original.max_drawdown:
            improvements += 1
        if optimized.win_rate > original.win_rate:
            improvements += 1
        
        if improvements >= 3:
            return "Highly Effective"
        elif improvements >= 2:
            return "Moderately Effective"
        else:
            return "Limited Effectiveness"
    
    def _generate_optimization_recommendation(self, original: BacktestResult, 
                                            optimized: BacktestResult) -> str:
        """Generate recommendation based on optimization results"""
        return_improvement = optimized.annualized_return - original.annualized_return
        
        if return_improvement > 0.05:  # 5% improvement
            return "Strong recommendation to adopt optimized parameters"
        elif return_improvement > 0.02:  # 2% improvement
            return "Moderate recommendation to adopt optimized parameters"
        else:
            return "Limited benefit from optimization, consider alternative approaches"
    
    def _add_parameter_noise(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Add noise to parameters for Monte Carlo simulation"""
        noisy_params = parameters.copy()
        
        # Add 5% noise to numeric parameters
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                noise = np.random.normal(0, 0.05) * value
                noisy_params[key] = max(0, value + noise)
        
        return noisy_params
    
    def _calculate_robustness_score(self, simulation_results: List[Dict[str, Any]]) -> float:
        """Calculate strategy robustness score"""
        returns = [r['annual_return'] for r in simulation_results]
        positive_returns = len([r for r in returns if r > 0])
        
        # Robustness based on consistency of positive returns
        return positive_returns / len(returns)
    
    def _assess_monte_carlo_risk(self, simulation_results: List[Dict[str, Any]]) -> str:
        """Assess risk based on Monte Carlo results"""
        returns = [r['annual_return'] for r in simulation_results]
        worst_case = min(returns)
        
        if worst_case < -0.3:
            return "High Risk"
        elif worst_case < -0.15:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def _compare_to_benchmarks(self, result: BacktestResult) -> Dict[str, Any]:
        """Compare results to standard benchmarks"""
        # Simplified benchmark comparison
        spy_return = 0.10  # Assume 10% SPY return
        bond_return = 0.03  # Assume 3% bond return
        
        return {
            'vs_spy': {
                'excess_return': result.annualized_return - spy_return,
                'better_performance': result.annualized_return > spy_return
            },
            'vs_bonds': {
                'excess_return': result.annualized_return - bond_return,
                'better_performance': result.annualized_return > bond_return
            },
            'risk_adjusted_vs_spy': {
                'excess_sharpe': result.sharpe_ratio - (spy_return / 0.16),  # Assume SPY Sharpe ~0.625
                'better_risk_adjusted': result.sharpe_ratio > (spy_return / 0.16)
            }
        }
    
    def _generate_performance_recommendations(self, result: BacktestResult) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if result.sharpe_ratio < 1.0:
            recommendations.append("Consider improving risk-adjusted returns through better position sizing")
        
        if result.max_drawdown > 0.20:
            recommendations.append("Implement stronger risk management to reduce maximum drawdown")
        
        if result.win_rate < 0.60:
            recommendations.append("Review entry criteria to improve win rate")
        
        return recommendations

def test_backtesting_engine():
    """Test the backtesting engine"""
    print("Testing Advanced Backtesting Engine...")
    
    engine = AdvancedBacktestingEngine()
    
    # Test comprehensive backtest
    print("\n--- Testing Comprehensive Backtest ---")
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    strategy_params = {
        'position_size': 10,
        'target_delta': 30,
        'max_dte': 45
    }
    
    backtest_result = engine.run_comprehensive_backtest(start_date, end_date, strategy_params)
    
    print(f"Backtest Period: {backtest_result.start_date} to {backtest_result.end_date}")
    print(f"Total Return: {backtest_result.total_return:.1%}")
    print(f"Annualized Return: {backtest_result.annualized_return:.1%}")
    print(f"Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {backtest_result.max_drawdown:.1%}")
    print(f"Win Rate: {backtest_result.win_rate:.1%}")
    print(f"Total Trades: {backtest_result.total_trades}")
    
    # Test stress testing
    print("\n--- Testing Stress Test ---")
    stress_scenario = engine.stress_scenarios[0]  # 2008 Financial Crisis
    stress_result = engine.run_stress_test(backtest_result, stress_scenario)
    
    print(f"Stress Scenario: {stress_result['scenario']}")
    print(f"Return Impact: {stress_result['return_impact']:.1%}")
    print(f"Sharpe Impact: {stress_result['sharpe_impact']:.2f}")
    print(f"Max DD Impact: {stress_result['max_dd_impact']:.1%}")
    print(f"Risk Assessment: {stress_result['risk_assessment']}")
    
    # Test ML optimization validation
    print("\n--- Testing ML Optimization Validation ---")
    original_params = {'position_size': 10, 'target_delta': 30}
    optimized_params = {'position_size': 12, 'target_delta': 35}
    validation_period = (datetime(2023, 6, 1), datetime(2023, 12, 31))
    
    validation_result = engine.validate_ml_optimization(
        original_params, optimized_params, validation_period
    )
    
    print(f"Return Improvement: {validation_result['improvement_metrics']['return_improvement']:.1%}")
    print(f"Sharpe Improvement: {validation_result['improvement_metrics']['sharpe_improvement']:.2f}")
    print(f"Optimization Validation: {validation_result['optimization_validation']}")
    print(f"Recommendation: {validation_result['recommendation']}")
    
    # Test Monte Carlo simulation (small sample)
    print("\n--- Testing Monte Carlo Simulation ---")
    monte_carlo_result = engine.run_monte_carlo_simulation(
        strategy_params, num_simulations=50, simulation_years=1
    )
    
    print(f"Mean Return: {monte_carlo_result['return_statistics']['mean']:.1%}")
    print(f"Return Std Dev: {monte_carlo_result['return_statistics']['std']:.1%}")
    print(f"5th Percentile: {monte_carlo_result['return_statistics']['percentile_5']:.1%}")
    print(f"95th Percentile: {monte_carlo_result['return_statistics']['percentile_95']:.1%}")
    print(f"Prob Positive Return: {monte_carlo_result['probability_analysis']['prob_positive_return']:.1%}")
    print(f"Robustness Score: {monte_carlo_result['robustness_score']:.1%}")
    
    # Test performance report
    print("\n--- Testing Performance Report ---")
    performance_report = engine.generate_performance_report(backtest_result)
    
    print(f"Executive Summary:")
    print(f"  Strategy: {performance_report['executive_summary']['strategy']}")
    print(f"  Period: {performance_report['executive_summary']['period']}")
    print(f"  Annual Return: {performance_report['executive_summary']['annualized_return']:.1%}")
    print(f"  Sharpe Ratio: {performance_report['executive_summary']['sharpe_ratio']:.2f}")
    
    print("\n✅ Advanced Backtesting Engine test completed successfully!")

if __name__ == "__main__":
    test_backtesting_engine()

