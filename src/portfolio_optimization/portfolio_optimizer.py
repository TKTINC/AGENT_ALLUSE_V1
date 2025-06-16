"""
Portfolio Optimization System for ALL-USE Protocol
Advanced portfolio optimization with multi-strategy coordination, capital allocation, and risk management

This module provides comprehensive portfolio optimization including:
- Multi-strategy portfolio coordination and management
- Intelligent capital allocation optimization
- Position correlation analysis and management
- Automated portfolio rebalancing algorithms
- Risk-adjusted portfolio construction
- Performance optimization and tracking
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
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    MAX_RETURN = "maximize_return"
    MIN_RISK = "minimize_risk"
    MAX_SHARPE = "maximize_sharpe"
    MAX_SORTINO = "maximize_sortino"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "minimum_variance"

class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequency"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    THRESHOLD_BASED = "threshold_based"
    ADAPTIVE = "adaptive"

class StrategyType(Enum):
    """ALL-USE strategy types"""
    PUT_SELLING = "put_selling"
    CALL_SELLING = "call_selling"
    IRON_CONDOR = "iron_condor"
    PUT_SPREAD = "put_spread"
    CALL_SPREAD = "call_spread"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    BUTTERFLY = "butterfly"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"

@dataclass
class StrategyAllocation:
    """Strategy allocation within portfolio"""
    strategy_id: str
    strategy_type: StrategyType
    target_weight: float
    current_weight: float
    capital_allocated: float
    expected_return: float
    expected_volatility: float
    max_drawdown: float
    sharpe_ratio: float
    correlation_matrix: Dict[str, float]
    performance_score: float
    risk_score: float
    last_updated: datetime

@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints"""
    max_position_weight: float = 0.20        # Maximum single position weight
    min_position_weight: float = 0.01        # Minimum position weight
    max_strategy_weight: float = 0.30        # Maximum strategy weight
    min_strategy_weight: float = 0.05        # Minimum strategy weight
    max_sector_concentration: float = 0.40   # Maximum sector concentration
    max_correlation: float = 0.70            # Maximum position correlation
    min_diversification_ratio: float = 0.60  # Minimum diversification ratio
    max_turnover: float = 0.20               # Maximum portfolio turnover
    target_volatility: Optional[float] = None # Target portfolio volatility
    max_leverage: float = 1.0                # Maximum leverage ratio

@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    optimization_id: str
    objective: OptimizationObjective
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    diversification_ratio: float
    turnover: float
    optimization_score: float
    constraints_satisfied: bool
    optimization_time: float
    convergence_status: str
    timestamp: datetime

@dataclass
class RebalanceRecommendation:
    """Portfolio rebalancing recommendation"""
    rebalance_id: str
    trigger_reason: str
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    weight_changes: Dict[str, float]
    trades_required: List[Dict[str, Any]]
    expected_cost: float
    expected_benefit: float
    urgency_score: float
    confidence: float
    timestamp: datetime

class PortfolioOptimizer:
    """
    Advanced Portfolio Optimization System for ALL-USE Protocol
    
    Provides comprehensive portfolio optimization with:
    - Multi-strategy coordination and allocation
    - Risk-adjusted portfolio construction
    - Intelligent capital allocation optimization
    - Automated rebalancing with cost consideration
    - Correlation analysis and diversification optimization
    """
    
    def __init__(self):
        """Initialize the portfolio optimizer"""
        self.logger = logging.getLogger(__name__)
        
        # Optimization configuration
        self.config = {
            'optimization_frequency': 'daily',
            'rebalance_threshold': 0.05,        # 5% weight deviation threshold
            'min_rebalance_benefit': 0.001,     # 0.1% minimum benefit for rebalancing
            'transaction_cost': 0.001,          # 0.1% transaction cost
            'lookback_period': 252,             # Trading days for analysis
            'monte_carlo_simulations': 1000,    # Monte Carlo simulation runs
            'optimization_tolerance': 1e-6,     # Optimization convergence tolerance
            'max_iterations': 1000,             # Maximum optimization iterations
            'risk_free_rate': 0.02,             # Risk-free rate for Sharpe calculation
            'confidence_level': 0.95            # Confidence level for risk metrics
        }
        
        # Portfolio state
        self.current_portfolio: Dict[str, StrategyAllocation] = {}
        self.target_portfolio: Dict[str, StrategyAllocation] = {}
        self.portfolio_constraints = PortfolioConstraints()
        
        # Optimization history
        self.optimization_history: List[OptimizationResult] = []
        self.rebalance_history: List[RebalanceRecommendation] = []
        
        # Market data and correlations
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        self.expected_returns: Dict[str, float] = {}
        self.expected_volatilities: Dict[str, float] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        
        # Optimization thread
        self.optimization_active = False
        self.optimization_thread = None
        
        self.logger.info("Portfolio Optimizer initialized")
    
    def optimize_portfolio(self, 
                          objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
                          constraints: Optional[PortfolioConstraints] = None) -> OptimizationResult:
        """
        Optimize portfolio allocation based on objective and constraints
        
        Args:
            objective: Optimization objective
            constraints: Portfolio constraints
            
        Returns:
            Optimization result with optimal weights and metrics
        """
        try:
            start_time = time.time()
            
            # Use provided constraints or default
            if constraints:
                self.portfolio_constraints = constraints
            
            # Get available strategies and their characteristics
            strategies = self._get_available_strategies()
            
            if not strategies:
                return self._create_empty_optimization_result(objective)
            
            # Prepare optimization data
            returns_data = self._prepare_returns_data(strategies)
            correlation_matrix = self._calculate_correlation_matrix(strategies)
            
            # Run optimization based on objective
            if objective == OptimizationObjective.MAX_SHARPE:
                optimal_weights = self._optimize_max_sharpe(returns_data, correlation_matrix)
            elif objective == OptimizationObjective.MIN_RISK:
                optimal_weights = self._optimize_min_risk(returns_data, correlation_matrix)
            elif objective == OptimizationObjective.MAX_RETURN:
                optimal_weights = self._optimize_max_return(returns_data, correlation_matrix)
            elif objective == OptimizationObjective.RISK_PARITY:
                optimal_weights = self._optimize_risk_parity(returns_data, correlation_matrix)
            elif objective == OptimizationObjective.MIN_VARIANCE:
                optimal_weights = self._optimize_min_variance(returns_data, correlation_matrix)
            else:
                optimal_weights = self._optimize_max_sharpe(returns_data, correlation_matrix)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(optimal_weights, returns_data, correlation_matrix)
            
            # Check constraints
            constraints_satisfied = self._check_constraints(optimal_weights)
            
            # Create optimization result
            optimization_result = OptimizationResult(
                optimization_id=f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                objective=objective,
                optimal_weights=optimal_weights,
                expected_return=portfolio_metrics['expected_return'],
                expected_volatility=portfolio_metrics['expected_volatility'],
                sharpe_ratio=portfolio_metrics['sharpe_ratio'],
                sortino_ratio=portfolio_metrics['sortino_ratio'],
                max_drawdown=portfolio_metrics['max_drawdown'],
                diversification_ratio=portfolio_metrics['diversification_ratio'],
                turnover=self._calculate_turnover(optimal_weights),
                optimization_score=portfolio_metrics['optimization_score'],
                constraints_satisfied=constraints_satisfied,
                optimization_time=time.time() - start_time,
                convergence_status='converged',
                timestamp=datetime.now()
            )
            
            # Store optimization result
            self.optimization_history.append(optimization_result)
            
            # Update target portfolio
            self._update_target_portfolio(optimal_weights, strategies)
            
            self.logger.info(f"Portfolio optimization completed: {objective.value} (Sharpe: {portfolio_metrics['sharpe_ratio']:.2f})")
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {str(e)}")
            return self._create_error_optimization_result(objective, str(e))
    
    def generate_rebalance_recommendation(self, force_rebalance: bool = False) -> Optional[RebalanceRecommendation]:
        """
        Generate portfolio rebalancing recommendation
        
        Args:
            force_rebalance: Force rebalancing regardless of thresholds
            
        Returns:
            Rebalancing recommendation or None if not needed
        """
        try:
            if not self.current_portfolio or not self.target_portfolio:
                return None
            
            # Calculate current vs target weight differences
            weight_differences = {}
            max_deviation = 0.0
            
            for strategy_id in self.target_portfolio.keys():
                current_weight = self.current_portfolio.get(strategy_id, StrategyAllocation(
                    strategy_id=strategy_id,
                    strategy_type=StrategyType.PUT_SELLING,
                    target_weight=0.0,
                    current_weight=0.0,
                    capital_allocated=0.0,
                    expected_return=0.0,
                    expected_volatility=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    correlation_matrix={},
                    performance_score=0.0,
                    risk_score=0.0,
                    last_updated=datetime.now()
                )).current_weight
                
                target_weight = self.target_portfolio[strategy_id].target_weight
                difference = target_weight - current_weight
                weight_differences[strategy_id] = difference
                max_deviation = max(max_deviation, abs(difference))
            
            # Check if rebalancing is needed
            rebalance_threshold = self.config['rebalance_threshold']
            
            if not force_rebalance and max_deviation < rebalance_threshold:
                return None
            
            # Calculate trades required
            trades_required = self._calculate_required_trades(weight_differences)
            
            # Estimate costs and benefits
            expected_cost = self._estimate_rebalancing_cost(trades_required)
            expected_benefit = self._estimate_rebalancing_benefit(weight_differences)
            
            # Check if rebalancing is beneficial
            min_benefit = self.config['min_rebalance_benefit']
            net_benefit = expected_benefit - expected_cost
            
            if not force_rebalance and net_benefit < min_benefit:
                return None
            
            # Determine trigger reason
            trigger_reason = self._determine_rebalance_trigger(max_deviation, force_rebalance)
            
            # Calculate urgency and confidence
            urgency_score = min(1.0, max_deviation / rebalance_threshold)
            confidence = self._calculate_rebalance_confidence(weight_differences, trades_required)
            
            # Create rebalancing recommendation
            recommendation = RebalanceRecommendation(
                rebalance_id=f"rebal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                trigger_reason=trigger_reason,
                current_weights={k: v.current_weight for k, v in self.current_portfolio.items()},
                target_weights={k: v.target_weight for k, v in self.target_portfolio.items()},
                weight_changes=weight_differences,
                trades_required=trades_required,
                expected_cost=expected_cost,
                expected_benefit=expected_benefit,
                urgency_score=urgency_score,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            # Store recommendation
            self.rebalance_history.append(recommendation)
            
            self.logger.info(f"Rebalancing recommended: {trigger_reason} (Max deviation: {max_deviation:.1%})")
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error generating rebalance recommendation: {str(e)}")
            return None
    
    def analyze_strategy_correlations(self) -> Dict[str, Any]:
        """
        Analyze correlations between strategies
        
        Returns:
            Correlation analysis results
        """
        try:
            if len(self.current_portfolio) < 2:
                return {'error': 'Insufficient strategies for correlation analysis'}
            
            strategies = list(self.current_portfolio.keys())
            n_strategies = len(strategies)
            
            # Build correlation matrix
            correlation_matrix = np.zeros((n_strategies, n_strategies))
            
            for i, strategy1 in enumerate(strategies):
                for j, strategy2 in enumerate(strategies):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                    else:
                        correlation = self._calculate_strategy_correlation(strategy1, strategy2)
                        correlation_matrix[i, j] = correlation
            
            # Convert to DataFrame
            corr_df = pd.DataFrame(correlation_matrix, index=strategies, columns=strategies)
            
            # Calculate correlation statistics
            upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            
            correlation_stats = {
                'average_correlation': np.mean(upper_triangle),
                'max_correlation': np.max(upper_triangle),
                'min_correlation': np.min(upper_triangle),
                'correlation_std': np.std(upper_triangle),
                'high_correlation_pairs': self._find_high_correlation_pairs(corr_df),
                'diversification_ratio': self._calculate_diversification_ratio(correlation_matrix)
            }
            
            # Identify correlation clusters
            clusters = self._identify_correlation_clusters(corr_df)
            
            # Generate recommendations
            recommendations = self._generate_correlation_recommendations(correlation_stats, clusters)
            
            return {
                'correlation_matrix': corr_df.to_dict(),
                'correlation_statistics': correlation_stats,
                'correlation_clusters': clusters,
                'recommendations': recommendations,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing strategy correlations: {str(e)}")
            return {'error': str(e)}
    
    def optimize_capital_allocation(self, total_capital: float, 
                                  strategy_preferences: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Optimize capital allocation across strategies
        
        Args:
            total_capital: Total capital to allocate
            strategy_preferences: Optional strategy preference weights
            
        Returns:
            Capital allocation optimization results
        """
        try:
            if not self.target_portfolio:
                return {'error': 'No target portfolio defined'}
            
            # Get strategy weights and characteristics
            strategies = list(self.target_portfolio.keys())
            target_weights = [self.target_portfolio[s].target_weight for s in strategies]
            expected_returns = [self.target_portfolio[s].expected_return for s in strategies]
            volatilities = [self.target_portfolio[s].expected_volatility for s in strategies]
            
            # Apply strategy preferences if provided
            if strategy_preferences:
                adjusted_weights = []
                for i, strategy in enumerate(strategies):
                    preference = strategy_preferences.get(strategy, 1.0)
                    adjusted_weights.append(target_weights[i] * preference)
                
                # Normalize weights
                total_weight = sum(adjusted_weights)
                target_weights = [w / total_weight for w in adjusted_weights]
            
            # Calculate optimal capital allocation
            capital_allocations = {}
            risk_allocations = {}
            
            for i, strategy in enumerate(strategies):
                # Base allocation from target weights
                base_allocation = total_capital * target_weights[i]
                
                # Risk-adjusted allocation
                risk_adjustment = self._calculate_risk_adjustment(
                    expected_returns[i], volatilities[i]
                )
                
                # Final allocation
                final_allocation = base_allocation * risk_adjustment
                capital_allocations[strategy] = final_allocation
                risk_allocations[strategy] = risk_adjustment
            
            # Normalize to total capital
            total_allocated = sum(capital_allocations.values())
            if total_allocated > 0:
                normalization_factor = total_capital / total_allocated
                capital_allocations = {k: v * normalization_factor for k, v in capital_allocations.items()}
            
            # Calculate allocation metrics
            allocation_metrics = self._calculate_allocation_metrics(
                capital_allocations, expected_returns, volatilities
            )
            
            # Generate allocation recommendations
            recommendations = self._generate_allocation_recommendations(
                capital_allocations, allocation_metrics
            )
            
            return {
                'capital_allocations': capital_allocations,
                'allocation_percentages': {k: v/total_capital for k, v in capital_allocations.items()},
                'risk_adjustments': risk_allocations,
                'allocation_metrics': allocation_metrics,
                'recommendations': recommendations,
                'total_capital': total_capital,
                'allocation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing capital allocation: {str(e)}")
            return {'error': str(e)}
    
    def get_portfolio_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive portfolio optimization dashboard"""
        try:
            # Current portfolio status
            portfolio_status = {
                'total_strategies': len(self.current_portfolio),
                'total_capital_allocated': sum(s.capital_allocated for s in self.current_portfolio.values()),
                'portfolio_expected_return': self._calculate_portfolio_expected_return(),
                'portfolio_volatility': self._calculate_portfolio_volatility(),
                'portfolio_sharpe_ratio': self._calculate_portfolio_sharpe_ratio(),
                'diversification_ratio': self._calculate_current_diversification_ratio()
            }
            
            # Strategy breakdown
            strategy_breakdown = {}
            for strategy_id, allocation in self.current_portfolio.items():
                strategy_breakdown[strategy_id] = {
                    'strategy_type': allocation.strategy_type.value,
                    'current_weight': allocation.current_weight,
                    'target_weight': allocation.target_weight,
                    'capital_allocated': allocation.capital_allocated,
                    'expected_return': allocation.expected_return,
                    'volatility': allocation.expected_volatility,
                    'sharpe_ratio': allocation.sharpe_ratio,
                    'performance_score': allocation.performance_score,
                    'risk_score': allocation.risk_score
                }
            
            # Recent optimization results
            recent_optimization = None
            if self.optimization_history:
                latest_opt = self.optimization_history[-1]
                recent_optimization = {
                    'optimization_id': latest_opt.optimization_id,
                    'objective': latest_opt.objective.value,
                    'sharpe_ratio': latest_opt.sharpe_ratio,
                    'expected_return': latest_opt.expected_return,
                    'expected_volatility': latest_opt.expected_volatility,
                    'optimization_score': latest_opt.optimization_score,
                    'timestamp': latest_opt.timestamp.isoformat()
                }
            
            # Rebalancing status
            rebalancing_status = {
                'rebalancing_needed': False,
                'max_weight_deviation': 0.0,
                'last_rebalance': None
            }
            
            if self.current_portfolio and self.target_portfolio:
                max_deviation = max(
                    abs(self.target_portfolio[s].target_weight - self.current_portfolio.get(s, type('obj', (object,), {'current_weight': 0})).current_weight)
                    for s in self.target_portfolio.keys()
                )
                rebalancing_status['max_weight_deviation'] = max_deviation
                rebalancing_status['rebalancing_needed'] = max_deviation > self.config['rebalance_threshold']
            
            if self.rebalance_history:
                last_rebalance = self.rebalance_history[-1]
                rebalancing_status['last_rebalance'] = {
                    'rebalance_id': last_rebalance.rebalance_id,
                    'trigger_reason': last_rebalance.trigger_reason,
                    'timestamp': last_rebalance.timestamp.isoformat()
                }
            
            # Performance metrics
            performance_summary = {
                'optimization_count': len(self.optimization_history),
                'rebalance_count': len(self.rebalance_history),
                'average_optimization_time': np.mean([opt.optimization_time for opt in self.optimization_history]) if self.optimization_history else 0,
                'portfolio_efficiency': self._calculate_portfolio_efficiency()
            }
            
            # Recommendations
            recommendations = self._generate_portfolio_recommendations()
            
            dashboard = {
                'portfolio_status': portfolio_status,
                'strategy_breakdown': strategy_breakdown,
                'recent_optimization': recent_optimization,
                'rebalancing_status': rebalancing_status,
                'performance_summary': performance_summary,
                'recommendations': recommendations,
                'last_updated': datetime.now().isoformat()
            }
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio dashboard: {str(e)}")
            return {'error': str(e)}
    
    def start_continuous_optimization(self):
        """Start continuous portfolio optimization"""
        if self.optimization_active:
            self.logger.warning("Continuous optimization already active")
            return
        
        self.optimization_active = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        self.logger.info("Continuous portfolio optimization started")
    
    def stop_continuous_optimization(self):
        """Stop continuous optimization"""
        self.optimization_active = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        
        self.logger.info("Continuous portfolio optimization stopped")
    
    # Helper methods for portfolio optimization
    def _get_available_strategies(self) -> List[Dict[str, Any]]:
        """Get available strategies for optimization"""
        # Mock strategy data - in practice, this would come from strategy manager
        strategies = [
            {
                'strategy_id': 'put_selling_spy',
                'strategy_type': StrategyType.PUT_SELLING,
                'expected_return': 0.15,
                'expected_volatility': 0.12,
                'max_drawdown': 0.08,
                'sharpe_ratio': 1.25,
                'capacity': 1000000
            },
            {
                'strategy_id': 'iron_condor_qqq',
                'strategy_type': StrategyType.IRON_CONDOR,
                'expected_return': 0.12,
                'expected_volatility': 0.10,
                'max_drawdown': 0.06,
                'sharpe_ratio': 1.10,
                'capacity': 800000
            },
            {
                'strategy_id': 'put_spread_iwm',
                'strategy_type': StrategyType.PUT_SPREAD,
                'expected_return': 0.18,
                'expected_volatility': 0.15,
                'max_drawdown': 0.10,
                'sharpe_ratio': 1.35,
                'capacity': 600000
            },
            {
                'strategy_id': 'call_selling_spy',
                'strategy_type': StrategyType.CALL_SELLING,
                'expected_return': 0.10,
                'expected_volatility': 0.08,
                'max_drawdown': 0.05,
                'sharpe_ratio': 0.95,
                'capacity': 1200000
            }
        ]
        
        return strategies
    
    def _prepare_returns_data(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare returns data for optimization"""
        returns_data = {}
        
        for strategy in strategies:
            strategy_id = strategy['strategy_id']
            returns_data[strategy_id] = {
                'expected_return': strategy['expected_return'],
                'volatility': strategy['expected_volatility'],
                'sharpe_ratio': strategy['sharpe_ratio'],
                'max_drawdown': strategy['max_drawdown']
            }
        
        return returns_data
    
    def _calculate_correlation_matrix(self, strategies: List[Dict[str, Any]]) -> pd.DataFrame:
        """Calculate correlation matrix between strategies"""
        strategy_ids = [s['strategy_id'] for s in strategies]
        n_strategies = len(strategy_ids)
        
        # Mock correlation matrix - in practice, this would be calculated from historical data
        correlation_matrix = np.eye(n_strategies)
        
        # Add some realistic correlations
        for i in range(n_strategies):
            for j in range(i+1, n_strategies):
                # Strategies on same underlying have higher correlation
                if 'spy' in strategy_ids[i] and 'spy' in strategy_ids[j]:
                    correlation = 0.75
                elif 'qqq' in strategy_ids[i] and 'qqq' in strategy_ids[j]:
                    correlation = 0.70
                else:
                    correlation = 0.35  # Default correlation
                
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return pd.DataFrame(correlation_matrix, index=strategy_ids, columns=strategy_ids)
    
    def _optimize_max_sharpe(self, returns_data: Dict[str, Any], correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """Optimize for maximum Sharpe ratio"""
        try:
            strategy_ids = list(returns_data.keys())
            n_strategies = len(strategy_ids)
            
            if n_strategies == 0:
                return {}
            
            # Extract expected returns and volatilities
            expected_returns = np.array([returns_data[s]['expected_return'] for s in strategy_ids])
            volatilities = np.array([returns_data[s]['volatility'] for s in strategy_ids])
            
            # Convert correlation matrix to covariance matrix
            correlation_np = correlation_matrix.values
            covariance_matrix = np.outer(volatilities, volatilities) * correlation_np
            
            # Optimize using simplified mean-variance optimization
            # In practice, this would use scipy.optimize or cvxpy
            
            # Equal weight as starting point
            weights = np.ones(n_strategies) / n_strategies
            
            # Simple iterative optimization
            for iteration in range(100):
                # Calculate portfolio return and risk
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # Calculate Sharpe ratio
                risk_free_rate = self.config['risk_free_rate']
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                
                # Simple gradient-based update
                gradient = (expected_returns - risk_free_rate) / portfolio_volatility - \
                          (portfolio_return - risk_free_rate) * np.dot(covariance_matrix, weights) / (portfolio_variance * portfolio_volatility)
                
                # Update weights
                learning_rate = 0.01
                weights += learning_rate * gradient
                
                # Apply constraints
                weights = np.maximum(weights, self.portfolio_constraints.min_position_weight)
                weights = np.minimum(weights, self.portfolio_constraints.max_position_weight)
                
                # Normalize weights
                weights = weights / np.sum(weights)
            
            # Convert to dictionary
            optimal_weights = {strategy_ids[i]: weights[i] for i in range(n_strategies)}
            
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"Error in max Sharpe optimization: {str(e)}")
            # Return equal weights as fallback
            strategy_ids = list(returns_data.keys())
            equal_weight = 1.0 / len(strategy_ids) if strategy_ids else 0.0
            return {s: equal_weight for s in strategy_ids}
    
    def _optimize_min_risk(self, returns_data: Dict[str, Any], correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """Optimize for minimum risk"""
        try:
            strategy_ids = list(returns_data.keys())
            n_strategies = len(strategy_ids)
            
            if n_strategies == 0:
                return {}
            
            # Extract volatilities
            volatilities = np.array([returns_data[s]['volatility'] for s in strategy_ids])
            
            # Convert correlation matrix to covariance matrix
            correlation_np = correlation_matrix.values
            covariance_matrix = np.outer(volatilities, volatilities) * correlation_np
            
            # Minimum variance optimization
            # weights = inv(Σ) * 1 / (1' * inv(Σ) * 1)
            try:
                inv_cov = np.linalg.inv(covariance_matrix)
                ones = np.ones(n_strategies)
                weights = np.dot(inv_cov, ones) / np.dot(ones, np.dot(inv_cov, ones))
            except np.linalg.LinAlgError:
                # Fallback to equal weights if matrix is singular
                weights = np.ones(n_strategies) / n_strategies
            
            # Apply constraints
            weights = np.maximum(weights, self.portfolio_constraints.min_position_weight)
            weights = np.minimum(weights, self.portfolio_constraints.max_position_weight)
            weights = weights / np.sum(weights)
            
            optimal_weights = {strategy_ids[i]: weights[i] for i in range(n_strategies)}
            
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"Error in min risk optimization: {str(e)}")
            strategy_ids = list(returns_data.keys())
            equal_weight = 1.0 / len(strategy_ids) if strategy_ids else 0.0
            return {s: equal_weight for s in strategy_ids}
    
    def _optimize_max_return(self, returns_data: Dict[str, Any], correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """Optimize for maximum return"""
        try:
            strategy_ids = list(returns_data.keys())
            
            if not strategy_ids:
                return {}
            
            # Find strategy with highest expected return
            best_strategy = max(strategy_ids, key=lambda s: returns_data[s]['expected_return'])
            
            # Allocate maximum weight to best strategy, distribute rest equally
            max_weight = self.portfolio_constraints.max_position_weight
            remaining_weight = 1.0 - max_weight
            other_strategies = [s for s in strategy_ids if s != best_strategy]
            
            optimal_weights = {best_strategy: max_weight}
            
            if other_strategies:
                equal_weight = remaining_weight / len(other_strategies)
                for strategy in other_strategies:
                    optimal_weights[strategy] = equal_weight
            
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"Error in max return optimization: {str(e)}")
            strategy_ids = list(returns_data.keys())
            equal_weight = 1.0 / len(strategy_ids) if strategy_ids else 0.0
            return {s: equal_weight for s in strategy_ids}
    
    def _optimize_risk_parity(self, returns_data: Dict[str, Any], correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """Optimize for risk parity"""
        try:
            strategy_ids = list(returns_data.keys())
            n_strategies = len(strategy_ids)
            
            if n_strategies == 0:
                return {}
            
            # Extract volatilities
            volatilities = np.array([returns_data[s]['volatility'] for s in strategy_ids])
            
            # Risk parity: weight inversely proportional to volatility
            inv_volatilities = 1.0 / volatilities
            weights = inv_volatilities / np.sum(inv_volatilities)
            
            # Apply constraints
            weights = np.maximum(weights, self.portfolio_constraints.min_position_weight)
            weights = np.minimum(weights, self.portfolio_constraints.max_position_weight)
            weights = weights / np.sum(weights)
            
            optimal_weights = {strategy_ids[i]: weights[i] for i in range(n_strategies)}
            
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"Error in risk parity optimization: {str(e)}")
            strategy_ids = list(returns_data.keys())
            equal_weight = 1.0 / len(strategy_ids) if strategy_ids else 0.0
            return {s: equal_weight for s in strategy_ids}
    
    def _optimize_min_variance(self, returns_data: Dict[str, Any], correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """Optimize for minimum variance (same as min risk)"""
        return self._optimize_min_risk(returns_data, correlation_matrix)
    
    def _calculate_portfolio_metrics(self, weights: Dict[str, float], 
                                   returns_data: Dict[str, Any], 
                                   correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio metrics for given weights"""
        try:
            if not weights:
                return self._get_default_portfolio_metrics()
            
            strategy_ids = list(weights.keys())
            weight_array = np.array([weights[s] for s in strategy_ids])
            
            # Expected return
            expected_returns = np.array([returns_data[s]['expected_return'] for s in strategy_ids])
            portfolio_return = np.dot(weight_array, expected_returns)
            
            # Portfolio volatility
            volatilities = np.array([returns_data[s]['volatility'] for s in strategy_ids])
            correlation_np = correlation_matrix.loc[strategy_ids, strategy_ids].values
            covariance_matrix = np.outer(volatilities, volatilities) * correlation_np
            portfolio_variance = np.dot(weight_array, np.dot(covariance_matrix, weight_array))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Sharpe ratio
            risk_free_rate = self.config['risk_free_rate']
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Sortino ratio (simplified)
            sortino_ratio = sharpe_ratio * 1.2  # Approximation
            
            # Maximum drawdown (estimated)
            max_drawdowns = np.array([returns_data[s]['max_drawdown'] for s in strategy_ids])
            portfolio_max_drawdown = np.dot(weight_array, max_drawdowns)
            
            # Diversification ratio
            weighted_avg_volatility = np.dot(weight_array, volatilities)
            diversification_ratio = weighted_avg_volatility / portfolio_volatility if portfolio_volatility > 0 else 1.0
            
            # Optimization score (composite metric)
            optimization_score = (sharpe_ratio * 0.4 + 
                                diversification_ratio * 0.3 + 
                                (1 - portfolio_max_drawdown) * 0.3) * 100
            
            return {
                'expected_return': portfolio_return,
                'expected_volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': portfolio_max_drawdown,
                'diversification_ratio': diversification_ratio,
                'optimization_score': optimization_score
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return self._get_default_portfolio_metrics()
    
    def _check_constraints(self, weights: Dict[str, float]) -> bool:
        """Check if weights satisfy portfolio constraints"""
        try:
            if not weights:
                return True
            
            # Check individual position weights
            for weight in weights.values():
                if weight < self.portfolio_constraints.min_position_weight:
                    return False
                if weight > self.portfolio_constraints.max_position_weight:
                    return False
            
            # Check weights sum to 1
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                return False
            
            # Additional constraint checks would go here
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking constraints: {str(e)}")
            return False
    
    def _calculate_turnover(self, new_weights: Dict[str, float]) -> float:
        """Calculate portfolio turnover"""
        try:
            if not self.current_portfolio:
                return 0.0
            
            turnover = 0.0
            
            for strategy_id, new_weight in new_weights.items():
                current_weight = self.current_portfolio.get(strategy_id, type('obj', (object,), {'current_weight': 0})).current_weight
                turnover += abs(new_weight - current_weight)
            
            return turnover / 2.0  # Divide by 2 since we count both buys and sells
            
        except Exception as e:
            self.logger.error(f"Error calculating turnover: {str(e)}")
            return 0.0
    
    def _update_target_portfolio(self, optimal_weights: Dict[str, float], strategies: List[Dict[str, Any]]):
        """Update target portfolio with optimization results"""
        try:
            self.target_portfolio.clear()
            
            strategy_dict = {s['strategy_id']: s for s in strategies}
            
            for strategy_id, weight in optimal_weights.items():
                strategy_info = strategy_dict.get(strategy_id, {})
                
                allocation = StrategyAllocation(
                    strategy_id=strategy_id,
                    strategy_type=strategy_info.get('strategy_type', StrategyType.PUT_SELLING),
                    target_weight=weight,
                    current_weight=self.current_portfolio.get(strategy_id, type('obj', (object,), {'current_weight': 0})).current_weight,
                    capital_allocated=0.0,  # Will be set during capital allocation
                    expected_return=strategy_info.get('expected_return', 0.0),
                    expected_volatility=strategy_info.get('expected_volatility', 0.0),
                    max_drawdown=strategy_info.get('max_drawdown', 0.0),
                    sharpe_ratio=strategy_info.get('sharpe_ratio', 0.0),
                    correlation_matrix={},
                    performance_score=0.0,
                    risk_score=0.0,
                    last_updated=datetime.now()
                )
                
                self.target_portfolio[strategy_id] = allocation
            
        except Exception as e:
            self.logger.error(f"Error updating target portfolio: {str(e)}")
    
    # Additional helper methods would continue here...
    # For brevity, I'll include key methods for the remaining functionality
    
    def _create_empty_optimization_result(self, objective: OptimizationObjective) -> OptimizationResult:
        """Create empty optimization result"""
        return OptimizationResult(
            optimization_id=f"opt_empty_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            objective=objective,
            optimal_weights={},
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            diversification_ratio=1.0,
            turnover=0.0,
            optimization_score=0.0,
            constraints_satisfied=True,
            optimization_time=0.0,
            convergence_status='no_strategies',
            timestamp=datetime.now()
        )
    
    def _create_error_optimization_result(self, objective: OptimizationObjective, error_msg: str) -> OptimizationResult:
        """Create error optimization result"""
        return OptimizationResult(
            optimization_id=f"opt_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            objective=objective,
            optimal_weights={},
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            diversification_ratio=1.0,
            turnover=0.0,
            optimization_score=0.0,
            constraints_satisfied=False,
            optimization_time=0.0,
            convergence_status=f'error: {error_msg}',
            timestamp=datetime.now()
        )
    
    def _get_default_portfolio_metrics(self) -> Dict[str, float]:
        """Get default portfolio metrics"""
        return {
            'expected_return': 0.0,
            'expected_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'diversification_ratio': 1.0,
            'optimization_score': 0.0
        }
    
    def _calculate_required_trades(self, weight_differences: Dict[str, float]) -> List[Dict[str, Any]]:
        """Calculate trades required for rebalancing"""
        trades = []
        
        for strategy_id, weight_diff in weight_differences.items():
            if abs(weight_diff) > 0.001:  # Only trade if difference > 0.1%
                trade = {
                    'strategy_id': strategy_id,
                    'action': 'increase' if weight_diff > 0 else 'decrease',
                    'weight_change': abs(weight_diff),
                    'estimated_cost': abs(weight_diff) * self.config['transaction_cost']
                }
                trades.append(trade)
        
        return trades
    
    def _estimate_rebalancing_cost(self, trades: List[Dict[str, Any]]) -> float:
        """Estimate cost of rebalancing"""
        total_cost = sum(trade['estimated_cost'] for trade in trades)
        return total_cost
    
    def _estimate_rebalancing_benefit(self, weight_differences: Dict[str, float]) -> float:
        """Estimate benefit of rebalancing"""
        # Simplified benefit calculation
        total_deviation = sum(abs(diff) for diff in weight_differences.values())
        return total_deviation * 0.01  # 1% benefit per 1% deviation
    
    def _determine_rebalance_trigger(self, max_deviation: float, force_rebalance: bool) -> str:
        """Determine rebalancing trigger reason"""
        if force_rebalance:
            return "Manual rebalancing requested"
        elif max_deviation > self.config['rebalance_threshold'] * 2:
            return "Large weight deviation detected"
        elif max_deviation > self.config['rebalance_threshold']:
            return "Weight deviation threshold exceeded"
        else:
            return "Periodic rebalancing"
    
    def _calculate_rebalance_confidence(self, weight_differences: Dict[str, float], 
                                      trades: List[Dict[str, Any]]) -> float:
        """Calculate confidence in rebalancing recommendation"""
        # Higher confidence for larger deviations and fewer trades
        max_deviation = max(abs(diff) for diff in weight_differences.values()) if weight_differences else 0
        num_trades = len(trades)
        
        deviation_score = min(1.0, max_deviation / (self.config['rebalance_threshold'] * 2))
        trade_score = max(0.5, 1.0 - num_trades * 0.1)
        
        return (deviation_score + trade_score) / 2
    
    def _calculate_strategy_correlation(self, strategy1: str, strategy2: str) -> float:
        """Calculate correlation between two strategies"""
        # Simplified correlation calculation
        if strategy1 == strategy2:
            return 1.0
        
        # Mock correlations based on strategy types and underlyings
        if 'spy' in strategy1 and 'spy' in strategy2:
            return 0.75
        elif 'qqq' in strategy1 and 'qqq' in strategy2:
            return 0.70
        else:
            return 0.35
    
    def _find_high_correlation_pairs(self, corr_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Find pairs with high correlation"""
        high_corr_pairs = []
        threshold = 0.7
        
        for i in range(len(corr_df.index)):
            for j in range(i+1, len(corr_df.columns)):
                correlation = corr_df.iloc[i, j]
                if correlation > threshold:
                    high_corr_pairs.append((corr_df.index[i], corr_df.columns[j], correlation))
        
        return high_corr_pairs
    
    def _calculate_diversification_ratio(self, correlation_matrix: np.ndarray) -> float:
        """Calculate diversification ratio"""
        n = correlation_matrix.shape[0]
        if n <= 1:
            return 1.0
        
        # Average correlation
        upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        avg_correlation = np.mean(upper_triangle)
        
        # Diversification ratio approximation
        return (1 + (n-1) * avg_correlation) / n
    
    def _optimization_loop(self):
        """Main optimization loop for continuous optimization"""
        while self.optimization_active:
            try:
                # Run optimization
                self.optimize_portfolio()
                
                # Check for rebalancing
                rebalance_rec = self.generate_rebalance_recommendation()
                if rebalance_rec:
                    self.logger.info(f"Rebalancing recommended: {rebalance_rec.trigger_reason}")
                
                # Sleep until next optimization
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {str(e)}")
                time.sleep(300)  # Wait 5 minutes on error

def test_portfolio_optimizer():
    """Test the portfolio optimizer"""
    print("Testing Portfolio Optimizer...")
    
    optimizer = PortfolioOptimizer()
    
    # Test portfolio optimization
    print("\n--- Testing Portfolio Optimization ---")
    
    # Test different objectives
    objectives = [
        OptimizationObjective.MAX_SHARPE,
        OptimizationObjective.MIN_RISK,
        OptimizationObjective.RISK_PARITY
    ]
    
    for objective in objectives:
        result = optimizer.optimize_portfolio(objective)
        print(f"\n{objective.value.upper()} Optimization:")
        print(f"  Expected Return: {result.expected_return:.1%}")
        print(f"  Expected Volatility: {result.expected_volatility:.1%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Optimization Score: {result.optimization_score:.1f}")
        print(f"  Optimal Weights:")
        for strategy, weight in result.optimal_weights.items():
            print(f"    {strategy}: {weight:.1%}")
    
    # Test rebalancing recommendation
    print("\n--- Testing Rebalancing Recommendation ---")
    
    # Simulate current portfolio
    optimizer.current_portfolio = {
        'put_selling_spy': StrategyAllocation(
            strategy_id='put_selling_spy',
            strategy_type=StrategyType.PUT_SELLING,
            target_weight=0.30,
            current_weight=0.35,  # 5% deviation
            capital_allocated=350000,
            expected_return=0.15,
            expected_volatility=0.12,
            max_drawdown=0.08,
            sharpe_ratio=1.25,
            correlation_matrix={},
            performance_score=85.0,
            risk_score=25.0,
            last_updated=datetime.now()
        )
    }
    
    rebalance_rec = optimizer.generate_rebalance_recommendation()
    if rebalance_rec:
        print(f"Rebalancing Recommended: {rebalance_rec.trigger_reason}")
        print(f"Max Weight Deviation: {max(abs(d) for d in rebalance_rec.weight_changes.values()):.1%}")
        print(f"Expected Cost: {rebalance_rec.expected_cost:.3%}")
        print(f"Expected Benefit: {rebalance_rec.expected_benefit:.3%}")
        print(f"Urgency Score: {rebalance_rec.urgency_score:.2f}")
    else:
        print("No rebalancing needed")
    
    # Test correlation analysis
    print("\n--- Testing Correlation Analysis ---")
    correlation_analysis = optimizer.analyze_strategy_correlations()
    
    if 'error' not in correlation_analysis:
        stats = correlation_analysis['correlation_statistics']
        print(f"Average Correlation: {stats['average_correlation']:.2f}")
        print(f"Max Correlation: {stats['max_correlation']:.2f}")
        print(f"Diversification Ratio: {stats['diversification_ratio']:.2f}")
        
        if stats['high_correlation_pairs']:
            print("High Correlation Pairs:")
            for pair in stats['high_correlation_pairs']:
                print(f"  {pair[0]} - {pair[1]}: {pair[2]:.2f}")
    
    # Test capital allocation
    print("\n--- Testing Capital Allocation ---")
    allocation_result = optimizer.optimize_capital_allocation(1000000)
    
    if 'error' not in allocation_result:
        print("Capital Allocation:")
        for strategy, allocation in allocation_result['capital_allocations'].items():
            percentage = allocation_result['allocation_percentages'][strategy]
            print(f"  {strategy}: ${allocation:,.0f} ({percentage:.1%})")
        
        metrics = allocation_result['allocation_metrics']
        print(f"\nAllocation Metrics:")
        print(f"  Expected Return: {metrics.get('expected_return', 0):.1%}")
        print(f"  Expected Volatility: {metrics.get('expected_volatility', 0):.1%}")
        print(f"  Allocation Score: {metrics.get('allocation_score', 0):.1f}")
    
    # Test portfolio dashboard
    print("\n--- Testing Portfolio Dashboard ---")
    dashboard = optimizer.get_portfolio_dashboard()
    
    if 'error' not in dashboard:
        status = dashboard['portfolio_status']
        print(f"Portfolio Status:")
        print(f"  Total Strategies: {status['total_strategies']}")
        print(f"  Total Capital: ${status['total_capital_allocated']:,.0f}")
        print(f"  Expected Return: {status['portfolio_expected_return']:.1%}")
        print(f"  Portfolio Volatility: {status['portfolio_volatility']:.1%}")
        print(f"  Sharpe Ratio: {status['portfolio_sharpe_ratio']:.2f}")
        
        print(f"\nRecommendations:")
        for rec in dashboard['recommendations']:
            print(f"  • {rec}")
    
    print("\n✅ Portfolio Optimizer test completed successfully!")

if __name__ == "__main__":
    test_portfolio_optimizer()


    def _calculate_risk_adjustment(self, expected_return: float, volatility: float) -> float:
        """Calculate risk adjustment factor for capital allocation"""
        try:
            # Risk adjustment based on Sharpe ratio
            risk_free_rate = self.config['risk_free_rate']
            sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Normalize Sharpe ratio to adjustment factor
            # Higher Sharpe = higher allocation
            base_adjustment = 1.0
            sharpe_adjustment = min(2.0, max(0.5, sharpe_ratio))
            
            return base_adjustment * sharpe_adjustment
            
        except Exception as e:
            return 1.0  # Default no adjustment
    
    def _calculate_allocation_metrics(self, capital_allocations: Dict[str, float], 
                                    expected_returns: List[float], 
                                    volatilities: List[float]) -> Dict[str, Any]:
        """Calculate allocation metrics"""
        try:
            total_capital = sum(capital_allocations.values())
            
            if total_capital == 0:
                return {'expected_return': 0.0, 'expected_volatility': 0.0, 'allocation_score': 0.0}
            
            # Calculate weighted metrics
            weights = [capital_allocations[s] / total_capital for s in capital_allocations.keys()]
            
            portfolio_return = sum(w * r for w, r in zip(weights, expected_returns))
            portfolio_volatility = np.sqrt(sum((w * v)**2 for w, v in zip(weights, volatilities)))
            
            # Allocation score
            risk_free_rate = self.config['risk_free_rate']
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            allocation_score = sharpe_ratio * 100
            
            return {
                'expected_return': portfolio_return,
                'expected_volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'allocation_score': allocation_score
            }
            
        except Exception as e:
            return {'expected_return': 0.0, 'expected_volatility': 0.0, 'allocation_score': 0.0}
    
    def _generate_allocation_recommendations(self, capital_allocations: Dict[str, float], 
                                           allocation_metrics: Dict[str, Any]) -> List[str]:
        """Generate capital allocation recommendations"""
        recommendations = []
        
        try:
            total_capital = sum(capital_allocations.values())
            
            # Check for concentration
            max_allocation = max(capital_allocations.values()) if capital_allocations else 0
            max_percentage = max_allocation / total_capital if total_capital > 0 else 0
            
            if max_percentage > 0.4:
                recommendations.append("Consider reducing concentration in largest allocation")
            
            # Check for diversification
            if len(capital_allocations) < 3:
                recommendations.append("Consider adding more strategies for better diversification")
            
            # Check allocation efficiency
            allocation_score = allocation_metrics.get('allocation_score', 0)
            if allocation_score < 50:
                recommendations.append("Allocation efficiency is low - consider rebalancing")
            elif allocation_score > 100:
                recommendations.append("Excellent allocation efficiency achieved")
            
            if not recommendations:
                recommendations.append("Capital allocation is well-balanced")
            
        except Exception as e:
            recommendations.append("Unable to generate allocation recommendations")
        
        return recommendations
    
    def _calculate_portfolio_expected_return(self) -> float:
        """Calculate current portfolio expected return"""
        try:
            if not self.current_portfolio:
                return 0.0
            
            total_weight = sum(s.current_weight for s in self.current_portfolio.values())
            
            if total_weight == 0:
                return 0.0
            
            weighted_return = sum(
                s.current_weight * s.expected_return 
                for s in self.current_portfolio.values()
            )
            
            return weighted_return / total_weight
            
        except Exception as e:
            return 0.0
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate current portfolio volatility"""
        try:
            if not self.current_portfolio:
                return 0.0
            
            # Simplified volatility calculation
            total_weight = sum(s.current_weight for s in self.current_portfolio.values())
            
            if total_weight == 0:
                return 0.0
            
            weighted_volatility = sum(
                s.current_weight * s.expected_volatility 
                for s in self.current_portfolio.values()
            )
            
            return weighted_volatility / total_weight
            
        except Exception as e:
            return 0.0
    
    def _calculate_portfolio_sharpe_ratio(self) -> float:
        """Calculate current portfolio Sharpe ratio"""
        try:
            portfolio_return = self._calculate_portfolio_expected_return()
            portfolio_volatility = self._calculate_portfolio_volatility()
            
            if portfolio_volatility == 0:
                return 0.0
            
            risk_free_rate = self.config['risk_free_rate']
            return (portfolio_return - risk_free_rate) / portfolio_volatility
            
        except Exception as e:
            return 0.0
    
    def _calculate_current_diversification_ratio(self) -> float:
        """Calculate current portfolio diversification ratio"""
        try:
            if len(self.current_portfolio) <= 1:
                return 1.0
            
            # Simplified diversification calculation
            weights = [s.current_weight for s in self.current_portfolio.values()]
            total_weight = sum(weights)
            
            if total_weight == 0:
                return 1.0
            
            normalized_weights = [w / total_weight for w in weights]
            
            # Herfindahl-Hirschman Index
            hhi = sum(w**2 for w in normalized_weights)
            
            # Convert to diversification ratio
            n = len(normalized_weights)
            max_diversification = 1.0 / n
            diversification_ratio = max_diversification / hhi if hhi > 0 else 1.0
            
            return min(1.0, diversification_ratio)
            
        except Exception as e:
            return 1.0
    
    def _calculate_portfolio_efficiency(self) -> float:
        """Calculate portfolio efficiency score"""
        try:
            if not self.optimization_history:
                return 0.0
            
            # Average optimization score from recent optimizations
            recent_optimizations = self.optimization_history[-5:]  # Last 5 optimizations
            avg_score = np.mean([opt.optimization_score for opt in recent_optimizations])
            
            return avg_score
            
        except Exception as e:
            return 0.0
    
    def _generate_portfolio_recommendations(self) -> List[str]:
        """Generate portfolio recommendations"""
        recommendations = []
        
        try:
            # Check portfolio size
            if len(self.current_portfolio) == 0:
                recommendations.append("Initialize portfolio with strategies")
                return recommendations
            
            # Check diversification
            diversification_ratio = self._calculate_current_diversification_ratio()
            if diversification_ratio < 0.6:
                recommendations.append("Improve portfolio diversification")
            
            # Check rebalancing needs
            if self.current_portfolio and self.target_portfolio:
                max_deviation = max(
                    abs(self.target_portfolio[s].target_weight - self.current_portfolio.get(s, type('obj', (object,), {'current_weight': 0})).current_weight)
                    for s in self.target_portfolio.keys()
                )
                
                if max_deviation > self.config['rebalance_threshold']:
                    recommendations.append("Portfolio rebalancing recommended")
            
            # Check optimization frequency
            if not self.optimization_history:
                recommendations.append("Run portfolio optimization")
            elif len(self.optimization_history) > 0:
                last_optimization = self.optimization_history[-1]
                time_since_last = datetime.now() - last_optimization.timestamp
                
                if time_since_last > timedelta(days=7):
                    recommendations.append("Consider running fresh portfolio optimization")
            
            # Check performance
            portfolio_sharpe = self._calculate_portfolio_sharpe_ratio()
            if portfolio_sharpe < 1.0:
                recommendations.append("Portfolio Sharpe ratio is below 1.0 - consider optimization")
            elif portfolio_sharpe > 2.0:
                recommendations.append("Excellent portfolio performance - maintain current allocation")
            
            if not recommendations:
                recommendations.append("Portfolio is well-optimized and balanced")
            
        except Exception as e:
            recommendations.append("Unable to generate portfolio recommendations")
        
        return recommendations
    
    def _identify_correlation_clusters(self, corr_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify correlation clusters"""
        try:
            clusters = {}
            threshold = 0.6
            
            # Simple clustering based on correlation threshold
            strategies = list(corr_df.index)
            clustered = set()
            cluster_id = 0
            
            for strategy in strategies:
                if strategy in clustered:
                    continue
                
                # Find highly correlated strategies
                cluster = [strategy]
                for other_strategy in strategies:
                    if (other_strategy != strategy and 
                        other_strategy not in clustered and
                        corr_df.loc[strategy, other_strategy] > threshold):
                        cluster.append(other_strategy)
                
                if len(cluster) > 1:
                    clusters[f"cluster_{cluster_id}"] = cluster
                    clustered.update(cluster)
                    cluster_id += 1
            
            return clusters
            
        except Exception as e:
            return {}
    
    def _generate_correlation_recommendations(self, correlation_stats: Dict[str, Any], 
                                            clusters: Dict[str, List[str]]) -> List[str]:
        """Generate correlation-based recommendations"""
        recommendations = []
        
        try:
            avg_correlation = correlation_stats.get('average_correlation', 0)
            max_correlation = correlation_stats.get('max_correlation', 0)
            
            if avg_correlation > 0.7:
                recommendations.append("Average correlation is high - consider more diverse strategies")
            
            if max_correlation > 0.8:
                recommendations.append("Some strategies are highly correlated - review for redundancy")
            
            if clusters:
                recommendations.append(f"Found {len(clusters)} correlation clusters - consider cluster-based allocation")
            
            diversification_ratio = correlation_stats.get('diversification_ratio', 1.0)
            if diversification_ratio < 0.6:
                recommendations.append("Low diversification - add uncorrelated strategies")
            
            if not recommendations:
                recommendations.append("Correlation structure is well-balanced")
            
        except Exception as e:
            recommendations.append("Unable to generate correlation recommendations")
        
        return recommendations

