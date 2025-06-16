"""
ALL-USE Portfolio Optimization: Advanced Portfolio Optimization Module

This module implements sophisticated portfolio optimization algorithms for the ALL-USE
trading system, including Modern Portfolio Theory, correlation analysis, efficient
frontier calculations, and dynamic rebalancing strategies.
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
import scipy.optimize as optimize
from scipy import linalg
import warnings

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
        logging.FileHandler('all_use_portfolio_optimizer.log')
    ]
)

logger = logging.getLogger('all_use_portfolio_optimizer')

# Suppress scipy optimization warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class OptimizationObjective(Enum):
    """Enumeration of optimization objectives."""
    MAX_SHARPE = "Max_Sharpe"
    MIN_VARIANCE = "Min_Variance"
    MAX_RETURN = "Max_Return"
    RISK_PARITY = "Risk_Parity"
    TARGET_RISK = "Target_Risk"
    TARGET_RETURN = "Target_Return"

class RebalanceFrequency(Enum):
    """Enumeration of rebalancing frequencies."""
    DAILY = "Daily"
    WEEKLY = "Weekly"
    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly"
    THRESHOLD_BASED = "Threshold_Based"

@dataclass
class OptimizationResult:
    """Portfolio optimization result data structure."""
    portfolio_id: str
    timestamp: datetime
    objective: OptimizationObjective
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    optimization_success: bool
    convergence_info: str
    constraints_satisfied: bool

@dataclass
class RebalanceRecommendation:
    """Portfolio rebalancing recommendation data structure."""
    portfolio_id: str
    timestamp: datetime
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    weight_changes: Dict[str, float]
    rebalance_magnitude: float
    expected_improvement: float
    transaction_costs: float
    net_benefit: float
    recommendation: str

class PortfolioOptimizer:
    """
    Advanced portfolio optimization system for ALL-USE trading strategy.
    
    This class provides:
    - Modern Portfolio Theory implementation
    - Multiple optimization objectives (Sharpe, min variance, risk parity)
    - Efficient frontier calculations
    - Dynamic rebalancing recommendations
    - Correlation and covariance analysis
    - Risk-return optimization with constraints
    """
    
    def __init__(self, rebalance_callback: Optional[Callable] = None):
        """Initialize the portfolio optimizer."""
        self.parameters = ALLUSEParameters
        self.rebalance_callback = rebalance_callback
        
        # Optimization configuration
        self.optimization_config = {
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'lookback_periods': {
                'returns': 252,      # 1 year for return estimation
                'correlation': 126,  # 6 months for correlation
                'volatility': 63     # 3 months for volatility
            },
            'constraints': {
                'min_weight': 0.0,      # Minimum position weight
                'max_weight': 0.40,     # Maximum position weight (40%)
                'max_concentration': 0.60,  # Maximum sector concentration
                'min_positions': 3,     # Minimum number of positions
                'max_positions': 10     # Maximum number of positions
            },
            'rebalancing': {
                'threshold': 0.05,      # 5% weight deviation threshold
                'min_improvement': 0.001,  # 0.1% minimum expected improvement
                'transaction_cost': 0.001  # 0.1% transaction cost
            },
            'optimization': {
                'max_iterations': 1000,
                'tolerance': 1e-8,
                'method': 'SLSQP'  # Sequential Least Squares Programming
            }
        }
        
        # Portfolio data
        self.portfolios = {}
        self.price_history = {}
        self.optimization_history = {}
        self.correlation_matrices = {}
        
        logger.info("Portfolio optimizer initialized")
    
    def add_portfolio(self, portfolio_id: str, symbols: List[str], 
                     current_weights: Dict[str, float]) -> None:
        """
        Add a portfolio to optimization tracking.
        
        Args:
            portfolio_id: Unique portfolio identifier
            symbols: List of symbols in the portfolio
            current_weights: Current portfolio weights
        """
        logger.info(f"Adding portfolio {portfolio_id} to optimization tracking")
        
        self.portfolios[portfolio_id] = {
            'symbols': symbols,
            'current_weights': current_weights,
            'target_weights': current_weights.copy(),
            'last_optimization': None,
            'last_rebalance': None,
            'optimization_objective': OptimizationObjective.MAX_SHARPE
        }
        
        self.optimization_history[portfolio_id] = []
        
        # Initialize price history for symbols
        for symbol in symbols:
            if symbol not in self.price_history:
                self.price_history[symbol] = self._generate_sample_price_history(symbol)
    
    def optimize_portfolio(self, portfolio_id: str, 
                          objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
                          target_return: Optional[float] = None,
                          target_risk: Optional[float] = None) -> OptimizationResult:
        """
        Optimize portfolio weights based on specified objective.
        
        Args:
            portfolio_id: Portfolio identifier
            objective: Optimization objective
            target_return: Target return for TARGET_RETURN objective
            target_risk: Target risk for TARGET_RISK objective
            
        Returns:
            OptimizationResult object
        """
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        logger.info(f"Optimizing portfolio {portfolio_id} with objective {objective.value}")
        
        portfolio = self.portfolios[portfolio_id]
        symbols = portfolio['symbols']
        
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = self._calculate_expected_returns(symbols)
            covariance_matrix = self._calculate_covariance_matrix(symbols)
            
            # Perform optimization based on objective
            if objective == OptimizationObjective.MAX_SHARPE:
                optimal_weights = self._optimize_max_sharpe(expected_returns, covariance_matrix)
            elif objective == OptimizationObjective.MIN_VARIANCE:
                optimal_weights = self._optimize_min_variance(covariance_matrix)
            elif objective == OptimizationObjective.MAX_RETURN:
                optimal_weights = self._optimize_max_return(expected_returns)
            elif objective == OptimizationObjective.RISK_PARITY:
                optimal_weights = self._optimize_risk_parity(covariance_matrix)
            elif objective == OptimizationObjective.TARGET_RETURN:
                if target_return is None:
                    raise ValueError("Target return must be specified for TARGET_RETURN objective")
                optimal_weights = self._optimize_target_return(expected_returns, covariance_matrix, target_return)
            elif objective == OptimizationObjective.TARGET_RISK:
                if target_risk is None:
                    raise ValueError("Target risk must be specified for TARGET_RISK objective")
                optimal_weights = self._optimize_target_risk(expected_returns, covariance_matrix, target_risk)
            else:
                raise ValueError(f"Unknown optimization objective: {objective}")
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_variance = np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate Sharpe ratio
            risk_free_rate = self.optimization_config['risk_free_rate']
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Create weight dictionary
            weight_dict = {symbol: weight for symbol, weight in zip(symbols, optimal_weights)}
            
            # Check constraints
            constraints_satisfied = self._check_constraints(weight_dict)
            
            # Create optimization result
            result = OptimizationResult(
                portfolio_id=portfolio_id,
                timestamp=datetime.now(),
                objective=objective,
                optimal_weights=weight_dict,
                expected_return=portfolio_return,
                expected_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                optimization_success=True,
                convergence_info="Optimization converged successfully",
                constraints_satisfied=constraints_satisfied
            )
            
            # Store results
            portfolio['target_weights'] = weight_dict
            portfolio['last_optimization'] = result
            portfolio['optimization_objective'] = objective
            self.optimization_history[portfolio_id].append(result)
            
            # Keep only recent history
            if len(self.optimization_history[portfolio_id]) > 100:
                self.optimization_history[portfolio_id] = self.optimization_history[portfolio_id][-100:]
            
            logger.info(f"Portfolio optimization completed: Sharpe={sharpe_ratio:.3f}, Return={portfolio_return:.2%}, Risk={portfolio_volatility:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed for {portfolio_id}: {str(e)}")
            
            # Return failed optimization result
            return OptimizationResult(
                portfolio_id=portfolio_id,
                timestamp=datetime.now(),
                objective=objective,
                optimal_weights=portfolio['current_weights'],
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                optimization_success=False,
                convergence_info=f"Optimization failed: {str(e)}",
                constraints_satisfied=False
            )
    
    def calculate_efficient_frontier(self, portfolio_id: str, 
                                   num_points: int = 50) -> Tuple[List[float], List[float], List[Dict[str, float]]]:
        """
        Calculate the efficient frontier for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            num_points: Number of points on the efficient frontier
            
        Returns:
            Tuple of (returns, risks, weights) for efficient frontier
        """
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        logger.info(f"Calculating efficient frontier for {portfolio_id}")
        
        portfolio = self.portfolios[portfolio_id]
        symbols = portfolio['symbols']
        
        # Calculate expected returns and covariance matrix
        expected_returns = self._calculate_expected_returns(symbols)
        covariance_matrix = self._calculate_covariance_matrix(symbols)
        
        # Calculate minimum and maximum returns
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, num_points)
        
        efficient_returns = []
        efficient_risks = []
        efficient_weights = []
        
        for target_return in target_returns:
            try:
                # Optimize for target return
                weights = self._optimize_target_return(expected_returns, covariance_matrix, target_return)
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                portfolio_risk = np.sqrt(portfolio_variance)
                
                # Store results
                efficient_returns.append(portfolio_return)
                efficient_risks.append(portfolio_risk)
                efficient_weights.append({symbol: weight for symbol, weight in zip(symbols, weights)})
                
            except Exception as e:
                logger.warning(f"Failed to optimize for target return {target_return:.3f}: {str(e)}")
                continue
        
        logger.info(f"Efficient frontier calculated with {len(efficient_returns)} points")
        return efficient_returns, efficient_risks, efficient_weights
    
    def analyze_correlation(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Analyze correlation structure of portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Dict containing correlation analysis
        """
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        portfolio = self.portfolios[portfolio_id]
        symbols = portfolio['symbols']
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(symbols)
        
        # Store correlation matrix
        self.correlation_matrices[portfolio_id] = {
            'matrix': correlation_matrix,
            'symbols': symbols,
            'timestamp': datetime.now()
        }
        
        # Calculate correlation statistics
        correlation_stats = self._analyze_correlation_statistics(correlation_matrix, symbols)
        
        return {
            'portfolio_id': portfolio_id,
            'timestamp': datetime.now(),
            'correlation_matrix': correlation_matrix.tolist(),
            'symbols': symbols,
            'statistics': correlation_stats
        }
    
    def generate_rebalance_recommendation(self, portfolio_id: str) -> RebalanceRecommendation:
        """
        Generate rebalancing recommendation for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            RebalanceRecommendation object
        """
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        portfolio = self.portfolios[portfolio_id]
        current_weights = portfolio['current_weights']
        target_weights = portfolio['target_weights']
        
        # Calculate weight changes
        weight_changes = {}
        total_change = 0.0
        
        for symbol in current_weights:
            current = current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)
            change = target - current
            weight_changes[symbol] = change
            total_change += abs(change)
        
        # Calculate rebalance magnitude
        rebalance_magnitude = total_change / 2.0  # Divide by 2 since buys = sells
        
        # Estimate transaction costs
        transaction_cost_rate = self.optimization_config['rebalancing']['transaction_cost']
        transaction_costs = rebalance_magnitude * transaction_cost_rate
        
        # Estimate expected improvement (simplified)
        expected_improvement = self._estimate_rebalance_improvement(portfolio_id)
        
        # Calculate net benefit
        net_benefit = expected_improvement - transaction_costs
        
        # Generate recommendation
        threshold = self.optimization_config['rebalancing']['threshold']
        min_improvement = self.optimization_config['rebalancing']['min_improvement']
        
        if rebalance_magnitude < threshold:
            recommendation = "No rebalancing needed - weights within threshold"
        elif net_benefit < min_improvement:
            recommendation = "Rebalancing not recommended - insufficient net benefit"
        else:
            recommendation = "Rebalancing recommended - significant improvement expected"
        
        return RebalanceRecommendation(
            portfolio_id=portfolio_id,
            timestamp=datetime.now(),
            current_weights=current_weights,
            target_weights=target_weights,
            weight_changes=weight_changes,
            rebalance_magnitude=rebalance_magnitude,
            expected_improvement=expected_improvement,
            transaction_costs=transaction_costs,
            net_benefit=net_benefit,
            recommendation=recommendation
        )
    
    def _calculate_expected_returns(self, symbols: List[str]) -> np.ndarray:
        """
        Calculate expected returns for symbols.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Array of expected returns
        """
        expected_returns = []
        
        for symbol in symbols:
            if symbol in self.price_history:
                prices = np.array(self.price_history[symbol])
                returns = np.diff(prices) / prices[:-1]
                
                # Annualize returns (assuming daily data)
                expected_return = np.mean(returns) * 252
                expected_returns.append(expected_return)
            else:
                # Default expected return
                expected_returns.append(0.10)  # 10% annual return
        
        return np.array(expected_returns)
    
    def _calculate_covariance_matrix(self, symbols: List[str]) -> np.ndarray:
        """
        Calculate covariance matrix for symbols.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Covariance matrix
        """
        returns_data = []
        
        for symbol in symbols:
            if symbol in self.price_history:
                prices = np.array(self.price_history[symbol])
                returns = np.diff(prices) / prices[:-1]
                returns_data.append(returns)
            else:
                # Generate default returns
                returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
                returns_data.append(returns)
        
        # Ensure all return series have the same length
        min_length = min(len(returns) for returns in returns_data)
        returns_data = [returns[-min_length:] for returns in returns_data]
        
        # Calculate covariance matrix
        returns_matrix = np.array(returns_data).T
        covariance_matrix = np.cov(returns_matrix.T) * 252  # Annualize
        
        return covariance_matrix
    
    def _calculate_correlation_matrix(self, symbols: List[str]) -> np.ndarray:
        """
        Calculate correlation matrix for symbols.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Correlation matrix
        """
        covariance_matrix = self._calculate_covariance_matrix(symbols)
        
        # Convert covariance to correlation
        std_devs = np.sqrt(np.diag(covariance_matrix))
        correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
        
        return correlation_matrix
    
    def _optimize_max_sharpe(self, expected_returns: np.ndarray, 
                           covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Optimize portfolio for maximum Sharpe ratio.
        
        Args:
            expected_returns: Expected returns array
            covariance_matrix: Covariance matrix
            
        Returns:
            Optimal weights array
        """
        n_assets = len(expected_returns)
        risk_free_rate = self.optimization_config['risk_free_rate']
        
        # Objective function (negative Sharpe ratio for minimization)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            if portfolio_volatility == 0:
                return -np.inf
            
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe_ratio  # Negative for minimization
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.optimization_config['constraints']['min_weight'],
                  self.optimization_config['constraints']['max_weight']) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = optimize.minimize(
            objective, initial_weights, method=self.optimization_config['optimization']['method'],
            bounds=bounds, constraints=constraints,
            options={'maxiter': self.optimization_config['optimization']['max_iterations']}
        )
        
        if not result.success:
            logger.warning(f"Max Sharpe optimization did not converge: {result.message}")
        
        return result.x
    
    def _optimize_min_variance(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Optimize portfolio for minimum variance.
        
        Args:
            covariance_matrix: Covariance matrix
            
        Returns:
            Optimal weights array
        """
        n_assets = covariance_matrix.shape[0]
        
        # Objective function (portfolio variance)
        def objective(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0}
        ]
        
        # Bounds
        bounds = [(self.optimization_config['constraints']['min_weight'],
                  self.optimization_config['constraints']['max_weight']) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = optimize.minimize(
            objective, initial_weights, method=self.optimization_config['optimization']['method'],
            bounds=bounds, constraints=constraints,
            options={'maxiter': self.optimization_config['optimization']['max_iterations']}
        )
        
        if not result.success:
            logger.warning(f"Min variance optimization did not converge: {result.message}")
        
        return result.x
    
    def _optimize_max_return(self, expected_returns: np.ndarray) -> np.ndarray:
        """
        Optimize portfolio for maximum return.
        
        Args:
            expected_returns: Expected returns array
            
        Returns:
            Optimal weights array
        """
        n_assets = len(expected_returns)
        
        # Objective function (negative return for minimization)
        def objective(weights):
            return -np.dot(weights, expected_returns)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0}
        ]
        
        # Bounds
        bounds = [(self.optimization_config['constraints']['min_weight'],
                  self.optimization_config['constraints']['max_weight']) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = optimize.minimize(
            objective, initial_weights, method=self.optimization_config['optimization']['method'],
            bounds=bounds, constraints=constraints,
            options={'maxiter': self.optimization_config['optimization']['max_iterations']}
        )
        
        if not result.success:
            logger.warning(f"Max return optimization did not converge: {result.message}")
        
        return result.x
    
    def _optimize_risk_parity(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Optimize portfolio for risk parity.
        
        Args:
            covariance_matrix: Covariance matrix
            
        Returns:
            Optimal weights array
        """
        n_assets = covariance_matrix.shape[0]
        
        # Objective function (risk parity)
        def objective(weights):
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            marginal_contrib = np.dot(covariance_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_variance
            
            # Minimize sum of squared deviations from equal risk contribution
            target_contrib = 1.0 / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0}
        ]
        
        # Bounds
        bounds = [(self.optimization_config['constraints']['min_weight'],
                  self.optimization_config['constraints']['max_weight']) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = optimize.minimize(
            objective, initial_weights, method=self.optimization_config['optimization']['method'],
            bounds=bounds, constraints=constraints,
            options={'maxiter': self.optimization_config['optimization']['max_iterations']}
        )
        
        if not result.success:
            logger.warning(f"Risk parity optimization did not converge: {result.message}")
        
        return result.x
    
    def _optimize_target_return(self, expected_returns: np.ndarray, 
                              covariance_matrix: np.ndarray, target_return: float) -> np.ndarray:
        """
        Optimize portfolio for target return with minimum risk.
        
        Args:
            expected_returns: Expected returns array
            covariance_matrix: Covariance matrix
            target_return: Target portfolio return
            
        Returns:
            Optimal weights array
        """
        n_assets = len(expected_returns)
        
        # Objective function (portfolio variance)
        def objective(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0},
            {'type': 'eq', 'fun': lambda weights: np.dot(weights, expected_returns) - target_return}
        ]
        
        # Bounds
        bounds = [(self.optimization_config['constraints']['min_weight'],
                  self.optimization_config['constraints']['max_weight']) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = optimize.minimize(
            objective, initial_weights, method=self.optimization_config['optimization']['method'],
            bounds=bounds, constraints=constraints,
            options={'maxiter': self.optimization_config['optimization']['max_iterations']}
        )
        
        if not result.success:
            logger.warning(f"Target return optimization did not converge: {result.message}")
        
        return result.x
    
    def _optimize_target_risk(self, expected_returns: np.ndarray, 
                            covariance_matrix: np.ndarray, target_risk: float) -> np.ndarray:
        """
        Optimize portfolio for target risk with maximum return.
        
        Args:
            expected_returns: Expected returns array
            covariance_matrix: Covariance matrix
            target_risk: Target portfolio risk (volatility)
            
        Returns:
            Optimal weights array
        """
        n_assets = len(expected_returns)
        
        # Objective function (negative return for minimization)
        def objective(weights):
            return -np.dot(weights, expected_returns)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0},
            {'type': 'eq', 'fun': lambda weights: np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights))) - target_risk}
        ]
        
        # Bounds
        bounds = [(self.optimization_config['constraints']['min_weight'],
                  self.optimization_config['constraints']['max_weight']) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = optimize.minimize(
            objective, initial_weights, method=self.optimization_config['optimization']['method'],
            bounds=bounds, constraints=constraints,
            options={'maxiter': self.optimization_config['optimization']['max_iterations']}
        )
        
        if not result.success:
            logger.warning(f"Target risk optimization did not converge: {result.message}")
        
        return result.x
    
    def _check_constraints(self, weights: Dict[str, float]) -> bool:
        """
        Check if portfolio weights satisfy constraints.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            True if constraints are satisfied
        """
        constraints = self.optimization_config['constraints']
        
        # Check individual weight constraints
        for weight in weights.values():
            if weight < constraints['min_weight'] or weight > constraints['max_weight']:
                return False
        
        # Check if weights sum to 1 (within tolerance)
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            return False
        
        # Check number of positions
        non_zero_positions = sum(1 for weight in weights.values() if weight > 1e-6)
        if non_zero_positions < constraints['min_positions'] or non_zero_positions > constraints['max_positions']:
            return False
        
        return True
    
    def _analyze_correlation_statistics(self, correlation_matrix: np.ndarray, 
                                      symbols: List[str]) -> Dict[str, Any]:
        """
        Analyze correlation matrix statistics.
        
        Args:
            correlation_matrix: Correlation matrix
            symbols: List of symbols
            
        Returns:
            Dict containing correlation statistics
        """
        # Extract upper triangle (excluding diagonal)
        upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        
        stats = {
            'average_correlation': np.mean(upper_triangle),
            'max_correlation': np.max(upper_triangle),
            'min_correlation': np.min(upper_triangle),
            'correlation_std': np.std(upper_triangle),
            'high_correlation_pairs': [],
            'low_correlation_pairs': []
        }
        
        # Find high and low correlation pairs
        n_assets = len(symbols)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                correlation = correlation_matrix[i, j]
                
                if correlation > 0.7:  # High correlation threshold
                    stats['high_correlation_pairs'].append({
                        'symbol1': symbols[i],
                        'symbol2': symbols[j],
                        'correlation': correlation
                    })
                elif correlation < 0.1:  # Low correlation threshold
                    stats['low_correlation_pairs'].append({
                        'symbol1': symbols[i],
                        'symbol2': symbols[j],
                        'correlation': correlation
                    })
        
        return stats
    
    def _estimate_rebalance_improvement(self, portfolio_id: str) -> float:
        """
        Estimate expected improvement from rebalancing.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Expected improvement (as decimal)
        """
        # Simplified improvement estimation
        # In practice, this would use more sophisticated models
        
        portfolio = self.portfolios[portfolio_id]
        last_optimization = portfolio.get('last_optimization')
        
        if last_optimization and last_optimization.optimization_success:
            # Estimate improvement based on Sharpe ratio difference
            current_sharpe = 0.8  # Simplified current Sharpe ratio
            target_sharpe = last_optimization.sharpe_ratio
            
            improvement = max(0, (target_sharpe - current_sharpe) * 0.01)  # Convert to return improvement
            return improvement
        
        return 0.005  # Default 0.5% improvement estimate
    
    def _generate_sample_price_history(self, symbol: str, days: int = 252) -> List[float]:
        """
        Generate sample price history for testing.
        
        Args:
            symbol: Symbol to generate prices for
            days: Number of days of price history
            
        Returns:
            List of prices
        """
        # Symbol-specific parameters
        symbol_params = {
            'AAPL': {'initial_price': 180, 'drift': 0.10, 'volatility': 0.25},
            'MSFT': {'initial_price': 350, 'drift': 0.12, 'volatility': 0.22},
            'GOOGL': {'initial_price': 2800, 'drift': 0.08, 'volatility': 0.28},
            'TSLA': {'initial_price': 250, 'drift': 0.15, 'volatility': 0.45},
            'NVDA': {'initial_price': 800, 'drift': 0.20, 'volatility': 0.40},
            'AMZN': {'initial_price': 3200, 'drift': 0.09, 'volatility': 0.30}
        }
        
        params = symbol_params.get(symbol, {'initial_price': 100, 'drift': 0.10, 'volatility': 0.25})
        
        # Generate geometric Brownian motion
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        
        dt = 1/252  # Daily time step
        prices = [params['initial_price']]
        
        for _ in range(days - 1):
            random_shock = np.random.normal(0, 1)
            price_change = prices[-1] * (params['drift'] * dt + params['volatility'] * np.sqrt(dt) * random_shock)
            new_price = prices[-1] + price_change
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        return prices
    
    def get_optimization_summary(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Get comprehensive optimization summary for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Dict containing optimization summary
        """
        if portfolio_id not in self.portfolios:
            return {'error': f'Portfolio {portfolio_id} not found'}
        
        portfolio = self.portfolios[portfolio_id]
        last_optimization = portfolio.get('last_optimization')
        
        # Get rebalance recommendation
        rebalance_rec = self.generate_rebalance_recommendation(portfolio_id)
        
        # Get correlation analysis
        correlation_analysis = self.analyze_correlation(portfolio_id)
        
        return {
            'portfolio_id': portfolio_id,
            'timestamp': datetime.now(),
            'current_weights': portfolio['current_weights'],
            'target_weights': portfolio['target_weights'],
            'optimization_objective': portfolio['optimization_objective'].value,
            'last_optimization': {
                'timestamp': last_optimization.timestamp if last_optimization else None,
                'expected_return': last_optimization.expected_return if last_optimization else None,
                'expected_volatility': last_optimization.expected_volatility if last_optimization else None,
                'sharpe_ratio': last_optimization.sharpe_ratio if last_optimization else None,
                'success': last_optimization.optimization_success if last_optimization else False
            } if last_optimization else None,
            'rebalance_recommendation': {
                'recommendation': rebalance_rec.recommendation,
                'rebalance_magnitude': rebalance_rec.rebalance_magnitude,
                'expected_improvement': rebalance_rec.expected_improvement,
                'net_benefit': rebalance_rec.net_benefit
            },
            'correlation_analysis': {
                'average_correlation': correlation_analysis['statistics']['average_correlation'],
                'max_correlation': correlation_analysis['statistics']['max_correlation'],
                'high_correlation_pairs': len(correlation_analysis['statistics']['high_correlation_pairs'])
            }
        }


# Example usage and testing
if __name__ == "__main__":
    def rebalance_handler(recommendation: RebalanceRecommendation):
        """Example rebalance handler."""
        print(f"REBALANCE: {recommendation.recommendation}")
    
    # Create portfolio optimizer
    optimizer = PortfolioOptimizer(rebalance_callback=rebalance_handler)
    
    # Sample portfolio
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    current_weights = {'AAPL': 0.25, 'MSFT': 0.20, 'GOOGL': 0.20, 'TSLA': 0.20, 'NVDA': 0.15}
    
    # Add portfolio
    optimizer.add_portfolio('test_portfolio', symbols, current_weights)
    
    print("=== Portfolio Optimization Test ===")
    
    # Test different optimization objectives
    objectives = [
        OptimizationObjective.MAX_SHARPE,
        OptimizationObjective.MIN_VARIANCE,
        OptimizationObjective.RISK_PARITY
    ]
    
    for objective in objectives:
        print(f"\n--- {objective.value} Optimization ---")
        
        result = optimizer.optimize_portfolio('test_portfolio', objective)
        
        print(f"Success: {result.optimization_success}")
        print(f"Expected Return: {result.expected_return:.2%}")
        print(f"Expected Volatility: {result.expected_volatility:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print("Optimal Weights:")
        for symbol, weight in result.optimal_weights.items():
            print(f"  {symbol}: {weight:.1%}")
    
    # Test efficient frontier
    print(f"\n--- Efficient Frontier ---")
    returns, risks, weights = optimizer.calculate_efficient_frontier('test_portfolio', 10)
    print(f"Efficient frontier calculated with {len(returns)} points")
    print(f"Return range: {min(returns):.2%} to {max(returns):.2%}")
    print(f"Risk range: {min(risks):.2%} to {max(risks):.2%}")
    
    # Test correlation analysis
    print(f"\n--- Correlation Analysis ---")
    correlation_analysis = optimizer.analyze_correlation('test_portfolio')
    stats = correlation_analysis['statistics']
    print(f"Average Correlation: {stats['average_correlation']:.3f}")
    print(f"Max Correlation: {stats['max_correlation']:.3f}")
    print(f"High Correlation Pairs: {len(stats['high_correlation_pairs'])}")
    
    # Test rebalancing recommendation
    print(f"\n--- Rebalancing Recommendation ---")
    rebalance_rec = optimizer.generate_rebalance_recommendation('test_portfolio')
    print(f"Recommendation: {rebalance_rec.recommendation}")
    print(f"Rebalance Magnitude: {rebalance_rec.rebalance_magnitude:.2%}")
    print(f"Expected Improvement: {rebalance_rec.expected_improvement:.3%}")
    print(f"Net Benefit: {rebalance_rec.net_benefit:.3%}")
    
    # Get optimization summary
    summary = optimizer.get_optimization_summary('test_portfolio')
    print(f"\nOptimization Summary:")
    print(f"Last Optimization Sharpe: {summary['last_optimization']['sharpe_ratio']:.3f}")
    print(f"Average Correlation: {summary['correlation_analysis']['average_correlation']:.3f}")
    print(f"Rebalance Recommendation: {summary['rebalance_recommendation']['recommendation']}")

