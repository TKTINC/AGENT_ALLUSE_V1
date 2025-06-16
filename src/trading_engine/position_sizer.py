"""
ALL-USE Trading Engine: Position Sizer Module

This module implements sophisticated position sizing algorithms for the ALL-USE trading system.
It provides risk-adjusted position sizing based on account balance, volatility, and market conditions.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import sys
import os

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
        logging.FileHandler('all_use_position_sizer.log')
    ]
)

logger = logging.getLogger('all_use_position_sizer')

class AccountType(Enum):
    """Enumeration of account types."""
    GEN_ACC = "GEN_ACC"      # Generation Account
    REV_ACC = "REV_ACC"      # Revenue Account
    COM_ACC = "COM_ACC"      # Compounding Account

class RiskLevel(Enum):
    """Enumeration of risk levels."""
    CONSERVATIVE = "Conservative"
    MODERATE = "Moderate"
    AGGRESSIVE = "Aggressive"

class PositionSizer:
    """
    Advanced position sizing system for the ALL-USE trading strategy.
    
    This class provides sophisticated position sizing algorithms that consider:
    - Account type and balance
    - Market volatility and conditions
    - Risk tolerance and drawdown limits
    - Portfolio correlation and concentration
    """
    
    def __init__(self):
        """Initialize the position sizer."""
        self.parameters = ALLUSEParameters
        
        # Position sizing parameters by account type
        self.account_risk_limits = {
            AccountType.GEN_ACC: {
                'max_position_pct': 0.15,      # 15% max per position
                'max_total_exposure': 0.80,    # 80% max total exposure
                'volatility_adjustment': 1.2,  # Higher vol adjustment
                'base_kelly_fraction': 0.25    # Base Kelly fraction
            },
            AccountType.REV_ACC: {
                'max_position_pct': 0.12,      # 12% max per position
                'max_total_exposure': 0.70,    # 70% max total exposure
                'volatility_adjustment': 1.0,  # Standard vol adjustment
                'base_kelly_fraction': 0.20    # Base Kelly fraction
            },
            AccountType.COM_ACC: {
                'max_position_pct': 0.10,      # 10% max per position
                'max_total_exposure': 0.60,    # 60% max total exposure
                'volatility_adjustment': 0.8,  # Lower vol adjustment
                'base_kelly_fraction': 0.15    # Base Kelly fraction
            }
        }
        
        # Risk adjustment factors
        self.risk_adjustments = {
            'market_condition': {
                'Green': 1.2,      # Increase size in favorable conditions
                'Red': 0.7,        # Decrease size in unfavorable conditions
                'Chop': 0.9        # Slightly decrease in uncertain conditions
            },
            'volatility_regime': {
                'Low': 1.1,        # Slightly increase in low vol
                'Medium': 1.0,     # Standard sizing
                'High': 0.8        # Decrease in high vol
            },
            'trend_strength': {
                'Strong': 1.1,     # Increase with strong trends
                'Moderate': 1.0,   # Standard sizing
                'Weak': 0.9        # Decrease with weak trends
            }
        }
        
        # Correlation limits
        self.correlation_limits = {
            'max_sector_exposure': 0.40,    # 40% max in any sector
            'max_correlated_positions': 0.30  # 30% max in highly correlated positions
        }
        
        logger.info("Position sizer initialized")
    
    def calculate_position_size(self, account_type: AccountType, account_balance: float,
                              symbol: str, market_analysis: Dict[str, Any],
                              portfolio_state: Dict[str, Any],
                              risk_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate optimal position size for a trade.
        
        Args:
            account_type: Type of account (GEN_ACC, REV_ACC, COM_ACC)
            account_balance: Current account balance
            symbol: Stock symbol for the trade
            market_analysis: Market analysis from MarketAnalyzer
            portfolio_state: Current portfolio state
            risk_preferences: User risk preferences
            
        Returns:
            Dict containing position sizing recommendation
        """
        logger.info(f"Calculating position size for {symbol} in {account_type.value}")
        
        try:
            # Get base parameters for account type
            account_params = self.account_risk_limits[account_type]
            
            # Calculate base position size
            base_size = self._calculate_base_size(
                account_balance, account_params, market_analysis
            )
            
            # Apply risk adjustments
            adjusted_size = self._apply_risk_adjustments(
                base_size, market_analysis, account_params
            )
            
            # Apply portfolio constraints
            final_size = self._apply_portfolio_constraints(
                adjusted_size, symbol, portfolio_state, account_balance, account_params
            )
            
            # Apply user risk preferences
            if risk_preferences:
                final_size = self._apply_user_preferences(
                    final_size, risk_preferences, account_type
                )
            
            # Calculate contract quantity and dollar amount
            option_price = market_analysis.get('option_price', 2.50)  # Default option price
            contracts = max(1, int(final_size / (option_price * 100)))  # 100 shares per contract
            dollar_amount = contracts * option_price * 100
            
            # Generate sizing recommendation
            recommendation = {
                'symbol': symbol,
                'account_type': account_type.value,
                'account_balance': account_balance,
                'recommended_contracts': contracts,
                'dollar_amount': dollar_amount,
                'position_percentage': (dollar_amount / account_balance) * 100,
                'base_size': base_size,
                'adjusted_size': adjusted_size,
                'final_size': final_size,
                'risk_metrics': self._calculate_risk_metrics(
                    dollar_amount, account_balance, market_analysis
                ),
                'sizing_rationale': self._generate_sizing_rationale(
                    base_size, adjusted_size, final_size, market_analysis
                ),
                'warnings': self._check_sizing_warnings(
                    dollar_amount, account_balance, portfolio_state, account_params
                ),
                'timestamp': datetime.now()
            }
            
            logger.info(f"Position sizing completed for {symbol}: {contracts} contracts (${dollar_amount:,.2f})")
            return recommendation
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {str(e)}")
            return self._get_default_sizing(account_type, account_balance, symbol)
    
    def _calculate_base_size(self, account_balance: float, account_params: Dict[str, Any],
                           market_analysis: Dict[str, Any]) -> float:
        """
        Calculate base position size using Kelly Criterion and volatility adjustment.
        
        Args:
            account_balance: Current account balance
            account_params: Account-specific parameters
            market_analysis: Market analysis results
            
        Returns:
            Base position size in dollars
        """
        # Get market data
        implied_volatility = market_analysis.get('volatility_analysis', {}).get('implied_volatility', 0.25)
        confidence_score = market_analysis.get('confidence_score', 0.5)
        
        # Calculate Kelly fraction
        # Simplified Kelly: f = (bp - q) / b
        # Where b = odds, p = probability of win, q = probability of loss
        win_probability = 0.55 + (confidence_score - 0.5) * 0.2  # Adjust based on confidence
        loss_probability = 1 - win_probability
        expected_return = 0.015  # 1.5% expected weekly return (simplified)
        
        kelly_fraction = (expected_return * win_probability - loss_probability) / expected_return
        kelly_fraction = max(0.05, min(0.30, kelly_fraction))  # Clamp between 5% and 30%
        
        # Apply base Kelly fraction from account parameters
        adjusted_kelly = kelly_fraction * account_params['base_kelly_fraction']
        
        # Volatility adjustment
        vol_adjustment = 1.0 / (1.0 + implied_volatility * account_params['volatility_adjustment'])
        
        # Calculate base size
        base_size = account_balance * adjusted_kelly * vol_adjustment
        
        # Apply maximum position percentage limit
        max_position_size = account_balance * account_params['max_position_pct']
        base_size = min(base_size, max_position_size)
        
        return base_size
    
    def _apply_risk_adjustments(self, base_size: float, market_analysis: Dict[str, Any],
                              account_params: Dict[str, Any]) -> float:
        """
        Apply risk adjustments based on market conditions.
        
        Args:
            base_size: Base position size
            market_analysis: Market analysis results
            account_params: Account-specific parameters
            
        Returns:
            Risk-adjusted position size
        """
        adjusted_size = base_size
        
        # Market condition adjustment
        market_condition = market_analysis.get('market_condition')
        if hasattr(market_condition, 'value'):
            condition_str = market_condition.value
        else:
            condition_str = str(market_condition)
        
        if condition_str in self.risk_adjustments['market_condition']:
            adjusted_size *= self.risk_adjustments['market_condition'][condition_str]
        
        # Volatility regime adjustment
        volatility_regime = market_analysis.get('volatility_analysis', {}).get('regime')
        if hasattr(volatility_regime, 'value'):
            regime_str = volatility_regime.value
        else:
            regime_str = str(volatility_regime)
        
        if regime_str in self.risk_adjustments['volatility_regime']:
            adjusted_size *= self.risk_adjustments['volatility_regime'][regime_str]
        
        # Trend strength adjustment
        trend_strength = market_analysis.get('trend_analysis', {}).get('strength')
        if hasattr(trend_strength, 'value'):
            strength_str = trend_strength.value
        else:
            strength_str = str(trend_strength)
        
        if strength_str in self.risk_adjustments['trend_strength']:
            adjusted_size *= self.risk_adjustments['trend_strength'][strength_str]
        
        # Confidence score adjustment
        confidence_score = market_analysis.get('confidence_score', 0.5)
        confidence_adjustment = 0.7 + (confidence_score * 0.6)  # Range: 0.7 to 1.3
        adjusted_size *= confidence_adjustment
        
        return adjusted_size
    
    def _apply_portfolio_constraints(self, adjusted_size: float, symbol: str,
                                   portfolio_state: Dict[str, Any], account_balance: float,
                                   account_params: Dict[str, Any]) -> float:
        """
        Apply portfolio-level constraints to position sizing.
        
        Args:
            adjusted_size: Risk-adjusted position size
            symbol: Stock symbol
            portfolio_state: Current portfolio state
            account_balance: Current account balance
            account_params: Account-specific parameters
            
        Returns:
            Portfolio-constrained position size
        """
        final_size = adjusted_size
        
        # Get current portfolio exposure
        current_positions = portfolio_state.get('positions', {})
        total_exposure = sum(pos.get('market_value', 0) for pos in current_positions.values())
        
        # Check total exposure limit
        max_total_exposure = account_balance * account_params['max_total_exposure']
        available_exposure = max_total_exposure - total_exposure
        
        if final_size > available_exposure:
            final_size = max(0, available_exposure)
        
        # Check individual position limit (if position already exists)
        if symbol in current_positions:
            current_position_value = current_positions[symbol].get('market_value', 0)
            max_position_value = account_balance * account_params['max_position_pct']
            available_position_size = max_position_value - current_position_value
            
            if final_size > available_position_size:
                final_size = max(0, available_position_size)
        
        # Check sector concentration (simplified)
        symbol_sector = self._get_symbol_sector(symbol)
        sector_exposure = self._calculate_sector_exposure(portfolio_state, symbol_sector)
        max_sector_exposure = account_balance * self.correlation_limits['max_sector_exposure']
        
        if sector_exposure + final_size > max_sector_exposure:
            available_sector_size = max(0, max_sector_exposure - sector_exposure)
            final_size = min(final_size, available_sector_size)
        
        return final_size
    
    def _apply_user_preferences(self, final_size: float, risk_preferences: Dict[str, Any],
                              account_type: AccountType) -> float:
        """
        Apply user risk preferences to position sizing.
        
        Args:
            final_size: Portfolio-constrained position size
            risk_preferences: User risk preferences
            account_type: Account type
            
        Returns:
            User-preference-adjusted position size
        """
        # Get user risk level
        risk_level = risk_preferences.get('risk_level', 'Moderate')
        
        # Risk level adjustments
        risk_multipliers = {
            'Conservative': 0.7,
            'Moderate': 1.0,
            'Aggressive': 1.3
        }
        
        if risk_level in risk_multipliers:
            final_size *= risk_multipliers[risk_level]
        
        # Account-specific user preferences
        account_preference = risk_preferences.get(f'{account_type.value}_preference', 1.0)
        final_size *= account_preference
        
        # Maximum position size override
        max_position_override = risk_preferences.get('max_position_size')
        if max_position_override:
            final_size = min(final_size, max_position_override)
        
        return final_size
    
    def _calculate_risk_metrics(self, position_size: float, account_balance: float,
                              market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk metrics for the position.
        
        Args:
            position_size: Position size in dollars
            account_balance: Account balance
            market_analysis: Market analysis results
            
        Returns:
            Dict containing risk metrics
        """
        position_percentage = (position_size / account_balance) * 100
        implied_volatility = market_analysis.get('volatility_analysis', {}).get('implied_volatility', 0.25)
        
        # Estimate potential loss (simplified)
        # Assume maximum loss is 50% of premium (for option selling)
        max_loss_estimate = position_size * 0.50
        max_loss_percentage = (max_loss_estimate / account_balance) * 100
        
        # Value at Risk (simplified 1-day VaR)
        daily_volatility = implied_volatility / np.sqrt(252)
        var_95 = position_size * daily_volatility * 1.645  # 95% confidence
        var_99 = position_size * daily_volatility * 2.326  # 99% confidence
        
        return {
            'position_percentage': position_percentage,
            'max_loss_estimate': max_loss_estimate,
            'max_loss_percentage': max_loss_percentage,
            'var_95_1day': var_95,
            'var_99_1day': var_99,
            'implied_volatility': implied_volatility,
            'risk_reward_ratio': self._calculate_risk_reward_ratio(market_analysis)
        }
    
    def _calculate_risk_reward_ratio(self, market_analysis: Dict[str, Any]) -> float:
        """
        Calculate risk-reward ratio for the trade.
        
        Args:
            market_analysis: Market analysis results
            
        Returns:
            Risk-reward ratio
        """
        # Simplified risk-reward calculation
        # Based on expected return vs. potential loss
        expected_return = 0.015  # 1.5% weekly expected return
        confidence_score = market_analysis.get('confidence_score', 0.5)
        
        # Adjust expected return by confidence
        adjusted_return = expected_return * confidence_score
        
        # Estimate risk (potential loss)
        estimated_risk = 0.05  # 5% estimated risk
        
        # Calculate ratio
        if estimated_risk > 0:
            return adjusted_return / estimated_risk
        else:
            return 0.0
    
    def _generate_sizing_rationale(self, base_size: float, adjusted_size: float,
                                 final_size: float, market_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate rationale for the position sizing decision.
        
        Args:
            base_size: Base position size
            adjusted_size: Risk-adjusted size
            final_size: Final position size
            market_analysis: Market analysis results
            
        Returns:
            List of rationale strings
        """
        rationale = []
        
        # Base sizing rationale
        rationale.append(f"Base size calculated using Kelly Criterion: ${base_size:,.2f}")
        
        # Risk adjustments
        if adjusted_size != base_size:
            adjustment_factor = adjusted_size / base_size
            if adjustment_factor > 1.05:
                rationale.append(f"Size increased by {((adjustment_factor - 1) * 100):.1f}% due to favorable market conditions")
            elif adjustment_factor < 0.95:
                rationale.append(f"Size decreased by {((1 - adjustment_factor) * 100):.1f}% due to market risks")
        
        # Portfolio constraints
        if final_size != adjusted_size:
            constraint_factor = final_size / adjusted_size
            if constraint_factor < 0.95:
                rationale.append(f"Size limited by portfolio constraints (reduced by {((1 - constraint_factor) * 100):.1f}%)")
        
        # Market condition rationale
        market_condition = market_analysis.get('market_condition')
        if hasattr(market_condition, 'value'):
            condition_str = market_condition.value
        else:
            condition_str = str(market_condition)
        
        rationale.append(f"Market condition: {condition_str}")
        
        # Confidence rationale
        confidence_score = market_analysis.get('confidence_score', 0.5)
        rationale.append(f"Analysis confidence: {confidence_score:.2f}")
        
        return rationale
    
    def _check_sizing_warnings(self, position_size: float, account_balance: float,
                             portfolio_state: Dict[str, Any], account_params: Dict[str, Any]) -> List[str]:
        """
        Check for position sizing warnings.
        
        Args:
            position_size: Position size in dollars
            account_balance: Account balance
            portfolio_state: Current portfolio state
            account_params: Account-specific parameters
            
        Returns:
            List of warning strings
        """
        warnings = []
        
        position_percentage = (position_size / account_balance) * 100
        
        # Large position warning
        if position_percentage > account_params['max_position_pct'] * 100 * 0.8:
            warnings.append(f"Large position size: {position_percentage:.1f}% of account")
        
        # High exposure warning
        current_positions = portfolio_state.get('positions', {})
        total_exposure = sum(pos.get('market_value', 0) for pos in current_positions.values())
        total_exposure_pct = ((total_exposure + position_size) / account_balance) * 100
        
        if total_exposure_pct > account_params['max_total_exposure'] * 100 * 0.9:
            warnings.append(f"High total exposure: {total_exposure_pct:.1f}% of account")
        
        # Small position warning
        if position_percentage < 1.0:
            warnings.append("Very small position size - consider minimum position requirements")
        
        # Cash availability warning
        available_cash = portfolio_state.get('available_cash', account_balance)
        if position_size > available_cash * 0.9:
            warnings.append("Position size approaches available cash limit")
        
        return warnings
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """
        Get sector for a symbol (simplified mapping).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Sector string
        """
        # Simplified sector mapping
        sector_mapping = {
            'TSLA': 'Automotive/Energy',
            'NVDA': 'Technology',
            'AAPL': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'META': 'Technology',
            'NFLX': 'Communication Services'
        }
        
        return sector_mapping.get(symbol, 'Unknown')
    
    def _calculate_sector_exposure(self, portfolio_state: Dict[str, Any], sector: str) -> float:
        """
        Calculate current exposure to a sector.
        
        Args:
            portfolio_state: Current portfolio state
            sector: Sector to calculate exposure for
            
        Returns:
            Current sector exposure in dollars
        """
        sector_exposure = 0.0
        current_positions = portfolio_state.get('positions', {})
        
        for symbol, position in current_positions.items():
            if self._get_symbol_sector(symbol) == sector:
                sector_exposure += position.get('market_value', 0)
        
        return sector_exposure
    
    def _get_default_sizing(self, account_type: AccountType, account_balance: float, symbol: str) -> Dict[str, Any]:
        """
        Get default position sizing when calculation fails.
        
        Args:
            account_type: Account type
            account_balance: Account balance
            symbol: Stock symbol
            
        Returns:
            Default sizing recommendation
        """
        # Conservative default sizing
        default_percentage = 0.05  # 5% of account
        default_size = account_balance * default_percentage
        default_contracts = max(1, int(default_size / 250))  # Assume $2.50 option price
        
        return {
            'symbol': symbol,
            'account_type': account_type.value,
            'account_balance': account_balance,
            'recommended_contracts': default_contracts,
            'dollar_amount': default_contracts * 250,
            'position_percentage': default_percentage * 100,
            'base_size': default_size,
            'adjusted_size': default_size,
            'final_size': default_size,
            'risk_metrics': {
                'position_percentage': default_percentage * 100,
                'max_loss_estimate': default_size * 0.5,
                'max_loss_percentage': default_percentage * 50
            },
            'sizing_rationale': ['Default conservative sizing due to calculation error'],
            'warnings': ['Using default sizing - manual review recommended'],
            'timestamp': datetime.now(),
            'error': 'Position sizing calculation failed'
        }
    
    def calculate_portfolio_sizing(self, account_balances: Dict[str, List[float]],
                                 symbols: List[str], market_analyses: Dict[str, Dict[str, Any]],
                                 portfolio_states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate position sizing for entire portfolio across all accounts.
        
        Args:
            account_balances: Dictionary of account balances by type
            symbols: List of symbols to analyze
            market_analyses: Market analyses for each symbol
            portfolio_states: Portfolio states for each account type
            
        Returns:
            Dict containing portfolio-wide sizing recommendations
        """
        logger.info(f"Calculating portfolio sizing for {len(symbols)} symbols across all accounts")
        
        portfolio_recommendations = {}
        
        for account_type_str, balances in account_balances.items():
            account_type = AccountType(account_type_str)
            account_recommendations = {}
            
            for i, balance in enumerate(balances):
                account_id = f"{account_type_str}_{i}"
                portfolio_state = portfolio_states.get(account_id, {})
                
                symbol_recommendations = {}
                for symbol in symbols:
                    if symbol in market_analyses:
                        recommendation = self.calculate_position_size(
                            account_type, balance, symbol,
                            market_analyses[symbol], portfolio_state
                        )
                        symbol_recommendations[symbol] = recommendation
                
                account_recommendations[account_id] = symbol_recommendations
            
            portfolio_recommendations[account_type_str] = account_recommendations
        
        # Calculate portfolio-level metrics
        portfolio_metrics = self._calculate_portfolio_metrics(portfolio_recommendations, account_balances)
        
        return {
            'timestamp': datetime.now(),
            'account_recommendations': portfolio_recommendations,
            'portfolio_metrics': portfolio_metrics,
            'total_symbols': len(symbols),
            'total_accounts': sum(len(balances) for balances in account_balances.values())
        }
    
    def _calculate_portfolio_metrics(self, portfolio_recommendations: Dict[str, Any],
                                   account_balances: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Calculate portfolio-level metrics.
        
        Args:
            portfolio_recommendations: Portfolio recommendations
            account_balances: Account balances
            
        Returns:
            Dict containing portfolio metrics
        """
        total_balance = sum(sum(balances) for balances in account_balances.values())
        total_recommended_size = 0.0
        total_positions = 0
        
        for account_type, account_recs in portfolio_recommendations.items():
            for account_id, symbol_recs in account_recs.items():
                for symbol, rec in symbol_recs.items():
                    total_recommended_size += rec.get('dollar_amount', 0)
                    total_positions += 1
        
        return {
            'total_portfolio_value': total_balance,
            'total_recommended_exposure': total_recommended_size,
            'portfolio_utilization': (total_recommended_size / total_balance) * 100 if total_balance > 0 else 0,
            'total_recommended_positions': total_positions,
            'average_position_size': total_recommended_size / max(total_positions, 1)
        }


# Example usage and testing
if __name__ == "__main__":
    # Create position sizer
    sizer = PositionSizer()
    
    # Sample market analysis (from MarketAnalyzer)
    sample_market_analysis = {
        'market_condition': 'Green',
        'volatility_analysis': {
            'implied_volatility': 0.30,
            'regime': 'Medium'
        },
        'trend_analysis': {
            'strength': 'Moderate'
        },
        'confidence_score': 0.75,
        'option_price': 3.50
    }
    
    # Sample portfolio state
    sample_portfolio_state = {
        'positions': {
            'AAPL': {'market_value': 15000},
            'MSFT': {'market_value': 12000}
        },
        'available_cash': 80000
    }
    
    # Test position sizing
    print("=== Position Sizing Test ===")
    
    for account_type in [AccountType.GEN_ACC, AccountType.REV_ACC, AccountType.COM_ACC]:
        account_balance = 100000  # $100k account
        
        recommendation = sizer.calculate_position_size(
            account_type, account_balance, 'TSLA',
            sample_market_analysis, sample_portfolio_state
        )
        
        print(f"\n{account_type.value} Recommendation:")
        print(f"Recommended Contracts: {recommendation['recommended_contracts']}")
        print(f"Dollar Amount: ${recommendation['dollar_amount']:,.2f}")
        print(f"Position Percentage: {recommendation['position_percentage']:.2f}%")
        print(f"Risk Metrics: Max Loss ${recommendation['risk_metrics']['max_loss_estimate']:,.2f}")
        print(f"Rationale: {recommendation['sizing_rationale'][0]}")
        
        if recommendation['warnings']:
            print(f"Warnings: {', '.join(recommendation['warnings'])}")
    
    print("\n=== Portfolio Sizing Test ===")
    
    # Test portfolio-wide sizing
    account_balances = {
        'GEN_ACC': [120000, 80000],
        'REV_ACC': [150000],
        'COM_ACC': [100000]
    }
    
    symbols = ['TSLA', 'AAPL']
    market_analyses = {
        'TSLA': sample_market_analysis,
        'AAPL': {**sample_market_analysis, 'volatility_analysis': {'implied_volatility': 0.20, 'regime': 'Low'}}
    }
    
    portfolio_states = {
        'GEN_ACC_0': sample_portfolio_state,
        'GEN_ACC_1': {'positions': {}, 'available_cash': 80000},
        'REV_ACC_0': {'positions': {}, 'available_cash': 150000},
        'COM_ACC_0': {'positions': {}, 'available_cash': 100000}
    }
    
    portfolio_sizing = sizer.calculate_portfolio_sizing(
        account_balances, symbols, market_analyses, portfolio_states
    )
    
    print(f"Total Portfolio Value: ${portfolio_sizing['portfolio_metrics']['total_portfolio_value']:,.2f}")
    print(f"Total Recommended Exposure: ${portfolio_sizing['portfolio_metrics']['total_recommended_exposure']:,.2f}")
    print(f"Portfolio Utilization: {portfolio_sizing['portfolio_metrics']['portfolio_utilization']:.1f}%")
    print(f"Total Recommended Positions: {portfolio_sizing['portfolio_metrics']['total_recommended_positions']}")

