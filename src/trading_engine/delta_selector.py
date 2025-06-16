"""
ALL-USE Trading Engine: Delta Selector Module

This module implements intelligent delta selection for options trading in the ALL-USE system.
It provides dynamic delta selection based on market conditions, volatility, and risk parameters.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
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
        logging.FileHandler('all_use_delta_selector.log')
    ]
)

logger = logging.getLogger('all_use_delta_selector')

class DeltaRange(Enum):
    """Enumeration of delta ranges."""
    CONSERVATIVE = "Conservative"    # 15-25 delta
    MODERATE = "Moderate"           # 25-35 delta
    AGGRESSIVE = "Aggressive"       # 35-50 delta
    VERY_AGGRESSIVE = "Very_Aggressive"  # 50+ delta

class OptionType(Enum):
    """Enumeration of option types."""
    PUT = "Put"
    CALL = "Call"

class DeltaSelector:
    """
    Intelligent delta selection system for the ALL-USE options trading strategy.
    
    This class provides sophisticated delta selection algorithms that consider:
    - Market conditions and volatility
    - Time to expiration
    - Underlying price movement and momentum
    - Risk tolerance and account type
    - Historical performance of different delta ranges
    """
    
    def __init__(self):
        """Initialize the delta selector."""
        self.parameters = ALLUSEParameters
        
        # Delta range definitions
        self.delta_ranges = {
            DeltaRange.CONSERVATIVE: {
                'min_delta': 15,
                'max_delta': 25,
                'target_delta': 20,
                'risk_level': 'Low',
                'expected_win_rate': 0.85,
                'expected_return': 0.008  # 0.8% weekly
            },
            DeltaRange.MODERATE: {
                'min_delta': 25,
                'max_delta': 35,
                'target_delta': 30,
                'risk_level': 'Medium',
                'expected_win_rate': 0.75,
                'expected_return': 0.012  # 1.2% weekly
            },
            DeltaRange.AGGRESSIVE: {
                'min_delta': 35,
                'max_delta': 50,
                'target_delta': 42,
                'risk_level': 'High',
                'expected_win_rate': 0.65,
                'expected_return': 0.018  # 1.8% weekly
            },
            DeltaRange.VERY_AGGRESSIVE: {
                'min_delta': 50,
                'max_delta': 70,
                'target_delta': 60,
                'risk_level': 'Very High',
                'expected_win_rate': 0.55,
                'expected_return': 0.025  # 2.5% weekly
            }
        }
        
        # Market condition delta adjustments
        self.market_condition_adjustments = {
            'Green': {
                'put_delta_adjustment': 5,      # Sell higher delta puts
                'call_delta_adjustment': -5,    # Sell lower delta calls
                'preferred_range': DeltaRange.AGGRESSIVE
            },
            'Red': {
                'put_delta_adjustment': -5,     # Sell lower delta puts
                'call_delta_adjustment': 5,     # Sell higher delta calls
                'preferred_range': DeltaRange.CONSERVATIVE
            },
            'Chop': {
                'put_delta_adjustment': 0,      # No adjustment
                'call_delta_adjustment': 0,     # No adjustment
                'preferred_range': DeltaRange.MODERATE
            }
        }
        
        # Volatility regime adjustments
        self.volatility_adjustments = {
            'Low': {
                'delta_adjustment': 5,          # Higher delta in low vol
                'range_preference': 'increase'
            },
            'Medium': {
                'delta_adjustment': 0,          # No adjustment
                'range_preference': 'maintain'
            },
            'High': {
                'delta_adjustment': -5,         # Lower delta in high vol
                'range_preference': 'decrease'
            }
        }
        
        # Time to expiration adjustments
        self.dte_adjustments = {
            'short': {  # < 7 days
                'delta_adjustment': -5,         # Lower delta for short DTE
                'risk_multiplier': 1.5
            },
            'medium': {  # 7-21 days
                'delta_adjustment': 0,          # Standard delta
                'risk_multiplier': 1.0
            },
            'long': {  # > 21 days
                'delta_adjustment': 3,          # Slightly higher delta
                'risk_multiplier': 0.8
            }
        }
        
        logger.info("Delta selector initialized")
    
    def select_optimal_delta(self, symbol: str, option_type: OptionType,
                           market_analysis: Dict[str, Any], account_type: str,
                           risk_preferences: Dict[str, Any] = None,
                           time_to_expiration: int = 14) -> Dict[str, Any]:
        """
        Select optimal delta for an options trade.
        
        Args:
            symbol: Stock symbol
            option_type: PUT or CALL
            market_analysis: Market analysis from MarketAnalyzer
            account_type: Account type (GEN_ACC, REV_ACC, COM_ACC)
            risk_preferences: User risk preferences
            time_to_expiration: Days to expiration
            
        Returns:
            Dict containing delta selection recommendation
        """
        logger.info(f"Selecting optimal delta for {symbol} {option_type.value}")
        
        try:
            # Determine base delta range
            base_range = self._determine_base_delta_range(
                market_analysis, account_type, risk_preferences
            )
            
            # Apply market condition adjustments
            adjusted_delta = self._apply_market_adjustments(
                base_range, option_type, market_analysis
            )
            
            # Apply volatility adjustments
            vol_adjusted_delta = self._apply_volatility_adjustments(
                adjusted_delta, market_analysis
            )
            
            # Apply time to expiration adjustments
            final_delta = self._apply_dte_adjustments(
                vol_adjusted_delta, time_to_expiration
            )
            
            # Validate and constrain delta
            constrained_delta = self._constrain_delta(final_delta, symbol, option_type)
            
            # Calculate expected metrics
            expected_metrics = self._calculate_expected_metrics(
                constrained_delta, market_analysis, time_to_expiration
            )
            
            # Generate recommendation
            recommendation = {
                'symbol': symbol,
                'option_type': option_type.value,
                'recommended_delta': constrained_delta,
                'delta_range': self._get_delta_range_for_value(constrained_delta),
                'base_delta': base_range['target_delta'],
                'market_adjusted_delta': adjusted_delta,
                'volatility_adjusted_delta': vol_adjusted_delta,
                'final_delta': final_delta,
                'time_to_expiration': time_to_expiration,
                'expected_metrics': expected_metrics,
                'selection_rationale': self._generate_selection_rationale(
                    base_range, adjusted_delta, vol_adjusted_delta, final_delta,
                    constrained_delta, market_analysis
                ),
                'risk_assessment': self._assess_delta_risk(
                    constrained_delta, market_analysis, option_type
                ),
                'alternative_deltas': self._suggest_alternative_deltas(
                    constrained_delta, market_analysis
                ),
                'timestamp': datetime.now()
            }
            
            logger.info(f"Delta selection completed for {symbol}: {constrained_delta} delta")
            return recommendation
            
        except Exception as e:
            logger.error(f"Error selecting delta for {symbol}: {str(e)}")
            return self._get_default_delta_selection(symbol, option_type, account_type)
    
    def _determine_base_delta_range(self, market_analysis: Dict[str, Any],
                                  account_type: str, risk_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Determine base delta range based on market conditions and account type.
        
        Args:
            market_analysis: Market analysis results
            account_type: Account type
            risk_preferences: User risk preferences
            
        Returns:
            Dict containing base delta range information
        """
        # Get market condition
        market_condition = market_analysis.get('market_condition')
        if hasattr(market_condition, 'value'):
            condition_str = market_condition.value
        else:
            condition_str = str(market_condition)
        
        # Start with market condition preference
        if condition_str in self.market_condition_adjustments:
            preferred_range = self.market_condition_adjustments[condition_str]['preferred_range']
        else:
            preferred_range = DeltaRange.MODERATE
        
        # Account type adjustments
        if account_type == 'COM_ACC':
            # More conservative for compounding account
            if preferred_range == DeltaRange.AGGRESSIVE:
                preferred_range = DeltaRange.MODERATE
            elif preferred_range == DeltaRange.VERY_AGGRESSIVE:
                preferred_range = DeltaRange.AGGRESSIVE
        elif account_type == 'GEN_ACC':
            # More aggressive for generation account
            if preferred_range == DeltaRange.CONSERVATIVE:
                preferred_range = DeltaRange.MODERATE
        
        # User risk preference adjustments
        if risk_preferences:
            user_risk_level = risk_preferences.get('risk_level', 'Moderate')
            if user_risk_level == 'Conservative':
                if preferred_range == DeltaRange.AGGRESSIVE:
                    preferred_range = DeltaRange.MODERATE
                elif preferred_range == DeltaRange.VERY_AGGRESSIVE:
                    preferred_range = DeltaRange.AGGRESSIVE
            elif user_risk_level == 'Aggressive':
                if preferred_range == DeltaRange.CONSERVATIVE:
                    preferred_range = DeltaRange.MODERATE
                elif preferred_range == DeltaRange.MODERATE:
                    preferred_range = DeltaRange.AGGRESSIVE
        
        return self.delta_ranges[preferred_range]
    
    def _apply_market_adjustments(self, base_range: Dict[str, Any], option_type: OptionType,
                                market_analysis: Dict[str, Any]) -> int:
        """
        Apply market condition adjustments to delta.
        
        Args:
            base_range: Base delta range information
            option_type: PUT or CALL
            market_analysis: Market analysis results
            
        Returns:
            Market-adjusted delta value
        """
        base_delta = base_range['target_delta']
        
        # Get market condition
        market_condition = market_analysis.get('market_condition')
        if hasattr(market_condition, 'value'):
            condition_str = market_condition.value
        else:
            condition_str = str(market_condition)
        
        # Apply market condition adjustment
        if condition_str in self.market_condition_adjustments:
            adjustments = self.market_condition_adjustments[condition_str]
            
            if option_type == OptionType.PUT:
                adjustment = adjustments['put_delta_adjustment']
            else:
                adjustment = adjustments['call_delta_adjustment']
            
            adjusted_delta = base_delta + adjustment
        else:
            adjusted_delta = base_delta
        
        # Trend strength adjustment
        trend_analysis = market_analysis.get('trend_analysis', {})
        trend_strength = trend_analysis.get('strength')
        
        if hasattr(trend_strength, 'value'):
            strength_str = trend_strength.value
        else:
            strength_str = str(trend_strength)
        
        if strength_str == 'Strong':
            # Strong trends allow for more aggressive deltas
            adjusted_delta += 3
        elif strength_str == 'Weak':
            # Weak trends call for more conservative deltas
            adjusted_delta -= 3
        
        # Momentum adjustment
        momentum_analysis = market_analysis.get('momentum_analysis', {})
        momentum_direction = momentum_analysis.get('momentum_direction', 'neutral')
        
        if momentum_direction == 'bullish' and option_type == OptionType.PUT:
            adjusted_delta += 2  # Higher delta puts in bullish momentum
        elif momentum_direction == 'bearish' and option_type == OptionType.CALL:
            adjusted_delta += 2  # Higher delta calls in bearish momentum
        
        return int(adjusted_delta)
    
    def _apply_volatility_adjustments(self, adjusted_delta: int, market_analysis: Dict[str, Any]) -> int:
        """
        Apply volatility regime adjustments to delta.
        
        Args:
            adjusted_delta: Market-adjusted delta
            market_analysis: Market analysis results
            
        Returns:
            Volatility-adjusted delta value
        """
        # Get volatility regime
        volatility_analysis = market_analysis.get('volatility_analysis', {})
        volatility_regime = volatility_analysis.get('regime')
        
        if hasattr(volatility_regime, 'value'):
            regime_str = volatility_regime.value
        else:
            regime_str = str(volatility_regime)
        
        # Apply volatility adjustment
        if regime_str in self.volatility_adjustments:
            vol_adjustment = self.volatility_adjustments[regime_str]['delta_adjustment']
            vol_adjusted_delta = adjusted_delta + vol_adjustment
        else:
            vol_adjusted_delta = adjusted_delta
        
        # IV percentile adjustment
        iv_percentile = volatility_analysis.get('percentile', 50)
        
        if iv_percentile > 80:
            # Very high IV - be more conservative
            vol_adjusted_delta -= 3
        elif iv_percentile < 20:
            # Very low IV - can be more aggressive
            vol_adjusted_delta += 3
        
        # IV/HV ratio adjustment
        iv_hv_ratio = volatility_analysis.get('iv_hv_ratio', 1.0)
        
        if iv_hv_ratio > 1.5:
            # IV much higher than HV - good for selling
            vol_adjusted_delta += 2
        elif iv_hv_ratio < 0.8:
            # IV lower than HV - be more conservative
            vol_adjusted_delta -= 2
        
        return int(vol_adjusted_delta)
    
    def _apply_dte_adjustments(self, vol_adjusted_delta: int, time_to_expiration: int) -> int:
        """
        Apply time to expiration adjustments to delta.
        
        Args:
            vol_adjusted_delta: Volatility-adjusted delta
            time_to_expiration: Days to expiration
            
        Returns:
            DTE-adjusted delta value
        """
        # Determine DTE category
        if time_to_expiration < 7:
            dte_category = 'short'
        elif time_to_expiration <= 21:
            dte_category = 'medium'
        else:
            dte_category = 'long'
        
        # Apply DTE adjustment
        dte_adjustment = self.dte_adjustments[dte_category]['delta_adjustment']
        final_delta = vol_adjusted_delta + dte_adjustment
        
        # Additional adjustments for very short DTE
        if time_to_expiration <= 3:
            final_delta -= 5  # Be more conservative with very short DTE
        
        # Additional adjustments for very long DTE
        if time_to_expiration > 45:
            final_delta -= 2  # Slightly more conservative with very long DTE
        
        return int(final_delta)
    
    def _constrain_delta(self, final_delta: int, symbol: str, option_type: OptionType) -> int:
        """
        Constrain delta within reasonable bounds.
        
        Args:
            final_delta: Final calculated delta
            symbol: Stock symbol
            option_type: PUT or CALL
            
        Returns:
            Constrained delta value
        """
        # Symbol-specific constraints
        symbol_constraints = {
            'TSLA': {'min_delta': 20, 'max_delta': 60},  # Higher volatility stock
            'NVDA': {'min_delta': 20, 'max_delta': 60},  # Higher volatility stock
            'AAPL': {'min_delta': 15, 'max_delta': 50},  # More stable stock
            'MSFT': {'min_delta': 15, 'max_delta': 50},  # More stable stock
        }
        
        # Get constraints for symbol
        if symbol in symbol_constraints:
            min_delta = symbol_constraints[symbol]['min_delta']
            max_delta = symbol_constraints[symbol]['max_delta']
        else:
            # Default constraints
            min_delta = 15
            max_delta = 55
        
        # Apply constraints
        constrained_delta = max(min_delta, min(max_delta, final_delta))
        
        # Ensure delta is reasonable for option type
        if option_type == OptionType.PUT:
            # For puts, delta should be negative, but we work with absolute values
            constrained_delta = max(10, min(70, constrained_delta))
        else:
            # For calls, similar constraints
            constrained_delta = max(10, min(70, constrained_delta))
        
        return int(constrained_delta)
    
    def _get_delta_range_for_value(self, delta_value: int) -> str:
        """
        Get delta range category for a specific delta value.
        
        Args:
            delta_value: Delta value
            
        Returns:
            Delta range category string
        """
        for range_enum, range_info in self.delta_ranges.items():
            if range_info['min_delta'] <= delta_value <= range_info['max_delta']:
                return range_enum.value
        
        # If outside defined ranges
        if delta_value < 25:
            return DeltaRange.CONSERVATIVE.value
        elif delta_value > 50:
            return DeltaRange.VERY_AGGRESSIVE.value
        else:
            return DeltaRange.MODERATE.value
    
    def _calculate_expected_metrics(self, delta: int, market_analysis: Dict[str, Any],
                                  time_to_expiration: int) -> Dict[str, Any]:
        """
        Calculate expected metrics for the selected delta.
        
        Args:
            delta: Selected delta
            market_analysis: Market analysis results
            time_to_expiration: Days to expiration
            
        Returns:
            Dict containing expected metrics
        """
        # Get range information for delta
        range_category = self._get_delta_range_for_value(delta)
        
        # Find matching range info
        range_info = None
        for range_enum, info in self.delta_ranges.items():
            if range_enum.value == range_category:
                range_info = info
                break
        
        if not range_info:
            range_info = self.delta_ranges[DeltaRange.MODERATE]
        
        # Base expected metrics
        base_win_rate = range_info['expected_win_rate']
        base_return = range_info['expected_return']
        
        # Adjust based on market conditions
        confidence_score = market_analysis.get('confidence_score', 0.5)
        
        # Adjust win rate based on confidence
        adjusted_win_rate = base_win_rate * (0.8 + confidence_score * 0.4)
        
        # Adjust return based on market conditions
        market_condition = market_analysis.get('market_condition')
        if hasattr(market_condition, 'value'):
            condition_str = market_condition.value
        else:
            condition_str = str(market_condition)
        
        if condition_str == 'Green':
            return_multiplier = 1.2
        elif condition_str == 'Red':
            return_multiplier = 0.8
        else:
            return_multiplier = 1.0
        
        adjusted_return = base_return * return_multiplier
        
        # Time decay benefit (theta)
        theta_benefit = self._calculate_theta_benefit(delta, time_to_expiration)
        
        # Risk metrics
        max_loss_probability = 1 - adjusted_win_rate
        expected_max_loss = 0.5  # Assume 50% max loss for option selling
        
        return {
            'expected_win_rate': adjusted_win_rate,
            'expected_return': adjusted_return,
            'expected_weekly_return': adjusted_return * (7 / time_to_expiration),
            'theta_benefit': theta_benefit,
            'max_loss_probability': max_loss_probability,
            'expected_max_loss': expected_max_loss,
            'risk_reward_ratio': adjusted_return / expected_max_loss,
            'confidence_adjusted': True
        }
    
    def _calculate_theta_benefit(self, delta: int, time_to_expiration: int) -> float:
        """
        Calculate expected theta benefit for the delta and DTE.
        
        Args:
            delta: Selected delta
            time_to_expiration: Days to expiration
            
        Returns:
            Expected theta benefit
        """
        # Simplified theta calculation
        # Higher delta options have more theta, but also more risk
        base_theta = 0.02 * (delta / 30)  # Base theta as percentage of premium
        
        # Time decay acceleration
        if time_to_expiration <= 7:
            time_multiplier = 2.0  # Accelerated decay in final week
        elif time_to_expiration <= 21:
            time_multiplier = 1.5  # Moderate acceleration
        else:
            time_multiplier = 1.0  # Standard decay
        
        return base_theta * time_multiplier
    
    def _generate_selection_rationale(self, base_range: Dict[str, Any], market_adjusted: int,
                                    vol_adjusted: int, dte_adjusted: int, final_delta: int,
                                    market_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate rationale for delta selection.
        
        Args:
            base_range: Base delta range information
            market_adjusted: Market-adjusted delta
            vol_adjusted: Volatility-adjusted delta
            dte_adjusted: DTE-adjusted delta
            final_delta: Final constrained delta
            market_analysis: Market analysis results
            
        Returns:
            List of rationale strings
        """
        rationale = []
        
        # Base selection
        rationale.append(f"Base delta range: {base_range['target_delta']} ({base_range['risk_level']} risk)")
        
        # Market adjustments
        if market_adjusted != base_range['target_delta']:
            adjustment = market_adjusted - base_range['target_delta']
            if adjustment > 0:
                rationale.append(f"Increased delta by {adjustment} due to favorable market conditions")
            else:
                rationale.append(f"Decreased delta by {abs(adjustment)} due to market risks")
        
        # Volatility adjustments
        if vol_adjusted != market_adjusted:
            vol_adjustment = vol_adjusted - market_adjusted
            if vol_adjustment > 0:
                rationale.append(f"Increased delta by {vol_adjustment} due to volatility conditions")
            else:
                rationale.append(f"Decreased delta by {abs(vol_adjustment)} due to high volatility")
        
        # DTE adjustments
        if dte_adjusted != vol_adjusted:
            dte_adjustment = dte_adjusted - vol_adjusted
            if dte_adjustment > 0:
                rationale.append(f"Increased delta by {dte_adjustment} for time to expiration")
            else:
                rationale.append(f"Decreased delta by {abs(dte_adjustment)} for short time to expiration")
        
        # Constraints
        if final_delta != dte_adjusted:
            rationale.append(f"Delta constrained to {final_delta} within acceptable risk limits")
        
        # Market condition summary
        market_condition = market_analysis.get('market_condition')
        if hasattr(market_condition, 'value'):
            condition_str = market_condition.value
        else:
            condition_str = str(market_condition)
        
        rationale.append(f"Selected for {condition_str} market conditions")
        
        return rationale
    
    def _assess_delta_risk(self, delta: int, market_analysis: Dict[str, Any],
                         option_type: OptionType) -> Dict[str, Any]:
        """
        Assess risk for the selected delta.
        
        Args:
            delta: Selected delta
            market_analysis: Market analysis results
            option_type: PUT or CALL
            
        Returns:
            Dict containing risk assessment
        """
        # Get range information
        range_category = self._get_delta_range_for_value(delta)
        
        # Base risk level
        if range_category == DeltaRange.CONSERVATIVE.value:
            base_risk = 'Low'
        elif range_category == DeltaRange.MODERATE.value:
            base_risk = 'Medium'
        elif range_category == DeltaRange.AGGRESSIVE.value:
            base_risk = 'High'
        else:
            base_risk = 'Very High'
        
        # Assignment probability (simplified)
        assignment_probability = delta / 100.0  # Rough approximation
        
        # Market risk factors
        volatility_analysis = market_analysis.get('volatility_analysis', {})
        implied_volatility = volatility_analysis.get('implied_volatility', 0.25)
        
        # Adjust assignment probability for volatility
        vol_adjusted_assignment = assignment_probability * (1 + implied_volatility)
        
        # Trend risk
        trend_analysis = market_analysis.get('trend_analysis', {})
        trend_direction = trend_analysis.get('direction', 'neutral')
        
        trend_risk = 'Medium'
        if option_type == OptionType.PUT and trend_direction == 'bearish':
            trend_risk = 'High'
        elif option_type == OptionType.CALL and trend_direction == 'bullish':
            trend_risk = 'High'
        elif trend_direction == 'neutral':
            trend_risk = 'Low'
        
        return {
            'overall_risk_level': base_risk,
            'assignment_probability': assignment_probability,
            'volatility_adjusted_assignment': vol_adjusted_assignment,
            'trend_risk': trend_risk,
            'max_loss_potential': 'High' if delta > 40 else 'Medium' if delta > 25 else 'Low',
            'time_decay_benefit': 'High' if delta > 30 else 'Medium',
            'liquidity_risk': 'Low' if delta > 20 else 'Medium'  # Higher delta = better liquidity
        }
    
    def _suggest_alternative_deltas(self, selected_delta: int, market_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest alternative delta options.
        
        Args:
            selected_delta: Selected delta
            market_analysis: Market analysis results
            
        Returns:
            List of alternative delta suggestions
        """
        alternatives = []
        
        # Conservative alternative
        conservative_delta = max(15, selected_delta - 10)
        alternatives.append({
            'delta': conservative_delta,
            'risk_level': 'Lower',
            'expected_return': 'Lower',
            'rationale': 'More conservative approach with lower risk'
        })
        
        # Aggressive alternative
        aggressive_delta = min(60, selected_delta + 10)
        alternatives.append({
            'delta': aggressive_delta,
            'risk_level': 'Higher',
            'expected_return': 'Higher',
            'rationale': 'More aggressive approach with higher potential return'
        })
        
        # Market-neutral alternative
        neutral_delta = 30
        if abs(neutral_delta - selected_delta) > 5:
            alternatives.append({
                'delta': neutral_delta,
                'risk_level': 'Balanced',
                'expected_return': 'Moderate',
                'rationale': 'Market-neutral approach regardless of conditions'
            })
        
        return alternatives
    
    def _get_default_delta_selection(self, symbol: str, option_type: OptionType, account_type: str) -> Dict[str, Any]:
        """
        Get default delta selection when calculation fails.
        
        Args:
            symbol: Stock symbol
            option_type: PUT or CALL
            account_type: Account type
            
        Returns:
            Default delta selection
        """
        # Conservative default
        default_delta = 25
        
        return {
            'symbol': symbol,
            'option_type': option_type.value,
            'recommended_delta': default_delta,
            'delta_range': DeltaRange.MODERATE.value,
            'base_delta': default_delta,
            'market_adjusted_delta': default_delta,
            'volatility_adjusted_delta': default_delta,
            'final_delta': default_delta,
            'time_to_expiration': 14,
            'expected_metrics': {
                'expected_win_rate': 0.70,
                'expected_return': 0.010,
                'expected_weekly_return': 0.005
            },
            'selection_rationale': ['Default conservative selection due to calculation error'],
            'risk_assessment': {
                'overall_risk_level': 'Medium',
                'assignment_probability': 0.25
            },
            'alternative_deltas': [],
            'timestamp': datetime.now(),
            'error': 'Delta selection calculation failed'
        }
    
    def select_portfolio_deltas(self, symbols: List[str], option_types: List[OptionType],
                              market_analyses: Dict[str, Dict[str, Any]], account_type: str,
                              risk_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Select optimal deltas for entire portfolio.
        
        Args:
            symbols: List of stock symbols
            option_types: List of option types for each symbol
            market_analyses: Market analyses for each symbol
            account_type: Account type
            risk_preferences: User risk preferences
            
        Returns:
            Dict containing portfolio delta selections
        """
        logger.info(f"Selecting portfolio deltas for {len(symbols)} symbols")
        
        portfolio_selections = {}
        delta_distribution = {'Conservative': 0, 'Moderate': 0, 'Aggressive': 0, 'Very_Aggressive': 0}
        total_risk_score = 0.0
        
        for i, symbol in enumerate(symbols):
            if symbol in market_analyses:
                option_type = option_types[i] if i < len(option_types) else OptionType.PUT
                
                selection = self.select_optimal_delta(
                    symbol, option_type, market_analyses[symbol],
                    account_type, risk_preferences
                )
                
                portfolio_selections[symbol] = selection
                
                # Track distribution
                delta_range = selection['delta_range']
                if delta_range in delta_distribution:
                    delta_distribution[delta_range] += 1
                
                # Calculate risk score
                risk_level = selection['risk_assessment']['overall_risk_level']
                risk_scores = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
                total_risk_score += risk_scores.get(risk_level, 2)
        
        # Calculate portfolio metrics
        avg_risk_score = total_risk_score / max(len(portfolio_selections), 1)
        portfolio_risk_level = 'Low' if avg_risk_score < 1.5 else 'Medium' if avg_risk_score < 2.5 else 'High' if avg_risk_score < 3.5 else 'Very High'
        
        return {
            'timestamp': datetime.now(),
            'account_type': account_type,
            'symbol_selections': portfolio_selections,
            'portfolio_metrics': {
                'total_positions': len(portfolio_selections),
                'delta_distribution': delta_distribution,
                'average_risk_score': avg_risk_score,
                'portfolio_risk_level': portfolio_risk_level,
                'diversification_score': self._calculate_diversification_score(delta_distribution)
            },
            'portfolio_recommendations': self._generate_portfolio_recommendations(
                delta_distribution, avg_risk_score
            )
        }
    
    def _calculate_diversification_score(self, delta_distribution: Dict[str, int]) -> float:
        """
        Calculate diversification score based on delta distribution.
        
        Args:
            delta_distribution: Distribution of delta ranges
            
        Returns:
            Diversification score (0-1)
        """
        total_positions = sum(delta_distribution.values())
        if total_positions == 0:
            return 0.0
        
        # Calculate entropy-based diversification
        entropy = 0.0
        for count in delta_distribution.values():
            if count > 0:
                probability = count / total_positions
                entropy -= probability * np.log2(probability)
        
        # Normalize to 0-1 scale
        max_entropy = np.log2(len(delta_distribution))
        diversification_score = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return diversification_score
    
    def _generate_portfolio_recommendations(self, delta_distribution: Dict[str, int],
                                          avg_risk_score: float) -> List[str]:
        """
        Generate portfolio-level recommendations.
        
        Args:
            delta_distribution: Distribution of delta ranges
            avg_risk_score: Average risk score
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        total_positions = sum(delta_distribution.values())
        
        # Risk level recommendations
        if avg_risk_score > 3.0:
            recommendations.append("Portfolio risk level is high - consider reducing some delta exposures")
        elif avg_risk_score < 1.5:
            recommendations.append("Portfolio risk level is conservative - consider increasing some exposures for better returns")
        
        # Diversification recommendations
        diversification_score = self._calculate_diversification_score(delta_distribution)
        if diversification_score < 0.5:
            recommendations.append("Consider diversifying across different delta ranges for better risk management")
        
        # Concentration warnings
        for range_name, count in delta_distribution.items():
            if count / total_positions > 0.7:
                recommendations.append(f"High concentration in {range_name} delta range - consider diversification")
        
        # Market condition alignment
        if delta_distribution['Aggressive'] + delta_distribution['Very_Aggressive'] > total_positions * 0.6:
            recommendations.append("High aggressive delta exposure - ensure market conditions support this strategy")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Create delta selector
    selector = DeltaSelector()
    
    # Sample market analysis
    sample_market_analysis = {
        'market_condition': 'Green',
        'volatility_analysis': {
            'implied_volatility': 0.35,
            'regime': 'Medium',
            'percentile': 60,
            'iv_hv_ratio': 1.2
        },
        'trend_analysis': {
            'direction': 'bullish',
            'strength': 'Moderate'
        },
        'momentum_analysis': {
            'momentum_direction': 'bullish'
        },
        'confidence_score': 0.75
    }
    
    # Test individual delta selection
    print("=== Individual Delta Selection ===")
    
    for option_type in [OptionType.PUT, OptionType.CALL]:
        for account_type in ['GEN_ACC', 'REV_ACC', 'COM_ACC']:
            selection = selector.select_optimal_delta(
                'TSLA', option_type, sample_market_analysis, account_type
            )
            
            print(f"\n{account_type} {option_type.value} Selection:")
            print(f"Recommended Delta: {selection['recommended_delta']}")
            print(f"Delta Range: {selection['delta_range']}")
            print(f"Expected Win Rate: {selection['expected_metrics']['expected_win_rate']:.2%}")
            print(f"Expected Return: {selection['expected_metrics']['expected_return']:.2%}")
            print(f"Risk Level: {selection['risk_assessment']['overall_risk_level']}")
            print(f"Rationale: {selection['selection_rationale'][0]}")
    
    # Test portfolio delta selection
    print("\n=== Portfolio Delta Selection ===")
    
    symbols = ['TSLA', 'AAPL', 'NVDA']
    option_types = [OptionType.PUT, OptionType.PUT, OptionType.CALL]
    market_analyses = {
        'TSLA': sample_market_analysis,
        'AAPL': {**sample_market_analysis, 'volatility_analysis': {'implied_volatility': 0.20, 'regime': 'Low'}},
        'NVDA': {**sample_market_analysis, 'market_condition': 'Chop'}
    }
    
    portfolio_selection = selector.select_portfolio_deltas(
        symbols, option_types, market_analyses, 'GEN_ACC'
    )
    
    print(f"Portfolio Risk Level: {portfolio_selection['portfolio_metrics']['portfolio_risk_level']}")
    print(f"Delta Distribution: {portfolio_selection['portfolio_metrics']['delta_distribution']}")
    print(f"Diversification Score: {portfolio_selection['portfolio_metrics']['diversification_score']:.2f}")
    
    if portfolio_selection['portfolio_recommendations']:
        print("Recommendations:")
        for rec in portfolio_selection['portfolio_recommendations']:
            print(f"- {rec}")

