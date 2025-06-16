"""
ATR-Based Adjustment System for ALL-USE Protocol Engine
Implements Average True Range (ATR) based volatility adjustments for dynamic parameter optimization

This module provides sophisticated ATR-based adjustment mechanisms that
dynamically adjust trading parameters based on market volatility conditions
to optimize performance and risk management.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolatilityRegime(Enum):
    """Volatility regime classifications"""
    VERY_LOW = "very_low"      # ATR < 0.5%
    LOW = "low"                # ATR 0.5% - 1.0%
    NORMAL = "normal"          # ATR 1.0% - 2.0%
    HIGH = "high"              # ATR 2.0% - 3.5%
    VERY_HIGH = "very_high"    # ATR > 3.5%

class AdjustmentType(Enum):
    """Types of ATR-based adjustments"""
    POSITION_SIZE = "position_size"
    DELTA_SELECTION = "delta_selection"
    TIME_HORIZON = "time_horizon"
    RISK_PARAMETERS = "risk_parameters"
    STOP_LOSS = "stop_loss"
    PROFIT_TARGET = "profit_target"

@dataclass
class ATRMetrics:
    """ATR calculation metrics"""
    current_atr: float
    atr_percentage: float
    atr_14: float
    atr_21: float
    atr_50: float
    volatility_regime: VolatilityRegime
    trend_direction: str
    volatility_trend: str
    percentile_rank: float
    z_score: float
    calculation_timestamp: datetime

@dataclass
class AdjustmentParameters:
    """Parameters for ATR-based adjustments"""
    base_position_size: float
    adjusted_position_size: float
    base_delta: float
    adjusted_delta: float
    base_dte: int
    adjusted_dte: int
    risk_multiplier: float
    stop_loss_multiplier: float
    profit_target_multiplier: float
    confidence_adjustment: float

@dataclass
class AdjustmentResult:
    """Result of ATR-based adjustment"""
    adjustment_type: AdjustmentType
    original_value: float
    adjusted_value: float
    adjustment_factor: float
    volatility_regime: VolatilityRegime
    rationale: str
    confidence: float
    expected_impact: str

class ATRAdjustmentSystem:
    """
    Advanced ATR-Based Adjustment System for ALL-USE Protocol
    
    Provides sophisticated volatility-based parameter adjustments including:
    - Real-time ATR calculation and analysis
    - Volatility regime classification
    - Dynamic position sizing adjustments
    - Delta selection optimization
    - Risk parameter scaling
    - Time horizon adjustments
    """
    
    def __init__(self, lookback_periods: List[int] = [14, 21, 50]):
        """Initialize the ATR adjustment system"""
        self.logger = logging.getLogger(__name__)
        self.lookback_periods = lookback_periods
        
        # ATR calculation parameters
        self.atr_smoothing = 0.1  # Exponential smoothing factor
        self.min_data_points = max(lookback_periods) + 5
        
        # Volatility regime thresholds (as percentage of price)
        self.volatility_thresholds = {
            VolatilityRegime.VERY_LOW: 0.005,   # 0.5%
            VolatilityRegime.LOW: 0.010,        # 1.0%
            VolatilityRegime.NORMAL: 0.020,     # 2.0%
            VolatilityRegime.HIGH: 0.035,       # 3.5%
            VolatilityRegime.VERY_HIGH: float('inf')
        }
        
        # Adjustment factors by volatility regime
        self.adjustment_factors = {
            VolatilityRegime.VERY_LOW: {
                'position_size': 1.2,      # Increase position size in low vol
                'delta': 1.1,              # Slightly higher delta
                'dte': 0.9,                # Shorter time horizon
                'risk': 0.8,               # Lower risk multiplier
                'stop_loss': 1.5,          # Wider stops
                'profit_target': 0.8       # Closer profit targets
            },
            VolatilityRegime.LOW: {
                'position_size': 1.1,
                'delta': 1.05,
                'dte': 0.95,
                'risk': 0.9,
                'stop_loss': 1.3,
                'profit_target': 0.9
            },
            VolatilityRegime.NORMAL: {
                'position_size': 1.0,      # No adjustment
                'delta': 1.0,
                'dte': 1.0,
                'risk': 1.0,
                'stop_loss': 1.0,
                'profit_target': 1.0
            },
            VolatilityRegime.HIGH: {
                'position_size': 0.8,      # Reduce position size in high vol
                'delta': 0.9,              # Lower delta
                'dte': 1.1,                # Longer time horizon
                'risk': 1.2,               # Higher risk multiplier
                'stop_loss': 0.8,          # Tighter stops
                'profit_target': 1.2       # Further profit targets
            },
            VolatilityRegime.VERY_HIGH: {
                'position_size': 0.6,
                'delta': 0.8,
                'dte': 1.2,
                'risk': 1.5,
                'stop_loss': 0.6,
                'profit_target': 1.4
            }
        }
        
        # Historical ATR data for percentile calculations
        self.atr_history: List[float] = []
        self.max_history_size = 252  # One year of daily data
        
        self.logger.info("ATR Adjustment System initialized")
    
    def calculate_atr_metrics(self, price_data: List[Dict[str, float]]) -> ATRMetrics:
        """
        Calculate comprehensive ATR metrics
        
        Args:
            price_data: List of price dictionaries with 'high', 'low', 'close'
            
        Returns:
            ATRMetrics: Comprehensive ATR analysis
        """
        try:
            if len(price_data) < self.min_data_points:
                raise ValueError(f"Insufficient data: need at least {self.min_data_points} points")
            
            # Convert to DataFrame for easier calculation
            df = pd.DataFrame(price_data)
            
            # Calculate True Range
            df['prev_close'] = df['close'].shift(1)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['prev_close'])
            df['tr3'] = abs(df['low'] - df['prev_close'])
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            # Calculate ATR for different periods
            atr_values = {}
            for period in self.lookback_periods:
                if len(df) >= period:
                    atr_values[f'atr_{period}'] = df['true_range'].rolling(window=period).mean().iloc[-1]
                else:
                    atr_values[f'atr_{period}'] = df['true_range'].mean()
            
            # Primary ATR (14-period)
            current_atr = atr_values.get('atr_14', df['true_range'].mean())
            current_price = df['close'].iloc[-1]
            atr_percentage = current_atr / current_price
            
            # Classify volatility regime
            volatility_regime = self._classify_volatility_regime(atr_percentage)
            
            # Calculate trend direction
            trend_direction = self._calculate_trend_direction(df['close'].tolist())
            
            # Calculate volatility trend
            volatility_trend = self._calculate_volatility_trend(df['true_range'].tolist())
            
            # Calculate percentile rank and z-score
            self.atr_history.append(atr_percentage)
            if len(self.atr_history) > self.max_history_size:
                self.atr_history.pop(0)
            
            percentile_rank = self._calculate_percentile_rank(atr_percentage, self.atr_history)
            z_score = self._calculate_z_score(atr_percentage, self.atr_history)
            
            metrics = ATRMetrics(
                current_atr=current_atr,
                atr_percentage=atr_percentage,
                atr_14=atr_values.get('atr_14', current_atr),
                atr_21=atr_values.get('atr_21', current_atr),
                atr_50=atr_values.get('atr_50', current_atr),
                volatility_regime=volatility_regime,
                trend_direction=trend_direction,
                volatility_trend=volatility_trend,
                percentile_rank=percentile_rank,
                z_score=z_score,
                calculation_timestamp=datetime.now()
            )
            
            self.logger.info(f"ATR calculated: {atr_percentage:.2%} ({volatility_regime.value})")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR metrics: {str(e)}")
            raise
    
    def adjust_position_size(self, base_position_size: float, atr_metrics: ATRMetrics,
                           account_type: str = "GEN_ACC") -> AdjustmentResult:
        """
        Adjust position size based on ATR volatility
        
        Args:
            base_position_size: Original position size
            atr_metrics: ATR metrics for adjustment
            account_type: Account type for specific adjustments
            
        Returns:
            AdjustmentResult: Position size adjustment result
        """
        try:
            regime = atr_metrics.volatility_regime
            adjustment_factor = self.adjustment_factors[regime]['position_size']
            
            # Account-specific adjustments
            account_multipliers = {
                'GEN_ACC': 1.0,
                'REV_ACC': 0.9,  # More conservative
                'COM_ACC': 1.1   # More aggressive
            }
            
            account_multiplier = account_multipliers.get(account_type, 1.0)
            final_adjustment_factor = adjustment_factor * account_multiplier
            
            adjusted_position_size = base_position_size * final_adjustment_factor
            
            # Generate rationale
            if regime in [VolatilityRegime.VERY_LOW, VolatilityRegime.LOW]:
                rationale = f"Increased position size due to {regime.value} volatility (ATR: {atr_metrics.atr_percentage:.2%})"
                expected_impact = "Higher returns with controlled risk in stable market"
            elif regime == VolatilityRegime.NORMAL:
                rationale = f"Standard position size for {regime.value} volatility (ATR: {atr_metrics.atr_percentage:.2%})"
                expected_impact = "Balanced risk-return profile"
            else:
                rationale = f"Reduced position size due to {regime.value} volatility (ATR: {atr_metrics.atr_percentage:.2%})"
                expected_impact = "Risk reduction in volatile market conditions"
            
            # Calculate confidence based on ATR stability
            confidence = self._calculate_adjustment_confidence(atr_metrics)
            
            return AdjustmentResult(
                adjustment_type=AdjustmentType.POSITION_SIZE,
                original_value=base_position_size,
                adjusted_value=adjusted_position_size,
                adjustment_factor=final_adjustment_factor,
                volatility_regime=regime,
                rationale=rationale,
                confidence=confidence,
                expected_impact=expected_impact
            )
            
        except Exception as e:
            self.logger.error(f"Error adjusting position size: {str(e)}")
            raise
    
    def adjust_delta_selection(self, base_delta: float, atr_metrics: ATRMetrics,
                             market_condition: str = "neutral") -> AdjustmentResult:
        """
        Adjust delta selection based on ATR volatility
        
        Args:
            base_delta: Original delta target
            atr_metrics: ATR metrics for adjustment
            market_condition: Current market condition
            
        Returns:
            AdjustmentResult: Delta adjustment result
        """
        try:
            regime = atr_metrics.volatility_regime
            adjustment_factor = self.adjustment_factors[regime]['delta']
            
            # Market condition adjustments
            market_adjustments = {
                'extremely_bullish': 1.1,
                'bullish': 1.05,
                'neutral': 1.0,
                'bearish': 0.95,
                'extremely_bearish': 0.9
            }
            
            market_multiplier = market_adjustments.get(market_condition, 1.0)
            final_adjustment_factor = adjustment_factor * market_multiplier
            
            adjusted_delta = base_delta * final_adjustment_factor
            
            # Ensure delta stays within reasonable bounds
            adjusted_delta = max(10, min(70, adjusted_delta))
            
            # Generate rationale
            if regime in [VolatilityRegime.VERY_LOW, VolatilityRegime.LOW]:
                rationale = f"Higher delta selection for {regime.value} volatility environment"
                expected_impact = "Increased premium collection with higher probability of success"
            elif regime == VolatilityRegime.NORMAL:
                rationale = f"Standard delta selection for {regime.value} volatility"
                expected_impact = "Balanced premium and probability trade-off"
            else:
                rationale = f"Lower delta selection for {regime.value} volatility protection"
                expected_impact = "Reduced assignment risk in volatile conditions"
            
            confidence = self._calculate_adjustment_confidence(atr_metrics)
            
            return AdjustmentResult(
                adjustment_type=AdjustmentType.DELTA_SELECTION,
                original_value=base_delta,
                adjusted_value=adjusted_delta,
                adjustment_factor=final_adjustment_factor,
                volatility_regime=regime,
                rationale=rationale,
                confidence=confidence,
                expected_impact=expected_impact
            )
            
        except Exception as e:
            self.logger.error(f"Error adjusting delta selection: {str(e)}")
            raise
    
    def adjust_time_horizon(self, base_dte: int, atr_metrics: ATRMetrics) -> AdjustmentResult:
        """
        Adjust time horizon (DTE) based on ATR volatility
        
        Args:
            base_dte: Original days to expiration
            atr_metrics: ATR metrics for adjustment
            
        Returns:
            AdjustmentResult: Time horizon adjustment result
        """
        try:
            regime = atr_metrics.volatility_regime
            adjustment_factor = self.adjustment_factors[regime]['dte']
            
            adjusted_dte = int(base_dte * adjustment_factor)
            
            # Ensure DTE stays within reasonable bounds
            adjusted_dte = max(10, min(90, adjusted_dte))
            
            # Generate rationale
            if regime in [VolatilityRegime.VERY_LOW, VolatilityRegime.LOW]:
                rationale = f"Shorter time horizon for {regime.value} volatility to capture theta decay"
                expected_impact = "Faster premium collection in stable conditions"
            elif regime == VolatilityRegime.NORMAL:
                rationale = f"Standard time horizon for {regime.value} volatility"
                expected_impact = "Balanced time decay and market exposure"
            else:
                rationale = f"Longer time horizon for {regime.value} volatility buffer"
                expected_impact = "More time for position recovery in volatile markets"
            
            confidence = self._calculate_adjustment_confidence(atr_metrics)
            
            return AdjustmentResult(
                adjustment_type=AdjustmentType.TIME_HORIZON,
                original_value=float(base_dte),
                adjusted_value=float(adjusted_dte),
                adjustment_factor=adjustment_factor,
                volatility_regime=regime,
                rationale=rationale,
                confidence=confidence,
                expected_impact=expected_impact
            )
            
        except Exception as e:
            self.logger.error(f"Error adjusting time horizon: {str(e)}")
            raise
    
    def adjust_risk_parameters(self, base_risk: float, atr_metrics: ATRMetrics) -> AdjustmentResult:
        """
        Adjust risk parameters based on ATR volatility
        
        Args:
            base_risk: Original risk parameter
            atr_metrics: ATR metrics for adjustment
            
        Returns:
            AdjustmentResult: Risk parameter adjustment result
        """
        try:
            regime = atr_metrics.volatility_regime
            adjustment_factor = self.adjustment_factors[regime]['risk']
            
            adjusted_risk = base_risk * adjustment_factor
            
            # Generate rationale
            if regime in [VolatilityRegime.VERY_LOW, VolatilityRegime.LOW]:
                rationale = f"Lower risk multiplier for {regime.value} volatility environment"
                expected_impact = "Optimized risk-return in stable conditions"
            elif regime == VolatilityRegime.NORMAL:
                rationale = f"Standard risk parameters for {regime.value} volatility"
                expected_impact = "Balanced risk management approach"
            else:
                rationale = f"Higher risk multiplier for {regime.value} volatility protection"
                expected_impact = "Enhanced risk management in volatile conditions"
            
            confidence = self._calculate_adjustment_confidence(atr_metrics)
            
            return AdjustmentResult(
                adjustment_type=AdjustmentType.RISK_PARAMETERS,
                original_value=base_risk,
                adjusted_value=adjusted_risk,
                adjustment_factor=adjustment_factor,
                volatility_regime=regime,
                rationale=rationale,
                confidence=confidence,
                expected_impact=expected_impact
            )
            
        except Exception as e:
            self.logger.error(f"Error adjusting risk parameters: {str(e)}")
            raise
    
    def generate_comprehensive_adjustments(self, base_parameters: Dict[str, Any], 
                                         atr_metrics: ATRMetrics,
                                         context: Dict[str, Any]) -> AdjustmentParameters:
        """
        Generate comprehensive ATR-based adjustments for all parameters
        
        Args:
            base_parameters: Original trading parameters
            atr_metrics: ATR metrics for adjustments
            context: Additional context (account_type, market_condition, etc.)
            
        Returns:
            AdjustmentParameters: Complete set of adjusted parameters
        """
        try:
            # Extract base parameters
            base_position_size = base_parameters.get('position_size', 1.0)
            base_delta = base_parameters.get('delta', 30.0)
            base_dte = base_parameters.get('dte', 35)
            base_risk = base_parameters.get('risk_multiplier', 1.0)
            
            # Extract context
            account_type = context.get('account_type', 'GEN_ACC')
            market_condition = context.get('market_condition', 'neutral')
            
            # Generate individual adjustments
            position_adjustment = self.adjust_position_size(base_position_size, atr_metrics, account_type)
            delta_adjustment = self.adjust_delta_selection(base_delta, atr_metrics, market_condition)
            dte_adjustment = self.adjust_time_horizon(base_dte, atr_metrics)
            risk_adjustment = self.adjust_risk_parameters(base_risk, atr_metrics)
            
            # Calculate stop loss and profit target multipliers
            regime = atr_metrics.volatility_regime
            stop_loss_multiplier = self.adjustment_factors[regime]['stop_loss']
            profit_target_multiplier = self.adjustment_factors[regime]['profit_target']
            
            # Calculate overall confidence adjustment
            confidence_adjustment = self._calculate_overall_confidence(
                [position_adjustment, delta_adjustment, dte_adjustment, risk_adjustment]
            )
            
            adjustments = AdjustmentParameters(
                base_position_size=base_position_size,
                adjusted_position_size=position_adjustment.adjusted_value,
                base_delta=base_delta,
                adjusted_delta=delta_adjustment.adjusted_value,
                base_dte=base_dte,
                adjusted_dte=int(dte_adjustment.adjusted_value),
                risk_multiplier=risk_adjustment.adjusted_value,
                stop_loss_multiplier=stop_loss_multiplier,
                profit_target_multiplier=profit_target_multiplier,
                confidence_adjustment=confidence_adjustment
            )
            
            self.logger.info(f"Generated comprehensive adjustments for {regime.value} volatility")
            return adjustments
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive adjustments: {str(e)}")
            raise
    
    def _classify_volatility_regime(self, atr_percentage: float) -> VolatilityRegime:
        """Classify volatility regime based on ATR percentage"""
        for regime, threshold in self.volatility_thresholds.items():
            if atr_percentage <= threshold:
                return regime
        return VolatilityRegime.VERY_HIGH
    
    def _calculate_trend_direction(self, prices: List[float]) -> str:
        """Calculate trend direction from price series"""
        if len(prices) < 10:
            return "neutral"
        
        recent_prices = prices[-10:]
        slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        
        if slope > prices[-1] * 0.001:  # 0.1% threshold
            return "uptrend"
        elif slope < -prices[-1] * 0.001:
            return "downtrend"
        else:
            return "neutral"
    
    def _calculate_volatility_trend(self, true_ranges: List[float]) -> str:
        """Calculate volatility trend from true range series"""
        if len(true_ranges) < 10:
            return "stable"
        
        recent_tr = true_ranges[-10:]
        earlier_tr = true_ranges[-20:-10] if len(true_ranges) >= 20 else true_ranges[:-10]
        
        if not earlier_tr:
            return "stable"
        
        recent_avg = np.mean(recent_tr)
        earlier_avg = np.mean(earlier_tr)
        
        change = (recent_avg - earlier_avg) / earlier_avg
        
        if change > 0.1:  # 10% increase
            return "increasing"
        elif change < -0.1:  # 10% decrease
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_percentile_rank(self, current_value: float, history: List[float]) -> float:
        """Calculate percentile rank of current value in historical context"""
        if len(history) < 10:
            return 50.0  # Default to median
        
        sorted_history = sorted(history)
        rank = sum(1 for x in sorted_history if x <= current_value)
        return (rank / len(sorted_history)) * 100
    
    def _calculate_z_score(self, current_value: float, history: List[float]) -> float:
        """Calculate z-score of current value"""
        if len(history) < 10:
            return 0.0
        
        mean_val = np.mean(history)
        std_val = np.std(history)
        
        if std_val == 0:
            return 0.0
        
        return (current_value - mean_val) / std_val
    
    def _calculate_adjustment_confidence(self, atr_metrics: ATRMetrics) -> float:
        """Calculate confidence in ATR-based adjustments"""
        base_confidence = 0.8
        
        # Adjust based on volatility trend stability
        if atr_metrics.volatility_trend == "stable":
            base_confidence += 0.1
        elif atr_metrics.volatility_trend in ["increasing", "decreasing"]:
            base_confidence -= 0.05
        
        # Adjust based on percentile rank (extreme values less confident)
        if 25 <= atr_metrics.percentile_rank <= 75:
            base_confidence += 0.05
        elif atr_metrics.percentile_rank < 10 or atr_metrics.percentile_rank > 90:
            base_confidence -= 0.1
        
        # Adjust based on z-score (extreme values less confident)
        if abs(atr_metrics.z_score) > 2:
            base_confidence -= 0.1
        elif abs(atr_metrics.z_score) < 1:
            base_confidence += 0.05
        
        return max(0.5, min(0.95, base_confidence))
    
    def _calculate_overall_confidence(self, adjustment_results: List[AdjustmentResult]) -> float:
        """Calculate overall confidence from multiple adjustments"""
        if not adjustment_results:
            return 0.5
        
        confidences = [result.confidence for result in adjustment_results]
        return np.mean(confidences)

def test_atr_adjustment_system():
    """Test the ATR adjustment system"""
    print("Testing ATR Adjustment System...")
    
    system = ATRAdjustmentSystem()
    
    # Generate sample price data
    def generate_price_data(volatility_level: str, days: int = 60) -> List[Dict[str, float]]:
        """Generate sample price data with specified volatility"""
        np.random.seed(42)  # For reproducible results
        
        base_price = 100.0
        volatility_map = {
            'low': 0.008,      # 0.8% daily volatility
            'normal': 0.015,   # 1.5% daily volatility
            'high': 0.030      # 3.0% daily volatility
        }
        
        daily_vol = volatility_map.get(volatility_level, 0.015)
        
        prices = []
        current_price = base_price
        
        for i in range(days):
            # Random walk with volatility
            change = np.random.normal(0, daily_vol)
            current_price *= (1 + change)
            
            # Generate OHLC data
            daily_range = current_price * daily_vol * np.random.uniform(0.5, 2.0)
            high = current_price + daily_range * np.random.uniform(0.3, 0.7)
            low = current_price - daily_range * np.random.uniform(0.3, 0.7)
            
            prices.append({
                'high': high,
                'low': low,
                'close': current_price
            })
        
        return prices
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Low Volatility Market',
            'price_data': generate_price_data('low'),
            'base_parameters': {
                'position_size': 10,
                'delta': 30,
                'dte': 35,
                'risk_multiplier': 1.0
            },
            'context': {
                'account_type': 'GEN_ACC',
                'market_condition': 'bullish'
            }
        },
        {
            'name': 'Normal Volatility Market',
            'price_data': generate_price_data('normal'),
            'base_parameters': {
                'position_size': 10,
                'delta': 30,
                'dte': 35,
                'risk_multiplier': 1.0
            },
            'context': {
                'account_type': 'REV_ACC',
                'market_condition': 'neutral'
            }
        },
        {
            'name': 'High Volatility Market',
            'price_data': generate_price_data('high'),
            'base_parameters': {
                'position_size': 10,
                'delta': 30,
                'dte': 35,
                'risk_multiplier': 1.0
            },
            'context': {
                'account_type': 'COM_ACC',
                'market_condition': 'bearish'
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        # Calculate ATR metrics
        atr_metrics = system.calculate_atr_metrics(scenario['price_data'])
        
        print(f"ATR Percentage: {atr_metrics.atr_percentage:.2%}")
        print(f"Volatility Regime: {atr_metrics.volatility_regime.value}")
        print(f"Trend Direction: {atr_metrics.trend_direction}")
        print(f"Volatility Trend: {atr_metrics.volatility_trend}")
        print(f"Percentile Rank: {atr_metrics.percentile_rank:.1f}%")
        print(f"Z-Score: {atr_metrics.z_score:.2f}")
        
        # Generate comprehensive adjustments
        adjustments = system.generate_comprehensive_adjustments(
            scenario['base_parameters'],
            atr_metrics,
            scenario['context']
        )
        
        print(f"\nAdjustments:")
        print(f"Position Size: {adjustments.base_position_size} → {adjustments.adjusted_position_size:.1f}")
        print(f"Delta: {adjustments.base_delta} → {adjustments.adjusted_delta:.1f}")
        print(f"DTE: {adjustments.base_dte} → {adjustments.adjusted_dte}")
        print(f"Risk Multiplier: {adjustments.risk_multiplier:.2f}")
        print(f"Stop Loss Multiplier: {adjustments.stop_loss_multiplier:.2f}")
        print(f"Profit Target Multiplier: {adjustments.profit_target_multiplier:.2f}")
        print(f"Confidence: {adjustments.confidence_adjustment:.1%}")
        
        # Test individual adjustments
        position_adj = system.adjust_position_size(
            adjustments.base_position_size, atr_metrics, scenario['context']['account_type']
        )
        print(f"\nPosition Size Adjustment:")
        print(f"Rationale: {position_adj.rationale}")
        print(f"Expected Impact: {position_adj.expected_impact}")
        print(f"Confidence: {position_adj.confidence:.1%}")
    
    print("\n✅ ATR Adjustment System test completed successfully!")

if __name__ == "__main__":
    test_atr_adjustment_system()

