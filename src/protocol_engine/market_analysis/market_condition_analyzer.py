"""
Market Condition Analyzer for ALL-USE Protocol Engine
Analyzes market conditions and provides probability-based scenario selection

This module provides sophisticated market condition analysis to support
the week classification system with probability-based decision making.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import statistics
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    RANGE_BOUND = "range_bound"

class MarketCondition(Enum):
    """Current market condition states"""
    EXTREMELY_BULLISH = "extremely_bullish"
    BULLISH = "bullish"
    NEUTRAL_BULLISH = "neutral_bullish"
    NEUTRAL = "neutral"
    NEUTRAL_BEARISH = "neutral_bearish"
    BEARISH = "bearish"
    EXTREMELY_BEARISH = "extremely_bearish"

@dataclass
class MarketMetrics:
    """Market metrics for analysis"""
    price_change_1d: float
    price_change_5d: float
    price_change_20d: float
    volatility_1d: float
    volatility_5d: float
    volatility_20d: float
    volume_ratio: float
    rsi: float
    macd_signal: float
    bollinger_position: float
    vix_level: float
    trend_strength: float
    momentum: float

@dataclass
class MarketAnalysis:
    """Complete market analysis result"""
    condition: MarketCondition
    regime: MarketRegime
    confidence: float
    metrics: MarketMetrics
    probability_distribution: Dict[str, float]
    risk_level: str
    volatility_regime: str
    trend_direction: str
    momentum_strength: str
    analysis_timestamp: datetime
    reasoning: str

class MarketConditionAnalyzer:
    """
    Advanced market condition analyzer for ALL-USE protocol
    
    Provides sophisticated market analysis including:
    - Market condition classification
    - Regime identification
    - Probability-based scenario selection
    - Risk assessment
    - Volatility analysis
    """
    
    def __init__(self):
        """Initialize the market condition analyzer"""
        self.logger = logging.getLogger(__name__)
        
        # Market condition thresholds
        self.condition_thresholds = {
            'extremely_bullish': {'min_change': 0.05, 'min_momentum': 0.8, 'max_vix': 15},
            'bullish': {'min_change': 0.02, 'min_momentum': 0.6, 'max_vix': 20},
            'neutral_bullish': {'min_change': 0.005, 'min_momentum': 0.4, 'max_vix': 25},
            'neutral': {'max_abs_change': 0.005, 'momentum_range': (-0.2, 0.2)},
            'neutral_bearish': {'max_change': -0.005, 'max_momentum': -0.4, 'min_vix': 25},
            'bearish': {'max_change': -0.02, 'max_momentum': -0.6, 'min_vix': 30},
            'extremely_bearish': {'max_change': -0.05, 'max_momentum': -0.8, 'min_vix': 35}
        }
        
        # Volatility regime thresholds
        self.volatility_thresholds = {
            'low': 0.15,
            'normal': 0.25,
            'high': 0.35,
            'extreme': 0.50
        }
        
        # Risk level thresholds
        self.risk_thresholds = {
            'low': 0.2,
            'moderate': 0.4,
            'high': 0.6,
            'extreme': 0.8
        }
        
        self.logger.info("Market Condition Analyzer initialized")
    
    def analyze_market_conditions(self, market_data: Dict[str, Any]) -> MarketAnalysis:
        """
        Perform comprehensive market condition analysis
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            MarketAnalysis: Complete market analysis result
        """
        try:
            # Calculate market metrics
            metrics = self._calculate_market_metrics(market_data)
            
            # Determine market condition
            condition = self._classify_market_condition(metrics)
            
            # Identify market regime
            regime = self._identify_market_regime(metrics)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(metrics, condition)
            
            # Generate probability distribution
            probability_dist = self._generate_probability_distribution(metrics)
            
            # Assess risk level
            risk_level = self._assess_risk_level(metrics)
            
            # Determine volatility regime
            volatility_regime = self._classify_volatility_regime(metrics)
            
            # Analyze trend direction
            trend_direction = self._analyze_trend_direction(metrics)
            
            # Assess momentum strength
            momentum_strength = self._assess_momentum_strength(metrics)
            
            # Generate reasoning
            reasoning = self._generate_analysis_reasoning(
                condition, regime, metrics, confidence
            )
            
            analysis = MarketAnalysis(
                condition=condition,
                regime=regime,
                confidence=confidence,
                metrics=metrics,
                probability_distribution=probability_dist,
                risk_level=risk_level,
                volatility_regime=volatility_regime,
                trend_direction=trend_direction,
                momentum_strength=momentum_strength,
                analysis_timestamp=datetime.now(),
                reasoning=reasoning
            )
            
            self.logger.info(f"Market analysis complete: {condition.value} with {confidence:.1%} confidence")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}")
            raise
    
    def _calculate_market_metrics(self, market_data: Dict[str, Any]) -> MarketMetrics:
        """Calculate comprehensive market metrics"""
        try:
            # Extract price data
            current_price = market_data.get('current_price', 100.0)
            price_history = market_data.get('price_history', [100.0] * 21)
            volume_history = market_data.get('volume_history', [1000000] * 21)
            vix_level = market_data.get('vix', 20.0)
            
            # Ensure we have enough data
            if len(price_history) < 21:
                price_history = [current_price] * 21
            if len(volume_history) < 21:
                volume_history = [1000000] * 21
            
            # Calculate price changes
            price_change_1d = (current_price - price_history[-2]) / price_history[-2] if len(price_history) > 1 else 0.0
            price_change_5d = (current_price - price_history[-6]) / price_history[-6] if len(price_history) > 5 else 0.0
            price_change_20d = (current_price - price_history[-21]) / price_history[-21] if len(price_history) > 20 else 0.0
            
            # Calculate volatilities
            returns_1d = [price_history[i] / price_history[i-1] - 1 for i in range(1, min(2, len(price_history)))]
            returns_5d = [price_history[i] / price_history[i-1] - 1 for i in range(1, min(6, len(price_history)))]
            returns_20d = [price_history[i] / price_history[i-1] - 1 for i in range(1, min(21, len(price_history)))]
            
            volatility_1d = np.std(returns_1d) * np.sqrt(252) if returns_1d else 0.2
            volatility_5d = np.std(returns_5d) * np.sqrt(252) if returns_5d else 0.2
            volatility_20d = np.std(returns_20d) * np.sqrt(252) if returns_20d else 0.2
            
            # Calculate volume ratio
            avg_volume_20d = np.mean(volume_history[-20:]) if len(volume_history) >= 20 else 1000000
            current_volume = volume_history[-1] if volume_history else 1000000
            volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1.0
            
            # Calculate technical indicators
            rsi = self._calculate_rsi(price_history)
            macd_signal = self._calculate_macd_signal(price_history)
            bollinger_position = self._calculate_bollinger_position(price_history)
            trend_strength = self._calculate_trend_strength(price_history)
            momentum = self._calculate_momentum(price_history)
            
            return MarketMetrics(
                price_change_1d=price_change_1d,
                price_change_5d=price_change_5d,
                price_change_20d=price_change_20d,
                volatility_1d=volatility_1d,
                volatility_5d=volatility_5d,
                volatility_20d=volatility_20d,
                volume_ratio=volume_ratio,
                rsi=rsi,
                macd_signal=macd_signal,
                bollinger_position=bollinger_position,
                vix_level=vix_level,
                trend_strength=trend_strength,
                momentum=momentum
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating market metrics: {str(e)}")
            # Return default metrics
            return MarketMetrics(
                price_change_1d=0.0, price_change_5d=0.0, price_change_20d=0.0,
                volatility_1d=0.2, volatility_5d=0.2, volatility_20d=0.2,
                volume_ratio=1.0, rsi=50.0, macd_signal=0.0, bollinger_position=0.5,
                vix_level=20.0, trend_strength=0.0, momentum=0.0
            )
    
    def _calculate_rsi(self, price_history: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(price_history) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, min(period + 1, len(price_history))):
            change = price_history[i] - price_history[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd_signal(self, price_history: List[float]) -> float:
        """Calculate MACD signal"""
        if len(price_history) < 26:
            return 0.0
        
        # Simple MACD calculation
        ema_12 = np.mean(price_history[-12:])
        ema_26 = np.mean(price_history[-26:])
        macd = ema_12 - ema_26
        
        # Normalize to -1 to 1 range
        return np.tanh(macd / price_history[-1]) if price_history[-1] != 0 else 0.0
    
    def _calculate_bollinger_position(self, price_history: List[float], period: int = 20) -> float:
        """Calculate position within Bollinger Bands"""
        if len(price_history) < period:
            return 0.5
        
        recent_prices = price_history[-period:]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        
        if std_price == 0:
            return 0.5
        
        current_price = price_history[-1]
        upper_band = mean_price + (2 * std_price)
        lower_band = mean_price - (2 * std_price)
        
        # Position within bands (0 = lower band, 1 = upper band)
        position = (current_price - lower_band) / (upper_band - lower_band)
        return max(0, min(1, position))
    
    def _calculate_trend_strength(self, price_history: List[float]) -> float:
        """Calculate trend strength (-1 to 1)"""
        if len(price_history) < 10:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(price_history[-10:]))
        y = price_history[-10:]
        
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope
        avg_price = np.mean(y)
        if avg_price == 0:
            return 0.0
        
        normalized_slope = slope / avg_price * 10  # Scale factor
        return np.tanh(normalized_slope)  # Bound to -1, 1
    
    def _calculate_momentum(self, price_history: List[float]) -> float:
        """Calculate price momentum"""
        if len(price_history) < 5:
            return 0.0
        
        # Rate of change over 5 periods
        roc = (price_history[-1] - price_history[-5]) / price_history[-5]
        return np.tanh(roc * 10)  # Normalize to -1, 1
    
    def _classify_market_condition(self, metrics: MarketMetrics) -> MarketCondition:
        """Classify current market condition"""
        # Composite score based on multiple factors
        price_score = (metrics.price_change_1d * 0.4 + 
                      metrics.price_change_5d * 0.4 + 
                      metrics.price_change_20d * 0.2)
        
        momentum_score = metrics.momentum
        vix_score = (30 - metrics.vix_level) / 30  # Normalize VIX (lower VIX = more bullish)
        
        # Weighted composite score
        composite_score = (price_score * 0.5 + 
                          momentum_score * 0.3 + 
                          vix_score * 0.2)
        
        # Classify based on composite score
        if composite_score > 0.05:
            return MarketCondition.EXTREMELY_BULLISH
        elif composite_score > 0.02:
            return MarketCondition.BULLISH
        elif composite_score > 0.005:
            return MarketCondition.NEUTRAL_BULLISH
        elif composite_score > -0.005:
            return MarketCondition.NEUTRAL
        elif composite_score > -0.02:
            return MarketCondition.NEUTRAL_BEARISH
        elif composite_score > -0.05:
            return MarketCondition.BEARISH
        else:
            return MarketCondition.EXTREMELY_BEARISH
    
    def _identify_market_regime(self, metrics: MarketMetrics) -> MarketRegime:
        """Identify current market regime"""
        # Volatility-based regime
        if metrics.volatility_20d > 0.35:
            return MarketRegime.HIGH_VOLATILITY
        elif metrics.volatility_20d < 0.15:
            return MarketRegime.LOW_VOLATILITY
        
        # Trend-based regime
        if abs(metrics.trend_strength) > 0.6:
            return MarketRegime.TRENDING
        elif abs(metrics.trend_strength) < 0.2:
            return MarketRegime.RANGE_BOUND
        
        # Price-based regime
        if metrics.price_change_20d > 0.1:
            return MarketRegime.BULL_MARKET
        elif metrics.price_change_20d < -0.1:
            return MarketRegime.BEAR_MARKET
        else:
            return MarketRegime.SIDEWAYS_MARKET
    
    def _calculate_confidence(self, metrics: MarketMetrics, condition: MarketCondition) -> float:
        """Calculate confidence in the market condition classification"""
        # Factors that increase confidence
        confidence_factors = []
        
        # Consistency across timeframes
        price_changes = [metrics.price_change_1d, metrics.price_change_5d, metrics.price_change_20d]
        if all(x > 0 for x in price_changes) or all(x < 0 for x in price_changes):
            confidence_factors.append(0.3)
        
        # Strong momentum alignment
        if abs(metrics.momentum) > 0.5:
            confidence_factors.append(0.2)
        
        # Clear trend
        if abs(metrics.trend_strength) > 0.5:
            confidence_factors.append(0.2)
        
        # Volume confirmation
        if metrics.volume_ratio > 1.2:
            confidence_factors.append(0.1)
        
        # RSI extremes
        if metrics.rsi > 70 or metrics.rsi < 30:
            confidence_factors.append(0.1)
        
        # VIX alignment
        if (condition in [MarketCondition.BULLISH, MarketCondition.EXTREMELY_BULLISH] and metrics.vix_level < 20) or \
           (condition in [MarketCondition.BEARISH, MarketCondition.EXTREMELY_BEARISH] and metrics.vix_level > 30):
            confidence_factors.append(0.1)
        
        base_confidence = 0.5
        total_confidence = base_confidence + sum(confidence_factors)
        return min(0.95, total_confidence)
    
    def _generate_probability_distribution(self, metrics: MarketMetrics) -> Dict[str, float]:
        """Generate probability distribution for different scenarios"""
        # Base probabilities for each condition
        base_probs = {
            'extremely_bullish': 0.05,
            'bullish': 0.15,
            'neutral_bullish': 0.20,
            'neutral': 0.20,
            'neutral_bearish': 0.20,
            'bearish': 0.15,
            'extremely_bearish': 0.05
        }
        
        # Adjust based on current metrics
        adjustments = {}
        
        # Price momentum adjustments
        if metrics.momentum > 0.5:
            adjustments['extremely_bullish'] = 0.1
            adjustments['bullish'] = 0.1
        elif metrics.momentum < -0.5:
            adjustments['extremely_bearish'] = 0.1
            adjustments['bearish'] = 0.1
        
        # Volatility adjustments
        if metrics.volatility_20d > 0.35:
            adjustments['extremely_bullish'] = adjustments.get('extremely_bullish', 0) + 0.05
            adjustments['extremely_bearish'] = adjustments.get('extremely_bearish', 0) + 0.05
        
        # Apply adjustments
        for condition, adjustment in adjustments.items():
            base_probs[condition] = min(0.8, base_probs[condition] + adjustment)
        
        # Normalize probabilities
        total = sum(base_probs.values())
        return {k: v / total for k, v in base_probs.items()}
    
    def _assess_risk_level(self, metrics: MarketMetrics) -> str:
        """Assess current risk level"""
        risk_score = 0
        
        # Volatility risk
        if metrics.volatility_20d > 0.35:
            risk_score += 0.3
        elif metrics.volatility_20d > 0.25:
            risk_score += 0.2
        
        # VIX risk
        if metrics.vix_level > 35:
            risk_score += 0.3
        elif metrics.vix_level > 25:
            risk_score += 0.2
        
        # Momentum risk
        if abs(metrics.momentum) > 0.7:
            risk_score += 0.2
        
        # Trend risk
        if abs(metrics.trend_strength) > 0.8:
            risk_score += 0.1
        
        # Volume risk
        if metrics.volume_ratio > 2.0:
            risk_score += 0.1
        
        if risk_score > 0.6:
            return "extreme"
        elif risk_score > 0.4:
            return "high"
        elif risk_score > 0.2:
            return "moderate"
        else:
            return "low"
    
    def _classify_volatility_regime(self, metrics: MarketMetrics) -> str:
        """Classify volatility regime"""
        vol = metrics.volatility_20d
        
        if vol > 0.50:
            return "extreme"
        elif vol > 0.35:
            return "high"
        elif vol > 0.25:
            return "normal"
        else:
            return "low"
    
    def _analyze_trend_direction(self, metrics: MarketMetrics) -> str:
        """Analyze trend direction"""
        if metrics.trend_strength > 0.3:
            return "uptrend"
        elif metrics.trend_strength < -0.3:
            return "downtrend"
        else:
            return "sideways"
    
    def _assess_momentum_strength(self, metrics: MarketMetrics) -> str:
        """Assess momentum strength"""
        momentum = abs(metrics.momentum)
        
        if momentum > 0.7:
            return "very_strong"
        elif momentum > 0.5:
            return "strong"
        elif momentum > 0.3:
            return "moderate"
        else:
            return "weak"
    
    def _generate_analysis_reasoning(self, condition: MarketCondition, regime: MarketRegime, 
                                   metrics: MarketMetrics, confidence: float) -> str:
        """Generate human-readable reasoning for the analysis"""
        reasoning_parts = []
        
        # Market condition reasoning
        reasoning_parts.append(f"Market classified as {condition.value.replace('_', ' ').title()}")
        
        # Price action reasoning
        if metrics.price_change_5d > 0.02:
            reasoning_parts.append(f"Strong 5-day price appreciation of {metrics.price_change_5d:.1%}")
        elif metrics.price_change_5d < -0.02:
            reasoning_parts.append(f"Significant 5-day price decline of {metrics.price_change_5d:.1%}")
        
        # Volatility reasoning
        if metrics.volatility_20d > 0.35:
            reasoning_parts.append(f"High volatility environment ({metrics.volatility_20d:.1%})")
        elif metrics.volatility_20d < 0.15:
            reasoning_parts.append(f"Low volatility environment ({metrics.volatility_20d:.1%})")
        
        # Momentum reasoning
        if abs(metrics.momentum) > 0.5:
            direction = "positive" if metrics.momentum > 0 else "negative"
            reasoning_parts.append(f"Strong {direction} momentum detected")
        
        # VIX reasoning
        if metrics.vix_level > 30:
            reasoning_parts.append(f"Elevated fear levels (VIX: {metrics.vix_level:.1f})")
        elif metrics.vix_level < 15:
            reasoning_parts.append(f"Low fear levels (VIX: {metrics.vix_level:.1f})")
        
        # Confidence reasoning
        reasoning_parts.append(f"Analysis confidence: {confidence:.1%}")
        
        return ". ".join(reasoning_parts) + "."

def test_market_condition_analyzer():
    """Test the market condition analyzer"""
    print("Testing Market Condition Analyzer...")
    
    analyzer = MarketConditionAnalyzer()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Bullish Market',
            'data': {
                'current_price': 105.0,
                'price_history': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0] + [100.0] * 15,
                'volume_history': [1000000] * 21,
                'vix': 18.0
            }
        },
        {
            'name': 'Bearish Market',
            'data': {
                'current_price': 95.0,
                'price_history': [100.0, 99.0, 98.0, 97.0, 96.0, 95.0] + [100.0] * 15,
                'volume_history': [1500000] * 21,
                'vix': 32.0
            }
        },
        {
            'name': 'High Volatility',
            'data': {
                'current_price': 100.0,
                'price_history': [100.0, 105.0, 95.0, 110.0, 90.0, 100.0] + [100.0] * 15,
                'volume_history': [2000000] * 21,
                'vix': 40.0
            }
        },
        {
            'name': 'Neutral Market',
            'data': {
                'current_price': 100.5,
                'price_history': [100.0, 100.2, 99.8, 100.1, 99.9, 100.5] + [100.0] * 15,
                'volume_history': [1000000] * 21,
                'vix': 20.0
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        analysis = analyzer.analyze_market_conditions(scenario['data'])
        
        print(f"Condition: {analysis.condition.value}")
        print(f"Regime: {analysis.regime.value}")
        print(f"Confidence: {analysis.confidence:.1%}")
        print(f"Risk Level: {analysis.risk_level}")
        print(f"Volatility Regime: {analysis.volatility_regime}")
        print(f"Trend Direction: {analysis.trend_direction}")
        print(f"Momentum Strength: {analysis.momentum_strength}")
        print(f"Reasoning: {analysis.reasoning}")
        
        # Show top 3 probabilities
        sorted_probs = sorted(analysis.probability_distribution.items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        print("Top Probabilities:")
        for condition, prob in sorted_probs:
            print(f"  {condition}: {prob:.1%}")
    
    print("\n✅ Market Condition Analyzer test completed successfully!")

if __name__ == "__main__":
    test_market_condition_analyzer()

