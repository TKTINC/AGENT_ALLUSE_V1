"""
ALL-USE Trading Engine: Market Analyzer Module

This module implements sophisticated market condition analysis for the ALL-USE trading system.
It provides real-time market assessment, volatility analysis, and trend detection capabilities.
"""

import logging
import numpy as np
import pandas as pd
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
        logging.FileHandler('all_use_market_analyzer.log')
    ]
)

logger = logging.getLogger('all_use_market_analyzer')

class MarketCondition(Enum):
    """Enumeration of market conditions."""
    GREEN = "Green"      # Strong uptrend, low volatility
    RED = "Red"          # Strong downtrend, high volatility
    CHOP = "Chop"        # Sideways/choppy, uncertain direction

class VolatilityRegime(Enum):
    """Enumeration of volatility regimes."""
    LOW = "Low"          # IV < 20%
    MEDIUM = "Medium"    # IV 20-40%
    HIGH = "High"        # IV > 40%

class TrendStrength(Enum):
    """Enumeration of trend strength."""
    WEAK = "Weak"        # Trend strength < 30%
    MODERATE = "Moderate" # Trend strength 30-70%
    STRONG = "Strong"    # Trend strength > 70%

class MarketAnalyzer:
    """
    Advanced market analyzer for the ALL-USE trading system.
    
    This class provides sophisticated market analysis including:
    - Market condition classification (Green/Red/Chop)
    - Volatility regime analysis
    - Trend strength assessment
    - Risk-adjusted recommendations
    """
    
    def __init__(self):
        """Initialize the market analyzer."""
        self.parameters = ALLUSEParameters
        self.analysis_cache = {}
        self.cache_expiry = timedelta(minutes=15)  # Cache expires after 15 minutes
        
        # Market analysis thresholds
        self.volatility_thresholds = {
            'low': 0.20,      # 20% IV
            'high': 0.40      # 40% IV
        }
        
        self.trend_thresholds = {
            'weak': 0.30,     # 30% trend strength
            'strong': 0.70    # 70% trend strength
        }
        
        # Moving average periods for trend analysis
        self.ma_periods = {
            'short': 10,      # 10-day MA
            'medium': 20,     # 20-day MA
            'long': 50        # 50-day MA
        }
        
        logger.info("Market analyzer initialized")
    
    def analyze_market_condition(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current market condition for a given symbol.
        
        Args:
            symbol: Stock symbol to analyze
            market_data: Dictionary containing market data
            
        Returns:
            Dict containing comprehensive market analysis
        """
        logger.info(f"Analyzing market condition for {symbol}")
        
        # Check cache first
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        if cache_key in self.analysis_cache:
            cached_analysis = self.analysis_cache[cache_key]
            if datetime.now() - cached_analysis['timestamp'] < self.cache_expiry:
                logger.info(f"Returning cached analysis for {symbol}")
                return cached_analysis['analysis']
        
        try:
            # Extract market data
            current_price = market_data.get('current_price', 0)
            implied_volatility = market_data.get('implied_volatility', 0.25)
            historical_prices = market_data.get('historical_prices', [])
            volume = market_data.get('volume', 0)
            
            # Perform comprehensive analysis
            volatility_analysis = self._analyze_volatility(implied_volatility, historical_prices)
            trend_analysis = self._analyze_trend(historical_prices)
            momentum_analysis = self._analyze_momentum(historical_prices)
            volume_analysis = self._analyze_volume(volume, market_data.get('avg_volume', volume))
            
            # Determine overall market condition
            market_condition = self._determine_market_condition(
                volatility_analysis, trend_analysis, momentum_analysis, volume_analysis
            )
            
            # Generate trading recommendations
            recommendations = self._generate_recommendations(
                symbol, market_condition, volatility_analysis, trend_analysis
            )
            
            # Compile comprehensive analysis
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': current_price,
                'market_condition': market_condition,
                'volatility_analysis': volatility_analysis,
                'trend_analysis': trend_analysis,
                'momentum_analysis': momentum_analysis,
                'volume_analysis': volume_analysis,
                'recommendations': recommendations,
                'confidence_score': self._calculate_confidence_score(
                    volatility_analysis, trend_analysis, momentum_analysis
                )
            }
            
            # Cache the analysis
            self.analysis_cache[cache_key] = {
                'timestamp': datetime.now(),
                'analysis': analysis
            }
            
            logger.info(f"Market analysis completed for {symbol}: {market_condition}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market condition for {symbol}: {str(e)}")
            return self._get_default_analysis(symbol, market_data)
    
    def _analyze_volatility(self, implied_volatility: float, historical_prices: List[float]) -> Dict[str, Any]:
        """
        Analyze volatility regime and characteristics.
        
        Args:
            implied_volatility: Current implied volatility
            historical_prices: List of historical prices
            
        Returns:
            Dict containing volatility analysis
        """
        # Calculate historical volatility if we have price data
        historical_volatility = 0.0
        if len(historical_prices) > 20:
            returns = np.diff(np.log(historical_prices))
            historical_volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Determine volatility regime
        if implied_volatility < self.volatility_thresholds['low']:
            regime = VolatilityRegime.LOW
        elif implied_volatility > self.volatility_thresholds['high']:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.MEDIUM
        
        # Calculate volatility percentile (simplified)
        volatility_percentile = min(100, (implied_volatility / 0.60) * 100)
        
        return {
            'implied_volatility': implied_volatility,
            'historical_volatility': historical_volatility,
            'regime': regime,
            'percentile': volatility_percentile,
            'iv_hv_ratio': implied_volatility / max(historical_volatility, 0.01),
            'is_elevated': implied_volatility > self.volatility_thresholds['high']
        }
    
    def _analyze_trend(self, historical_prices: List[float]) -> Dict[str, Any]:
        """
        Analyze trend direction and strength.
        
        Args:
            historical_prices: List of historical prices
            
        Returns:
            Dict containing trend analysis
        """
        if len(historical_prices) < self.ma_periods['long']:
            return {
                'direction': 'neutral',
                'strength': TrendStrength.WEAK,
                'strength_score': 0.0,
                'moving_averages': {},
                'trend_quality': 'insufficient_data'
            }
        
        prices = np.array(historical_prices)
        
        # Calculate moving averages
        ma_short = np.mean(prices[-self.ma_periods['short']:])
        ma_medium = np.mean(prices[-self.ma_periods['medium']:])
        ma_long = np.mean(prices[-self.ma_periods['long']:])
        
        current_price = prices[-1]
        
        # Determine trend direction
        if ma_short > ma_medium > ma_long and current_price > ma_short:
            direction = 'bullish'
        elif ma_short < ma_medium < ma_long and current_price < ma_short:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # Calculate trend strength (simplified linear regression slope)
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        normalized_slope = abs(slope) / current_price  # Normalize by price
        
        # Determine trend strength
        if normalized_slope < self.trend_thresholds['weak']:
            strength = TrendStrength.WEAK
        elif normalized_slope > self.trend_thresholds['strong']:
            strength = TrendStrength.STRONG
        else:
            strength = TrendStrength.MODERATE
        
        # Calculate trend quality (how consistent the trend is)
        price_changes = np.diff(prices)
        positive_changes = np.sum(price_changes > 0)
        trend_consistency = positive_changes / len(price_changes) if direction == 'bullish' else (len(price_changes) - positive_changes) / len(price_changes)
        
        return {
            'direction': direction,
            'strength': strength,
            'strength_score': normalized_slope,
            'moving_averages': {
                'short': ma_short,
                'medium': ma_medium,
                'long': ma_long
            },
            'trend_quality': trend_consistency,
            'slope': slope
        }
    
    def _analyze_momentum(self, historical_prices: List[float]) -> Dict[str, Any]:
        """
        Analyze price momentum indicators.
        
        Args:
            historical_prices: List of historical prices
            
        Returns:
            Dict containing momentum analysis
        """
        if len(historical_prices) < 14:
            return {
                'rsi': 50.0,
                'momentum_score': 0.0,
                'momentum_direction': 'neutral',
                'is_overbought': False,
                'is_oversold': False
            }
        
        prices = np.array(historical_prices)
        
        # Calculate RSI (simplified)
        price_changes = np.diff(prices)
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Calculate momentum score (rate of change)
        if len(prices) >= 10:
            momentum_score = (prices[-1] - prices[-10]) / prices[-10]
        else:
            momentum_score = 0.0
        
        # Determine momentum direction
        if rsi > 60 and momentum_score > 0.02:
            momentum_direction = 'bullish'
        elif rsi < 40 and momentum_score < -0.02:
            momentum_direction = 'bearish'
        else:
            momentum_direction = 'neutral'
        
        return {
            'rsi': rsi,
            'momentum_score': momentum_score,
            'momentum_direction': momentum_direction,
            'is_overbought': rsi > 70,
            'is_oversold': rsi < 30
        }
    
    def _analyze_volume(self, current_volume: float, average_volume: float) -> Dict[str, Any]:
        """
        Analyze volume characteristics.
        
        Args:
            current_volume: Current trading volume
            average_volume: Average trading volume
            
        Returns:
            Dict containing volume analysis
        """
        volume_ratio = current_volume / max(average_volume, 1)
        
        if volume_ratio > 1.5:
            volume_condition = 'high'
        elif volume_ratio < 0.5:
            volume_condition = 'low'
        else:
            volume_condition = 'normal'
        
        return {
            'current_volume': current_volume,
            'average_volume': average_volume,
            'volume_ratio': volume_ratio,
            'volume_condition': volume_condition,
            'is_unusual': volume_ratio > 2.0 or volume_ratio < 0.3
        }
    
    def _determine_market_condition(self, volatility_analysis: Dict[str, Any], 
                                  trend_analysis: Dict[str, Any],
                                  momentum_analysis: Dict[str, Any],
                                  volume_analysis: Dict[str, Any]) -> MarketCondition:
        """
        Determine overall market condition based on all analyses.
        
        Args:
            volatility_analysis: Volatility analysis results
            trend_analysis: Trend analysis results
            momentum_analysis: Momentum analysis results
            volume_analysis: Volume analysis results
            
        Returns:
            MarketCondition enum value
        """
        # Scoring system for market condition
        green_score = 0
        red_score = 0
        chop_score = 0
        
        # Volatility scoring
        if volatility_analysis['regime'] == VolatilityRegime.LOW:
            green_score += 2
        elif volatility_analysis['regime'] == VolatilityRegime.HIGH:
            red_score += 2
            chop_score += 1
        else:
            chop_score += 1
        
        # Trend scoring
        if trend_analysis['direction'] == 'bullish' and trend_analysis['strength'] in [TrendStrength.MODERATE, TrendStrength.STRONG]:
            green_score += 3
        elif trend_analysis['direction'] == 'bearish' and trend_analysis['strength'] in [TrendStrength.MODERATE, TrendStrength.STRONG]:
            red_score += 3
        else:
            chop_score += 2
        
        # Momentum scoring
        if momentum_analysis['momentum_direction'] == 'bullish':
            green_score += 2
        elif momentum_analysis['momentum_direction'] == 'bearish':
            red_score += 2
        else:
            chop_score += 1
        
        # Volume scoring (confirmation)
        if volume_analysis['volume_condition'] == 'high':
            # High volume confirms the trend
            if green_score > red_score:
                green_score += 1
            elif red_score > green_score:
                red_score += 1
        
        # Determine final condition
        max_score = max(green_score, red_score, chop_score)
        
        if max_score == green_score and green_score > 4:
            return MarketCondition.GREEN
        elif max_score == red_score and red_score > 4:
            return MarketCondition.RED
        else:
            return MarketCondition.CHOP
    
    def _generate_recommendations(self, symbol: str, market_condition: MarketCondition,
                                volatility_analysis: Dict[str, Any],
                                trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading recommendations based on market analysis.
        
        Args:
            symbol: Stock symbol
            market_condition: Determined market condition
            volatility_analysis: Volatility analysis results
            trend_analysis: Trend analysis results
            
        Returns:
            Dict containing trading recommendations
        """
        recommendations = {
            'market_condition': market_condition.value,
            'recommended_strategy': None,
            'delta_range': None,
            'position_size_multiplier': 1.0,
            'risk_adjustment': 'normal',
            'notes': []
        }
        
        # Determine strategy based on market condition
        if market_condition == MarketCondition.GREEN:
            recommendations['recommended_strategy'] = 'aggressive_premium_selling'
            recommendations['delta_range'] = (40, 50) if symbol in ['TSLA', 'NVDA'] else (30, 40)
            recommendations['position_size_multiplier'] = 1.2
            recommendations['risk_adjustment'] = 'increase'
            recommendations['notes'].append('Favorable conditions for premium selling')
            
        elif market_condition == MarketCondition.RED:
            recommendations['recommended_strategy'] = 'defensive_premium_selling'
            recommendations['delta_range'] = (20, 30)
            recommendations['position_size_multiplier'] = 0.7
            recommendations['risk_adjustment'] = 'decrease'
            recommendations['notes'].append('Defensive positioning recommended')
            
        else:  # CHOP
            recommendations['recommended_strategy'] = 'conservative_premium_selling'
            recommendations['delta_range'] = (25, 35)
            recommendations['position_size_multiplier'] = 0.9
            recommendations['risk_adjustment'] = 'normal'
            recommendations['notes'].append('Choppy conditions - proceed with caution')
        
        # Volatility-based adjustments
        if volatility_analysis['regime'] == VolatilityRegime.HIGH:
            recommendations['notes'].append('High volatility - consider smaller position sizes')
            recommendations['position_size_multiplier'] *= 0.8
        elif volatility_analysis['regime'] == VolatilityRegime.LOW:
            recommendations['notes'].append('Low volatility - may increase position sizes')
            recommendations['position_size_multiplier'] *= 1.1
        
        # Trend-based adjustments
        if trend_analysis['strength'] == TrendStrength.STRONG:
            recommendations['notes'].append('Strong trend - align with direction')
        elif trend_analysis['strength'] == TrendStrength.WEAK:
            recommendations['notes'].append('Weak trend - neutral positioning preferred')
        
        return recommendations
    
    def _calculate_confidence_score(self, volatility_analysis: Dict[str, Any],
                                  trend_analysis: Dict[str, Any],
                                  momentum_analysis: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the analysis.
        
        Args:
            volatility_analysis: Volatility analysis results
            trend_analysis: Trend analysis results
            momentum_analysis: Momentum analysis results
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence
        
        # Trend confidence
        if trend_analysis['strength'] == TrendStrength.STRONG:
            confidence += 0.2
        elif trend_analysis['strength'] == TrendStrength.MODERATE:
            confidence += 0.1
        
        # Trend quality
        if trend_analysis.get('trend_quality', 0) > 0.7:
            confidence += 0.1
        
        # Volatility confidence
        if volatility_analysis['regime'] in [VolatilityRegime.LOW, VolatilityRegime.HIGH]:
            confidence += 0.1  # Clear volatility regime
        
        # Momentum confirmation
        if (trend_analysis['direction'] == 'bullish' and momentum_analysis['momentum_direction'] == 'bullish') or \
           (trend_analysis['direction'] == 'bearish' and momentum_analysis['momentum_direction'] == 'bearish'):
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _get_default_analysis(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get default analysis when error occurs.
        
        Args:
            symbol: Stock symbol
            market_data: Market data dictionary
            
        Returns:
            Default analysis dictionary
        """
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'current_price': market_data.get('current_price', 0),
            'market_condition': MarketCondition.CHOP,
            'volatility_analysis': {
                'implied_volatility': 0.25,
                'regime': VolatilityRegime.MEDIUM,
                'is_elevated': False
            },
            'trend_analysis': {
                'direction': 'neutral',
                'strength': TrendStrength.WEAK
            },
            'momentum_analysis': {
                'rsi': 50.0,
                'momentum_direction': 'neutral'
            },
            'volume_analysis': {
                'volume_condition': 'normal'
            },
            'recommendations': {
                'market_condition': 'Chop',
                'recommended_strategy': 'conservative_premium_selling',
                'delta_range': (25, 35),
                'position_size_multiplier': 0.9,
                'risk_adjustment': 'normal',
                'notes': ['Default analysis due to insufficient data']
            },
            'confidence_score': 0.3,
            'error': 'Analysis failed, using default values'
        }
    
    def get_market_summary(self, symbols: List[str], market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get market summary for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            market_data: Dictionary of market data for each symbol
            
        Returns:
            Dict containing market summary
        """
        logger.info(f"Generating market summary for {len(symbols)} symbols")
        
        analyses = {}
        conditions_count = {'Green': 0, 'Red': 0, 'Chop': 0}
        avg_volatility = 0.0
        
        for symbol in symbols:
            if symbol in market_data:
                analysis = self.analyze_market_condition(symbol, market_data[symbol])
                analyses[symbol] = analysis
                
                # Count conditions
                condition = analysis['market_condition'].value if hasattr(analysis['market_condition'], 'value') else analysis['market_condition']
                conditions_count[condition] += 1
                
                # Sum volatility
                avg_volatility += analysis['volatility_analysis']['implied_volatility']
        
        # Calculate averages
        if len(analyses) > 0:
            avg_volatility /= len(analyses)
        
        # Determine overall market sentiment
        max_condition = max(conditions_count, key=conditions_count.get)
        
        summary = {
            'timestamp': datetime.now(),
            'symbols_analyzed': len(analyses),
            'individual_analyses': analyses,
            'overall_sentiment': max_condition,
            'condition_distribution': conditions_count,
            'average_volatility': avg_volatility,
            'market_health': self._assess_market_health(conditions_count, avg_volatility)
        }
        
        logger.info(f"Market summary completed: {max_condition} sentiment, {avg_volatility:.2f} avg volatility")
        return summary
    
    def _assess_market_health(self, conditions_count: Dict[str, int], avg_volatility: float) -> str:
        """
        Assess overall market health.
        
        Args:
            conditions_count: Count of each market condition
            avg_volatility: Average volatility across symbols
            
        Returns:
            Market health assessment string
        """
        total_symbols = sum(conditions_count.values())
        
        if total_symbols == 0:
            return 'unknown'
        
        green_ratio = conditions_count['Green'] / total_symbols
        red_ratio = conditions_count['Red'] / total_symbols
        chop_ratio = conditions_count['Chop'] / total_symbols
        
        if green_ratio > 0.6 and avg_volatility < 0.3:
            return 'excellent'
        elif green_ratio > 0.4 and avg_volatility < 0.4:
            return 'good'
        elif red_ratio > 0.6 or avg_volatility > 0.5:
            return 'poor'
        elif chop_ratio > 0.5:
            return 'uncertain'
        else:
            return 'fair'


# Example usage and testing
if __name__ == "__main__":
    # Create market analyzer
    analyzer = MarketAnalyzer()
    
    # Sample market data
    sample_data = {
        'TSLA': {
            'current_price': 250.0,
            'implied_volatility': 0.45,
            'historical_prices': [240, 242, 245, 248, 250, 252, 255, 253, 251, 250] * 5,
            'volume': 50000000,
            'avg_volume': 40000000
        },
        'AAPL': {
            'current_price': 180.0,
            'implied_volatility': 0.25,
            'historical_prices': [175, 176, 177, 178, 179, 180, 181, 180, 179, 180] * 5,
            'volume': 30000000,
            'avg_volume': 35000000
        }
    }
    
    # Test individual analysis
    print("=== Individual Market Analysis ===")
    for symbol, data in sample_data.items():
        analysis = analyzer.analyze_market_condition(symbol, data)
        print(f"\n{symbol} Analysis:")
        print(f"Market Condition: {analysis['market_condition']}")
        print(f"Confidence Score: {analysis['confidence_score']:.2f}")
        print(f"Recommended Strategy: {analysis['recommendations']['recommended_strategy']}")
        print(f"Delta Range: {analysis['recommendations']['delta_range']}")
    
    # Test market summary
    print("\n=== Market Summary ===")
    summary = analyzer.get_market_summary(['TSLA', 'AAPL'], sample_data)
    print(f"Overall Sentiment: {summary['overall_sentiment']}")
    print(f"Market Health: {summary['market_health']}")
    print(f"Average Volatility: {summary['average_volatility']:.2f}")
    print(f"Condition Distribution: {summary['condition_distribution']}")

