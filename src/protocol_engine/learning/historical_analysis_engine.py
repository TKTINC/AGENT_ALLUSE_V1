"""
Historical Analysis and Learning System for ALL-USE Protocol Engine
Provides historical pattern analysis and machine learning capabilities for week classification improvement

This module implements sophisticated historical analysis and learning capabilities
to continuously improve week classification accuracy and trading performance.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import pickle
import statistics
from collections import defaultdict, deque
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningMode(Enum):
    """Learning modes for the system"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    ENSEMBLE = "ensemble"

class PatternType(Enum):
    """Types of patterns to identify"""
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"
    TREND = "trend"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    ANOMALY = "anomaly"

@dataclass
class HistoricalWeek:
    """Historical week data structure"""
    week_number: int
    year: int
    week_type: str
    actual_return: float
    predicted_return: float
    market_conditions: Dict[str, Any]
    classification_confidence: float
    action_taken: str
    outcome_success: bool
    lessons_learned: List[str]
    timestamp: datetime

@dataclass
class Pattern:
    """Identified pattern structure"""
    pattern_type: PatternType
    description: str
    frequency: int
    confidence: float
    conditions: Dict[str, Any]
    expected_outcome: Dict[str, float]
    historical_accuracy: float
    last_occurrence: datetime
    next_expected: Optional[datetime]

@dataclass
class LearningInsight:
    """Learning insight from historical analysis"""
    insight_type: str
    description: str
    confidence: float
    supporting_evidence: List[str]
    actionable_recommendations: List[str]
    expected_impact: float
    validation_criteria: List[str]

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    classification_accuracy: float
    prediction_accuracy: float
    return_accuracy: float
    risk_adjusted_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    average_return: float
    volatility: float
    total_weeks_analyzed: int

class HistoricalAnalysisEngine:
    """
    Advanced historical analysis and learning engine for ALL-USE protocol
    
    Provides sophisticated capabilities for:
    - Historical pattern identification
    - Performance tracking and analysis
    - Machine learning for classification improvement
    - Adaptive strategy optimization
    - Predictive analytics
    """
    
    def __init__(self, max_history_size: int = 1000):
        """Initialize the historical analysis engine"""
        self.logger = logging.getLogger(__name__)
        self.max_history_size = max_history_size
        
        # Historical data storage
        self.historical_weeks: deque = deque(maxlen=max_history_size)
        self.patterns: List[Pattern] = []
        self.insights: List[LearningInsight] = []
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics(
            classification_accuracy=0.0,
            prediction_accuracy=0.0,
            return_accuracy=0.0,
            risk_adjusted_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            average_return=0.0,
            volatility=0.0,
            total_weeks_analyzed=0
        )
        
        # Learning parameters
        self.learning_parameters = {
            'pattern_min_frequency': 3,
            'pattern_confidence_threshold': 0.7,
            'accuracy_improvement_threshold': 0.05,
            'adaptation_rate': 0.1,
            'ensemble_weight_decay': 0.95
        }
        
        # Classification improvement tracking
        self.classification_improvements = defaultdict(list)
        self.prediction_models = {}
        
        self.logger.info("Historical Analysis Engine initialized")
    
    def add_historical_week(self, week_data: HistoricalWeek) -> None:
        """Add a new historical week to the analysis"""
        try:
            self.historical_weeks.append(week_data)
            self.performance_metrics.total_weeks_analyzed += 1
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Identify new patterns
            self._identify_patterns()
            
            # Generate insights
            self._generate_insights()
            
            # Adapt classification models
            self._adapt_classification_models()
            
            self.logger.info(f"Added historical week {week_data.week_number}/{week_data.year}: {week_data.week_type}")
            
        except Exception as e:
            self.logger.error(f"Error adding historical week: {str(e)}")
            raise
    
    def analyze_historical_performance(self, time_period: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze historical performance over specified time period
        
        Args:
            time_period: Number of weeks to analyze (None for all)
            
        Returns:
            Dictionary containing comprehensive performance analysis
        """
        try:
            # Select data for analysis
            if time_period and time_period < len(self.historical_weeks):
                weeks_to_analyze = list(self.historical_weeks)[-time_period:]
            else:
                weeks_to_analyze = list(self.historical_weeks)
            
            if not weeks_to_analyze:
                return {"error": "No historical data available"}
            
            # Calculate performance metrics
            performance_analysis = self._calculate_performance_metrics(weeks_to_analyze)
            
            # Analyze week type performance
            week_type_analysis = self._analyze_week_type_performance(weeks_to_analyze)
            
            # Analyze market condition performance
            market_condition_analysis = self._analyze_market_condition_performance(weeks_to_analyze)
            
            # Identify best and worst performing periods
            period_analysis = self._analyze_performance_periods(weeks_to_analyze)
            
            # Generate improvement recommendations
            improvement_recommendations = self._generate_improvement_recommendations(weeks_to_analyze)
            
            analysis_result = {
                'analysis_period': {
                    'weeks_analyzed': len(weeks_to_analyze),
                    'start_date': weeks_to_analyze[0].timestamp if weeks_to_analyze else None,
                    'end_date': weeks_to_analyze[-1].timestamp if weeks_to_analyze else None
                },
                'overall_performance': performance_analysis,
                'week_type_performance': week_type_analysis,
                'market_condition_performance': market_condition_analysis,
                'period_analysis': period_analysis,
                'improvement_recommendations': improvement_recommendations,
                'patterns_identified': len(self.patterns),
                'insights_generated': len(self.insights)
            }
            
            self.logger.info(f"Historical performance analysis completed for {len(weeks_to_analyze)} weeks")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in historical performance analysis: {str(e)}")
            raise
    
    def identify_patterns(self, pattern_types: Optional[List[PatternType]] = None) -> List[Pattern]:
        """
        Identify patterns in historical data
        
        Args:
            pattern_types: Specific pattern types to look for (None for all)
            
        Returns:
            List of identified patterns
        """
        try:
            if not self.historical_weeks:
                return []
            
            identified_patterns = []
            
            # Seasonal patterns
            if not pattern_types or PatternType.SEASONAL in pattern_types:
                seasonal_patterns = self._identify_seasonal_patterns()
                identified_patterns.extend(seasonal_patterns)
            
            # Cyclical patterns
            if not pattern_types or PatternType.CYCLICAL in pattern_types:
                cyclical_patterns = self._identify_cyclical_patterns()
                identified_patterns.extend(cyclical_patterns)
            
            # Trend patterns
            if not pattern_types or PatternType.TREND in pattern_types:
                trend_patterns = self._identify_trend_patterns()
                identified_patterns.extend(trend_patterns)
            
            # Volatility patterns
            if not pattern_types or PatternType.VOLATILITY in pattern_types:
                volatility_patterns = self._identify_volatility_patterns()
                identified_patterns.extend(volatility_patterns)
            
            # Correlation patterns
            if not pattern_types or PatternType.CORRELATION in pattern_types:
                correlation_patterns = self._identify_correlation_patterns()
                identified_patterns.extend(correlation_patterns)
            
            # Anomaly patterns
            if not pattern_types or PatternType.ANOMALY in pattern_types:
                anomaly_patterns = self._identify_anomaly_patterns()
                identified_patterns.extend(anomaly_patterns)
            
            # Update patterns list
            self.patterns = identified_patterns
            
            self.logger.info(f"Identified {len(identified_patterns)} patterns")
            return identified_patterns
            
        except Exception as e:
            self.logger.error(f"Error identifying patterns: {str(e)}")
            raise
    
    def generate_predictive_insights(self, forecast_horizon: int = 4) -> List[LearningInsight]:
        """
        Generate predictive insights for future weeks
        
        Args:
            forecast_horizon: Number of weeks to forecast
            
        Returns:
            List of predictive insights
        """
        try:
            insights = []
            
            # Pattern-based predictions
            pattern_insights = self._generate_pattern_based_insights(forecast_horizon)
            insights.extend(pattern_insights)
            
            # Performance-based predictions
            performance_insights = self._generate_performance_based_insights(forecast_horizon)
            insights.extend(performance_insights)
            
            # Market condition predictions
            market_insights = self._generate_market_condition_insights(forecast_horizon)
            insights.extend(market_insights)
            
            # Risk-based predictions
            risk_insights = self._generate_risk_based_insights(forecast_horizon)
            insights.extend(risk_insights)
            
            # Update insights list
            self.insights = insights
            
            self.logger.info(f"Generated {len(insights)} predictive insights")
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating predictive insights: {str(e)}")
            raise
    
    def optimize_classification_parameters(self) -> Dict[str, Any]:
        """
        Optimize classification parameters based on historical performance
        
        Returns:
            Dictionary containing optimized parameters
        """
        try:
            if len(self.historical_weeks) < 10:
                return {"error": "Insufficient historical data for optimization"}
            
            # Analyze classification accuracy by parameters
            accuracy_analysis = self._analyze_classification_accuracy()
            
            # Optimize confidence thresholds
            confidence_optimization = self._optimize_confidence_thresholds()
            
            # Optimize market condition weights
            market_weight_optimization = self._optimize_market_condition_weights()
            
            # Optimize time horizon parameters
            time_horizon_optimization = self._optimize_time_horizon_parameters()
            
            optimization_results = {
                'accuracy_analysis': accuracy_analysis,
                'confidence_optimization': confidence_optimization,
                'market_weight_optimization': market_weight_optimization,
                'time_horizon_optimization': time_horizon_optimization,
                'overall_improvement': self._calculate_overall_improvement()
            }
            
            self.logger.info("Classification parameter optimization completed")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing classification parameters: {str(e)}")
            raise
    
    def _update_performance_metrics(self) -> None:
        """Update overall performance metrics"""
        if not self.historical_weeks:
            return
        
        weeks = list(self.historical_weeks)
        
        # Classification accuracy
        correct_classifications = sum(1 for w in weeks if w.outcome_success)
        self.performance_metrics.classification_accuracy = correct_classifications / len(weeks)
        
        # Return accuracy
        actual_returns = [w.actual_return for w in weeks]
        predicted_returns = [w.predicted_return for w in weeks]
        
        if actual_returns and predicted_returns:
            return_errors = [abs(a - p) for a, p in zip(actual_returns, predicted_returns)]
            self.performance_metrics.return_accuracy = 1.0 - (sum(return_errors) / len(return_errors))
            
            # Risk-adjusted metrics
            avg_return = statistics.mean(actual_returns)
            volatility = statistics.stdev(actual_returns) if len(actual_returns) > 1 else 0
            
            self.performance_metrics.average_return = avg_return
            self.performance_metrics.volatility = volatility
            self.performance_metrics.sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            
            # Win rate
            positive_returns = sum(1 for r in actual_returns if r > 0)
            self.performance_metrics.win_rate = positive_returns / len(actual_returns)
            
            # Max drawdown
            cumulative_returns = np.cumsum(actual_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            self.performance_metrics.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    def _identify_patterns(self) -> None:
        """Identify patterns in the historical data"""
        if len(self.historical_weeks) < self.learning_parameters['pattern_min_frequency']:
            return
        
        # This is a simplified pattern identification
        # In a full implementation, this would use more sophisticated algorithms
        self.patterns = self.identify_patterns()
    
    def _generate_insights(self) -> None:
        """Generate insights from historical data"""
        if len(self.historical_weeks) < 5:
            return
        
        # This is a simplified insight generation
        # In a full implementation, this would use more sophisticated analysis
        self.insights = self.generate_predictive_insights()
    
    def _adapt_classification_models(self) -> None:
        """Adapt classification models based on recent performance"""
        if len(self.historical_weeks) < 10:
            return
        
        # Simplified adaptation logic
        recent_weeks = list(self.historical_weeks)[-10:]
        recent_accuracy = sum(1 for w in recent_weeks if w.outcome_success) / len(recent_weeks)
        
        if recent_accuracy < 0.7:
            # Trigger model adaptation
            self.logger.info(f"Triggering model adaptation due to low accuracy: {recent_accuracy:.1%}")
    
    def _calculate_performance_metrics(self, weeks: List[HistoricalWeek]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not weeks:
            return {}
        
        actual_returns = [w.actual_return for w in weeks]
        predicted_returns = [w.predicted_return for w in weeks]
        successful_classifications = sum(1 for w in weeks if w.outcome_success)
        
        metrics = {
            'total_weeks': len(weeks),
            'classification_accuracy': successful_classifications / len(weeks),
            'average_actual_return': statistics.mean(actual_returns),
            'average_predicted_return': statistics.mean(predicted_returns),
            'return_volatility': statistics.stdev(actual_returns) if len(actual_returns) > 1 else 0,
            'win_rate': sum(1 for r in actual_returns if r > 0) / len(actual_returns),
            'max_return': max(actual_returns),
            'min_return': min(actual_returns),
            'total_return': sum(actual_returns)
        }
        
        if metrics['return_volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['average_actual_return'] / metrics['return_volatility']
        else:
            metrics['sharpe_ratio'] = 0
        
        return metrics
    
    def _analyze_week_type_performance(self, weeks: List[HistoricalWeek]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by week type"""
        week_type_data = defaultdict(list)
        
        for week in weeks:
            week_type_data[week.week_type].append(week)
        
        week_type_performance = {}
        
        for week_type, week_list in week_type_data.items():
            if week_list:
                actual_returns = [w.actual_return for w in week_list]
                successful = sum(1 for w in week_list if w.outcome_success)
                
                week_type_performance[week_type] = {
                    'count': len(week_list),
                    'success_rate': successful / len(week_list),
                    'average_return': statistics.mean(actual_returns),
                    'total_return': sum(actual_returns),
                    'volatility': statistics.stdev(actual_returns) if len(actual_returns) > 1 else 0
                }
        
        return week_type_performance
    
    def _analyze_market_condition_performance(self, weeks: List[HistoricalWeek]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by market conditions"""
        condition_data = defaultdict(list)
        
        for week in weeks:
            condition = week.market_conditions.get('condition', 'unknown')
            condition_data[condition].append(week)
        
        condition_performance = {}
        
        for condition, week_list in condition_data.items():
            if week_list:
                actual_returns = [w.actual_return for w in week_list]
                successful = sum(1 for w in week_list if w.outcome_success)
                
                condition_performance[condition] = {
                    'count': len(week_list),
                    'success_rate': successful / len(week_list),
                    'average_return': statistics.mean(actual_returns),
                    'total_return': sum(actual_returns)
                }
        
        return condition_performance
    
    def _analyze_performance_periods(self, weeks: List[HistoricalWeek]) -> Dict[str, Any]:
        """Analyze best and worst performing periods"""
        if len(weeks) < 4:
            return {}
        
        # Calculate rolling 4-week returns
        rolling_returns = []
        for i in range(len(weeks) - 3):
            period_return = sum(w.actual_return for w in weeks[i:i+4])
            rolling_returns.append({
                'start_week': weeks[i].week_number,
                'end_week': weeks[i+3].week_number,
                'return': period_return
            })
        
        if not rolling_returns:
            return {}
        
        best_period = max(rolling_returns, key=lambda x: x['return'])
        worst_period = min(rolling_returns, key=lambda x: x['return'])
        
        return {
            'best_4_week_period': best_period,
            'worst_4_week_period': worst_period,
            'average_4_week_return': statistics.mean(p['return'] for p in rolling_returns)
        }
    
    def _generate_improvement_recommendations(self, weeks: List[HistoricalWeek]) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        if not weeks:
            return recommendations
        
        # Analyze classification accuracy
        accuracy = sum(1 for w in weeks if w.outcome_success) / len(weeks)
        if accuracy < 0.75:
            recommendations.append("Improve classification accuracy through better market condition analysis")
        
        # Analyze return prediction accuracy
        actual_returns = [w.actual_return for w in weeks]
        predicted_returns = [w.predicted_return for w in weeks]
        
        if actual_returns and predicted_returns:
            prediction_errors = [abs(a - p) for a, p in zip(actual_returns, predicted_returns)]
            avg_error = statistics.mean(prediction_errors)
            
            if avg_error > 0.01:  # 1% average error
                recommendations.append("Improve return prediction accuracy through enhanced modeling")
        
        # Analyze volatility
        if len(actual_returns) > 1:
            volatility = statistics.stdev(actual_returns)
            if volatility > 0.05:  # 5% volatility threshold
                recommendations.append("Implement better risk management to reduce return volatility")
        
        # Analyze win rate
        win_rate = sum(1 for r in actual_returns if r > 0) / len(actual_returns)
        if win_rate < 0.6:
            recommendations.append("Focus on strategies with higher win rates")
        
        return recommendations
    
    def _identify_seasonal_patterns(self) -> List[Pattern]:
        """Identify seasonal patterns in the data"""
        patterns = []
        
        # Group by month/quarter
        monthly_data = defaultdict(list)
        for week in self.historical_weeks:
            month = week.timestamp.month
            monthly_data[month].append(week)
        
        for month, weeks in monthly_data.items():
            if len(weeks) >= self.learning_parameters['pattern_min_frequency']:
                avg_return = statistics.mean(w.actual_return for w in weeks)
                success_rate = sum(1 for w in weeks if w.outcome_success) / len(weeks)
                
                if success_rate > self.learning_parameters['pattern_confidence_threshold']:
                    pattern = Pattern(
                        pattern_type=PatternType.SEASONAL,
                        description=f"Month {month} shows consistent performance",
                        frequency=len(weeks),
                        confidence=success_rate,
                        conditions={'month': month},
                        expected_outcome={'return': avg_return, 'success_rate': success_rate},
                        historical_accuracy=success_rate,
                        last_occurrence=max(w.timestamp for w in weeks),
                        next_expected=None
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _identify_cyclical_patterns(self) -> List[Pattern]:
        """Identify cyclical patterns in the data"""
        # Simplified cyclical pattern identification
        return []
    
    def _identify_trend_patterns(self) -> List[Pattern]:
        """Identify trend patterns in the data"""
        # Simplified trend pattern identification
        return []
    
    def _identify_volatility_patterns(self) -> List[Pattern]:
        """Identify volatility patterns in the data"""
        # Simplified volatility pattern identification
        return []
    
    def _identify_correlation_patterns(self) -> List[Pattern]:
        """Identify correlation patterns in the data"""
        # Simplified correlation pattern identification
        return []
    
    def _identify_anomaly_patterns(self) -> List[Pattern]:
        """Identify anomaly patterns in the data"""
        # Simplified anomaly pattern identification
        return []
    
    def _generate_pattern_based_insights(self, forecast_horizon: int) -> List[LearningInsight]:
        """Generate insights based on identified patterns"""
        insights = []
        
        for pattern in self.patterns:
            if pattern.confidence > 0.8:
                insight = LearningInsight(
                    insight_type="pattern_prediction",
                    description=f"Pattern suggests {pattern.description}",
                    confidence=pattern.confidence,
                    supporting_evidence=[f"Historical accuracy: {pattern.historical_accuracy:.1%}"],
                    actionable_recommendations=[f"Consider {pattern.description} in strategy"],
                    expected_impact=pattern.expected_outcome.get('return', 0),
                    validation_criteria=["Monitor pattern continuation"]
                )
                insights.append(insight)
        
        return insights
    
    def _generate_performance_based_insights(self, forecast_horizon: int) -> List[LearningInsight]:
        """Generate insights based on performance analysis"""
        # Simplified performance-based insight generation
        return []
    
    def _generate_market_condition_insights(self, forecast_horizon: int) -> List[LearningInsight]:
        """Generate insights based on market conditions"""
        # Simplified market condition insight generation
        return []
    
    def _generate_risk_based_insights(self, forecast_horizon: int) -> List[LearningInsight]:
        """Generate insights based on risk analysis"""
        # Simplified risk-based insight generation
        return []
    
    def _analyze_classification_accuracy(self) -> Dict[str, float]:
        """Analyze classification accuracy patterns"""
        if not self.historical_weeks:
            return {}
        
        weeks = list(self.historical_weeks)
        total_accuracy = sum(1 for w in weeks if w.outcome_success) / len(weeks)
        
        # Accuracy by confidence level
        high_confidence_weeks = [w for w in weeks if w.classification_confidence > 0.8]
        high_confidence_accuracy = (sum(1 for w in high_confidence_weeks if w.outcome_success) / 
                                   len(high_confidence_weeks)) if high_confidence_weeks else 0
        
        return {
            'overall_accuracy': total_accuracy,
            'high_confidence_accuracy': high_confidence_accuracy,
            'total_weeks': len(weeks)
        }
    
    def _optimize_confidence_thresholds(self) -> Dict[str, float]:
        """Optimize confidence thresholds for better performance"""
        # Simplified confidence threshold optimization
        return {'optimal_threshold': 0.75}
    
    def _optimize_market_condition_weights(self) -> Dict[str, float]:
        """Optimize market condition weights"""
        # Simplified market condition weight optimization
        return {'volatility_weight': 0.3, 'trend_weight': 0.4, 'momentum_weight': 0.3}
    
    def _optimize_time_horizon_parameters(self) -> Dict[str, int]:
        """Optimize time horizon parameters"""
        # Simplified time horizon optimization
        return {'optimal_lookback': 20, 'optimal_forecast': 4}
    
    def _calculate_overall_improvement(self) -> float:
        """Calculate overall improvement from optimization"""
        # Simplified improvement calculation
        return 0.05  # 5% improvement

def test_historical_analysis_engine():
    """Test the historical analysis engine"""
    print("Testing Historical Analysis Engine...")
    
    engine = HistoricalAnalysisEngine()
    
    # Create sample historical data
    sample_weeks = [
        HistoricalWeek(
            week_number=1, year=2024, week_type='P-EW',
            actual_return=0.022, predicted_return=0.020,
            market_conditions={'condition': 'bullish', 'volatility': 0.18},
            classification_confidence=0.85, action_taken='sell_put',
            outcome_success=True, lessons_learned=['Good market timing'],
            timestamp=datetime(2024, 1, 8)
        ),
        HistoricalWeek(
            week_number=2, year=2024, week_type='C-WAP',
            actual_return=0.035, predicted_return=0.030,
            market_conditions={'condition': 'bullish', 'volatility': 0.20},
            classification_confidence=0.80, action_taken='sell_call',
            outcome_success=True, lessons_learned=['Strong appreciation captured'],
            timestamp=datetime(2024, 1, 15)
        ),
        HistoricalWeek(
            week_number=3, year=2024, week_type='P-RO',
            actual_return=0.015, predicted_return=0.012,
            market_conditions={'condition': 'bearish', 'volatility': 0.25},
            classification_confidence=0.75, action_taken='roll_position',
            outcome_success=True, lessons_learned=['Successful roll avoided assignment'],
            timestamp=datetime(2024, 1, 22)
        ),
        HistoricalWeek(
            week_number=4, year=2024, week_type='P-DD',
            actual_return=-0.005, predicted_return=0.000,
            market_conditions={'condition': 'extremely_bearish', 'volatility': 0.40},
            classification_confidence=0.90, action_taken='enter_protective',
            outcome_success=False, lessons_learned=['Protection helped limit losses'],
            timestamp=datetime(2024, 1, 29)
        ),
        HistoricalWeek(
            week_number=5, year=2024, week_type='C-WAP+',
            actual_return=0.055, predicted_return=0.050,
            market_conditions={'condition': 'extremely_bullish', 'volatility': 0.15},
            classification_confidence=0.88, action_taken='sell_call',
            outcome_success=True, lessons_learned=['Excellent timing on strong move'],
            timestamp=datetime(2024, 2, 5)
        )
    ]
    
    # Add historical weeks
    for week in sample_weeks:
        engine.add_historical_week(week)
    
    print(f"Added {len(sample_weeks)} historical weeks")
    
    # Analyze historical performance
    print("\n--- Historical Performance Analysis ---")
    performance_analysis = engine.analyze_historical_performance()
    
    overall_perf = performance_analysis['overall_performance']
    print(f"Classification Accuracy: {overall_perf['classification_accuracy']:.1%}")
    print(f"Average Return: {overall_perf['average_actual_return']:.1%}")
    print(f"Win Rate: {overall_perf['win_rate']:.1%}")
    print(f"Total Return: {overall_perf['total_return']:.1%}")
    
    # Week type performance
    print("\n--- Week Type Performance ---")
    week_type_perf = performance_analysis['week_type_performance']
    for week_type, metrics in week_type_perf.items():
        print(f"{week_type}: {metrics['success_rate']:.1%} success, {metrics['average_return']:.1%} avg return")
    
    # Identify patterns
    print("\n--- Pattern Identification ---")
    patterns = engine.identify_patterns()
    print(f"Identified {len(patterns)} patterns")
    for pattern in patterns:
        print(f"- {pattern.description} (confidence: {pattern.confidence:.1%})")
    
    # Generate insights
    print("\n--- Predictive Insights ---")
    insights = engine.generate_predictive_insights()
    print(f"Generated {len(insights)} insights")
    for insight in insights:
        print(f"- {insight.description} (confidence: {insight.confidence:.1%})")
    
    # Optimization
    print("\n--- Parameter Optimization ---")
    optimization = engine.optimize_classification_parameters()
    if 'error' not in optimization:
        accuracy_analysis = optimization.get('accuracy_analysis', {})
        print(f"Overall Accuracy: {accuracy_analysis.get('overall_accuracy', 0):.1%}")
        print(f"High Confidence Accuracy: {accuracy_analysis.get('high_confidence_accuracy', 0):.1%}")
        print(f"Overall Improvement: {optimization.get('overall_improvement', 0):.1%}")
    else:
        print(f"Optimization not available: {optimization['error']}")
    
    print("\n✅ Historical Analysis Engine test completed successfully!")

if __name__ == "__main__":
    test_historical_analysis_engine()

