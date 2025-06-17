"""
ALL-USE Account Analytics Engine
Advanced analytics and predictive capabilities for account management

This module provides sophisticated analytics, performance analysis, trend detection,
and predictive modeling for the ALL-USE Account Management System.
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import statistics
from collections import defaultdict
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyticsType(Enum):
    """Types of analytics available"""
    PERFORMANCE = "performance"
    TREND = "trend"
    RISK = "risk"
    PREDICTIVE = "predictive"
    COMPARATIVE = "comparative"
    OPTIMIZATION = "optimization"

class TimeFrame(Enum):
    """Time frame options for analytics"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"

@dataclass
class AnalyticsResult:
    """Result of analytics operation"""
    analytics_id: str
    account_id: str
    analytics_type: AnalyticsType
    timeframe: TimeFrame
    metrics: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    confidence_score: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    """Performance metrics for account analysis"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    average_trade: float
    total_trades: int
    profitable_trades: int

@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    trend_direction: str  # "bullish", "bearish", "sideways"
    trend_strength: float  # 0-1 scale
    momentum: float
    support_level: float
    resistance_level: float
    trend_duration: int  # days
    reversal_probability: float

@dataclass
class RiskAssessment:
    """Risk assessment results"""
    risk_score: float  # 0-100 scale
    risk_level: str  # "low", "medium", "high", "critical"
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    market_risk: float

@dataclass
class PredictiveModel:
    """Predictive modeling results"""
    predicted_return: float
    prediction_confidence: float
    time_horizon: int  # days
    scenario_analysis: Dict[str, float]
    probability_distribution: Dict[str, float]
    key_factors: List[str]
    model_accuracy: float

class AccountAnalyticsEngine:
    """
    Advanced analytics engine for ALL-USE Account Management System
    
    Provides comprehensive analytics including:
    - Performance analysis and benchmarking
    - Trend detection and forecasting
    - Risk assessment and optimization
    - Predictive modeling
    - Comparative analysis
    - Real-time analytics dashboard
    """
    
    def __init__(self, db_path: str = "/home/ubuntu/AGENT_ALLUSE_V1/data/account_management.db"):
        """Initialize the analytics engine"""
        self.db_path = db_path
        self.analytics_cache = {}
        self.model_cache = {}
        self._initialize_analytics_schema()
        logger.info("Account Analytics Engine initialized")
    
    def _initialize_analytics_schema(self):
        """Initialize analytics database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Analytics results table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics_results (
                    analytics_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    analytics_type TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    insights TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (account_id) REFERENCES accounts (account_id)
                )
                """)
                
                # Performance metrics table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    total_return REAL,
                    annualized_return REAL,
                    volatility REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    average_trade REAL,
                    total_trades INTEGER,
                    profitable_trades INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (account_id) REFERENCES accounts (account_id)
                )
                """)
                
                # Trend analysis table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS trend_analysis (
                    trend_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    trend_direction TEXT NOT NULL,
                    trend_strength REAL NOT NULL,
                    momentum REAL,
                    support_level REAL,
                    resistance_level REAL,
                    trend_duration INTEGER,
                    reversal_probability REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (account_id) REFERENCES accounts (account_id)
                )
                """)
                
                # Risk assessment table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_assessments (
                    risk_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    var_95 REAL,
                    expected_shortfall REAL,
                    correlation_risk REAL,
                    concentration_risk REAL,
                    liquidity_risk REAL,
                    market_risk REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (account_id) REFERENCES accounts (account_id)
                )
                """)
                
                # Predictive models table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictive_models (
                    model_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    predicted_return REAL NOT NULL,
                    prediction_confidence REAL NOT NULL,
                    time_horizon INTEGER NOT NULL,
                    scenario_analysis TEXT,
                    probability_distribution TEXT,
                    key_factors TEXT,
                    model_accuracy REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (account_id) REFERENCES accounts (account_id)
                )
                """)
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_analytics_account_type ON analytics_results (account_id, analytics_type)",
                    "CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics_results (timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_performance_account_timeframe ON performance_metrics (account_id, timeframe)",
                    "CREATE INDEX IF NOT EXISTS idx_trend_account ON trend_analysis (account_id)",
                    "CREATE INDEX IF NOT EXISTS idx_risk_account ON risk_assessments (account_id)",
                    "CREATE INDEX IF NOT EXISTS idx_predictive_account ON predictive_models (account_id)"
                ]
                
                for index in indexes:
                    cursor.execute(index)
                
                conn.commit()
                logger.info("Analytics database schema initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing analytics schema: {e}")
            raise
    
    def analyze_account_performance(self, account_id: str, timeframe: TimeFrame = TimeFrame.ALL_TIME) -> PerformanceMetrics:
        """
        Analyze account performance with comprehensive metrics
        
        Args:
            account_id: Account to analyze
            timeframe: Time period for analysis
            
        Returns:
            PerformanceMetrics object with detailed performance data
        """
        try:
            logger.info(f"Analyzing performance for account {account_id} over {timeframe.value}")
            
            # Get account transaction history
            transactions = self._get_account_transactions(account_id, timeframe)
            if not transactions:
                logger.warning(f"No transactions found for account {account_id}")
                return self._create_default_performance_metrics()
            
            # Calculate performance metrics
            returns = self._calculate_returns(transactions)
            metrics = self._calculate_performance_metrics(returns, transactions)
            
            # Store results
            self._store_performance_metrics(account_id, timeframe, metrics)
            
            logger.info(f"Performance analysis completed for account {account_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing account performance: {e}")
            return self._create_default_performance_metrics()
    
    def detect_trends(self, account_id: str, lookback_days: int = 30) -> TrendAnalysis:
        """
        Detect trends in account performance
        
        Args:
            account_id: Account to analyze
            lookback_days: Number of days to analyze
            
        Returns:
            TrendAnalysis object with trend information
        """
        try:
            logger.info(f"Detecting trends for account {account_id} over {lookback_days} days")
            
            # Get historical data
            historical_data = self._get_historical_performance(account_id, lookback_days)
            if len(historical_data) < 5:
                logger.warning(f"Insufficient data for trend analysis: {len(historical_data)} points")
                return self._create_default_trend_analysis()
            
            # Analyze trends
            trend_analysis = self._analyze_trends(historical_data)
            
            # Store results
            self._store_trend_analysis(account_id, trend_analysis)
            
            logger.info(f"Trend analysis completed for account {account_id}")
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error detecting trends: {e}")
            return self._create_default_trend_analysis()
    
    def assess_risk(self, account_id: str, confidence_level: float = 0.95) -> RiskAssessment:
        """
        Comprehensive risk assessment for account
        
        Args:
            account_id: Account to assess
            confidence_level: Confidence level for VaR calculation
            
        Returns:
            RiskAssessment object with risk metrics
        """
        try:
            logger.info(f"Assessing risk for account {account_id}")
            
            # Get account data
            account_data = self._get_account_risk_data(account_id)
            if not account_data:
                logger.warning(f"No risk data available for account {account_id}")
                return self._create_default_risk_assessment()
            
            # Calculate risk metrics
            risk_assessment = self._calculate_risk_metrics(account_data, confidence_level)
            
            # Store results
            self._store_risk_assessment(account_id, risk_assessment)
            
            logger.info(f"Risk assessment completed for account {account_id}")
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return self._create_default_risk_assessment()
    
    def generate_predictions(self, account_id: str, time_horizon: int = 30) -> PredictiveModel:
        """
        Generate predictive model for account performance
        
        Args:
            account_id: Account to model
            time_horizon: Prediction horizon in days
            
        Returns:
            PredictiveModel object with predictions
        """
        try:
            logger.info(f"Generating predictions for account {account_id} over {time_horizon} days")
            
            # Get historical data for modeling
            historical_data = self._get_modeling_data(account_id)
            if len(historical_data) < 10:
                logger.warning(f"Insufficient data for predictive modeling: {len(historical_data)} points")
                return self._create_default_predictive_model(time_horizon)
            
            # Build predictive model
            model = self._build_predictive_model(historical_data, time_horizon)
            
            # Store results
            self._store_predictive_model(account_id, model)
            
            logger.info(f"Predictive modeling completed for account {account_id}")
            return model
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return self._create_default_predictive_model(time_horizon)
    
    def compare_accounts(self, account_ids: List[str], timeframe: TimeFrame = TimeFrame.MONTHLY) -> Dict[str, Any]:
        """
        Compare performance across multiple accounts
        
        Args:
            account_ids: List of accounts to compare
            timeframe: Time period for comparison
            
        Returns:
            Dictionary with comparative analysis
        """
        try:
            logger.info(f"Comparing {len(account_ids)} accounts over {timeframe.value}")
            
            comparison_results = {
                "accounts": account_ids,
                "timeframe": timeframe.value,
                "performance_comparison": {},
                "rankings": {},
                "insights": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Analyze each account
            account_metrics = {}
            for account_id in account_ids:
                metrics = self.analyze_account_performance(account_id, timeframe)
                account_metrics[account_id] = metrics
            
            # Generate comparative analysis
            comparison_results["performance_comparison"] = self._generate_comparative_analysis(account_metrics)
            comparison_results["rankings"] = self._generate_rankings(account_metrics)
            comparison_results["insights"] = self._generate_comparative_insights(account_metrics)
            
            logger.info(f"Account comparison completed for {len(account_ids)} accounts")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error comparing accounts: {e}")
            return {"error": str(e), "accounts": account_ids}
    
    def get_analytics_dashboard(self, account_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive analytics dashboard for account
        
        Args:
            account_id: Account to analyze
            
        Returns:
            Dictionary with dashboard data
        """
        try:
            logger.info(f"Generating analytics dashboard for account {account_id}")
            
            dashboard = {
                "account_id": account_id,
                "timestamp": datetime.now().isoformat(),
                "performance": {},
                "trends": {},
                "risk": {},
                "predictions": {},
                "summary": {},
                "alerts": []
            }
            
            # Get all analytics
            dashboard["performance"] = asdict(self.analyze_account_performance(account_id))
            dashboard["trends"] = asdict(self.detect_trends(account_id))
            dashboard["risk"] = asdict(self.assess_risk(account_id))
            dashboard["predictions"] = asdict(self.generate_predictions(account_id))
            
            # Generate summary and alerts
            dashboard["summary"] = self._generate_dashboard_summary(dashboard)
            dashboard["alerts"] = self._generate_alerts(dashboard)
            
            logger.info(f"Analytics dashboard generated for account {account_id}")
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating analytics dashboard: {e}")
            return {"error": str(e), "account_id": account_id}
    
    def _get_account_transactions(self, account_id: str, timeframe: TimeFrame) -> List[Dict]:
        """Get account transactions for specified timeframe"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate date range
                end_date = datetime.now()
                if timeframe == TimeFrame.DAILY:
                    start_date = end_date - timedelta(days=1)
                elif timeframe == TimeFrame.WEEKLY:
                    start_date = end_date - timedelta(weeks=1)
                elif timeframe == TimeFrame.MONTHLY:
                    start_date = end_date - timedelta(days=30)
                elif timeframe == TimeFrame.QUARTERLY:
                    start_date = end_date - timedelta(days=90)
                elif timeframe == TimeFrame.YEARLY:
                    start_date = end_date - timedelta(days=365)
                else:  # ALL_TIME
                    start_date = datetime(2020, 1, 1)
                
                cursor.execute("""
                SELECT * FROM transactions 
                WHERE account_id = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
                """, (account_id, start_date, end_date))
                
                columns = [desc[0] for desc in cursor.description]
                transactions = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                return transactions
                
        except Exception as e:
            logger.error(f"Error getting account transactions: {e}")
            return []
    
    def _calculate_returns(self, transactions: List[Dict]) -> List[float]:
        """Calculate returns from transaction history"""
        if not transactions:
            return []
        
        # Simple return calculation based on transaction amounts
        returns = []
        running_balance = 0
        
        for transaction in transactions:
            amount = float(transaction.get('amount', 0))
            if running_balance > 0:
                return_pct = amount / running_balance
                returns.append(return_pct)
            running_balance += amount
        
        return returns
    
    def _calculate_performance_metrics(self, returns: List[float], transactions: List[Dict]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not returns or not transactions:
            return self._create_default_performance_metrics()
        
        # Calculate basic metrics
        total_return = sum(returns) if returns else 0.0
        avg_return = statistics.mean(returns) if returns else 0.0
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0
        
        # Annualized return (assuming daily returns)
        annualized_return = avg_return * 252 if avg_return else 0.0
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        
        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # Trade statistics
        positive_returns = [r for r in returns if r > 0]
        total_trades = len(returns)
        profitable_trades = len(positive_returns)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0.0
        
        # Profit factor
        gross_profit = sum(positive_returns) if positive_returns else 0.0
        gross_loss = abs(sum([r for r in returns if r < 0])) if returns else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # Average trade
        average_trade = avg_return
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_trade=average_trade,
            total_trades=total_trades,
            profitable_trades=profitable_trades
        )
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
        
        cumulative = [1.0]
        for r in returns:
            cumulative.append(cumulative[-1] * (1 + r))
        
        max_dd = 0.0
        peak = cumulative[0]
        
        for value in cumulative[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _get_historical_performance(self, account_id: str, lookback_days: int) -> List[Dict]:
        """Get historical performance data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                start_date = datetime.now() - timedelta(days=lookback_days)
                
                cursor.execute("""
                SELECT DATE(timestamp) as date, SUM(amount) as daily_pnl
                FROM transactions 
                WHERE account_id = ? AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
                """, (account_id, start_date))
                
                columns = [desc[0] for desc in cursor.description]
                data = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                return data
                
        except Exception as e:
            logger.error(f"Error getting historical performance: {e}")
            return []
    
    def _analyze_trends(self, historical_data: List[Dict]) -> TrendAnalysis:
        """Analyze trends in historical data"""
        if len(historical_data) < 3:
            return self._create_default_trend_analysis()
        
        # Extract values
        values = [float(d.get('daily_pnl', 0)) for d in historical_data]
        
        # Calculate trend direction using linear regression
        x = list(range(len(values)))
        n = len(values)
        
        if n < 2:
            return self._create_default_trend_analysis()
        
        # Simple linear regression
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        
        # Determine trend direction
        if slope > 0.01:
            trend_direction = "bullish"
        elif slope < -0.01:
            trend_direction = "bearish"
        else:
            trend_direction = "sideways"
        
        # Calculate trend strength (0-1)
        trend_strength = min(abs(slope) * 100, 1.0)
        
        # Calculate momentum (rate of change)
        momentum = (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
        
        # Support and resistance levels
        support_level = min(values)
        resistance_level = max(values)
        
        # Trend duration (simplified)
        trend_duration = len(historical_data)
        
        # Reversal probability (simplified heuristic)
        recent_volatility = statistics.stdev(values[-5:]) if len(values) >= 5 else 0
        avg_volatility = statistics.stdev(values) if len(values) > 1 else 0
        reversal_probability = min(recent_volatility / avg_volatility, 1.0) if avg_volatility > 0 else 0.5
        
        return TrendAnalysis(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            momentum=momentum,
            support_level=support_level,
            resistance_level=resistance_level,
            trend_duration=trend_duration,
            reversal_probability=reversal_probability
        )
    
    def _get_account_risk_data(self, account_id: str) -> Dict:
        """Get account data for risk assessment"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get account information
                cursor.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
                account = cursor.fetchone()
                
                if not account:
                    return {}
                
                # Get recent transactions
                cursor.execute("""
                SELECT * FROM transactions 
                WHERE account_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 100
                """, (account_id,))
                
                transactions = cursor.fetchall()
                
                return {
                    "account": account,
                    "transactions": transactions,
                    "balance": float(account[3]) if account else 0.0  # Assuming balance is at index 3
                }
                
        except Exception as e:
            logger.error(f"Error getting account risk data: {e}")
            return {}
    
    def _calculate_risk_metrics(self, account_data: Dict, confidence_level: float) -> RiskAssessment:
        """Calculate comprehensive risk metrics"""
        if not account_data or not account_data.get('transactions'):
            return self._create_default_risk_assessment()
        
        transactions = account_data['transactions']
        balance = account_data.get('balance', 0.0)
        
        # Extract returns from transactions
        returns = []
        for i, txn in enumerate(transactions):
            if i > 0:
                prev_amount = float(transactions[i-1][2]) if len(transactions[i-1]) > 2 else 0  # Assuming amount at index 2
                curr_amount = float(txn[2]) if len(txn) > 2 else 0
                if prev_amount != 0:
                    returns.append((curr_amount - prev_amount) / prev_amount)
        
        if not returns:
            return self._create_default_risk_assessment()
        
        # Calculate VaR (Value at Risk)
        returns_sorted = sorted(returns)
        var_index = int((1 - confidence_level) * len(returns_sorted))
        var_95 = abs(returns_sorted[var_index]) * balance if var_index < len(returns_sorted) else 0.0
        
        # Expected Shortfall (Conditional VaR)
        tail_returns = returns_sorted[:var_index] if var_index > 0 else [0]
        expected_shortfall = abs(statistics.mean(tail_returns)) * balance if tail_returns else 0.0
        
        # Risk components (simplified calculations)
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0
        
        correlation_risk = min(volatility * 0.3, 1.0)  # Simplified
        concentration_risk = min(balance / 100000, 1.0)  # Risk increases with account size
        liquidity_risk = 0.1  # Assumed low for now
        market_risk = min(volatility * 0.5, 1.0)
        
        # Overall risk score (0-100)
        risk_components = [correlation_risk, concentration_risk, liquidity_risk, market_risk]
        risk_score = sum(risk_components) / len(risk_components) * 100
        
        # Risk level classification
        if risk_score < 25:
            risk_level = "low"
        elif risk_score < 50:
            risk_level = "medium"
        elif risk_score < 75:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        return RiskAssessment(
            risk_score=risk_score,
            risk_level=risk_level,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk,
            market_risk=market_risk
        )
    
    def _get_modeling_data(self, account_id: str) -> List[Dict]:
        """Get data for predictive modeling"""
        return self._get_historical_performance(account_id, 90)  # 90 days of data
    
    def _build_predictive_model(self, historical_data: List[Dict], time_horizon: int) -> PredictiveModel:
        """Build predictive model for account performance"""
        if len(historical_data) < 5:
            return self._create_default_predictive_model(time_horizon)
        
        # Extract values
        values = [float(d.get('daily_pnl', 0)) for d in historical_data]
        
        # Simple trend-based prediction
        recent_values = values[-10:] if len(values) >= 10 else values
        avg_recent = statistics.mean(recent_values) if recent_values else 0.0
        
        # Predicted return (simplified)
        predicted_return = avg_recent * time_horizon
        
        # Prediction confidence based on consistency
        volatility = statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0
        prediction_confidence = max(0.1, 1.0 - min(volatility / abs(avg_recent), 1.0)) if avg_recent != 0 else 0.5
        
        # Scenario analysis
        scenario_analysis = {
            "bull_case": predicted_return * 1.5,
            "base_case": predicted_return,
            "bear_case": predicted_return * 0.5
        }
        
        # Probability distribution (simplified)
        probability_distribution = {
            "very_positive": 0.1,
            "positive": 0.3,
            "neutral": 0.2,
            "negative": 0.3,
            "very_negative": 0.1
        }
        
        # Key factors (simplified)
        key_factors = ["historical_trend", "recent_volatility", "account_balance"]
        
        # Model accuracy (simplified)
        model_accuracy = prediction_confidence * 0.8  # Conservative estimate
        
        return PredictiveModel(
            predicted_return=predicted_return,
            prediction_confidence=prediction_confidence,
            time_horizon=time_horizon,
            scenario_analysis=scenario_analysis,
            probability_distribution=probability_distribution,
            key_factors=key_factors,
            model_accuracy=model_accuracy
        )
    
    def _generate_comparative_analysis(self, account_metrics: Dict[str, PerformanceMetrics]) -> Dict:
        """Generate comparative analysis between accounts"""
        if not account_metrics:
            return {}
        
        comparison = {}
        
        # Compare key metrics
        metrics_to_compare = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        for metric in metrics_to_compare:
            values = {acc_id: getattr(metrics, metric) for acc_id, metrics in account_metrics.items()}
            comparison[metric] = {
                "values": values,
                "best": max(values.items(), key=lambda x: x[1]) if values else None,
                "worst": min(values.items(), key=lambda x: x[1]) if values else None,
                "average": statistics.mean(values.values()) if values else 0
            }
        
        return comparison
    
    def _generate_rankings(self, account_metrics: Dict[str, PerformanceMetrics]) -> Dict:
        """Generate account rankings"""
        if not account_metrics:
            return {}
        
        # Rank by different criteria
        rankings = {}
        
        # By total return
        total_returns = {acc_id: metrics.total_return for acc_id, metrics in account_metrics.items()}
        rankings["by_total_return"] = sorted(total_returns.items(), key=lambda x: x[1], reverse=True)
        
        # By Sharpe ratio
        sharpe_ratios = {acc_id: metrics.sharpe_ratio for acc_id, metrics in account_metrics.items()}
        rankings["by_sharpe_ratio"] = sorted(sharpe_ratios.items(), key=lambda x: x[1], reverse=True)
        
        # By win rate
        win_rates = {acc_id: metrics.win_rate for acc_id, metrics in account_metrics.items()}
        rankings["by_win_rate"] = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _generate_comparative_insights(self, account_metrics: Dict[str, PerformanceMetrics]) -> List[str]:
        """Generate insights from comparative analysis"""
        insights = []
        
        if not account_metrics:
            return insights
        
        # Find best and worst performers
        total_returns = {acc_id: metrics.total_return for acc_id, metrics in account_metrics.items()}
        if total_returns:
            best_performer = max(total_returns.items(), key=lambda x: x[1])
            worst_performer = min(total_returns.items(), key=lambda x: x[1])
            
            insights.append(f"Best performer: {best_performer[0]} with {best_performer[1]:.2%} return")
            insights.append(f"Worst performer: {worst_performer[0]} with {worst_performer[1]:.2%} return")
        
        # Analyze risk-adjusted returns
        sharpe_ratios = {acc_id: metrics.sharpe_ratio for acc_id, metrics in account_metrics.items()}
        if sharpe_ratios:
            best_risk_adjusted = max(sharpe_ratios.items(), key=lambda x: x[1])
            insights.append(f"Best risk-adjusted return: {best_risk_adjusted[0]} with Sharpe ratio of {best_risk_adjusted[1]:.2f}")
        
        return insights
    
    def _generate_dashboard_summary(self, dashboard: Dict) -> Dict:
        """Generate summary for analytics dashboard"""
        summary = {
            "overall_status": "healthy",
            "key_metrics": {},
            "alerts_count": len(dashboard.get("alerts", [])),
            "last_updated": dashboard.get("timestamp")
        }
        
        # Extract key metrics
        performance = dashboard.get("performance", {})
        if performance:
            summary["key_metrics"]["total_return"] = performance.get("total_return", 0)
            summary["key_metrics"]["sharpe_ratio"] = performance.get("sharpe_ratio", 0)
            summary["key_metrics"]["win_rate"] = performance.get("win_rate", 0)
        
        risk = dashboard.get("risk", {})
        if risk:
            summary["key_metrics"]["risk_level"] = risk.get("risk_level", "unknown")
            summary["key_metrics"]["risk_score"] = risk.get("risk_score", 0)
        
        return summary
    
    def _generate_alerts(self, dashboard: Dict) -> List[str]:
        """Generate alerts based on dashboard data"""
        alerts = []
        
        # Risk-based alerts
        risk = dashboard.get("risk", {})
        if risk:
            risk_level = risk.get("risk_level", "")
            if risk_level in ["high", "critical"]:
                alerts.append(f"High risk level detected: {risk_level}")
            
            risk_score = risk.get("risk_score", 0)
            if risk_score > 75:
                alerts.append(f"Risk score elevated: {risk_score:.1f}/100")
        
        # Performance alerts
        performance = dashboard.get("performance", {})
        if performance:
            max_drawdown = performance.get("max_drawdown", 0)
            if max_drawdown > 0.2:  # 20% drawdown
                alerts.append(f"Significant drawdown detected: {max_drawdown:.1%}")
            
            sharpe_ratio = performance.get("sharpe_ratio", 0)
            if sharpe_ratio < 0:
                alerts.append("Negative risk-adjusted returns")
        
        # Trend alerts
        trends = dashboard.get("trends", {})
        if trends:
            reversal_prob = trends.get("reversal_probability", 0)
            if reversal_prob > 0.7:
                alerts.append(f"High trend reversal probability: {reversal_prob:.1%}")
        
        return alerts
    
    def _store_performance_metrics(self, account_id: str, timeframe: TimeFrame, metrics: PerformanceMetrics):
        """Store performance metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                metric_id = f"perf_{account_id}_{timeframe.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                cursor.execute("""
                INSERT INTO performance_metrics (
                    metric_id, account_id, timeframe, total_return, annualized_return,
                    volatility, sharpe_ratio, max_drawdown, win_rate, profit_factor,
                    average_trade, total_trades, profitable_trades
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric_id, account_id, timeframe.value, metrics.total_return,
                    metrics.annualized_return, metrics.volatility, metrics.sharpe_ratio,
                    metrics.max_drawdown, metrics.win_rate, metrics.profit_factor,
                    metrics.average_trade, metrics.total_trades, metrics.profitable_trades
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
    
    def _store_trend_analysis(self, account_id: str, trend_analysis: TrendAnalysis):
        """Store trend analysis in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                trend_id = f"trend_{account_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                cursor.execute("""
                INSERT INTO trend_analysis (
                    trend_id, account_id, trend_direction, trend_strength, momentum,
                    support_level, resistance_level, trend_duration, reversal_probability
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trend_id, account_id, trend_analysis.trend_direction, trend_analysis.trend_strength,
                    trend_analysis.momentum, trend_analysis.support_level, trend_analysis.resistance_level,
                    trend_analysis.trend_duration, trend_analysis.reversal_probability
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing trend analysis: {e}")
    
    def _store_risk_assessment(self, account_id: str, risk_assessment: RiskAssessment):
        """Store risk assessment in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                risk_id = f"risk_{account_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                cursor.execute("""
                INSERT INTO risk_assessments (
                    risk_id, account_id, risk_score, risk_level, var_95, expected_shortfall,
                    correlation_risk, concentration_risk, liquidity_risk, market_risk
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    risk_id, account_id, risk_assessment.risk_score, risk_assessment.risk_level,
                    risk_assessment.var_95, risk_assessment.expected_shortfall,
                    risk_assessment.correlation_risk, risk_assessment.concentration_risk,
                    risk_assessment.liquidity_risk, risk_assessment.market_risk
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing risk assessment: {e}")
    
    def _store_predictive_model(self, account_id: str, model: PredictiveModel):
        """Store predictive model in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                model_id = f"pred_{account_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                cursor.execute("""
                INSERT INTO predictive_models (
                    model_id, account_id, predicted_return, prediction_confidence, time_horizon,
                    scenario_analysis, probability_distribution, key_factors, model_accuracy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_id, account_id, model.predicted_return, model.prediction_confidence,
                    model.time_horizon, json.dumps(model.scenario_analysis),
                    json.dumps(model.probability_distribution), json.dumps(model.key_factors),
                    model.model_accuracy
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing predictive model: {e}")
    
    def _create_default_performance_metrics(self) -> PerformanceMetrics:
        """Create default performance metrics"""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            average_trade=0.0,
            total_trades=0,
            profitable_trades=0
        )
    
    def _create_default_trend_analysis(self) -> TrendAnalysis:
        """Create default trend analysis"""
        return TrendAnalysis(
            trend_direction="sideways",
            trend_strength=0.0,
            momentum=0.0,
            support_level=0.0,
            resistance_level=0.0,
            trend_duration=0,
            reversal_probability=0.5
        )
    
    def _create_default_risk_assessment(self) -> RiskAssessment:
        """Create default risk assessment"""
        return RiskAssessment(
            risk_score=50.0,
            risk_level="medium",
            var_95=0.0,
            expected_shortfall=0.0,
            correlation_risk=0.0,
            concentration_risk=0.0,
            liquidity_risk=0.0,
            market_risk=0.0
        )
    
    def _create_default_predictive_model(self, time_horizon: int) -> PredictiveModel:
        """Create default predictive model"""
        return PredictiveModel(
            predicted_return=0.0,
            prediction_confidence=0.5,
            time_horizon=time_horizon,
            scenario_analysis={"bull_case": 0.0, "base_case": 0.0, "bear_case": 0.0},
            probability_distribution={"positive": 0.5, "negative": 0.5},
            key_factors=["insufficient_data"],
            model_accuracy=0.5
        )

# Example usage and testing
if __name__ == "__main__":
    # Initialize analytics engine
    analytics = AccountAnalyticsEngine()
    
    # Test with sample account (assuming it exists)
    test_account_id = "test_gen_acc_001"
    
    print("=== Account Analytics Engine Test ===")
    print(f"Testing with account: {test_account_id}")
    
    # Test performance analysis
    print("\n1. Performance Analysis:")
    performance = analytics.analyze_account_performance(test_account_id)
    print(f"   Total Return: {performance.total_return:.2%}")
    print(f"   Sharpe Ratio: {performance.sharpe_ratio:.2f}")
    print(f"   Win Rate: {performance.win_rate:.2%}")
    
    # Test trend detection
    print("\n2. Trend Analysis:")
    trends = analytics.detect_trends(test_account_id)
    print(f"   Trend Direction: {trends.trend_direction}")
    print(f"   Trend Strength: {trends.trend_strength:.2f}")
    print(f"   Momentum: {trends.momentum:.2%}")
    
    # Test risk assessment
    print("\n3. Risk Assessment:")
    risk = analytics.assess_risk(test_account_id)
    print(f"   Risk Level: {risk.risk_level}")
    print(f"   Risk Score: {risk.risk_score:.1f}/100")
    print(f"   VaR (95%): ${risk.var_95:.2f}")
    
    # Test predictions
    print("\n4. Predictive Model:")
    predictions = analytics.generate_predictions(test_account_id)
    print(f"   Predicted Return (30 days): {predictions.predicted_return:.2%}")
    print(f"   Confidence: {predictions.prediction_confidence:.2%}")
    print(f"   Model Accuracy: {predictions.model_accuracy:.2%}")
    
    # Test dashboard
    print("\n5. Analytics Dashboard:")
    dashboard = analytics.get_analytics_dashboard(test_account_id)
    print(f"   Dashboard generated with {len(dashboard)} sections")
    print(f"   Alerts: {len(dashboard.get('alerts', []))}")
    
    print("\n=== Account Analytics Engine Test Complete ===")

