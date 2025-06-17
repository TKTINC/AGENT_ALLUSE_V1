#!/usr/bin/env python3
"""
ALL-USE Agent Real-time Market Analytics and Performance Tracking
WS4-P5 Phase 5: Real-time Market Analytics and Performance Tracking

This module implements advanced analytics and visualization capabilities for
market integration performance monitoring, including:
- Statistical analysis and trend detection
- Performance visualization with professional charts
- Predictive analytics and anomaly detection
- Real-time dashboards with interactive data
"""

import asyncio
import time
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for high-quality plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

class PerformanceDataAnalyzer:
    """
    Advanced analytics engine for market integration performance data
    """
    
    def __init__(self, db_path: str = "docs/market_integration/monitoring_metrics.db"):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        
    def load_performance_data(self, hours: int = 1) -> pd.DataFrame:
        """Load performance data from database"""
        since_timestamp = (datetime.now() - timedelta(hours=hours)).timestamp()
        
        query = """
            SELECT timestamp, component, metric_name, value, unit, threshold_status, tags
            FROM metrics
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, self.connection, params=(since_timestamp,))
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['tags'] = df['tags'].apply(json.loads)
        
        return df
    
    def calculate_performance_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance statistics"""
        stats = {}
        
        # Group by component and metric
        for component in df['component'].unique():
            component_data = df[df['component'] == component]
            stats[component] = {}
            
            for metric in component_data['metric_name'].unique():
                metric_data = component_data[component_data['metric_name'] == metric]['value']
                
                if len(metric_data) > 0:
                    stats[component][metric] = {
                        'count': len(metric_data),
                        'mean': float(metric_data.mean()),
                        'median': float(metric_data.median()),
                        'std': float(metric_data.std()) if len(metric_data) > 1 else 0.0,
                        'min': float(metric_data.min()),
                        'max': float(metric_data.max()),
                        'q25': float(metric_data.quantile(0.25)),
                        'q75': float(metric_data.quantile(0.75)),
                        'latest': float(metric_data.iloc[-1]) if len(metric_data) > 0 else 0.0
                    }
        
        return stats
    
    def detect_performance_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect performance trends using linear regression"""
        trends = {}
        
        for component in df['component'].unique():
            component_data = df[df['component'] == component]
            trends[component] = {}
            
            for metric in component_data['metric_name'].unique():
                metric_data = component_data[component_data['metric_name'] == metric].copy()
                
                if len(metric_data) >= 5:  # Need at least 5 points for trend analysis
                    # Prepare data for regression
                    metric_data['timestamp_numeric'] = metric_data['timestamp'].astype(np.int64) // 10**9
                    X = metric_data['timestamp_numeric'].values.reshape(-1, 1)
                    y = metric_data['value'].values
                    
                    # Fit linear regression
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Calculate trend metrics
                    slope = model.coef_[0]
                    r_squared = model.score(X, y)
                    
                    # Determine trend direction
                    if abs(slope) < 0.001:
                        trend_direction = "stable"
                    elif slope > 0:
                        trend_direction = "increasing"
                    else:
                        trend_direction = "decreasing"
                    
                    trends[component][metric] = {
                        'slope': float(slope),
                        'r_squared': float(r_squared),
                        'trend_direction': trend_direction,
                        'trend_strength': 'strong' if r_squared > 0.7 else 'moderate' if r_squared > 0.3 else 'weak'
                    }
        
        return trends
    
    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect performance anomalies using Isolation Forest"""
        anomalies = {}
        
        for component in df['component'].unique():
            component_data = df[df['component'] == component]
            anomalies[component] = {}
            
            for metric in component_data['metric_name'].unique():
                metric_data = component_data[component_data['metric_name'] == metric]['value']
                
                if len(metric_data) >= 10:  # Need sufficient data for anomaly detection
                    # Prepare data
                    X = metric_data.values.reshape(-1, 1)
                    
                    # Fit Isolation Forest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_labels = iso_forest.fit_predict(X)
                    
                    # Count anomalies
                    anomaly_count = np.sum(anomaly_labels == -1)
                    anomaly_percentage = (anomaly_count / len(anomaly_labels)) * 100
                    
                    anomalies[component][metric] = {
                        'total_points': len(metric_data),
                        'anomaly_count': int(anomaly_count),
                        'anomaly_percentage': float(anomaly_percentage),
                        'anomaly_threshold': 'high' if anomaly_percentage > 15 else 'normal'
                    }
        
        return anomalies
    
    def generate_performance_forecast(self, df: pd.DataFrame, forecast_minutes: int = 30) -> Dict[str, Any]:
        """Generate performance forecasts using trend analysis"""
        forecasts = {}
        
        for component in df['component'].unique():
            component_data = df[df['component'] == component]
            forecasts[component] = {}
            
            for metric in component_data['metric_name'].unique():
                metric_data = component_data[component_data['metric_name'] == metric].copy()
                
                if len(metric_data) >= 10:  # Need sufficient data for forecasting
                    # Prepare data
                    metric_data['timestamp_numeric'] = metric_data['timestamp'].astype(np.int64) // 10**9
                    X = metric_data['timestamp_numeric'].values.reshape(-1, 1)
                    y = metric_data['value'].values
                    
                    # Fit model
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Generate forecast
                    current_time = datetime.now().timestamp()
                    future_time = current_time + (forecast_minutes * 60)
                    forecast_value = model.predict([[future_time]])[0]
                    
                    # Calculate confidence based on recent variance
                    recent_values = metric_data['value'].tail(10)
                    confidence = max(0, min(100, 100 - (recent_values.std() / recent_values.mean() * 100)))
                    
                    forecasts[component][metric] = {
                        'current_value': float(metric_data['value'].iloc[-1]),
                        'forecast_value': float(forecast_value),
                        'forecast_time_minutes': forecast_minutes,
                        'confidence_percentage': float(confidence),
                        'trend_direction': 'increasing' if forecast_value > metric_data['value'].iloc[-1] else 'decreasing'
                    }
        
        return forecasts

class PerformanceVisualizer:
    """
    Advanced visualization engine for market integration performance data
    """
    
    def __init__(self):
        self.color_palette = {
            'trading_system': '#2E86AB',
            'market_data_system': '#A23B72',
            'ibkr_integration': '#F18F01',
            'system_resources': '#C73E1D'
        }
    
    def create_comprehensive_dashboard(self, df: pd.DataFrame, stats: Dict, trends: Dict, 
                                    anomalies: Dict, forecasts: Dict) -> str:
        """Create comprehensive performance dashboard"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Define grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Trading System Performance (Top Left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_trading_system_metrics(ax1, df)
        
        # 2. Market Data System Performance (Top Right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_market_data_metrics(ax2, df)
        
        # 3. System Resources Overview (Second Row Left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_system_resources(ax3, df)
        
        # 4. Performance Statistics Summary (Second Row Right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_performance_statistics(ax4, stats)
        
        # 5. Trend Analysis (Third Row Left)
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_trend_analysis(ax5, trends)
        
        # 6. Anomaly Detection (Third Row Right)
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_anomaly_analysis(ax6, anomalies)
        
        # 7. Performance Forecasts (Bottom Row)
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_performance_forecasts(ax7, forecasts)
        
        # Add main title
        fig.suptitle('Market Integration Performance Analytics Dashboard\nWS4-P5 Real-time Analytics and Performance Tracking', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Save dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_path = f"docs/market_integration/market_analytics_dashboard_{timestamp}.png"
        Path("docs/market_integration").mkdir(parents=True, exist_ok=True)
        
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return dashboard_path
    
    def _plot_trading_system_metrics(self, ax, df):
        """Plot trading system performance metrics"""
        trading_data = df[df['component'] == 'trading_system']
        
        if not trading_data.empty:
            # Plot error rate and latency
            for metric in ['error_rate', 'latency_ms']:
                metric_data = trading_data[trading_data['metric_name'] == metric]
                if not metric_data.empty:
                    ax.plot(metric_data['timestamp'], metric_data['value'], 
                           label=f'{metric.replace("_", " ").title()}', linewidth=2, marker='o', markersize=4)
        
        ax.set_title('Trading System Performance', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_market_data_metrics(self, ax, df):
        """Plot market data system performance metrics"""
        market_data = df[df['component'] == 'market_data_system']
        
        if not market_data.empty:
            # Create secondary y-axis for throughput
            ax2 = ax.twinx()
            
            # Plot latency on primary axis
            latency_data = market_data[market_data['metric_name'] == 'latency_ms']
            if not latency_data.empty:
                ax.plot(latency_data['timestamp'], latency_data['value'], 
                       color='red', label='Latency (ms)', linewidth=2, marker='o', markersize=4)
            
            # Plot throughput on secondary axis
            throughput_data = market_data[market_data['metric_name'] == 'throughput_ops_sec']
            if not throughput_data.empty:
                ax2.plot(throughput_data['timestamp'], throughput_data['value'], 
                        color='blue', label='Throughput (ops/sec)', linewidth=2, marker='s', markersize=4)
        
        ax.set_title('Market Data System Performance', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Latency (ms)', color='red')
        ax2.set_ylabel('Throughput (ops/sec)', color='blue')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_system_resources(self, ax, df):
        """Plot system resource utilization"""
        resource_data = df[df['component'] == 'system_resources']
        
        if not resource_data.empty:
            for metric in ['memory_usage_mb', 'cpu_percent']:
                metric_data = resource_data[resource_data['metric_name'] == metric]
                if not metric_data.empty:
                    ax.plot(metric_data['timestamp'], metric_data['value'], 
                           label=f'{metric.replace("_", " ").title()}', linewidth=2, marker='o', markersize=4)
        
        ax.set_title('System Resource Utilization', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_statistics(self, ax, stats):
        """Plot performance statistics summary"""
        # Create bar chart of key performance metrics
        components = []
        metrics = []
        values = []
        
        for component, component_stats in stats.items():
            for metric, metric_stats in component_stats.items():
                if metric in ['throughput_ops_sec', 'latency_ms', 'error_rate']:
                    components.append(component.replace('_', ' ').title())
                    metrics.append(metric.replace('_', ' ').title())
                    values.append(metric_stats['latest'])
        
        if values:
            # Create grouped bar chart
            x_pos = np.arange(len(components))
            bars = ax.bar(x_pos, values, color=[self.color_palette.get(comp.lower().replace(' ', '_'), '#666666') for comp in components])
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Current Performance Statistics', fontweight='bold')
        ax.set_xlabel('Component - Metric')
        ax.set_ylabel('Value')
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels([f'{comp}\n{metric}' for comp, metric in zip(components, metrics)], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_trend_analysis(self, ax, trends):
        """Plot trend analysis results"""
        # Create trend strength visualization
        components = []
        trend_strengths = []
        trend_directions = []
        
        for component, component_trends in trends.items():
            for metric, trend_data in component_trends.items():
                components.append(f"{component.replace('_', ' ').title()}\n{metric.replace('_', ' ').title()}")
                
                # Convert trend strength to numeric
                strength_map = {'weak': 1, 'moderate': 2, 'strong': 3}
                trend_strengths.append(strength_map.get(trend_data['trend_strength'], 1))
                trend_directions.append(trend_data['trend_direction'])
        
        if trend_strengths:
            # Create color map based on trend direction
            colors = []
            for direction in trend_directions:
                if direction == 'increasing':
                    colors.append('#2E8B57')  # Green
                elif direction == 'decreasing':
                    colors.append('#DC143C')  # Red
                else:
                    colors.append('#4682B4')  # Blue
            
            bars = ax.bar(range(len(components)), trend_strengths, color=colors)
            
            # Add direction labels
            for bar, direction in zip(bars, trend_directions):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       direction.title(), ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_title('Performance Trend Analysis', fontweight='bold')
        ax.set_xlabel('Component - Metric')
        ax.set_ylabel('Trend Strength')
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.set_ylim(0, 4)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Weak', 'Moderate', 'Strong'])
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_anomaly_analysis(self, ax, anomalies):
        """Plot anomaly detection results"""
        components = []
        anomaly_percentages = []
        
        for component, component_anomalies in anomalies.items():
            for metric, anomaly_data in component_anomalies.items():
                components.append(f"{component.replace('_', ' ').title()}\n{metric.replace('_', ' ').title()}")
                anomaly_percentages.append(anomaly_data['anomaly_percentage'])
        
        if anomaly_percentages:
            # Create color map based on anomaly threshold
            colors = ['#DC143C' if pct > 15 else '#FFA500' if pct > 5 else '#2E8B57' for pct in anomaly_percentages]
            
            bars = ax.bar(range(len(components)), anomaly_percentages, color=colors)
            
            # Add percentage labels
            for bar, pct in zip(bars, anomaly_percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                       f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Anomaly Detection Analysis', fontweight='bold')
        ax.set_xlabel('Component - Metric')
        ax.set_ylabel('Anomaly Percentage')
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='High Threshold (15%)')
        ax.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold (5%)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_performance_forecasts(self, ax, forecasts):
        """Plot performance forecasts"""
        components = []
        current_values = []
        forecast_values = []
        confidence_levels = []
        
        for component, component_forecasts in forecasts.items():
            for metric, forecast_data in component_forecasts.items():
                components.append(f"{component.replace('_', ' ').title()}\n{metric.replace('_', ' ').title()}")
                current_values.append(forecast_data['current_value'])
                forecast_values.append(forecast_data['forecast_value'])
                confidence_levels.append(forecast_data['confidence_percentage'])
        
        if current_values:
            x_pos = np.arange(len(components))
            width = 0.35
            
            # Plot current vs forecast values
            bars1 = ax.bar(x_pos - width/2, current_values, width, label='Current Value', alpha=0.8)
            bars2 = ax.bar(x_pos + width/2, forecast_values, width, label='Forecast Value', alpha=0.8)
            
            # Add confidence level annotations
            for i, (current, forecast, confidence) in enumerate(zip(current_values, forecast_values, confidence_levels)):
                max_val = max(current, forecast)
                ax.text(i, max_val + max_val*0.05, f'{confidence:.0f}% conf.', 
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_title('Performance Forecasts (30-minute prediction)', fontweight='bold')
        ax.set_xlabel('Component - Metric')
        ax.set_ylabel('Value')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

class RealTimeMarketAnalytics:
    """
    Comprehensive real-time market analytics and performance tracking system
    """
    
    def __init__(self):
        self.analyzer = PerformanceDataAnalyzer()
        self.visualizer = PerformanceVisualizer()
        self.analytics_stats = {
            "analytics_sessions": 0,
            "visualizations_generated": 0,
            "data_points_analyzed": 0,
            "trends_detected": 0,
            "anomalies_detected": 0,
            "forecasts_generated": 0
        }
    
    async def run_comprehensive_analytics(self, data_hours: int = 1) -> Dict[str, Any]:
        """Run comprehensive analytics on market integration performance data"""
        print(f"🚀 Starting comprehensive analytics for last {data_hours} hour(s)...")
        analytics_start = time.time()
        
        # Load performance data
        print("📊 Loading performance data...")
        df = self.analyzer.load_performance_data(hours=data_hours)
        
        if df.empty:
            print("⚠️  No performance data available for analysis")
            return {"error": "No data available"}
        
        print(f"✅ Loaded {len(df)} data points for analysis")
        self.analytics_stats["data_points_analyzed"] += len(df)
        
        # Calculate performance statistics
        print("📈 Calculating performance statistics...")
        stats = self.analyzer.calculate_performance_statistics(df)
        
        # Detect performance trends
        print("📊 Detecting performance trends...")
        trends = self.analyzer.detect_performance_trends(df)
        trend_count = sum(len(component_trends) for component_trends in trends.values())
        self.analytics_stats["trends_detected"] += trend_count
        
        # Detect anomalies
        print("🔍 Detecting performance anomalies...")
        anomalies = self.analyzer.detect_anomalies(df)
        anomaly_count = sum(len(component_anomalies) for component_anomalies in anomalies.values())
        self.analytics_stats["anomalies_detected"] += anomaly_count
        
        # Generate forecasts
        print("🔮 Generating performance forecasts...")
        forecasts = self.analyzer.generate_performance_forecast(df)
        forecast_count = sum(len(component_forecasts) for component_forecasts in forecasts.values())
        self.analytics_stats["forecasts_generated"] += forecast_count
        
        # Create comprehensive dashboard
        print("📊 Creating comprehensive analytics dashboard...")
        dashboard_path = self.visualizer.create_comprehensive_dashboard(df, stats, trends, anomalies, forecasts)
        self.analytics_stats["visualizations_generated"] += 1
        
        analytics_duration = time.time() - analytics_start
        self.analytics_stats["analytics_sessions"] += 1
        
        # Prepare analytics report
        analytics_report = {
            "analytics_summary": {
                "duration_seconds": analytics_duration,
                "data_points_analyzed": len(df),
                "trends_detected": trend_count,
                "anomalies_detected": anomaly_count,
                "forecasts_generated": forecast_count,
                "dashboard_generated": dashboard_path
            },
            "performance_statistics": stats,
            "trend_analysis": trends,
            "anomaly_detection": anomalies,
            "performance_forecasts": forecasts,
            "analytics_stats": self.analytics_stats.copy()
        }
        
        # Save analytics report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"docs/market_integration/market_analytics_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(analytics_report, f, indent=2, default=str)
        
        print(f"💾 Analytics report saved to: {report_file}")
        print(f"📊 Dashboard visualization saved to: {dashboard_path}")
        
        return analytics_report
    
    def generate_analytics_summary(self, analytics_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of analytics results"""
        summary = analytics_report["analytics_summary"]
        stats = analytics_report["performance_statistics"]
        trends = analytics_report["trend_analysis"]
        anomalies = analytics_report["anomaly_detection"]
        forecasts = analytics_report["performance_forecasts"]
        
        # Key performance insights
        key_insights = []
        
        # Trading system insights
        if "trading_system" in stats:
            trading_stats = stats["trading_system"]
            if "error_rate" in trading_stats:
                error_rate = trading_stats["error_rate"]["latest"]
                if error_rate == 0.0:
                    key_insights.append("🎯 Trading system achieving 0% error rate - exceptional performance")
                elif error_rate < 1.0:
                    key_insights.append(f"✅ Trading system error rate at {error_rate:.2f}% - within optimal range")
            
            if "latency_ms" in trading_stats:
                latency = trading_stats["latency_ms"]["latest"]
                if latency < 20.0:
                    key_insights.append(f"⚡ Trading system latency at {latency:.1f}ms - excellent responsiveness")
        
        # Market data insights
        if "market_data_system" in stats:
            market_stats = stats["market_data_system"]
            if "throughput_ops_sec" in market_stats:
                throughput = market_stats["throughput_ops_sec"]["latest"]
                if throughput > 30000:
                    key_insights.append(f"🚀 Market data throughput at {throughput:.0f} ops/sec - exceptional performance")
            
            if "latency_ms" in market_stats:
                latency = market_stats["latency_ms"]["latest"]
                if latency < 0.1:
                    key_insights.append(f"⚡ Market data latency at {latency:.3f}ms - sub-millisecond performance")
        
        # Trend insights
        strong_trends = []
        for component, component_trends in trends.items():
            for metric, trend_data in component_trends.items():
                if trend_data["trend_strength"] == "strong":
                    strong_trends.append(f"{component}.{metric}: {trend_data['trend_direction']}")
        
        if strong_trends:
            key_insights.append(f"📈 Strong trends detected: {', '.join(strong_trends)}")
        
        # Anomaly insights
        high_anomaly_components = []
        for component, component_anomalies in anomalies.items():
            for metric, anomaly_data in component_anomalies.items():
                if anomaly_data["anomaly_threshold"] == "high":
                    high_anomaly_components.append(f"{component}.{metric}")
        
        if high_anomaly_components:
            key_insights.append(f"⚠️  High anomaly rates detected: {', '.join(high_anomaly_components)}")
        else:
            key_insights.append("✅ All components showing normal anomaly rates")
        
        return {
            "executive_summary": {
                "total_data_points": summary["data_points_analyzed"],
                "analysis_duration": f"{summary['duration_seconds']:.2f} seconds",
                "key_insights": key_insights,
                "overall_health": "excellent" if len(high_anomaly_components) == 0 else "good",
                "performance_grade": "A+" if all("exceptional" in insight or "excellent" in insight for insight in key_insights[:2]) else "A"
            },
            "performance_highlights": {
                "trading_system_status": "optimal" if "trading_system" in stats and stats["trading_system"].get("error_rate", {}).get("latest", 1) == 0 else "good",
                "market_data_status": "exceptional" if "market_data_system" in stats and stats["market_data_system"].get("throughput_ops_sec", {}).get("latest", 0) > 30000 else "good",
                "system_stability": "stable" if len(strong_trends) <= 2 else "variable"
            }
        }

async def main():
    """
    Main execution function for real-time market analytics and performance tracking
    """
    print("🚀 Starting WS4-P5 Phase 5: Real-time Market Analytics and Performance Tracking")
    print("=" * 80)
    
    try:
        # Initialize analytics system
        analytics_system = RealTimeMarketAnalytics()
        
        # Run comprehensive analytics
        analytics_report = await analytics_system.run_comprehensive_analytics(data_hours=1)
        
        if "error" in analytics_report:
            print(f"❌ Analytics failed: {analytics_report['error']}")
            return False
        
        # Generate executive summary
        executive_summary = analytics_system.generate_analytics_summary(analytics_report)
        
        # Save executive summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"docs/market_integration/analytics_executive_summary_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(executive_summary, f, indent=2)
        
        print(f"💾 Executive summary saved to: {summary_file}")
        
        # Print comprehensive results
        print("\n" + "=" * 80)
        print("📊 REAL-TIME MARKET ANALYTICS AND PERFORMANCE TRACKING SUMMARY")
        print("=" * 80)
        
        summary = analytics_report["analytics_summary"]
        exec_summary = executive_summary["executive_summary"]
        highlights = executive_summary["performance_highlights"]
        
        print(f"⏱️  Analytics Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"📊 Data Points Analyzed: {summary['data_points_analyzed']}")
        print(f"📈 Trends Detected: {summary['trends_detected']}")
        print(f"🔍 Anomalies Detected: {summary['anomalies_detected']}")
        print(f"🔮 Forecasts Generated: {summary['forecasts_generated']}")
        print(f"📊 Dashboard Generated: ✅ {summary['dashboard_generated']}")
        
        print(f"\n🎯 EXECUTIVE SUMMARY:")
        print(f"  • Overall Health: {exec_summary['overall_health'].upper()}")
        print(f"  • Performance Grade: {exec_summary['performance_grade']}")
        print(f"  • Trading System: {highlights['trading_system_status'].upper()}")
        print(f"  • Market Data System: {highlights['market_data_status'].upper()}")
        print(f"  • System Stability: {highlights['system_stability'].upper()}")
        
        print(f"\n💡 KEY INSIGHTS:")
        for insight in exec_summary['key_insights']:
            print(f"  • {insight}")
        
        print(f"\n📈 ANALYTICS CAPABILITIES:")
        print(f"  • Statistical Analysis: ✅ Comprehensive performance statistics")
        print(f"  • Trend Detection: ✅ Linear regression-based trend analysis")
        print(f"  • Anomaly Detection: ✅ Isolation Forest anomaly detection")
        print(f"  • Performance Forecasting: ✅ 30-minute predictive analytics")
        print(f"  • Advanced Visualization: ✅ Professional dashboard with 7 chart types")
        
        print("\n🚀 READY FOR PHASE 6: Optimization Validation and Documentation")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in analytics system: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

