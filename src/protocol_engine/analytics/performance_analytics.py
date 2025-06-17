#!/usr/bin/env python3
"""
ALL-USE Agent - Protocol Engine Performance Analytics
P5 of WS2: Performance Optimization and Monitoring - Phase 5

This module provides advanced performance analytics and real-time tracking
for the Protocol Engine, implementing trend analysis, visualization, and
optimization impact measurement capabilities.

Analytics Components:
1. Performance Analyzer - Advanced performance trend analysis
2. Real-time Tracker - Live performance tracking and streaming
3. Optimization Impact Analyzer - Quantifying optimization effectiveness
4. Trend Detector - Pattern recognition in performance data
5. Performance Visualizer - Charts and graphs for performance data
6. Analytics Dashboard - Comprehensive analytics interface
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import json
import sqlite3
import threading
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class PerformanceTrend:
    """Performance trend analysis result"""
    metric_name: str
    trend_direction: str  # increasing, decreasing, stable
    trend_strength: float  # 0-1, strength of trend
    slope: float
    r_squared: float
    prediction_24h: float
    confidence_interval: Tuple[float, float]
    analysis_period: str


@dataclass
class OptimizationImpact:
    """Optimization impact measurement"""
    optimization_name: str
    metric_name: str
    before_value: float
    after_value: float
    improvement_percent: float
    improvement_absolute: float
    statistical_significance: float
    confidence_level: float


@dataclass
class PerformanceAlert:
    """Performance analytics alert"""
    alert_type: str  # trend, anomaly, threshold
    severity: str    # low, medium, high, critical
    message: str
    metric_name: str
    current_value: float
    expected_value: float
    deviation_percent: float
    timestamp: datetime


class PerformanceAnalyzer:
    """Advanced performance trend analysis"""
    
    def __init__(self, db_path: str = "/tmp/protocol_engine_metrics.db"):
        self.db_path = db_path
        self.analysis_cache = {}
        self.trend_models = {}
        
        logger.info("Performance Analyzer initialized")
    
    def analyze_metric_trends(self, metric_name: str, hours: int = 24) -> PerformanceTrend:
        """Analyze trends for a specific metric"""
        # Get metric data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        data = self._get_metric_data(metric_name, start_time, end_time)
        
        if len(data) < 10:
            logger.warning(f"Insufficient data for trend analysis of {metric_name}")
            return None
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Prepare data for regression
        df['timestamp_numeric'] = df['timestamp'].astype(np.int64) // 10**9
        X = df['timestamp_numeric'].values.reshape(-1, 1)
        y = df['value'].values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate trend metrics
        slope = model.coef_[0]
        r_squared = model.score(X, y)
        
        # Determine trend direction and strength
        if abs(slope) < 0.001:
            trend_direction = "stable"
            trend_strength = 0.0
        elif slope > 0:
            trend_direction = "increasing"
            trend_strength = min(abs(slope) * 1000, 1.0)
        else:
            trend_direction = "decreasing"
            trend_strength = min(abs(slope) * 1000, 1.0)
        
        # Predict 24h ahead
        future_timestamp = (end_time + timedelta(hours=24)).timestamp()
        prediction_24h = model.predict([[future_timestamp]])[0]
        
        # Calculate confidence interval (simplified)
        residuals = y - model.predict(X)
        mse = np.mean(residuals**2)
        std_error = np.sqrt(mse)
        confidence_interval = (prediction_24h - 1.96*std_error, prediction_24h + 1.96*std_error)
        
        trend = PerformanceTrend(
            metric_name=metric_name,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            slope=slope,
            r_squared=r_squared,
            prediction_24h=prediction_24h,
            confidence_interval=confidence_interval,
            analysis_period=f"{hours}h"
        )
        
        # Cache the model for future predictions
        self.trend_models[metric_name] = model
        
        logger.info(f"Trend analysis complete for {metric_name}: {trend_direction} ({trend_strength:.2f})")
        
        return trend
    
    def _get_metric_data(self, metric_name: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get metric data from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT timestamp, value FROM metrics 
                WHERE name = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """, (metric_name, start_time.isoformat(), end_time.isoformat()))
            
            return [{'timestamp': row['timestamp'], 'value': row['value']} for row in cursor.fetchall()]
    
    def detect_anomalies(self, metric_name: str, hours: int = 24, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect anomalies in metric data using statistical methods"""
        data = self._get_metric_data(metric_name, datetime.now() - timedelta(hours=hours), datetime.now())
        
        if len(data) < 20:
            return []
        
        values = [d['value'] for d in data]
        timestamps = [d['timestamp'] for d in data]
        
        # Calculate z-scores
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        anomalies = []
        for i, (value, timestamp) in enumerate(zip(values, timestamps)):
            z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
            
            if z_score > threshold:
                anomalies.append({
                    'timestamp': timestamp,
                    'value': value,
                    'z_score': z_score,
                    'deviation_percent': ((value - mean_val) / mean_val) * 100 if mean_val != 0 else 0
                })
        
        logger.info(f"Detected {len(anomalies)} anomalies in {metric_name}")
        return anomalies
    
    def analyze_optimization_impact(self, optimization_name: str, metric_name: str, 
                                  before_period_hours: int = 24, after_period_hours: int = 24) -> OptimizationImpact:
        """Analyze the impact of an optimization on performance metrics"""
        now = datetime.now()
        
        # Get before and after data (assuming optimization happened at current time)
        before_start = now - timedelta(hours=before_period_hours * 2)
        before_end = now - timedelta(hours=before_period_hours)
        after_start = now - timedelta(hours=after_period_hours)
        after_end = now
        
        before_data = self._get_metric_data(metric_name, before_start, before_end)
        after_data = self._get_metric_data(metric_name, after_start, after_end)
        
        if not before_data or not after_data:
            logger.warning(f"Insufficient data for optimization impact analysis")
            return None
        
        before_values = [d['value'] for d in before_data]
        after_values = [d['value'] for d in after_data]
        
        before_mean = np.mean(before_values)
        after_mean = np.mean(after_values)
        
        improvement_absolute = before_mean - after_mean
        improvement_percent = (improvement_absolute / before_mean) * 100 if before_mean != 0 else 0
        
        # Statistical significance test (t-test)
        t_stat, p_value = stats.ttest_ind(before_values, after_values)
        statistical_significance = 1 - p_value
        confidence_level = 95.0 if p_value < 0.05 else 90.0 if p_value < 0.1 else 0.0
        
        impact = OptimizationImpact(
            optimization_name=optimization_name,
            metric_name=metric_name,
            before_value=before_mean,
            after_value=after_mean,
            improvement_percent=improvement_percent,
            improvement_absolute=improvement_absolute,
            statistical_significance=statistical_significance,
            confidence_level=confidence_level
        )
        
        logger.info(f"Optimization impact analysis: {improvement_percent:.2f}% improvement in {metric_name}")
        
        return impact


class RealTimeTracker:
    """Real-time performance tracking and streaming"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.tracking_active = False
        self.tracked_metrics = {}
        self.metric_streams = defaultdict(lambda: deque(maxlen=1000))
        self.tracking_thread = None
        self.callbacks = []
        
        logger.info(f"Real-time Tracker initialized (interval: {update_interval}s)")
    
    def add_metric_to_track(self, metric_name: str, source_func: callable):
        """Add a metric to real-time tracking"""
        self.tracked_metrics[metric_name] = source_func
        logger.info(f"Added metric to real-time tracking: {metric_name}")
    
    def add_update_callback(self, callback: callable):
        """Add callback for real-time updates"""
        self.callbacks.append(callback)
    
    def start_tracking(self):
        """Start real-time tracking"""
        if self.tracking_active:
            return
        
        self.tracking_active = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        
        logger.info("Real-time tracking started")
    
    def stop_tracking(self):
        """Stop real-time tracking"""
        self.tracking_active = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=5.0)
        
        logger.info("Real-time tracking stopped")
    
    def _tracking_loop(self):
        """Main tracking loop"""
        while self.tracking_active:
            try:
                timestamp = datetime.now()
                
                # Collect current values for all tracked metrics
                current_values = {}
                for metric_name, source_func in self.tracked_metrics.items():
                    try:
                        value = source_func()
                        current_values[metric_name] = value
                        
                        # Add to stream
                        self.metric_streams[metric_name].append({
                            'timestamp': timestamp,
                            'value': value
                        })
                        
                    except Exception as e:
                        logger.error(f"Error collecting metric {metric_name}: {e}")
                
                # Call update callbacks
                for callback in self.callbacks:
                    try:
                        callback(timestamp, current_values)
                    except Exception as e:
                        logger.error(f"Error in tracking callback: {e}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
                time.sleep(self.update_interval)
    
    def get_metric_stream(self, metric_name: str, last_n: int = 100) -> List[Dict[str, Any]]:
        """Get recent values from metric stream"""
        stream = self.metric_streams.get(metric_name, deque())
        return list(stream)[-last_n:]
    
    def get_current_values(self) -> Dict[str, float]:
        """Get current values for all tracked metrics"""
        current_values = {}
        for metric_name, source_func in self.tracked_metrics.items():
            try:
                current_values[metric_name] = source_func()
            except Exception as e:
                logger.error(f"Error getting current value for {metric_name}: {e}")
                current_values[metric_name] = None
        
        return current_values


class PerformanceVisualizer:
    """Performance data visualization and charting"""
    
    def __init__(self, output_dir: str = "/home/ubuntu/AGENT_ALLUSE_V1/docs/optimization"):
        self.output_dir = output_dir
        self.figure_size = (12, 8)
        
        # Ensure output directory exists
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Performance Visualizer initialized (output: {output_dir})")
    
    def create_trend_chart(self, metric_name: str, data: List[Dict[str, Any]], 
                          trend: PerformanceTrend = None) -> str:
        """Create a trend chart for metric data"""
        if not data:
            logger.warning(f"No data provided for trend chart of {metric_name}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot data
        ax.plot(df['timestamp'], df['value'], 'b-', linewidth=2, alpha=0.7, label='Actual')
        
        # Add trend line if provided
        if trend:
            # Create trend line
            x_numeric = df['timestamp'].astype(np.int64) // 10**9
            trend_line = trend.slope * x_numeric + (df['value'].iloc[0] - trend.slope * x_numeric.iloc[0])
            ax.plot(df['timestamp'], trend_line, 'r--', linewidth=2, alpha=0.8, 
                   label=f'Trend ({trend.trend_direction})')
        
        # Formatting
        ax.set_title(f'Performance Trend: {metric_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.xticks(rotation=45)
        
        # Add statistics text
        if trend:
            stats_text = f'Trend: {trend.trend_direction}\nStrength: {trend.trend_strength:.2f}\nR²: {trend.r_squared:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save chart
        filename = f"trend_{metric_name.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Trend chart saved: {filepath}")
        return filepath
    
    def create_optimization_impact_chart(self, impacts: List[OptimizationImpact]) -> str:
        """Create a chart showing optimization impacts"""
        if not impacts:
            return None
        
        # Prepare data
        metrics = [impact.metric_name for impact in impacts]
        improvements = [impact.improvement_percent for impact in impacts]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create bar chart
        bars = ax.bar(metrics, improvements, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                   f'{improvement:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # Formatting
        ax.set_title('Optimization Impact Analysis', fontsize=16, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save chart
        filename = f"optimization_impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Optimization impact chart saved: {filepath}")
        return filepath
    
    def create_real_time_dashboard(self, metric_streams: Dict[str, List[Dict[str, Any]]]) -> str:
        """Create a real-time dashboard with multiple metrics"""
        if not metric_streams:
            return None
        
        # Create subplots
        n_metrics = len(metric_streams)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for i, (metric_name, data) in enumerate(metric_streams.items()):
            if not data:
                continue
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            axes[i].plot(df['timestamp'], df['value'], 'b-', linewidth=2)
            axes[i].set_title(f'{metric_name}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)
            
            # Format x-axis
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            axes[i].xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            
            # Add current value annotation
            if len(df) > 0:
                current_value = df['value'].iloc[-1]
                axes[i].annotate(f'Current: {current_value:.2f}', 
                               xy=(df['timestamp'].iloc[-1], current_value),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.suptitle('Real-time Performance Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save dashboard
        filename = f"realtime_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Real-time dashboard saved: {filepath}")
        return filepath


class AnalyticsDashboard:
    """Comprehensive analytics dashboard and interface"""
    
    def __init__(self, db_path: str = "/tmp/protocol_engine_metrics.db"):
        self.analyzer = PerformanceAnalyzer(db_path)
        self.tracker = RealTimeTracker(update_interval=1.0)
        self.visualizer = PerformanceVisualizer()
        
        self.dashboard_data = {}
        self.optimization_history = []
        
        logger.info("Analytics Dashboard initialized")
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analytics report"""
        logger.info("Generating comprehensive performance analytics report...")
        
        # Key metrics to analyze
        key_metrics = [
            "system.memory.rss",
            "system.cpu.percent",
            "function.test_calculation.duration"
        ]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'trends': {},
            'anomalies': {},
            'optimization_impacts': [],
            'visualizations': [],
            'summary': {}
        }
        
        # Analyze trends for each metric
        for metric in key_metrics:
            try:
                trend = self.analyzer.analyze_metric_trends(metric, hours=24)
                if trend:
                    report['trends'][metric] = {
                        'direction': trend.trend_direction,
                        'strength': trend.trend_strength,
                        'r_squared': trend.r_squared,
                        'prediction_24h': trend.prediction_24h
                    }
                    
                    # Create visualization
                    data = self.analyzer._get_metric_data(metric, 
                                                        datetime.now() - timedelta(hours=24), 
                                                        datetime.now())
                    if data:
                        chart_path = self.visualizer.create_trend_chart(metric, data, trend)
                        if chart_path:
                            report['visualizations'].append(chart_path)
                
                # Detect anomalies
                anomalies = self.analyzer.detect_anomalies(metric, hours=24)
                if anomalies:
                    report['anomalies'][metric] = len(anomalies)
                
            except Exception as e:
                logger.error(f"Error analyzing metric {metric}: {e}")
        
        # Generate summary
        report['summary'] = {
            'total_metrics_analyzed': len(key_metrics),
            'metrics_with_trends': len(report['trends']),
            'metrics_with_anomalies': len(report['anomalies']),
            'total_visualizations': len(report['visualizations'])
        }
        
        self.dashboard_data = report
        
        logger.info(f"Comprehensive report generated with {len(report['trends'])} trend analyses")
        return report
    
    def start_real_time_analytics(self):
        """Start real-time analytics tracking"""
        # Add metrics to track
        import psutil
        process = psutil.Process()
        
        self.tracker.add_metric_to_track("memory_usage", lambda: process.memory_info().rss / 1024 / 1024)
        self.tracker.add_metric_to_track("cpu_usage", lambda: process.cpu_percent())
        self.tracker.add_metric_to_track("thread_count", lambda: threading.active_count())
        
        # Add callback for real-time dashboard updates
        self.tracker.add_update_callback(self._real_time_update_callback)
        
        # Start tracking
        self.tracker.start_tracking()
        
        logger.info("Real-time analytics started")
    
    def stop_real_time_analytics(self):
        """Stop real-time analytics tracking"""
        self.tracker.stop_tracking()
        logger.info("Real-time analytics stopped")
    
    def _real_time_update_callback(self, timestamp: datetime, values: Dict[str, float]):
        """Callback for real-time updates"""
        # Update dashboard data
        self.dashboard_data['last_update'] = timestamp.isoformat()
        self.dashboard_data['current_values'] = values
        
        # Log significant changes
        for metric, value in values.items():
            if metric == "memory_usage" and value > 150:
                logger.warning(f"High memory usage detected: {value:.2f} MB")
            elif metric == "cpu_usage" and value > 70:
                logger.warning(f"High CPU usage detected: {value:.2f}%")
    
    def create_real_time_dashboard_visualization(self) -> str:
        """Create real-time dashboard visualization"""
        # Get recent data for all tracked metrics
        metric_streams = {}
        for metric_name in self.tracker.tracked_metrics.keys():
            stream_data = self.tracker.get_metric_stream(metric_name, last_n=60)  # Last 60 points
            if stream_data:
                metric_streams[metric_name] = stream_data
        
        if metric_streams:
            return self.visualizer.create_real_time_dashboard(metric_streams)
        
        return None
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard summary data"""
        current_values = self.tracker.get_current_values()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'real_time_tracking_active': self.tracker.tracking_active,
            'tracked_metrics': list(self.tracker.tracked_metrics.keys()),
            'current_values': current_values,
            'dashboard_data_available': bool(self.dashboard_data),
            'optimization_history_count': len(self.optimization_history)
        }
        
        return summary


if __name__ == '__main__':
    print("📊 Testing Performance Analytics and Real-time Tracking (P5 of WS2 - Phase 5)")
    print("=" * 85)
    
    # Initialize analytics dashboard
    dashboard = AnalyticsDashboard()
    
    print("\n🚀 Starting Real-time Analytics:")
    dashboard.start_real_time_analytics()
    
    # Simulate some performance data
    print("\n📈 Simulating Performance Data:")
    import psutil
    process = psutil.Process()
    
    for i in range(10):
        # Simulate some work
        time.sleep(0.5)
        
        # Log current metrics
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        print(f"   Iteration {i+1}: Memory={memory_mb:.1f}MB, CPU={cpu_percent:.1f}%")
    
    print("\n📊 Generating Comprehensive Analytics Report:")
    report = dashboard.generate_comprehensive_report()
    
    print(f"   Report Summary:")
    print(f"   - Metrics Analyzed: {report['summary']['total_metrics_analyzed']}")
    print(f"   - Trends Detected: {report['summary']['metrics_with_trends']}")
    print(f"   - Anomalies Found: {report['summary']['metrics_with_anomalies']}")
    print(f"   - Visualizations Created: {report['summary']['total_visualizations']}")
    
    # Show trend analysis results
    if report['trends']:
        print(f"\n📈 Trend Analysis Results:")
        for metric, trend_data in report['trends'].items():
            print(f"   {metric}: {trend_data['direction']} (strength: {trend_data['strength']:.2f})")
    
    # Create real-time dashboard visualization
    print("\n📊 Creating Real-time Dashboard Visualization:")
    dashboard_viz = dashboard.create_real_time_dashboard_visualization()
    if dashboard_viz:
        print(f"   Dashboard visualization saved: {dashboard_viz}")
    else:
        print("   No real-time data available for visualization")
    
    # Get dashboard summary
    print("\n📋 Dashboard Summary:")
    summary = dashboard.get_dashboard_summary()
    print(f"   Real-time Tracking: {'Active' if summary['real_time_tracking_active'] else 'Inactive'}")
    print(f"   Tracked Metrics: {len(summary['tracked_metrics'])}")
    print(f"   Current Values: {summary['current_values']}")
    
    print("\n🛑 Stopping Real-time Analytics:")
    dashboard.stop_real_time_analytics()
    
    print("\n✅ P5 of WS2 - Phase 5: Performance Analytics and Real-time Tracking COMPLETE")
    print("🚀 Ready for Phase 6: Optimization Validation and Documentation")

