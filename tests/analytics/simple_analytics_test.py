"""
Simplified Account Analytics Engine Test
Test analytics engine with minimal setup
"""

import sys
import os
sys.path.append('/home/ubuntu/AGENT_ALLUSE_V1/src')

from account_management.analytics.account_analytics_engine import AccountAnalyticsEngine, TimeFrame
import sqlite3

def test_analytics_engine():
    """Test analytics engine with simulated data"""
    print("=== Testing Account Analytics Engine ===")
    
    # Initialize analytics engine
    analytics = AccountAnalyticsEngine()
    
    # Create test data directly in database
    print("\n1. Setting up test data...")
    
    db_path = "/home/ubuntu/AGENT_ALLUSE_V1/data/account_management.db"
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Create accounts table if not exists
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            account_id TEXT PRIMARY KEY,
            account_name TEXT,
            account_type TEXT,
            balance REAL
        )
        """)
        
        # Create transactions table if not exists
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            account_id TEXT,
            amount REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Insert test account
        test_account_id = "test_analytics_account_001"
        cursor.execute("""
        INSERT OR REPLACE INTO accounts (account_id, account_name, account_type, balance)
        VALUES (?, ?, ?, ?)
        """, (test_account_id, "Test Analytics Account", "generation", 100000.0))
        
        # Insert test transactions
        transactions = [
            (f"txn_001_{test_account_id}", test_account_id, 2500.0),
            (f"txn_002_{test_account_id}", test_account_id, -500.0),
            (f"txn_003_{test_account_id}", test_account_id, 1800.0),
            (f"txn_004_{test_account_id}", test_account_id, 3200.0),
            (f"txn_005_{test_account_id}", test_account_id, -800.0),
            (f"txn_006_{test_account_id}", test_account_id, 2200.0),
        ]
        
        for txn_id, acc_id, amount in transactions:
            cursor.execute("""
            INSERT OR REPLACE INTO transactions (transaction_id, account_id, amount)
            VALUES (?, ?, ?)
            """, (txn_id, acc_id, amount))
        
        conn.commit()
        print(f"   Created test account: {test_account_id}")
        print(f"   Added {len(transactions)} test transactions")
    
    # Test analytics components
    print(f"\n2. Testing Analytics Components for {test_account_id}:")
    
    # Performance Analysis
    print("   a) Performance Analysis:")
    performance = analytics.analyze_account_performance(test_account_id, TimeFrame.ALL_TIME)
    print(f"      Total Return: {performance.total_return:.2%}")
    print(f"      Annualized Return: {performance.annualized_return:.2%}")
    print(f"      Sharpe Ratio: {performance.sharpe_ratio:.2f}")
    print(f"      Win Rate: {performance.win_rate:.2%}")
    print(f"      Total Trades: {performance.total_trades}")
    print(f"      Profitable Trades: {performance.profitable_trades}")
    
    # Trend Analysis
    print("   b) Trend Analysis:")
    trends = analytics.detect_trends(test_account_id, 30)
    print(f"      Trend Direction: {trends.trend_direction}")
    print(f"      Trend Strength: {trends.trend_strength:.2f}")
    print(f"      Momentum: {trends.momentum:.2%}")
    print(f"      Support Level: ${trends.support_level:.2f}")
    print(f"      Resistance Level: ${trends.resistance_level:.2f}")
    
    # Risk Assessment
    print("   c) Risk Assessment:")
    risk = analytics.assess_risk(test_account_id)
    print(f"      Risk Level: {risk.risk_level}")
    print(f"      Risk Score: {risk.risk_score:.1f}/100")
    print(f"      VaR (95%): ${risk.var_95:.2f}")
    print(f"      Expected Shortfall: ${risk.expected_shortfall:.2f}")
    
    # Predictive Model
    print("   d) Predictive Model:")
    predictions = analytics.generate_predictions(test_account_id, 30)
    print(f"      Predicted Return (30 days): {predictions.predicted_return:.2%}")
    print(f"      Confidence: {predictions.prediction_confidence:.2%}")
    print(f"      Model Accuracy: {predictions.model_accuracy:.2%}")
    print(f"      Bull Case: ${predictions.scenario_analysis['bull_case']:.2f}")
    print(f"      Base Case: ${predictions.scenario_analysis['base_case']:.2f}")
    print(f"      Bear Case: ${predictions.scenario_analysis['bear_case']:.2f}")
    
    # Analytics Dashboard
    print("   e) Analytics Dashboard:")
    dashboard = analytics.get_analytics_dashboard(test_account_id)
    
    if 'summary' in dashboard:
        summary = dashboard['summary']
        print(f"      Overall Status: {summary.get('overall_status', 'unknown')}")
        print(f"      Alerts Count: {summary.get('alerts_count', 0)}")
        
        key_metrics = summary.get('key_metrics', {})
        if key_metrics:
            print(f"      Key Metrics:")
            for metric, value in key_metrics.items():
                if isinstance(value, (int, float)):
                    if 'rate' in metric or 'return' in metric:
                        print(f"        {metric}: {value:.2%}")
                    else:
                        print(f"        {metric}: {value}")
                else:
                    print(f"        {metric}: {value}")
    
    # Display alerts
    alerts = dashboard.get('alerts', [])
    if alerts:
        print(f"      Active Alerts:")
        for alert in alerts:
            print(f"        - {alert}")
    else:
        print(f"      No active alerts")
    
    # Test comparative analysis
    print(f"\n3. Comparative Analysis:")
    comparison = analytics.compare_accounts([test_account_id], TimeFrame.ALL_TIME)
    print(f"   Comparing 1 account")
    
    if 'performance_comparison' in comparison:
        perf_comp = comparison['performance_comparison']
        if 'total_return' in perf_comp:
            avg_return = perf_comp['total_return']['average']
            print(f"   Average Return: {avg_return:.2%}")
    
    print("\n=== Account Analytics Engine Test Complete ===")
    print("✅ All analytics components tested successfully!")

if __name__ == "__main__":
    test_analytics_engine()

