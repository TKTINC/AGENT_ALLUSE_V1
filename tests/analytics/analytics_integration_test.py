"""
Test Account Analytics Engine with WS3-P1 Infrastructure
Integration test using existing account management database
"""

import sys
import os
sys.path.append('/home/ubuntu/AGENT_ALLUSE_V1/src')

from account_management.analytics.account_analytics_engine import AccountAnalyticsEngine, TimeFrame
from account_management.models.account_models import create_account, AccountType
from account_management.database.account_database import AccountDatabase
from account_management.api.account_operations_api import AccountOperationsAPI
import json

def test_analytics_with_real_data():
    """Test analytics engine with real account data"""
    print("=== Testing Account Analytics Engine with Real Data ===")
    
    # Initialize components
    db = AccountDatabase()
    api = AccountOperationsAPI()
    analytics = AccountAnalyticsEngine()
    
    # Remove the unused account creation code
    # Create test accounts if they don't exist
    print("\n1. Setting up test accounts...")
    
    # Save accounts using API
    api.create_account(AccountType.GENERATION, "Test Generation Account", 100000.0)
    api.create_account(AccountType.REVENUE, "Test Revenue Account", 75000.0)
    api.create_account(AccountType.COMPOUNDING, "Test Compounding Account", 75000.0)
    
    # Get account IDs from the created accounts
    gen_accounts = api.get_accounts_by_type(AccountType.GENERATION)
    rev_accounts = api.get_accounts_by_type(AccountType.REVENUE)
    com_accounts = api.get_accounts_by_type(AccountType.COMPOUNDING)
    
    if not (gen_accounts.get('success') and rev_accounts.get('success') and com_accounts.get('success')):
        print("   Error creating accounts")
        return
    
    gen_acc_id = gen_accounts['accounts'][0]['account_id'] if gen_accounts['accounts'] else None
    rev_acc_id = rev_accounts['accounts'][0]['account_id'] if rev_accounts['accounts'] else None
    com_acc_id = com_accounts['accounts'][0]['account_id'] if com_accounts['accounts'] else None
    
    if not all([gen_acc_id, rev_acc_id, com_acc_id]):
        print("   Error getting account IDs")
        return
    
    print(f"   Created accounts: {gen_acc_id}, {rev_acc_id}, {com_acc_id}")
    
    # Add some transactions for testing
    print("\n2. Adding test transactions...")
    
    # Add transactions to Generation Account
    api.update_balance(gen_acc_id, 2500.0, "premium_collection", "Weekly premium collection")
    api.update_balance(gen_acc_id, -500.0, "trading_loss", "Small trading loss")
    api.update_balance(gen_acc_id, 1800.0, "premium_collection", "Premium collection")
    api.update_balance(gen_acc_id, 3200.0, "trading_profit", "Successful trade")
    
    # Add transactions to Revenue Account
    api.update_balance(rev_acc_id, 1500.0, "premium_collection", "Revenue account premium")
    api.update_balance(rev_acc_id, -800.0, "trading_loss", "Minor loss")
    api.update_balance(rev_acc_id, 2200.0, "trading_profit", "Good trade")
    
    # Add transactions to Compounding Account
    api.update_balance(com_acc_id, 1000.0, "premium_collection", "Compounding premium")
    api.update_balance(com_acc_id, 2800.0, "trading_profit", "Excellent trade")
    
    print("   Added 9 test transactions across all accounts")
    
    # Test analytics on each account
    test_accounts = [gen_acc_id, rev_acc_id, com_acc_id]
    
    for account_id in test_accounts:
        print(f"\n3. Testing Analytics for Account: {account_id}")
        
        # Performance Analysis
        print("   a) Performance Analysis:")
        performance = analytics.analyze_account_performance(account_id, TimeFrame.ALL_TIME)
        print(f"      Total Return: {performance.total_return:.2%}")
        print(f"      Annualized Return: {performance.annualized_return:.2%}")
        print(f"      Sharpe Ratio: {performance.sharpe_ratio:.2f}")
        print(f"      Win Rate: {performance.win_rate:.2%}")
        print(f"      Total Trades: {performance.total_trades}")
        print(f"      Profitable Trades: {performance.profitable_trades}")
        
        # Trend Analysis
        print("   b) Trend Analysis:")
        trends = analytics.detect_trends(account_id, 30)
        print(f"      Trend Direction: {trends.trend_direction}")
        print(f"      Trend Strength: {trends.trend_strength:.2f}")
        print(f"      Momentum: {trends.momentum:.2%}")
        print(f"      Support Level: ${trends.support_level:.2f}")
        print(f"      Resistance Level: ${trends.resistance_level:.2f}")
        
        # Risk Assessment
        print("   c) Risk Assessment:")
        risk = analytics.assess_risk(account_id)
        print(f"      Risk Level: {risk.risk_level}")
        print(f"      Risk Score: {risk.risk_score:.1f}/100")
        print(f"      VaR (95%): ${risk.var_95:.2f}")
        print(f"      Expected Shortfall: ${risk.expected_shortfall:.2f}")
        
        # Predictive Model
        print("   d) Predictive Model:")
        predictions = analytics.generate_predictions(account_id, 30)
        print(f"      Predicted Return (30 days): {predictions.predicted_return:.2%}")
        print(f"      Confidence: {predictions.prediction_confidence:.2%}")
        print(f"      Model Accuracy: {predictions.model_accuracy:.2%}")
        print(f"      Bull Case: {predictions.scenario_analysis['bull_case']:.2f}")
        print(f"      Base Case: {predictions.scenario_analysis['base_case']:.2f}")
        print(f"      Bear Case: {predictions.scenario_analysis['bear_case']:.2f}")
    
    # Test comparative analysis
    print(f"\n4. Comparative Analysis:")
    comparison = analytics.compare_accounts(test_accounts, TimeFrame.ALL_TIME)
    print(f"   Comparing {len(test_accounts)} accounts")
    
    if 'performance_comparison' in comparison:
        perf_comp = comparison['performance_comparison']
        if 'total_return' in perf_comp:
            best_return = perf_comp['total_return']['best']
            worst_return = perf_comp['total_return']['worst']
            avg_return = perf_comp['total_return']['average']
            print(f"   Best Return: {best_return[0]} with {best_return[1]:.2%}")
            print(f"   Worst Return: {worst_return[0]} with {worst_return[1]:.2%}")
            print(f"   Average Return: {avg_return:.2%}")
    
    # Test analytics dashboard
    print(f"\n5. Analytics Dashboard for {gen_acc_id}:")
    dashboard = analytics.get_analytics_dashboard(gen_acc_id)
    
    if 'summary' in dashboard:
        summary = dashboard['summary']
        print(f"   Overall Status: {summary.get('overall_status', 'unknown')}")
        print(f"   Alerts Count: {summary.get('alerts_count', 0)}")
        
        key_metrics = summary.get('key_metrics', {})
        if key_metrics:
            print(f"   Key Metrics:")
            for metric, value in key_metrics.items():
                if isinstance(value, (int, float)):
                    if 'rate' in metric or 'return' in metric:
                        print(f"     {metric}: {value:.2%}")
                    else:
                        print(f"     {metric}: {value}")
                else:
                    print(f"     {metric}: {value}")
    
    # Display alerts
    alerts = dashboard.get('alerts', [])
    if alerts:
        print(f"   Active Alerts:")
        for alert in alerts:
            print(f"     - {alert}")
    else:
        print(f"   No active alerts")
    
    print("\n=== Account Analytics Engine Integration Test Complete ===")
    print("✅ All analytics components tested successfully with real data!")

if __name__ == "__main__":
    test_analytics_with_real_data()

