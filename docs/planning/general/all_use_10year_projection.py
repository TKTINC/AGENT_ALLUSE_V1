"""
ALL-USE 10-Year Projection Model

This script generates a 10-year projection for the ALL-USE investment system,
showing both income generation and growth based on the protocol rules and
implementation plan.

Key parameters:
- Initial investment: $300,000
- Account structure: Gen-Acc (40%), Rev-Acc (30%), Com-Acc (30%)
- Weekly returns: Gen-Acc (1.5%), Rev-Acc (1.0%), Com-Acc (0.5%)
- Forking threshold: $50,000 surplus in Gen-Acc
- Merging threshold: $500,000 in forked account
- Reinvestment: Quarterly (75% contracts, 25% LEAPS) for Rev-Acc and Com-Acc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Constants
INITIAL_INVESTMENT = 300000
GEN_ACC_ALLOCATION = 0.40
REV_ACC_ALLOCATION = 0.30
COM_ACC_ALLOCATION = 0.30
GEN_ACC_WEEKLY_RETURN = 0.015  # 1.5%
REV_ACC_WEEKLY_RETURN = 0.010  # 1.0%
COM_ACC_WEEKLY_RETURN = 0.005  # 0.5%
FORKING_THRESHOLD = 50000
MERGING_THRESHOLD = 500000
WEEKS_PER_YEAR = 52
YEARS = 10
CASH_BUFFER = 0.05  # 5% cash buffer

# Initialize accounts
class Account:
    def __init__(self, name, initial_balance, weekly_return, is_parent=True, parent=None):
        self.name = name
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.weekly_return = weekly_return
        self.is_parent = is_parent
        self.parent = parent
        self.history = [initial_balance]
        self.income_history = [0]
        self.week_count = 0
        self.effective_balance = initial_balance * (1 - CASH_BUFFER)  # Account for cash buffer
        
    def update_weekly(self):
        weekly_income = self.effective_balance * self.weekly_return
        self.balance += weekly_income
        self.effective_balance = self.balance * (1 - CASH_BUFFER)
        self.history.append(self.balance)
        self.income_history.append(weekly_income)
        self.week_count += 1
        return weekly_income
        
    def reinvest_quarterly(self, week):
        # Quarterly reinvestment for Rev-Acc and Com-Acc
        if week % 13 == 0 and week > 0:  # Every 13 weeks (quarterly)
            # No specific action needed as growth is already accounted for in weekly updates
            pass
            
    def check_fork_threshold(self):
        # Check if Gen-Acc has reached forking threshold
        if self.name.startswith("Gen-Acc") and self.balance >= self.initial_balance + FORKING_THRESHOLD:
            surplus = self.balance - self.initial_balance
            self.balance = self.initial_balance
            self.effective_balance = self.balance * (1 - CASH_BUFFER)
            return surplus
        return 0
        
    def check_merge_threshold(self):
        # Check if forked account has reached merging threshold
        if not self.is_parent and self.balance >= MERGING_THRESHOLD:
            return True
        return False

class ForkedAccount:
    def __init__(self, id, initial_balance, parent_com_acc):
        self.id = id
        self.initial_balance = initial_balance
        self.gen_acc = Account(f"Gen-Acc-Fork-{id}", initial_balance/2, GEN_ACC_WEEKLY_RETURN, False, None)
        self.com_acc = Account(f"Com-Acc-Fork-{id}", initial_balance/2, COM_ACC_WEEKLY_RETURN, False, parent_com_acc)
        self.history = [initial_balance]
        self.income_history = [0]
        self.merged = False
        
    def update_weekly(self):
        if self.merged:
            return 0
            
        gen_income = self.gen_acc.update_weekly()
        com_income = self.com_acc.update_weekly()
        total_income = gen_income + com_income
        
        self.history.append(self.gen_acc.balance + self.com_acc.balance)
        self.income_history.append(total_income)
        
        return total_income
        
    def reinvest_quarterly(self, week):
        if not self.merged:
            self.gen_acc.reinvest_quarterly(week)
            self.com_acc.reinvest_quarterly(week)
            
    def check_fork_threshold(self):
        if self.merged:
            return 0
        return self.gen_acc.check_fork_threshold()
        
    def check_merge_threshold(self):
        if self.merged:
            return False
        return self.gen_acc.balance + self.com_acc.balance >= MERGING_THRESHOLD
        
    def merge_to_parent(self):
        if not self.merged:
            self.merged = True
            return self.gen_acc.balance + self.com_acc.balance
        return 0

def run_simulation():
    # Initialize main accounts
    gen_acc = Account("Gen-Acc-Main", INITIAL_INVESTMENT * GEN_ACC_ALLOCATION, GEN_ACC_WEEKLY_RETURN)
    rev_acc = Account("Rev-Acc-Main", INITIAL_INVESTMENT * REV_ACC_ALLOCATION, REV_ACC_WEEKLY_RETURN)
    com_acc = Account("Com-Acc-Main", INITIAL_INVESTMENT * COM_ACC_ALLOCATION, COM_ACC_WEEKLY_RETURN)
    
    # Track forked accounts
    forked_accounts = []
    fork_id = 1
    
    # Track weekly and yearly data
    weekly_data = []
    yearly_data = []
    fork_events = []
    merge_events = []
    
    # Run simulation for 10 years (520 weeks)
    for week in range(1, WEEKS_PER_YEAR * YEARS + 1):
        # Update main accounts
        gen_income = gen_acc.update_weekly()
        rev_income = rev_acc.update_weekly()
        com_income = com_acc.update_weekly()
        
        # Update forked accounts
        forked_income = 0
        for forked_acc in forked_accounts:
            if not forked_acc.merged:
                forked_income += forked_acc.update_weekly()
        
        # Check for quarterly reinvestment
        gen_acc.reinvest_quarterly(week)
        rev_acc.reinvest_quarterly(week)
        com_acc.reinvest_quarterly(week)
        for forked_acc in forked_accounts:
            forked_acc.reinvest_quarterly(week)
        
        # Check for forking in main Gen-Acc
        surplus = gen_acc.check_fork_threshold()
        if surplus > 0:
            new_forked_acc = ForkedAccount(fork_id, surplus, com_acc)
            forked_accounts.append(new_forked_acc)
            fork_events.append({
                'week': week,
                'year': (week - 1) // WEEKS_PER_YEAR + 1,
                'fork_id': fork_id,
                'amount': surplus
            })
            fork_id += 1
        
        # Check for forking in forked Gen-Acc accounts
        for forked_acc in forked_accounts:
            if not forked_acc.merged:
                surplus = forked_acc.check_fork_threshold()
                if surplus > 0:
                    new_forked_acc = ForkedAccount(fork_id, surplus, com_acc)
                    forked_accounts.append(new_forked_acc)
                    fork_events.append({
                        'week': week,
                        'year': (week - 1) // WEEKS_PER_YEAR + 1,
                        'fork_id': fork_id,
                        'amount': surplus
                    })
                    fork_id += 1
        
        # Check for merging
        for forked_acc in forked_accounts:
            if not forked_acc.merged and forked_acc.check_merge_threshold():
                merge_amount = forked_acc.merge_to_parent()
                com_acc.balance += merge_amount
                com_acc.effective_balance = com_acc.balance * (1 - CASH_BUFFER)
                merge_events.append({
                    'week': week,
                    'year': (week - 1) // WEEKS_PER_YEAR + 1,
                    'fork_id': forked_acc.id,
                    'amount': merge_amount
                })
        
        # Calculate total portfolio value and income
        active_forked_accounts = [acc for acc in forked_accounts if not acc.merged]
        total_forked_value = sum(acc.gen_acc.balance + acc.com_acc.balance for acc in active_forked_accounts)
        total_portfolio_value = gen_acc.balance + rev_acc.balance + com_acc.balance + total_forked_value
        total_weekly_income = gen_income + rev_income + com_income + forked_income
        
        # Record weekly data
        weekly_data.append({
            'week': week,
            'year': (week - 1) // WEEKS_PER_YEAR + 1,
            'gen_acc_balance': gen_acc.balance,
            'rev_acc_balance': rev_acc.balance,
            'com_acc_balance': com_acc.balance,
            'forked_accounts_balance': total_forked_value,
            'total_portfolio_value': total_portfolio_value,
            'weekly_income': total_weekly_income,
            'active_forked_accounts': len(active_forked_accounts),
            'total_forked_accounts': len(forked_accounts)
        })
        
        # Record yearly data at the end of each year
        if week % WEEKS_PER_YEAR == 0:
            year = week // WEEKS_PER_YEAR
            
            # Calculate annual income by summing weekly income for the year
            annual_income = sum(item['weekly_income'] for item in weekly_data[(week-WEEKS_PER_YEAR):week])
            
            yearly_data.append({
                'year': year,
                'gen_acc_balance': gen_acc.balance,
                'rev_acc_balance': rev_acc.balance,
                'com_acc_balance': com_acc.balance,
                'forked_accounts_balance': total_forked_value,
                'total_portfolio_value': total_portfolio_value,
                'annual_income': annual_income,
                'active_forked_accounts': len(active_forked_accounts),
                'total_forked_accounts': len(forked_accounts),
                'forks_this_year': sum(1 for event in fork_events if event['year'] == year),
                'merges_this_year': sum(1 for event in merge_events if event['year'] == year)
            })
    
    # Convert to DataFrames
    weekly_df = pd.DataFrame(weekly_data)
    yearly_df = pd.DataFrame(yearly_data)
    fork_df = pd.DataFrame(fork_events) if fork_events else pd.DataFrame(columns=['week', 'year', 'fork_id', 'amount'])
    merge_df = pd.DataFrame(merge_events) if merge_events else pd.DataFrame(columns=['week', 'year', 'fork_id', 'amount'])
    
    # Calculate CAGR
    initial_value = INITIAL_INVESTMENT
    final_value = yearly_df.iloc[-1]['total_portfolio_value']
    years = YEARS
    cagr = (final_value / initial_value) ** (1 / years) - 1
    
    return {
        'weekly_data': weekly_df,
        'yearly_data': yearly_df,
        'fork_events': fork_df,
        'merge_events': merge_df,
        'cagr': cagr,
        'final_portfolio': {
            'gen_acc_balance': gen_acc.balance,
            'rev_acc_balance': rev_acc.balance,
            'com_acc_balance': com_acc.balance,
            'active_forked_accounts': len([acc for acc in forked_accounts if not acc.merged]),
            'total_forked_accounts': len(forked_accounts),
            'total_portfolio_value': final_value
        }
    }

def generate_report(results):
    yearly_data = results['yearly_data']
    fork_events = results['fork_events']
    merge_events = results['merge_events']
    cagr = results['cagr']
    final_portfolio = results['final_portfolio']
    
    # Create output directory if it doesn't exist
    output_dir = '/home/ubuntu/AGENT_ALLUSE_V1/docs/planning/projection_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save yearly data to CSV
    yearly_data.to_csv(f'{output_dir}/yearly_projection.csv', index=False)
    
    # Generate summary report
    with open(f'{output_dir}/projection_summary.md', 'w') as f:
        f.write("# ALL-USE 10-Year Projection Summary\n\n")
        f.write("## Initial Parameters\n\n")
        f.write(f"- Initial Investment: ${INITIAL_INVESTMENT:,.2f}\n")
        f.write(f"- Gen-Acc Allocation: {GEN_ACC_ALLOCATION:.0%} (${INITIAL_INVESTMENT * GEN_ACC_ALLOCATION:,.2f})\n")
        f.write(f"- Rev-Acc Allocation: {REV_ACC_ALLOCATION:.0%} (${INITIAL_INVESTMENT * REV_ACC_ALLOCATION:,.2f})\n")
        f.write(f"- Com-Acc Allocation: {COM_ACC_ALLOCATION:.0%} (${INITIAL_INVESTMENT * COM_ACC_ALLOCATION:,.2f})\n")
        f.write(f"- Gen-Acc Weekly Return: {GEN_ACC_WEEKLY_RETURN:.1%}\n")
        f.write(f"- Rev-Acc Weekly Return: {REV_ACC_WEEKLY_RETURN:.1%}\n")
        f.write(f"- Com-Acc Weekly Return: {COM_ACC_WEEKLY_RETURN:.1%}\n")
        f.write(f"- Forking Threshold: ${FORKING_THRESHOLD:,.2f}\n")
        f.write(f"- Merging Threshold: ${MERGING_THRESHOLD:,.2f}\n")
        f.write(f"- Cash Buffer: {CASH_BUFFER:.0%}\n\n")
        
        f.write("## 10-Year Growth Summary\n\n")
        f.write("| Year | Total Portfolio Value | Annual Growth | Annual Income | Active Forked Accounts | Forks | Merges |\n")
        f.write("|------|----------------------|---------------|---------------|------------------------|-------|--------|\n")
        
        prev_value = INITIAL_INVESTMENT
        for _, row in yearly_data.iterrows():
            annual_growth = (row['total_portfolio_value'] / prev_value) - 1 if prev_value > 0 else 0
            f.write(f"| {int(row['year'])} | ${row['total_portfolio_value']:,.2f} | {annual_growth:.2%} | ${row['annual_income']:,.2f} | {int(row['active_forked_accounts'])} | {int(row['forks_this_year'])} | {int(row['merges_this_year'])} |\n")
            prev_value = row['total_portfolio_value']
        
        f.write("\n## Final Portfolio (After 10 Years)\n\n")
        f.write(f"- Gen-Acc Balance: ${final_portfolio['gen_acc_balance']:,.2f}\n")
        f.write(f"- Rev-Acc Balance: ${final_portfolio['rev_acc_balance']:,.2f}\n")
        f.write(f"- Com-Acc Balance: ${final_portfolio['com_acc_balance']:,.2f}\n")
        f.write(f"- Active Forked Accounts: {final_portfolio['active_forked_accounts']}\n")
        f.write(f"- Total Forked Accounts Created: {final_portfolio['total_forked_accounts']}\n")
        f.write(f"- Total Portfolio Value: ${final_portfolio['total_portfolio_value']:,.2f}\n")
        f.write(f"- Compound Annual Growth Rate (CAGR): {cagr:.2%}\n\n")
        
        f.write("## Account Forking Events\n\n")
        if len(fork_events) > 0:
            f.write("| Year | Week | Fork ID | Amount |\n")
            f.write("|------|------|---------|--------|\n")
            for _, row in fork_events.iterrows():
                f.write(f"| {int(row['year'])} | {int(row['week'])} | {int(row['fork_id'])} | ${row['amount']:,.2f} |\n")
        else:
            f.write("No forking events occurred.\n")
        
        f.write("\n## Account Merging Events\n\n")
        if len(merge_events) > 0:
            f.write("| Year | Week | Fork ID | Amount |\n")
            f.write("|------|------|---------|--------|\n")
            for _, row in merge_events.iterrows():
                f.write(f"| {int(row['year'])} | {int(row['week'])} | {int(row['fork_id'])} | ${row['amount']:,.2f} |\n")
        else:
            f.write("No merging events occurred.\n")
        
        f.write("\n## Income Projection\n\n")
        f.write("| Year | Weekly Income (Start) | Weekly Income (End) | Annual Income | Cumulative Income |\n")
        f.write("|------|----------------------|---------------------|---------------|-------------------|\n")
        
        cumulative_income = 0
        for i, row in yearly_data.iterrows():
            year = int(row['year'])
            annual_income = row['annual_income']
            cumulative_income += annual_income
            
            # Get weekly income at start and end of year
            start_week = (year - 1) * WEEKS_PER_YEAR + 1
            end_week = year * WEEKS_PER_YEAR
            
            weekly_start = results['weekly_data'].loc[results['weekly_data']['week'] == start_week, 'weekly_income'].values[0]
            weekly_end = results['weekly_data'].loc[results['weekly_data']['week'] == end_week, 'weekly_income'].values[0]
            
            f.write(f"| {year} | ${weekly_start:,.2f} | ${weekly_end:,.2f} | ${annual_income:,.2f} | ${cumulative_income:,.2f} |\n")
    
    # Generate plots
    plt.figure(figsize=(12, 8))
    plt.plot(yearly_data['year'], yearly_data['total_portfolio_value'] / 1000000, marker='o', linewidth=2)
    plt.title('ALL-USE 10-Year Portfolio Growth', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Portfolio Value ($ Millions)', fontsize=14)
    plt.grid(True)
    plt.xticks(yearly_data['year'])
    plt.savefig(f'{output_dir}/portfolio_growth.png', dpi=300, bbox_inches='tight')
    
    # Plot account distribution
    plt.figure(figsize=(12, 8))
    plt.bar(yearly_data['year'], yearly_data['gen_acc_balance'] / 1000, label='Gen-Acc')
    plt.bar(yearly_data['year'], yearly_data['rev_acc_balance'] / 1000, bottom=yearly_data['gen_acc_balance'] / 1000, label='Rev-Acc')
    plt.bar(yearly_data['year'], yearly_data['com_acc_balance'] / 1000, 
            bottom=(yearly_data['gen_acc_balance'] + yearly_data['rev_acc_balance']) / 1000, label='Com-Acc')
    plt.bar(yearly_data['year'], yearly_data['forked_accounts_balance'] / 1000, 
            bottom=(yearly_data['gen_acc_balance'] + yearly_data['rev_acc_balance'] + yearly_data['com_acc_balance']) / 1000, 
            label='Forked Accounts')
    plt.title('Account Distribution Over Time', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Value ($ Thousands)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xticks(yearly_data['year'])
    plt.savefig(f'{output_dir}/account_distribution.png', dpi=300, bbox_inches='tight')
    
    # Plot annual income
    plt.figure(figsize=(12, 8))
    plt.bar(yearly_data['year'], yearly_data['annual_income'] / 1000)
    plt.title('Annual Income', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Annual Income ($ Thousands)', fontsize=14)
    plt.grid(True)
    plt.xticks(yearly_data['year'])
    plt.savefig(f'{output_dir}/annual_income.png', dpi=300, bbox_inches='tight')
    
    # Plot number of forked accounts
    plt.figure(figsize=(12, 8))
    plt.plot(yearly_data['year'], yearly_data['active_forked_accounts'], marker='o', linewidth=2, label='Active Forked Accounts')
    plt.plot(yearly_data['year'], yearly_data['total_forked_accounts'], marker='s', linewidth=2, label='Total Forked Accounts')
    plt.title('Forked Accounts Over Time', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Accounts', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xticks(yearly_data['year'])
    plt.savefig(f'{output_dir}/forked_accounts.png', dpi=300, bbox_inches='tight')
    
    return f'{output_dir}/projection_summary.md'

if __name__ == "__main__":
    print("Running ALL-USE 10-Year Projection...")
    results = run_simulation()
    summary_file = generate_report(results)
    print(f"Projection complete. Summary report saved to: {summary_file}")
