"""
ALL-USE 10-Year Projection Model (Conservative)

This script generates a 10-year projection for the ALL-USE investment system,
incorporating highly conservative assumptions:

Key conservative features:
1. Significantly reduced high-return scenario multipliers
2. Increased drawdown frequency (4 years out of 10)
3. Strict 60% success rate parameter for weekly returns
4. Account-specific drawdown exposure (Com-Acc fully exposed, Gen-Acc and Rev-Acc minimally exposed)
5. Quarterly reinvestment of 80-85% of accumulated premiums
6. $100K forking threshold

Core parameters:
- Initial investment: $300,000
- Account structure: Gen-Acc (40%), Rev-Acc (30%), Com-Acc (30%)
- Target weekly returns: Gen-Acc (1.5%), Rev-Acc (1.0%), Com-Acc (0.5%)
- Forking threshold: $100,000 surplus in Gen-Acc
- Merging threshold: $500,000 in forked account
- Reinvestment: Quarterly (80-85% of accumulated premiums)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Constants
INITIAL_INVESTMENT = 300000
GEN_ACC_ALLOCATION = 0.40
REV_ACC_ALLOCATION = 0.30
COM_ACC_ALLOCATION = 0.30
# Target weekly returns (used as a base for scenario-driven returns)
TARGET_GEN_ACC_WEEKLY_RETURN = 0.015
TARGET_REV_ACC_WEEKLY_RETURN = 0.010
TARGET_COM_ACC_WEEKLY_RETURN = 0.005

FORKING_THRESHOLD = 100000
MERGING_THRESHOLD = 500000
WEEKS_PER_YEAR = 52
YEARS = 10
CASH_BUFFER = 0.05  # 5% cash buffer
REINVESTMENT_MIN = 0.80
REINVESTMENT_MAX = 0.85

# Weekly Scenarios Table with REDUCED multipliers for conservative projection
# Columns: scenario_name, probability, min_return_factor, max_return_factor
SCENARIOS = [
    # Put-based scenarios (approx. 50% of weeks)
    {"name": "P-EW", "probability": 0.31, "min_return_factor": 0.8, "max_return_factor": 1.0},  # Puts Expired Worthless (reduced)
    {"name": "P-RO", "probability": 0.15, "min_return_factor": 0.3, "max_return_factor": 0.5},  # Puts Roll Over (reduced)
    {"name": "P-DD", "probability": 0.04, "min_return_factor": -0.8, "max_return_factor": -0.3}, # Puts Deep Drawdown (worse)
    # Call-based scenarios (approx. 50% of weeks, assuming P-AWL/P-AOL lead to calls)
    {"name": "C-WAP", "probability": 0.27, "min_return_factor": 1.0, "max_return_factor": 1.3}, # Calls With Appreciation Profit (reduced)
    {"name": "C-WAP+", "probability": 0.11, "min_return_factor": 1.5, "max_return_factor": 2.0},# Calls With Strong Appreciation (reduced)
    {"name": "C-PNO", "probability": 0.08, "min_return_factor": 0.7, "max_return_factor": 0.9}, # Calls Premium-Only (reduced)
    # Note: C-RO and C-REC are implicitly covered by the factors above or P-RO/P-DD if calls are not opened.
    # W-IDL (Idle weeks) - remaining probability
]
# Normalize probabilities and add Idle scenario
current_prob_sum = sum(s["probability"] for s in SCENARIOS)
if current_prob_sum < 1.0:
    SCENARIOS.append({"name": "W-IDL", "probability": 1.0 - current_prob_sum, "min_return_factor": 0.0, "max_return_factor": 0.0})

# Simulate MORE Drawdown Years (4 out of 10 years)
DRAWDOWN_YEARS = [2, 4, 7, 9]  # Increased from 2 to 4 years

# Account-specific drawdown impact factors
# Com-Acc is fully exposed to drawdowns, Gen-Acc and Rev-Acc have limited exposure
DRAWDOWN_IMPACT_FACTOR_COM_ACC = 0.4  # Severe impact (fully exposed)
DRAWDOWN_IMPACT_FACTOR_GEN_ACC = 0.9  # Minimal impact (24-48hr exposure only)
DRAWDOWN_IMPACT_FACTOR_REV_ACC = 0.8  # Limited impact (24-48hr exposure only)

# Strict 60% success rate implementation
SUCCESS_RATE = 0.60  # Only 60% of weeks will achieve scenario-based returns

class Account:
    def __init__(self, name, initial_balance, target_weekly_return, is_parent=True, parent=None):
        self.name = name
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.target_weekly_return = target_weekly_return
        self.is_parent = is_parent
        self.parent = parent
        self.history = [initial_balance]
        self.income_history = [0]
        self.week_count = 0
        self.effective_balance = initial_balance * (1 - CASH_BUFFER)
        self.pending_reinvestment = 0

    def get_scenario_driven_return(self, current_year, is_success_week, market_condition):
        # Apply market condition (drawdown year effect) based on account type
        if current_year in DRAWDOWN_YEARS:
            if "Gen-Acc" in self.name:
                market_factor = DRAWDOWN_IMPACT_FACTOR_GEN_ACC
            elif "Rev-Acc" in self.name:
                market_factor = DRAWDOWN_IMPACT_FACTOR_REV_ACC
            elif "Com-Acc" in self.name:
                market_factor = DRAWDOWN_IMPACT_FACTOR_COM_ACC
            else:
                market_factor = 0.7  # Default for unknown account types
        else:
            market_factor = 1.0
            
        # Apply success rate check
        if not is_success_week:
            # On non-success weeks, return between -0.2% and +0.2% (near break-even)
            return random.uniform(-0.002, 0.002), "BREAK-EVEN"
            
        # Choose a scenario based on probabilities
        rand_val = random.random()
        cumulative_prob = 0
        chosen_scenario = None
        for scenario in SCENARIOS:
            cumulative_prob += scenario["probability"]
            if rand_val <= cumulative_prob:
                chosen_scenario = scenario
                break
        
        if not chosen_scenario: # Fallback, should not happen if probabilities sum to 1
            chosen_scenario = SCENARIOS[-1]

        # Calculate return based on scenario
        return_factor = random.uniform(chosen_scenario["min_return_factor"], chosen_scenario["max_return_factor"])
        actual_weekly_return = self.target_weekly_return * return_factor * market_factor
            
        return actual_weekly_return, chosen_scenario["name"]

    def update_weekly(self, current_year, is_success_week, market_condition):
        actual_weekly_return, scenario_name = self.get_scenario_driven_return(current_year, is_success_week, market_condition)
        weekly_income = self.effective_balance * actual_weekly_return
        
        self.pending_reinvestment += weekly_income
        self.balance += weekly_income
        self.effective_balance = self.balance * (1 - CASH_BUFFER) # Update effective balance for next week
        
        self.history.append(self.balance)
        self.income_history.append(weekly_income)
        self.week_count += 1
        return weekly_income, scenario_name

    def reinvest_quarterly(self, week):
        if week % 13 == 0 and week > 0:
            if self.pending_reinvestment > 0:
                reinvestment_rate = random.uniform(REINVESTMENT_MIN, REINVESTMENT_MAX)
                reinvestment_amount = self.pending_reinvestment * reinvestment_rate
                
                non_reinvested_amount = self.pending_reinvestment * (1 - reinvestment_rate) # Amount not reinvested
                self.balance -= non_reinvested_amount # Adjust balance to reflect only reinvested income
                self.effective_balance = self.balance * (1 - CASH_BUFFER) # Update effective balance
                reinvested_this_quarter = self.pending_reinvestment * reinvestment_rate
                self.pending_reinvestment = 0
                return reinvested_this_quarter
        return 0

    def check_fork_threshold(self):
        if self.name.startswith("Gen-Acc") and self.balance >= self.initial_balance + FORKING_THRESHOLD:
            surplus = self.balance - self.initial_balance
            self.balance = self.initial_balance
            self.effective_balance = self.balance * (1 - CASH_BUFFER)
            self.pending_reinvestment = 0 # Reset pending after fork
            return surplus
        return 0

    def check_merge_threshold(self):
        if not self.is_parent and self.balance >= MERGING_THRESHOLD:
            return True
        return False

class ForkedAccount:
    def __init__(self, id, initial_balance, parent_com_acc):
        self.id = id
        self.initial_balance = initial_balance
        self.gen_acc = Account(f"Gen-Acc-Fork-{id}", initial_balance/2, TARGET_GEN_ACC_WEEKLY_RETURN, False, None)
        self.com_acc = Account(f"Com-Acc-Fork-{id}", initial_balance/2, TARGET_COM_ACC_WEEKLY_RETURN, False, parent_com_acc)
        self.history = [initial_balance]
        self.income_history = [0]
        self.merged = False

    def update_weekly(self, current_year, is_success_week, market_condition):
        if self.merged: return 0, "N/A"
        gen_income, gen_scenario = self.gen_acc.update_weekly(current_year, is_success_week, market_condition)
        com_income, com_scenario = self.com_acc.update_weekly(current_year, is_success_week, market_condition)
        total_income = gen_income + com_income
        self.history.append(self.gen_acc.balance + self.com_acc.balance)
        self.income_history.append(total_income)
        # For simplicity, we'll just return total income and an aggregate scenario name
        return total_income, f"{gen_scenario}/{com_scenario}"

    def reinvest_quarterly(self, week):
        if not self.merged:
            gen_reinvestment = self.gen_acc.reinvest_quarterly(week)
            com_reinvestment = self.com_acc.reinvest_quarterly(week)
            return gen_reinvestment + com_reinvestment
        return 0

    def check_fork_threshold(self):
        if self.merged: return 0
        return self.gen_acc.check_fork_threshold()

    def check_merge_threshold(self):
        if self.merged: return False
        return self.gen_acc.balance + self.com_acc.balance >= MERGING_THRESHOLD

    def merge_to_parent(self):
        if not self.merged:
            self.merged = True
            return self.gen_acc.balance + self.com_acc.balance
        return 0

def run_simulation():
    gen_acc = Account("Gen-Acc-Main", INITIAL_INVESTMENT * GEN_ACC_ALLOCATION, TARGET_GEN_ACC_WEEKLY_RETURN)
    rev_acc = Account("Rev-Acc-Main", INITIAL_INVESTMENT * REV_ACC_ALLOCATION, TARGET_REV_ACC_WEEKLY_RETURN)
    com_acc = Account("Com-Acc-Main", INITIAL_INVESTMENT * COM_ACC_ALLOCATION, TARGET_COM_ACC_WEEKLY_RETURN)
    
    forked_accounts = []
    fork_id_counter = 1
    
    weekly_data = []
    yearly_data = []
    fork_events = []
    merge_events = []
    scenario_counts = {s["name"]: 0 for s in SCENARIOS}
    scenario_counts["BREAK-EVEN"] = 0  # Add break-even tracking
    
    success_weeks = 0
    total_weeks = 0

    for week in range(1, WEEKS_PER_YEAR * YEARS + 1):
        current_year = (week - 1) // WEEKS_PER_YEAR + 1
        total_weeks += 1
        
        # Determine if this is a success week (60% chance)
        is_success_week = random.random() < SUCCESS_RATE
        if is_success_week:
            success_weeks += 1
            
        # Market condition is shared across all accounts (correlation)
        market_condition = "drawdown" if current_year in DRAWDOWN_YEARS else "normal"
        
        gen_income, gen_scenario = gen_acc.update_weekly(current_year, is_success_week, market_condition)
        rev_income, rev_scenario = rev_acc.update_weekly(current_year, is_success_week, market_condition)
        com_income, com_scenario = com_acc.update_weekly(current_year, is_success_week, market_condition)
        
        # Track scenarios
        if gen_scenario == "BREAK-EVEN":
            scenario_counts["BREAK-EVEN"] += 1
        else:
            scenario_counts[gen_scenario] = scenario_counts.get(gen_scenario, 0) + 1
            
        if rev_scenario == "BREAK-EVEN":
            scenario_counts["BREAK-EVEN"] += 1
        else:
            scenario_counts[rev_scenario] = scenario_counts.get(rev_scenario, 0) + 1
            
        if com_scenario == "BREAK-EVEN":
            scenario_counts["BREAK-EVEN"] += 1
        else:
            scenario_counts[com_scenario] = scenario_counts.get(com_scenario, 0) + 1

        forked_income_total = 0
        for fa in forked_accounts:
            if not fa.merged:
                fi, fs = fa.update_weekly(current_year, is_success_week, market_condition)
                forked_income_total += fi
                # Scenario counting for forked accounts can be complex, skipping detailed breakdown for now

        total_reinvestment_this_quarter = 0
        if week % 13 == 0 and week > 0:
            total_reinvestment_this_quarter += gen_acc.reinvest_quarterly(week)
            total_reinvestment_this_quarter += rev_acc.reinvest_quarterly(week)
            total_reinvestment_this_quarter += com_acc.reinvest_quarterly(week)
            for fa in forked_accounts:
                if not fa.merged:
                    total_reinvestment_this_quarter += fa.reinvest_quarterly(week)
        
        surplus = gen_acc.check_fork_threshold()
        if surplus > 0:
            new_fork = ForkedAccount(fork_id_counter, surplus, com_acc)
            forked_accounts.append(new_fork)
            fork_events.append({"week": week, "year": current_year, "fork_id": fork_id_counter, "amount": surplus})
            fork_id_counter += 1

        for fa in forked_accounts:
            if not fa.merged:
                surplus_fa = fa.check_fork_threshold()
                if surplus_fa > 0:
                    new_fork_fa = ForkedAccount(fork_id_counter, surplus_fa, com_acc) # All forked Com-Accs point to main Com-Acc
                    forked_accounts.append(new_fork_fa)
                    fork_events.append({"week": week, "year": current_year, "fork_id": fork_id_counter, "amount": surplus_fa})
                    fork_id_counter += 1
        
        for fa_idx, fa in enumerate(forked_accounts):
            if not fa.merged and fa.check_merge_threshold():
                merged_amount = fa.merge_to_parent()
                com_acc.balance += merged_amount
                com_acc.effective_balance = com_acc.balance * (1-CASH_BUFFER)
                merge_events.append({"week": week, "year": current_year, "fork_id": fa.id, "amount": merged_amount})
        
        active_forks = [f for f in forked_accounts if not f.merged]
        total_forked_val = sum(f.gen_acc.balance + f.com_acc.balance for f in active_forks)
        total_portfolio_val = gen_acc.balance + rev_acc.balance + com_acc.balance + total_forked_val
        total_weekly_inc = gen_income + rev_income + com_income + forked_income_total

        weekly_data.append({
            "week": week, "year": current_year,
            "gen_acc_balance": gen_acc.balance, "rev_acc_balance": rev_acc.balance, "com_acc_balance": com_acc.balance,
            "forked_accounts_balance": total_forked_val, "total_portfolio_value": total_portfolio_val,
            "weekly_income": total_weekly_inc, "active_forked_accounts": len(active_forks),
            "total_forked_accounts_created": len(forked_accounts),
            "quarterly_reinvestment": total_reinvestment_this_quarter if week % 13 == 0 else 0,
            "is_success_week": is_success_week, "market_condition": market_condition
        })

        if week % WEEKS_PER_YEAR == 0:
            year = current_year
            annual_inc = sum(item["weekly_income"] for item in weekly_data[(week - WEEKS_PER_YEAR):week])
            annual_reinv = sum(item["quarterly_reinvestment"] for item in weekly_data[(week - WEEKS_PER_YEAR):week])
            yearly_data.append({
                "year": year, "gen_acc_balance": gen_acc.balance, "rev_acc_balance": rev_acc.balance,
                "com_acc_balance": com_acc.balance, "forked_accounts_balance": total_forked_val,
                "total_portfolio_value": total_portfolio_val, "annual_income": annual_inc,
                "annual_reinvestment": annual_reinv,
                "active_forked_accounts": len(active_forks), "total_forked_accounts_created": len(forked_accounts),
                "forks_this_year": sum(1 for fe in fork_events if fe["year"] == year),
                "merges_this_year": sum(1 for me in merge_events if me["year"] == year),
                "pending_reinvestment_eof": gen_acc.pending_reinvestment + rev_acc.pending_reinvestment + com_acc.pending_reinvestment + sum(f.gen_acc.pending_reinvestment + f.com_acc.pending_reinvestment for f in active_forks),
                "is_drawdown_year": year in DRAWDOWN_YEARS
            })

    weekly_df = pd.DataFrame(weekly_data)
    yearly_df = pd.DataFrame(yearly_data)
    fork_df = pd.DataFrame(fork_events) if fork_events else pd.DataFrame(columns=["week", "year", "fork_id", "amount"])
    merge_df = pd.DataFrame(merge_events) if merge_events else pd.DataFrame(columns=["week", "year", "fork_id", "amount"])
    
    final_val = yearly_df.iloc[-1]["total_portfolio_value"]
    cagr = (final_val / INITIAL_INVESTMENT)**(1/YEARS) - 1 if INITIAL_INVESTMENT > 0 else 0
    
    # Calculate actual success rate
    actual_success_rate = success_weeks / total_weeks if total_weeks > 0 else 0
    
    return {
        "weekly_data": weekly_df, "yearly_data": yearly_df,
        "fork_events": fork_df, "merge_events": merge_df, "cagr": cagr,
        "final_portfolio": {
            "gen_acc_balance": gen_acc.balance, "rev_acc_balance": rev_acc.balance, "com_acc_balance": com_acc.balance,
            "active_forked_accounts": len([f for f in forked_accounts if not f.merged]),
            "total_forked_accounts_created": len(forked_accounts),
            "total_portfolio_value": final_val
        },
        "scenario_counts": scenario_counts,
        "actual_success_rate": actual_success_rate,
        "success_weeks": success_weeks,
        "total_weeks": total_weeks
    }

def generate_report(results):
    yearly_data = results["yearly_data"]
    fork_events = results["fork_events"]
    merge_events = results["merge_events"]
    cagr = results["cagr"]
    final_portfolio = results["final_portfolio"]
    scenario_counts = results["scenario_counts"]
    actual_success_rate = results["actual_success_rate"]
    success_weeks = results["success_weeks"]
    total_weeks = results["total_weeks"]
    
    output_dir = "/home/ubuntu/AGENT_ALLUSE_V1/docs/planning/projection_results_conservative"
    os.makedirs(output_dir, exist_ok=True)
    
    yearly_data.to_csv(f"{output_dir}/yearly_projection_conservative.csv", index=False)
    results["weekly_data"].to_csv(f"{output_dir}/weekly_projection_conservative.csv", index=False)

    with open(f"{output_dir}/projection_summary_conservative.md", "w") as f:
        f.write("# ALL-USE 10-Year Projection Summary (Conservative Model)\n\n")
        f.write("## Conservative Model Enhancements\n\n")
        f.write("1. **Significantly reduced high-return scenario multipliers**\n")
        f.write("   - C-WAP: Reduced from 1.5-2.0x to 1.0-1.3x target returns\n")
        f.write("   - C-WAP+: Reduced from 2.5-3.0x to 1.5-2.0x target returns\n")
        f.write("   - P-EW: Reduced from 1.0-1.2x to 0.8-1.0x target returns\n")
        f.write("   - All other scenarios similarly reduced\n\n")
        
        f.write("2. **Increased drawdown frequency**\n")
        f.write(f"   - {len(DRAWDOWN_YEARS)} drawdown years out of 10 (Years {', '.join(map(str, DRAWDOWN_YEARS))})\n")
        f.write(f"   - Account-specific drawdown impact:\n")
        f.write(f"     - Com-Acc: {DRAWDOWN_IMPACT_FACTOR_COM_ACC:.1f}x (fully exposed)\n")
        f.write(f"     - Gen-Acc: {DRAWDOWN_IMPACT_FACTOR_GEN_ACC:.1f}x (minimal impact, 24-48hr exposure)\n")
        f.write(f"     - Rev-Acc: {DRAWDOWN_IMPACT_FACTOR_REV_ACC:.1f}x (limited impact, 24-48hr exposure)\n\n")
        
        f.write("3. **Strict 60% success rate implementation**\n")
        f.write(f"   - Target: {SUCCESS_RATE:.0%} of weeks achieve scenario-based returns\n")
        f.write(f"   - Actual: {actual_success_rate:.1%} of weeks achieved scenario-based returns ({success_weeks} out of {total_weeks})\n")
        f.write("   - Non-success weeks limited to -0.2% to +0.2% returns (near break-even)\n\n")
        
        f.write("4. **Market correlation**\n")
        f.write("   - Drawdowns affect all accounts simultaneously (with account-specific impact factors)\n")
        f.write("   - Shared market conditions across all accounts\n\n")
        
        f.write("## Initial Parameters\n\n")
        f.write(f"- Initial Investment: ${INITIAL_INVESTMENT:,.2f}\n")
        f.write(f"- Target Weekly Returns: Gen: {TARGET_GEN_ACC_WEEKLY_RETURN:.1%}, Rev: {TARGET_REV_ACC_WEEKLY_RETURN:.1%}, Com: {TARGET_COM_ACC_WEEKLY_RETURN:.1%}\n")
        f.write(f"- Forking Threshold: ${FORKING_THRESHOLD:,.2f}\n")
        f.write(f"- Merging Threshold: ${MERGING_THRESHOLD:,.2f}\n")
        f.write(f"- Cash Buffer: {CASH_BUFFER:.0%}\n")
        f.write(f"- Reinvestment Rate: {REINVESTMENT_MIN:.0%}-{REINVESTMENT_MAX:.0%}\n\n")
        
        f.write("## 10-Year Growth Summary\n\n")
        f.write("| Year | Portfolio Value | Annual Growth | Annual Income | Annual Reinvestment | Active Forks | Forks | Merges | Drawdown Year |\n")
        f.write("|------|-----------------|---------------|---------------|---------------------|--------------|-------|--------|---------------|\n")
        prev_value = INITIAL_INVESTMENT
        for _, row in yearly_data.iterrows():
            growth = (row["total_portfolio_value"] / prev_value - 1) if prev_value else 0
            is_drawdown = "Yes" if row["is_drawdown_year"] else "No"
            f.write(f"| {int(row['year'])} | ${row['total_portfolio_value']:,.2f} | {growth:.2%} | ${row['annual_income']:,.2f} | ${row['annual_reinvestment']:,.2f} | {int(row['active_forked_accounts'])} | {int(row['forks_this_year'])} | {int(row['merges_this_year'])} | {is_drawdown} |\n")
            prev_value = row["total_portfolio_value"]
        
        f.write("\n## Final Portfolio (After 10 Years)\n\n")
        f.write(f"- Gen-Acc Balance: ${final_portfolio['gen_acc_balance']:,.2f}\n")
        f.write(f"- Rev-Acc Balance: ${final_portfolio['rev_acc_balance']:,.2f}\n")
        f.write(f"- Com-Acc Balance: ${final_portfolio['com_acc_balance']:,.2f}\n")
        f.write(f"- Active Forked Accounts: {final_portfolio['active_forked_accounts']}\n")
        f.write(f"- Total Forked Accounts Created: {final_portfolio['total_forked_accounts_created']}\n")
        f.write(f"- Total Portfolio Value: ${final_portfolio['total_portfolio_value']:,.2f}\n")
        f.write(f"- Compound Annual Growth Rate (CAGR): {cagr:.2%}\n\n")

        f.write("## Scenario Occurrence Counts\n\n")
        total_scenario_weeks = sum(scenario_counts.values())
        for sc_name, count in scenario_counts.items():
            f.write(f"- {sc_name}: {count} occurrences ({count/total_scenario_weeks:.2%} of total scenario weeks)\n")
        f.write(f"Total scenario weeks processed: {total_scenario_weeks}\n\n")

        # Plots (similar to previous, titles updated)
        plt.figure(figsize=(12, 8))
        plt.plot(yearly_data["year"], yearly_data["total_portfolio_value"] / 1000000, marker='o')
        plt.title("ALL-USE 10-Year Portfolio Growth (Conservative Model)")
        plt.xlabel("Year"); plt.ylabel("Portfolio Value ($ Millions)"); plt.grid(True); plt.xticks(yearly_data["year"])
        plt.savefig(f"{output_dir}/portfolio_growth_conservative.png")

        plt.figure(figsize=(12, 8))
        plt.bar(yearly_data["year"], yearly_data["annual_income"] / 1000, label="Annual Income")
        plt.bar(yearly_data["year"], yearly_data["annual_reinvestment"] / 1000, label="Annual Reinvestment")
        plt.title("Annual Income vs. Reinvestment (Conservative Model)")
        plt.xlabel("Year"); plt.ylabel("Amount ($ Thousands)"); plt.legend(); plt.grid(True); plt.xticks(yearly_data["year"])
        plt.savefig(f"{output_dir}/income_reinvestment_conservative.png")
        
        # Compare with previous models if data exists
        try:
            prev_revised_summary = pd.read_csv("/home/ubuntu/AGENT_ALLUSE_V1/docs/planning/projection_results_revised/yearly_projection.csv")
            prev_enhanced_summary = pd.read_csv("/home/ubuntu/AGENT_ALLUSE_V1/docs/planning/projection_results_enhanced/yearly_projection_enhanced.csv")
            
            plt.figure(figsize=(12,8))
            plt.plot(yearly_data["year"], yearly_data["total_portfolio_value"] / 1000000, marker='o', label="Conservative Model")
            plt.plot(prev_enhanced_summary["year"], prev_enhanced_summary["total_portfolio_value"] / 1000000, marker='s', label="Enhanced Model")
            plt.plot(prev_revised_summary["year"], prev_revised_summary["total_portfolio_value"] / 1000000, marker='^', label="Revised Model")
            plt.title("Portfolio Growth: Comparison of All Models")
            plt.xlabel("Year"); plt.ylabel("Portfolio Value ($ Millions)"); plt.legend(); plt.grid(True); plt.xticks(yearly_data["year"])
            plt.savefig(f"{output_dir}/model_comparison_all.png")

            with open(f"{output_dir}/model_comparison_all.md", "w") as comp_f:
                comp_f.write("# Comparison of All Projection Models\n\n")
                comp_f.write("## Final Portfolio Values (After 10 Years)\n\n")
                comp_f.write("| Model | Final Portfolio Value | CAGR |\n")
                comp_f.write("|-------|----------------------|------|\n")
                comp_f.write(f"| Conservative Model | ${final_portfolio['total_portfolio_value']:,.2f} | {cagr:.2%} |\n")
                
                # Extract values from previous models if available
                try:
                    enhanced_final = prev_enhanced_summary.iloc[-1]["total_portfolio_value"]
                    enhanced_cagr = (enhanced_final / INITIAL_INVESTMENT)**(1/YEARS) - 1
                    comp_f.write(f"| Enhanced Model | ${enhanced_final:,.2f} | {enhanced_cagr:.2%} |\n")
                except:
                    comp_f.write("| Enhanced Model | Data unavailable | - |\n")
                    
                try:
                    revised_final = prev_revised_summary.iloc[-1]["total_portfolio_value"]
                    revised_cagr = (revised_final / INITIAL_INVESTMENT)**(1/YEARS) - 1
                    comp_f.write(f"| Revised Model | ${revised_final:,.2f} | {revised_cagr:.2%} |\n")
                except:
                    comp_f.write("| Revised Model | Data unavailable | - |\n")
                
                comp_f.write("\n## Key Differences Between Models\n\n")
                comp_f.write("### Conservative Model\n")
                comp_f.write("- Significantly reduced high-return scenario multipliers\n")
                comp_f.write(f"- {len(DRAWDOWN_YEARS)} drawdown years out of 10\n")
                comp_f.write(f"- Strict {SUCCESS_RATE:.0%} success rate for weekly returns\n")
                comp_f.write("- Account-specific drawdown impact (Com-Acc fully exposed, Gen-Acc and Rev-Acc minimally exposed)\n\n")
                
                comp_f.write("### Enhanced Model\n")
                comp_f.write("- Variable weekly returns based on scenario probabilities\n")
                comp_f.write("- 2 drawdown years out of 10\n")
                comp_f.write("- No explicit success rate limitation\n")
                comp_f.write("- Equal drawdown impact across all accounts\n\n")
                
                comp_f.write("### Revised Model\n")
                comp_f.write("- Quarterly reinvestment of 80-85% of accumulated premiums\n")
                comp_f.write("- $100K forking threshold\n")
                comp_f.write("- No drawdown years or scenario variability\n")
                comp_f.write("- Consistent weekly returns\n")
        except Exception as e:
            print(f"Could not create comparison with previous models: {e}")

    return f"{output_dir}/projection_summary_conservative.md"

if __name__ == "__main__":
    print("Running ALL-USE 10-Year Projection (Conservative Model)...")
    sim_results = run_simulation()
    summary_report_file = generate_report(sim_results)
    print(f"Conservative projection complete. Summary report: {summary_report_file}")
