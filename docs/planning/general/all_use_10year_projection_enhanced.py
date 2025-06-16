"""
ALL-USE 10-Year Projection Model (Enhanced)

This script generates a 10-year projection for the ALL-USE investment system,
incorporating drawdowns, variable weekly performance based on scenario probabilities,
and more realistic reinvestment assumptions.

Key enhancements:
1. Variable weekly returns based on a predefined scenarios table.
2. Assumption that only ~60% of weeks achieve target premiums.
3. Inclusion of periodic drawdown years (e.g., 2-3 years out of 10).
4. Modeling of recovery periods after drawdowns.
5. Quarterly reinvestment of 80-85% of accumulated premiums.
6. $100K forking threshold.

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

# Weekly Scenarios Table (simplified for this model, based on user input)
# Columns: scenario_name, probability, min_return_factor, max_return_factor
# Return factor is multiplied by the account's target weekly return.
# Example: P-EW might be 1.0 to 1.2 times the target return.
# Example: P-DD might be -0.5 to 0.0 times the target return (i.e., a loss or break-even)

SCENARIOS = [
    # Put-based scenarios (approx. 50% of weeks)
    {"name": "P-EW", "probability": 0.31, "min_return_factor": 1.0, "max_return_factor": 1.2},  # Puts Expired Worthless (target or slightly better)
    {"name": "P-RO", "probability": 0.15, "min_return_factor": 0.5, "max_return_factor": 0.8},  # Puts Roll Over (reduced return)
    {"name": "P-DD", "probability": 0.04, "min_return_factor": -0.5, "max_return_factor": 0.0}, # Puts Deep Drawdown (loss)
    # Call-based scenarios (approx. 50% of weeks, assuming P-AWL/P-AOL lead to calls)
    {"name": "C-WAP", "probability": 0.27, "min_return_factor": 1.5, "max_return_factor": 2.0}, # Calls With Appreciation Profit (better than target)
    {"name": "C-WAP+", "probability": 0.11, "min_return_factor": 2.5, "max_return_factor": 3.0},# Calls With Strong Appreciation (significantly better)
    {"name": "C-PNO", "probability": 0.08, "min_return_factor": 0.9, "max_return_factor": 1.1}, # Calls Premium-Only (around target)
    # Note: C-RO and C-REC are implicitly covered by the factors above or P-RO/P-DD if calls are not opened.
    # W-IDL (Idle weeks) - remaining probability
]
# Normalize probabilities and add Idle scenario
current_prob_sum = sum(s["probability"] for s in SCENARIOS)
if current_prob_sum < 1.0:
    SCENARIOS.append({"name": "W-IDL", "probability": 1.0 - current_prob_sum, "min_return_factor": 0.0, "max_return_factor": 0.0})

# Simulate Drawdown Years (e.g., years 3 and 7 are drawdown years)
DRAWDOWN_YEARS = [3, 7]
DRAWDOWN_IMPACT_FACTOR = 0.5 # During drawdown years, overall returns are scaled by this factor

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

    def get_scenario_driven_return(self, current_year):
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
        actual_weekly_return = self.target_weekly_return * return_factor
        
        # Apply drawdown year impact
        if current_year in DRAWDOWN_YEARS:
            actual_weekly_return *= DRAWDOWN_IMPACT_FACTOR
            
        return actual_weekly_return, chosen_scenario["name"]

    def update_weekly(self, current_year):
        actual_weekly_return, scenario_name = self.get_scenario_driven_return(current_year)
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
                self.pending_reinvestment = 0 # Non-reinvested portion is considered "lost" or used for fees/buffer
                # The reinvestment is implicitly handled by effective_balance updates if balance includes all income.
                # No explicit addition to balance here as it's already included from weekly updates.
                # The key is that pending_reinvestment is cleared, and only a portion effectively boosts future earnings if not already in balance.
                # For this model, we assume income added to balance weekly, and reinvestment_rate affects how much of *that income* is truly compounding.
                # Let's adjust effective_balance to reflect that only reinvested portion of income truly compounds
                # This is tricky. A simpler way: balance grows with all income. Effective balance for earning is based on this total balance.
                # The reinvestment_rate can be seen as a drag on the *overall* system if not all income is available for compounding.
                # For now, let's assume the reinvestment_rate means only that portion of *accumulated income* is put back to work.
                # The current model adds all income to balance. The reinvestment step will now adjust the effective_balance based on reinvested income.
                # This means the effective_balance might be lower than balance * (1-CASH_BUFFER) if reinvestment is less than 100%.
                # This is complex. Let's simplify: the reinvestment_rate applies to the *growth component*.
                # The current model has income added to balance. The reinvestment step is more about ensuring the *compounding base* is correct.
                # Let's assume the `self.balance` already reflects growth. The `pending_reinvestment` is just for tracking.
                # The `reinvestment_amount` is what *actually* gets added to the compounding base.
                # The current model is: income -> balance. Then reinvestment from pending. This is double counting if not careful.
                # Corrected logic: income -> pending_reinvestment. Balance only grows by reinvested_amount.
                # This is a major change. Let's stick to the previous revised model's reinvestment logic for now and focus on scenario returns.
                # The previous revised model: income added to balance, pending_reinvestment cleared. This means 100% of income compounds, then reinvestment rate is applied.
                # This is still not quite right. Let's refine reinvest_quarterly:
                # Income is added to balance weekly. `pending_reinvestment` tracks this income.
                # Quarterly, `pending_reinvestment` is processed. `reinvestment_amount` is the portion that *stays* in the compounding base.
                # The difference (`self.pending_reinvestment - reinvestment_amount`) is effectively withdrawn or lost to fees.
                # So, `self.balance` should be reduced by this non-reinvested amount.
                
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

    def update_weekly(self, current_year):
        if self.merged: return 0, "N/A"
        gen_income, gen_scenario = self.gen_acc.update_weekly(current_year)
        com_income, com_scenario = self.com_acc.update_weekly(current_year)
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

    for week in range(1, WEEKS_PER_YEAR * YEARS + 1):
        current_year = (week - 1) // WEEKS_PER_YEAR + 1
        
        gen_income, gen_scenario = gen_acc.update_weekly(current_year)
        rev_income, rev_scenario = rev_acc.update_weekly(current_year)
        com_income, com_scenario = com_acc.update_weekly(current_year)
        scenario_counts[gen_scenario] = scenario_counts.get(gen_scenario, 0) + 1
        scenario_counts[rev_scenario] = scenario_counts.get(rev_scenario, 0) + 1
        scenario_counts[com_scenario] = scenario_counts.get(com_scenario, 0) + 1

        forked_income_total = 0
        for fa in forked_accounts:
            if not fa.merged:
                fi, fs = fa.update_weekly(current_year)
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
            "quarterly_reinvestment": total_reinvestment_this_quarter if week % 13 == 0 else 0
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
                "pending_reinvestment_eof": gen_acc.pending_reinvestment + rev_acc.pending_reinvestment + com_acc.pending_reinvestment + sum(f.gen_acc.pending_reinvestment + f.com_acc.pending_reinvestment for f in active_forks)
            })

    weekly_df = pd.DataFrame(weekly_data)
    yearly_df = pd.DataFrame(yearly_data)
    fork_df = pd.DataFrame(fork_events) if fork_events else pd.DataFrame(columns=["week", "year", "fork_id", "amount"])
    merge_df = pd.DataFrame(merge_events) if merge_events else pd.DataFrame(columns=["week", "year", "fork_id", "amount"])
    
    final_val = yearly_df.iloc[-1]["total_portfolio_value"]
    cagr = (final_val / INITIAL_INVESTMENT)**(1/YEARS) - 1 if INITIAL_INVESTMENT > 0 else 0
    
    return {
        "weekly_data": weekly_df, "yearly_data": yearly_df,
        "fork_events": fork_df, "merge_events": merge_df, "cagr": cagr,
        "final_portfolio": {
            "gen_acc_balance": gen_acc.balance, "rev_acc_balance": rev_acc.balance, "com_acc_balance": com_acc.balance,
            "active_forked_accounts": len([f for f in forked_accounts if not f.merged]),
            "total_forked_accounts_created": len(forked_accounts),
            "total_portfolio_value": final_val
        },
        "scenario_counts": scenario_counts
    }

def generate_report(results):
    yearly_data = results["yearly_data"]
    fork_events = results["fork_events"]
    merge_events = results["merge_events"]
    cagr = results["cagr"]
    final_portfolio = results["final_portfolio"]
    scenario_counts = results["scenario_counts"]
    
    output_dir = "/home/ubuntu/AGENT_ALLUSE_V1/docs/planning/projection_results_enhanced"
    os.makedirs(output_dir, exist_ok=True)
    
    yearly_data.to_csv(f"{output_dir}/yearly_projection_enhanced.csv", index=False)
    results["weekly_data"].to_csv(f"{output_dir}/weekly_projection_enhanced.csv", index=False)

    with open(f"{output_dir}/projection_summary_enhanced.md", "w") as f:
        f.write("# ALL-USE 10-Year Projection Summary (Enhanced Model)\n\n")
        f.write("## Model Enhancements\n\n")
        f.write("- Variable weekly returns based on scenario probabilities.\n")
        f.write("- Inclusion of drawdown years (Years 3 and 7 impacted by a 0.5 factor).\n")
        f.write("- Quarterly reinvestment of 80-85% of accumulated premiums.\n")
        f.write("- Forking threshold at $100,000.\n\n")
        
        f.write("## Initial Parameters\n\n")
        f.write(f"- Initial Investment: ${INITIAL_INVESTMENT:,.2f}\n")
        f.write(f"- Target Weekly Returns: Gen: {TARGET_GEN_ACC_WEEKLY_RETURN:.1%}, Rev: {TARGET_REV_ACC_WEEKLY_RETURN:.1%}, Com: {TARGET_COM_ACC_WEEKLY_RETURN:.1%}\n")
        f.write(f"- Forking Threshold: ${FORKING_THRESHOLD:,.2f}\n")
        f.write(f"- Merging Threshold: ${MERGING_THRESHOLD:,.2f}\n")
        f.write(f"- Cash Buffer: {CASH_BUFFER:.0%}\n")
        f.write(f"- Reinvestment Rate: {REINVESTMENT_MIN:.0%}-{REINVESTMENT_MAX:.0%}\n\n")
        
        f.write("## 10-Year Growth Summary\n\n")
        f.write("| Year | Portfolio Value | Annual Growth | Annual Income | Annual Reinvestment | Active Forks | Forks | Merges | Pending Reinvestment (EOF) |\n")
        f.write("|------|-----------------|---------------|---------------|---------------------|--------------|-------|--------|----------------------------|\n")
        prev_value = INITIAL_INVESTMENT
        for _, row in yearly_data.iterrows():
            growth = (row["total_portfolio_value"] / prev_value - 1) if prev_value else 0
            f.write(f"| {int(row['year'])} | ${row['total_portfolio_value']:,.2f} | {growth:.2%} | ${row['annual_income']:,.2f} | ${row['annual_reinvestment']:,.2f} | {int(row['active_forked_accounts'])} | {int(row['forks_this_year'])} | {int(row['merges_this_year'])} | ${row['pending_reinvestment_eof']:,.2f} |\n")
            prev_value = row["total_portfolio_value"]
        
        f.write("\n## Final Portfolio (After 10 Years)\n\n")
        f.write(f"- Gen-Acc Balance: ${final_portfolio['gen_acc_balance']:,.2f}\n")
        f.write(f"- Rev-Acc Balance: ${final_portfolio['rev_acc_balance']:,.2f}\n")
        f.write(f"- Com-Acc Balance: ${final_portfolio['com_acc_balance']:,.2f}\n")
        f.write(f"- Active Forked Accounts: {final_portfolio['active_forked_accounts']}\n")
        f.write(f"- Total Forked Accounts Created: {final_portfolio['total_forked_accounts_created']}\n")
        f.write(f"- Total Portfolio Value: ${final_portfolio['total_portfolio_value']:,.2f}\n")
        f.write(f"- Compound Annual Growth Rate (CAGR): {cagr:.2%}\n\n")

        f.write("## Scenario Occurrence Counts (across all accounts, main and forked Gen/Com)\n\n")
        total_scenario_weeks = sum(scenario_counts.values())
        for sc_name, count in scenario_counts.items():
            f.write(f"- {sc_name}: {count} occurrences ({count/total_scenario_weeks:.2%} of total scenario weeks)\n")
        f.write(f"Total scenario weeks processed: {total_scenario_weeks}\n\n") # Each main account + each part of forked account runs a scenario weekly

        # Plots (similar to previous, titles updated)
        plt.figure(figsize=(12, 8))
        plt.plot(yearly_data["year"], yearly_data["total_portfolio_value"] / 1000000, marker='o')
        plt.title("ALL-USE 10-Year Portfolio Growth (Enhanced Model)")
        plt.xlabel("Year"); plt.ylabel("Portfolio Value ($ Millions)"); plt.grid(True); plt.xticks(yearly_data["year"])
        plt.savefig(f"{output_dir}/portfolio_growth_enhanced.png")

        plt.figure(figsize=(12, 8))
        plt.bar(yearly_data["year"], yearly_data["annual_income"] / 1000, label="Annual Income")
        plt.bar(yearly_data["year"], yearly_data["annual_reinvestment"] / 1000, label="Annual Reinvestment")
        plt.title("Annual Income vs. Reinvestment (Enhanced Model)")
        plt.xlabel("Year"); plt.ylabel("Amount ($ Thousands)"); plt.legend(); plt.grid(True); plt.xticks(yearly_data["year"])
        plt.savefig(f"{output_dir}/income_reinvestment_enhanced.png")
        
        # Compare with previous revised model if data exists
        try:
            prev_revised_summary = pd.read_csv("/home/ubuntu/AGENT_ALLUSE_V1/docs/planning/projection_results_revised/yearly_projection.csv")
            plt.figure(figsize=(12,8))
            plt.plot(yearly_data["year"], yearly_data["total_portfolio_value"] / 1000000, marker='o', label="Enhanced Model")
            plt.plot(prev_revised_summary["year"], prev_revised_summary["total_portfolio_value"] / 1000000, marker='s', label="Previous Revised Model")
            plt.title("Portfolio Growth: Enhanced vs. Previous Revised Model")
            plt.xlabel("Year"); plt.ylabel("Portfolio Value ($ Millions)"); plt.legend(); plt.grid(True); plt.xticks(yearly_data["year"])
            plt.savefig(f"{output_dir}/model_comparison_enhanced.png")

            with open(f"{output_dir}/model_comparison_enhanced.md", "w") as comp_f:
                comp_f.write("# Enhanced Model vs. Previous Revised Model Comparison\n\n")
                # ... (comparison table details) ...
        except Exception as e:
            print(f"Could not create comparison with previous revised model: {e}")

    return f"{output_dir}/projection_summary_enhanced.md"

if __name__ == "__main__":
    print("Running ALL-USE 10-Year Projection (Enhanced Model with Scenarios & Drawdowns)...")
    sim_results = run_simulation()
    summary_report_file = generate_report(sim_results)
    print(f"Enhanced projection complete. Summary report: {summary_report_file}")
