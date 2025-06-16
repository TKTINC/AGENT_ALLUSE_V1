# ALL-USE Core Parameters and Logic

This document defines the core parameters and logic of the ALL-USE system that must be strictly followed in all implementation aspects of the ALL-USE agent.

## 1. Weekly Return Rates

The ALL-USE system is built on specific weekly return rates tied to option delta selections:

| Account Type | Weekly Return | Delta Range | Target Stocks |
|-------------|---------------|-------------|---------------|
| Gen-Acc | 1.5% | 40-50 delta | Volatile stocks (e.g., TSLA, NVDA) |
| Rev-Acc | 1.0% | 30-40 delta | Stable market leaders (e.g., AAPL, AMZN, MSFT) |
| Com-Acc | 0.5% | 20-30 delta | Same stable market leaders |

These weekly return rates are foundational to the ALL-USE system and must be maintained in all calculations and projections.

## 2. Account Structure

The ALL-USE system uses a three-tiered account structure:

1. **Generation Account (Gen-Acc)**
   - Primary purpose: Weekly premium harvesting
   - Initial allocation: 40% of total investment (with 5% cash buffer)
   - Weekly return: 1.5% (40-50 delta options on volatile stocks)
   - Entry protocol: Thursday entry

2. **Revenue Account (Rev-Acc)**
   - Primary purpose: Stable income generation
   - Initial allocation: 30% of total investment (with 5% cash buffer)
   - Weekly return: 1.0% (30-40 delta options on stable market leaders)
   - Entry protocol: Monday-Wednesday entry

3. **Compounding Account (Com-Acc)**
   - Primary purpose: Long-term geometric growth
   - Initial allocation: 30% of total investment (with 5% cash buffer)
   - Weekly return: 0.5% (20-30 delta options on stable market leaders)
   - No withdrawals permitted

## 3. Reinvestment Protocol

Reinvestment follows a strict quarterly schedule with specific allocations:

1. **Gen-Acc Reinvestment**
   - Frequency: Variable based on forking threshold
   - Allocation: Reinvests as feasible based on forking requirements

2. **Rev-Acc Reinvestment**
   - Frequency: Quarterly
   - Allocation: 75% to contracts, 25% to LEAPS

3. **Com-Acc Reinvestment**
   - Frequency: Quarterly
   - Allocation: 75% to contracts, 25% to LEAPS

## 4. Account Forking and Merging

The ALL-USE system implements a specific forking and merging mechanism:

1. **Forking Protocol**
   - Trigger: $50,000 surplus in Gen-Acc
   - Split: 50% remains in Gen-Acc, 50% creates new account
   - New account allocation: 50% to new Gen-Acc, 50% to Com-Acc
   - Purpose: Creates geometric rather than linear growth

2. **Merging Protocol**
   - Trigger: Forked account reaches $500,000
   - Process: Forked account merges into parent Com-Acc
   - Purpose: Consolidates growth into the compounding engine

## 5. Annual Effective Rates

When calculating long-term projections, the weekly rates convert to the following effective annual rates:

| Account Type | Weekly Rate | Effective Annual Rate |
|-------------|-------------|------------------------|
| Gen-Acc | 1.5% | 117% |
| Rev-Acc | 1.0% | 68% |
| Com-Acc | 0.5% | 30% |

## 6. Income vs. Growth Split

For projection purposes, the ALL-USE system assumes:

- 80% of returns are available as income
- 20% of returns are reinvested for portfolio growth

## 7. Market Variation

The ALL-USE system accounts for market variation:

- Good years (60%): 10% better returns
- Average years (30%): Expected returns
- Poor years (10%): 20% lower returns

## Implementation Requirements

All components of the ALL-USE agent must strictly adhere to these core parameters. These are not guidelines but fundamental rules that define the ALL-USE system. Any deviation from these parameters must be explicitly approved and documented.

The implementation must ensure:

1. All calculations and projections use these exact parameters
2. The protocol engine enforces these rules in all trading decisions
3. Account management follows the specified structure and forking/merging logic
4. Reinvestment adheres to the quarterly schedule and allocation percentages
5. All visualizations and explanations accurately reflect these parameters

These core parameters represent the mathematical edge of the ALL-USE system and are essential to its wealth-building capabilities.
