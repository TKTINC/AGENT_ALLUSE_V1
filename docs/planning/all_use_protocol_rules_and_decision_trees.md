# ALL-USE Protocol Rules and Decision Trees

## Overview

This document formalizes the protocol rules and decision trees for the ALL-USE agent, based on the comprehensive implementation framework, weekly scenario table, and narrative provided by the user. These rules and decision trees serve as the definitive reference for the agent's decision-making processes.

## 1. Week Classification System

### 1.1 Week Type Taxonomy

The ALL-USE system uses a detailed week classification system with 11 distinct week types:

#### Put-Based Scenarios (P-prefix)

1. **P-EW: Puts Expired Worthless**
   - **Scenario**: Sell 1DTE Put → expires worthless
   - **Stock Movement**: Stock up or flat
   - **Action**: Collect premium, repeat put selling
   - **Frequency**: 16 weeks (~31% annually)
   - **Expected Weekly Return**: 1.8-2.2%

2. **P-AWL: Puts Assigned Within Limit**
   - **Scenario**: Put assigned at strike
   - **Stock Movement**: Downside 0-5%
   - **Action**: Accept assignment, prepare to sell calls
   - **Frequency**: Combined with call weeks
   - **Expected Weekly Return**: 1.8-2.2%

3. **P-RO: Puts Roll Over**
   - **Scenario**: Put at risk of assignment
   - **Stock Movement**: Downside >5%
   - **Action**: Roll put out (and possibly down) to avoid assignment
   - **Frequency**: 6-8 weeks (~12-15% annually)
   - **Expected Weekly Return**: 1.0-1.5%

4. **P-AOL: Puts Assigned Over Limit**
   - **Scenario**: Put assigned despite exceeding threshold
   - **Stock Movement**: Downside >5%
   - **Action**: Accept assignment when rolling wasn't possible
   - **Frequency**: Combined with call weeks
   - **Expected Weekly Return**: 1.8-2.2%

5. **P-DD: Puts Deep Drawdown**
   - **Scenario**: Extreme market conditions
   - **Stock Movement**: Downside >15%
   - **Action**: Implement GBR-4 protocol for capital preservation
   - **Frequency**: 2 weeks (~4% annually)
   - **Expected Weekly Return**: -0.5-0.0%

#### Call-Based Scenarios (C-prefix)

6. **C-WAP: Calls With Appreciation Profit**
   - **Scenario**: Assigned stock → ATM call sold
   - **Stock Movement**: Stock up 0-5%
   - **Action**: Accept call assignment for profit
   - **Frequency**: 14 weeks (~27% annually)
   - **Expected Weekly Return**: 3.0-4.0%

7. **C-WAP+: Calls With Strong Appreciation Profit**
   - **Scenario**: Assigned stock → ATM call sold
   - **Stock Movement**: Stock up 5-10%
   - **Action**: Accept call assignment for greater profit
   - **Frequency**: 6 weeks (~11% annually)
   - **Expected Weekly Return**: 5.0-6.0%

8. **C-PNO: Calls Premium-Only**
   - **Scenario**: Assigned stock → ATM call sold
   - **Stock Movement**: Stock flat/down 0-5%
   - **Action**: Collect premium for cost basis reduction
   - **Frequency**: 8 weeks (~15% annually)
   - **Expected Weekly Return**: 1.8-2.2%

9. **C-RO: Calls Roll Over**
   - **Scenario**: Assigned stock → ATM call sold
   - **Stock Movement**: Stock down 5-10%
   - **Action**: Roll call down to match current price
   - **Frequency**: 4 weeks (~8% annually)
   - **Expected Weekly Return**: 0.8-1.2%

10. **C-REC: Calls Recovery Mode**
    - **Scenario**: Assigned stock → 20-30 delta call sold
    - **Stock Movement**: Stock down 10-15%
    - **Action**: Sell lower delta calls for recovery room
    - **Frequency**: 2 weeks (~4% annually)
    - **Expected Weekly Return**: 0.5-0.8%

#### Other Scenarios

11. **W-IDL: Week Idle**
    - **Scenario**: No active trades
    - **Stock Movement**: N/A
    - **Action**: Strategic pause due to market conditions
    - **Frequency**: 0-2 weeks (~0-4% annually)
    - **Expected Weekly Return**: 0%

### 1.2 Week Classification Decision Tree

```
START
|
+-- Check Current Position Type
|   |
|   +-- No Position (Cash) → Evaluate for Put Entry
|   |   |
|   |   +-- Check Market Conditions
|   |       |
|   |       +-- Favorable → Sell Put (P-EW expected)
|   |       |
|   |       +-- Unfavorable → W-IDL
|   |
|   +-- Holding Put Position → Evaluate Put Outcome
|   |   |
|   |   +-- Check Stock Movement vs. Strike
|   |       |
|   |       +-- Stock Above Strike → P-EW
|   |       |
|   |       +-- Stock 0-5% Below Strike → P-AWL
|   |       |
|   |       +-- Stock >5% Below Strike
|   |           |
|   |           +-- Can Roll? → P-RO
|   |           |
|   |           +-- Cannot Roll → P-AOL
|   |           |
|   |           +-- Stock >15% Below Strike → P-DD
|   |
|   +-- Holding Stock Position → Evaluate for Call Entry/Outcome
|       |
|       +-- Check Stock Movement vs. Cost Basis
|           |
|           +-- Stock Up 0-5% → C-WAP
|           |
|           +-- Stock Up 5-10% → C-WAP+
|           |
|           +-- Stock Flat/Down 0-5% → C-PNO
|           |
|           +-- Stock Down 5-10% → C-RO
|           |
|           +-- Stock Down 10-15% → C-REC
|
END
```

## 2. Account-Specific Protocol Rules

### 2.1 Generation Account (Gen-Acc) Protocol

1. **Weekly Operation Rules**
   - **Entry Day**: Thursday (standard protocol)
   - **Option Strategy**: 40-50 delta puts
   - **Target Stocks**: Volatile stocks (e.g., TSLA, NVDA)
   - **Position Sizing**: 90-95% of account value (with 5-10% cash buffer)
   - **Expected Weekly Return**: 1.5% (least expected)

2. **Week Type-Specific Actions**
   - **P-EW**: Collect premium, prepare for next Thursday entry
   - **P-AWL**: Accept assignment, prepare to sell calls next week
   - **P-RO**: Roll put out (and possibly down) to avoid assignment
   - **P-AOL**: Accept assignment when rolling wasn't possible
   - **P-DD**: Implement GBR-4 protocol for capital preservation
   - **C-WAP/C-WAP+**: Accept call assignment for profit
   - **C-PNO**: Collect premium for cost basis reduction
   - **C-RO**: Roll call down to match current price
   - **C-REC**: Sell lower delta calls for recovery room
   - **W-IDL**: Hold cash, prepare for next opportunity

3. **Forking Protocol**
   - **Trigger**: $50,000 surplus in Gen-Acc
   - **Action**: Create new forked account
   - **Allocation**: 50% to new Gen-Acc, 50% to Com-Acc within the forked account
   - **Operation**: Forked account follows same weekly protocol as parent accounts

### 2.2 Revenue Account (Rev-Acc) Protocol

1. **Weekly Operation Rules**
   - **Entry Day**: Monday-Wednesday (standard protocol)
   - **Option Strategy**: 30-40 delta puts
   - **Target Stocks**: Stable market leaders (e.g., AAPL, AMZN, MSFT)
   - **Position Sizing**: 90-95% of account value (with 5-10% cash buffer)
   - **Expected Weekly Return**: 1.0% (least expected)

2. **Week Type-Specific Actions**
   - Similar to Gen-Acc but with more conservative delta ranges
   - More emphasis on premium collection than stock assignment

3. **Reinvestment Protocol**
   - **Frequency**: Quarterly
   - **Allocation**: 75% to contracts, 25% to LEAPS
   - **Timing**: March, June, September, December

### 2.3 Compounding Account (Com-Acc) Protocol

1. **Weekly Operation Rules**
   - **Option Strategy**: 20-30 delta puts/calls
   - **Target Stocks**: Same stable market leaders as Rev-Acc
   - **Position Sizing**: Based on holdings and quarterly plan
   - **Expected Weekly Return**: 0.5% (least expected)

2. **Week Type-Specific Actions**
   - Focus on long-term growth rather than weekly income
   - Sell covered calls on held positions

3. **Reinvestment Protocol**
   - **Frequency**: Quarterly
   - **Allocation**: 75% to contracts, 25% to LEAPS
   - **Timing**: March, June, September, December

4. **Merging Protocol**
   - **Trigger**: Forked account reaches $500,000
   - **Action**: Merge forked account into parent Com-Acc
   - **Purpose**: Consolidate growth into the compounding engine

## 3. Trade Management Decision Trees

### 3.1 Put Position Management Decision Tree

```
START
|
+-- Check Days to Expiration
|   |
|   +-- DTE > 1
|   |   |
|   |   +-- Monitor Position
|   |
|   +-- DTE = 1 (Expiration Tomorrow)
|       |
|       +-- Check Stock Price vs. Strike
|           |
|           +-- Stock Price > Strike + Buffer
|           |   |
|           |   +-- Let Expire Worthless (P-EW)
|           |
|           +-- Strike - 5% < Stock Price < Strike + Buffer
|           |   |
|           |   +-- Prepare for Potential Assignment (P-AWL)
|           |
|           +-- Stock Price < Strike - 5%
|               |
|               +-- Check Roll Opportunity
|                   |
|                   +-- Roll Available → Roll Position (P-RO)
|                   |
|                   +-- Roll Not Available → Prepare for Assignment (P-AOL)
|                   |
|                   +-- Stock Price < Strike - 15% → Implement GBR-4 Protocol (P-DD)
|
END
```

### 3.2 Call Position Management Decision Tree

```
START
|
+-- Check Days to Expiration
|   |
|   +-- DTE > 1
|   |   |
|   |   +-- Monitor Position
|   |
|   +-- DTE = 1 (Expiration Tomorrow)
|       |
|       +-- Check Stock Price vs. Strike
|           |
|           +-- Stock Price < Strike - Buffer
|           |   |
|           |   +-- Let Expire Worthless, Keep Stock
|           |
|           +-- Strike - Buffer < Stock Price < Strike + 5%
|           |   |
|           |   +-- Prepare for Assignment (C-WAP)
|           |
|           +-- Strike + 5% < Stock Price < Strike + 10%
|           |   |
|           |   +-- Prepare for Assignment (C-WAP+)
|           |
|           +-- Stock Price < Cost Basis - 5%
|               |
|               +-- Check Roll Opportunity
|                   |
|                   +-- Roll Available → Roll Position (C-RO)
|                   |
|                   +-- Stock Price < Cost Basis - 10% → Switch to Recovery Mode (C-REC)
|
END
```

### 3.3 ATR-Based Adjustment Rules

1. **Monitoring Thresholds**
   - **Close Monitoring**: Underlying moves against position by 0.5-1.0 ATR
   - **Adjustment Consideration**: Underlying moves against position by 1.0-1.5 ATR
   - **Mandatory Action**: Underlying moves against position by >1.5 ATR

2. **Adjustment Actions**
   - **For Puts**:
     - Roll out to later expiration
     - Roll down to lower strike if necessary
     - Consider closing position if roll not favorable
   
   - **For Calls**:
     - Roll out to later expiration
     - Roll down to lower strike to match current price
     - Switch to lower delta calls for recovery room

3. **Profit-Taking Rules**
   - Take profit at 50-80% of maximum premium
   - Close positions by expiration day if not already closed
   - Roll positions if within profit target and expiration approaching

## 4. Account Management Rules

### 4.1 Forking Implementation Rules

1. **Forking Trigger**
   - Monitor Gen-Acc balance after each week
   - Trigger fork when balance exceeds initial balance by $50,000

2. **Forking Process**
   - Calculate fork amount (surplus over initial balance)
   - Create new forked account
   - Allocate 50% of fork amount to new Gen-Acc within forked account
   - Allocate 50% of fork amount to Com-Acc within forked account

3. **Forked Account Operation**
   - Forked account operates independently
   - Follows same weekly protocol as parent accounts
   - Tracks performance separately

### 4.2 Merging Implementation Rules

1. **Merging Trigger**
   - Monitor forked account total balance
   - Trigger merge when forked account reaches $500,000

2. **Merging Process**
   - Close all positions in forked account
   - Transfer full balance to parent Com-Acc
   - Update account tracking and performance metrics

### 4.3 Tax Management Rules

1. **Per-Trade Tax Tracking**
   - Record each trade's tax classification (short-term vs. long-term)
   - Calculate estimated tax liability per trade
   - Track wash sale considerations

2. **Per-Account Tax Efficiency**
   - Prioritize tax-efficient trades when multiple options available
   - Balance tax considerations with protocol requirements
   - Track tax efficiency metrics per account

3. **Annual Tax Planning**
   - Generate year-end tax summary in December
   - Identify tax-loss harvesting opportunities in Q4
   - Prepare documentation for tax filing

## 5. Risk Management Rules

### 5.1 Position-Level Risk Rules

1. **Delta Management**
   - Gen-Acc: Strict 40-50 delta range for puts
   - Rev-Acc: Strict 30-40 delta range for puts
   - Com-Acc: Strict 20-30 delta range for puts/calls
   - Adjust delta based on week classification:
     - Red weeks: Reduce delta by 5
     - Chop weeks: Reduce delta by 10

2. **Position Sizing Rules**
   - Gen-Acc: 90-95% of account value (standard)
   - Rev-Acc: 90-95% of account value (standard)
   - Adjust based on week classification:
     - Green Week: Standard sizing
     - Red Week: Reduce to 70-80% of standard
     - Chop Week: Reduce to 50-60% of standard or no new entries

3. **Stop-Loss Implementation**
   - Implement GBR-4 protocol when stock moves >15% against position
   - Close position if roll would result in unfavorable risk/reward
   - Set maximum drawdown thresholds per account type

### 5.2 Account-Level Risk Rules

1. **Volatility Management**
   - Adjust position sizing based on VIX:
     - VIX < historical average: Standard sizing
     - VIX > historical average: Reduced sizing
     - VIX > 1.5x historical average: Minimum sizing or W-IDL

2. **Drawdown Protection**
   - Set maximum drawdown thresholds:
     - Gen-Acc: 15% maximum drawdown
     - Rev-Acc: 10% maximum drawdown
     - Com-Acc: 7% maximum drawdown
   - Implement recovery protocols when thresholds breached

3. **Black Swan Protocols**
   - Implement GBR-4 protocol for extreme market events
   - Reduce position sizing across all accounts
   - Increase cash buffer temporarily
   - Focus on capital preservation until conditions normalize

## 6. Reporting Rules

### 6.1 Weekly Reporting Requirements

1. **Week Classification Report**
   - Identify and document week type (P-EW, C-WAP, etc.)
   - Record stock movement and action taken
   - Calculate actual weekly return vs. expected

2. **Account Balance Report**
   - Update all account balances
   - Track performance vs. targets
   - Identify any protocol deviations

3. **Next Week Preparation**
   - Forecast next week's actions
   - Identify potential adjustments needed
   - Prepare for upcoming entries

### 6.2 Monthly Reporting Requirements

1. **Performance Summary**
   - Aggregate weekly performance
   - Compare to monthly targets
   - Analyze week type distribution

2. **Protocol Adherence Analysis**
   - Review all protocol decisions
   - Document any exceptions
   - Identify optimization opportunities

3. **Account Growth Tracking**
   - Track month-over-month growth
   - Compare to expected growth rates
   - Identify accounts approaching fork/merge thresholds

### 6.3 Quarterly Reporting Requirements

1. **Quarterly Performance Analysis**
   - Detailed performance breakdown
   - Week type frequency analysis
   - Return attribution by account and strategy

2. **Reinvestment Execution**
   - Document all reinvestment actions
   - Track allocation to contracts vs. LEAPS
   - Measure impact on account growth

3. **Strategic Recommendations**
   - Identify protocol optimization opportunities
   - Suggest parameter adjustments
   - Recommend strategic changes if needed

### 6.4 Annual Reporting Requirements

1. **Yearly Performance Review**
   - Comprehensive performance analysis
   - Compare to annual targets
   - Calculate effective annual return rates

2. **Tax Implications Summary**
   - Summarize tax liabilities
   - Document tax-efficient actions taken
   - Prepare tax documentation

3. **Long-term Growth Analysis**
   - Track progress toward long-term goals
   - Analyze geometric growth patterns
   - Project future performance

## 7. Learning System Rules

### 7.1 Performance Tracking Rules

1. **Trade Outcome Database**
   - Record all trade parameters:
     - Entry date, strike, expiration, premium
     - Week type classification
     - Outcome (expired, assigned, rolled)
     - Actual return vs. expected
   - Analyze success factors by week type
   - Identify patterns in successful vs. unsuccessful trades

2. **Protocol Effectiveness Analysis**
   - Track protocol application by week type
   - Measure effectiveness of protocol decisions
   - Identify optimal parameter settings by market condition

3. **User Interaction Analysis**
   - Track user satisfaction with recommendations
   - Identify common questions and concerns
   - Measure explanation clarity and effectiveness

### 7.2 Adaptation Rules

1. **Parameter Optimization Process**
   - Quarterly review of delta ranges
   - Analyze performance by delta range
   - Suggest adjustments based on historical performance

2. **Protocol Enhancement Process**
   - Identify edge cases requiring special handling
   - Develop new protocol variations for testing
   - Implement improvements based on performance data

3. **Personalization Process**
   - Adapt to user risk tolerance over time
   - Customize communication style based on user preferences
   - Tailor recommendations to user's specific goals

## 8. Decision Tree Integration

The week classification, trade management, and account management decision trees work together in an integrated system:

```
START
|
+-- Weekly Protocol Cycle
|   |
|   +-- Determine Account Status
|   |   |
|   |   +-- Check for Forking Opportunities
|   |   |   |
|   |   |   +-- Gen-Acc Surplus ≥ $50K → Execute Fork
|   |   |
|   |   +-- Check for Merging Opportunities
|   |       |
|   |       +-- Forked Account ≥ $500K → Execute Merge
|   |
|   +-- Apply Week Classification
|   |   |
|   |   +-- Evaluate Current Positions → Determine Week Type
|   |
|   +-- Execute Trade Management
|       |
|       +-- Apply Position-Specific Decision Tree
|       |
|       +-- Implement ATR-Based Adjustments
|
+-- Reporting Cycle
|   |
|   +-- Generate Weekly Report
|   |
|   +-- Generate Monthly Report (if applicable)
|   |
|   +-- Generate Quarterly Report (if applicable)
|   |
|   +-- Generate Annual Report (if applicable)
|
+-- Learning Cycle
    |
    +-- Update Trade Outcome Database
    |
    +-- Analyze Protocol Effectiveness
    |
    +-- Suggest Adaptations (if applicable)
|
END
```

## 9. Implementation Notes

1. **Parameter Configuration**
   - All parameters should be configurable for future optimization
   - Decision trees should be implemented as explicit rule sets
   - Protocol rules should be centralized for consistency

2. **Verification Requirements**
   - All calculations should be logged for verification
   - Protocol decisions should include rationale
   - Exceptions should be documented with justification

3. **User Preference Integration**
   - User preferences should override defaults where appropriate
   - Risk tolerance should influence position sizing
   - Communication preferences should adapt over time

4. **Core Assumptions**
   - Invest in market leaders of tech for long-term
   - Selling puts and calls each week will have positive returns overall annually
   - Market has fewer negative weeks than positive weeks in a year
   - Consistent reinvestment creates compounding growth
   - Account splitting manages risk better
   - Premiums from top tech will help achieve expected returns
