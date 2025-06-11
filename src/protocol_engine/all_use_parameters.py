"""
ALL-USE Core Parameters Module

This module defines the core parameters of the ALL-USE system that must be
strictly followed in all implementation aspects of the ALL-USE agent.
"""

class ALLUSEParameters:
    """Core parameters for the ALL-USE system."""
    
    # Weekly return rates
    GEN_ACC_WEEKLY_RETURN = 0.015  # 1.5% weekly from 40-50 delta options
    REV_ACC_WEEKLY_RETURN = 0.01   # 1.0% weekly from 30-40 delta options
    COM_ACC_WEEKLY_RETURN = 0.005  # 0.5% weekly from 20-30 delta options
    
    # Delta ranges
    GEN_ACC_DELTA_RANGE = (40, 50)  # 40-50 delta for Gen-Acc
    REV_ACC_DELTA_RANGE = (30, 40)  # 30-40 delta for Rev-Acc
    COM_ACC_DELTA_RANGE = (20, 30)  # 20-30 delta for Com-Acc
    
    # Account structure
    INITIAL_ALLOCATION = {
        'GEN_ACC': 0.40,  # 40% of initial investment
        'REV_ACC': 0.30,  # 30% of initial investment
        'COM_ACC': 0.30,  # 30% of initial investment
    }
    
    CASH_BUFFER = 0.05  # 5% cash buffer in each account
    
    # Forking and merging thresholds
    FORK_THRESHOLD = 50000  # $50K surplus in Gen-Acc triggers fork
    MERGE_THRESHOLD = 500000  # $500K threshold for merging
    
    # Reinvestment protocol
    REINVESTMENT_FREQUENCY = {
        'GEN_ACC': 'variable',  # Variable based on forking threshold
        'REV_ACC': 'quarterly',  # Quarterly reinvestment
        'COM_ACC': 'quarterly',  # Quarterly reinvestment
    }
    
    REINVESTMENT_ALLOCATION = {
        'CONTRACTS': 0.75,  # 75% to contracts
        'LEAPS': 0.25,      # 25% to LEAPS
    }
    
    # Entry protocols
    ENTRY_PROTOCOL = {
        'GEN_ACC': 'thursday',       # Thursday entry for Gen-Acc
        'REV_ACC': 'mon_to_wed',     # Monday-Wednesday entry for Rev-Acc
        'COM_ACC': 'not_applicable', # No specific entry day for Com-Acc
    }
    
    # Target stocks
    TARGET_STOCKS = {
        'GEN_ACC': ['TSLA', 'NVDA'],  # Volatile stocks for Gen-Acc
        'REV_ACC': ['AAPL', 'AMZN', 'MSFT'],  # Stable market leaders for Rev-Acc
        'COM_ACC': ['AAPL', 'AMZN', 'MSFT'],  # Same stable market leaders for Com-Acc
    }
    
    # Annual effective rates (for projection purposes)
    @classmethod
    def get_annual_rate(cls, account_type):
        """Calculate effective annual rate from weekly rate."""
        if account_type == 'GEN_ACC':
            return (1 + cls.GEN_ACC_WEEKLY_RETURN) ** 52 - 1
        elif account_type == 'REV_ACC':
            return (1 + cls.REV_ACC_WEEKLY_RETURN) ** 52 - 1
        elif account_type == 'COM_ACC':
            return (1 + cls.COM_ACC_WEEKLY_RETURN) ** 52 - 1
        else:
            raise ValueError(f"Unknown account type: {account_type}")
    
    # Income vs. growth split
    INCOME_RATIO = 0.80  # 80% of returns available as income
    GROWTH_RATIO = 0.20  # 20% of returns reinvested for portfolio growth
    
    # Market variation (for projection purposes)
    MARKET_VARIATION = {
        'GOOD_YEAR': 1.10,   # 10% better returns (60% of years)
        'AVERAGE_YEAR': 1.0,  # Expected returns (30% of years)
        'POOR_YEAR': 0.80,    # 20% lower returns (10% of years)
    }
    
    @classmethod
    def get_effective_allocation(cls, account_type):
        """Calculate effective allocation after cash buffer."""
        base_allocation = cls.INITIAL_ALLOCATION[account_type]
        return base_allocation * (1 - cls.CASH_BUFFER)
    
    @classmethod
    def get_delta_range(cls, account_type):
        """Get delta range for specified account type."""
        if account_type == 'GEN_ACC':
            return cls.GEN_ACC_DELTA_RANGE
        elif account_type == 'REV_ACC':
            return cls.REV_ACC_DELTA_RANGE
        elif account_type == 'COM_ACC':
            return cls.COM_ACC_DELTA_RANGE
        else:
            raise ValueError(f"Unknown account type: {account_type}")
    
    @classmethod
    def get_weekly_return(cls, account_type):
        """Get weekly return rate for specified account type."""
        if account_type == 'GEN_ACC':
            return cls.GEN_ACC_WEEKLY_RETURN
        elif account_type == 'REV_ACC':
            return cls.REV_ACC_WEEKLY_RETURN
        elif account_type == 'COM_ACC':
            return cls.COM_ACC_WEEKLY_RETURN
        else:
            raise ValueError(f"Unknown account type: {account_type}")
