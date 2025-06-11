"""
ALL-USE Protocol Engine Test Suite

This module contains tests to validate that the protocol engine correctly
implements the core ALL-USE parameters and logic.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from all_use_parameters import ALLUSEParameters

class TestALLUSEParameters(unittest.TestCase):
    """Test suite for ALL-USE core parameters."""
    
    def test_weekly_return_rates(self):
        """Test that weekly return rates match the specification."""
        self.assertEqual(ALLUSEParameters.GEN_ACC_WEEKLY_RETURN, 0.015)
        self.assertEqual(ALLUSEParameters.REV_ACC_WEEKLY_RETURN, 0.01)
        self.assertEqual(ALLUSEParameters.COM_ACC_WEEKLY_RETURN, 0.005)
    
    def test_delta_ranges(self):
        """Test that delta ranges match the specification."""
        self.assertEqual(ALLUSEParameters.GEN_ACC_DELTA_RANGE, (40, 50))
        self.assertEqual(ALLUSEParameters.REV_ACC_DELTA_RANGE, (30, 40))
        self.assertEqual(ALLUSEParameters.COM_ACC_DELTA_RANGE, (20, 30))
    
    def test_initial_allocation(self):
        """Test that initial allocation percentages match the specification."""
        self.assertEqual(ALLUSEParameters.INITIAL_ALLOCATION['GEN_ACC'], 0.40)
        self.assertEqual(ALLUSEParameters.INITIAL_ALLOCATION['REV_ACC'], 0.30)
        self.assertEqual(ALLUSEParameters.INITIAL_ALLOCATION['COM_ACC'], 0.30)
    
    def test_cash_buffer(self):
        """Test that cash buffer percentage matches the specification."""
        self.assertEqual(ALLUSEParameters.CASH_BUFFER, 0.05)
    
    def test_fork_threshold(self):
        """Test that fork threshold matches the specification."""
        self.assertEqual(ALLUSEParameters.FORK_THRESHOLD, 50000)
    
    def test_merge_threshold(self):
        """Test that merge threshold matches the specification."""
        self.assertEqual(ALLUSEParameters.MERGE_THRESHOLD, 500000)
    
    def test_reinvestment_allocation(self):
        """Test that reinvestment allocation matches the specification."""
        self.assertEqual(ALLUSEParameters.REINVESTMENT_ALLOCATION['CONTRACTS'], 0.75)
        self.assertEqual(ALLUSEParameters.REINVESTMENT_ALLOCATION['LEAPS'], 0.25)
    
    def test_annual_rates(self):
        """Test that annual rates are calculated correctly from weekly rates."""
        gen_annual = (1 + ALLUSEParameters.GEN_ACC_WEEKLY_RETURN) ** 52 - 1
        rev_annual = (1 + ALLUSEParameters.REV_ACC_WEEKLY_RETURN) ** 52 - 1
        com_annual = (1 + ALLUSEParameters.COM_ACC_WEEKLY_RETURN) ** 52 - 1
        
        self.assertAlmostEqual(ALLUSEParameters.get_annual_rate('GEN_ACC'), gen_annual)
        self.assertAlmostEqual(ALLUSEParameters.get_annual_rate('REV_ACC'), rev_annual)
        self.assertAlmostEqual(ALLUSEParameters.get_annual_rate('COM_ACC'), com_annual)
    
    def test_effective_allocation(self):
        """Test that effective allocation is calculated correctly after cash buffer."""
        gen_effective = ALLUSEParameters.INITIAL_ALLOCATION['GEN_ACC'] * (1 - ALLUSEParameters.CASH_BUFFER)
        rev_effective = ALLUSEParameters.INITIAL_ALLOCATION['REV_ACC'] * (1 - ALLUSEParameters.CASH_BUFFER)
        com_effective = ALLUSEParameters.INITIAL_ALLOCATION['COM_ACC'] * (1 - ALLUSEParameters.CASH_BUFFER)
        
        self.assertAlmostEqual(ALLUSEParameters.get_effective_allocation('GEN_ACC'), gen_effective)
        self.assertAlmostEqual(ALLUSEParameters.get_effective_allocation('REV_ACC'), rev_effective)
        self.assertAlmostEqual(ALLUSEParameters.get_effective_allocation('COM_ACC'), com_effective)
    
    def test_get_delta_range(self):
        """Test that get_delta_range returns correct values."""
        self.assertEqual(ALLUSEParameters.get_delta_range('GEN_ACC'), (40, 50))
        self.assertEqual(ALLUSEParameters.get_delta_range('REV_ACC'), (30, 40))
        self.assertEqual(ALLUSEParameters.get_delta_range('COM_ACC'), (20, 30))
        
        with self.assertRaises(ValueError):
            ALLUSEParameters.get_delta_range('INVALID_ACC')
    
    def test_get_weekly_return(self):
        """Test that get_weekly_return returns correct values."""
        self.assertEqual(ALLUSEParameters.get_weekly_return('GEN_ACC'), 0.015)
        self.assertEqual(ALLUSEParameters.get_weekly_return('REV_ACC'), 0.01)
        self.assertEqual(ALLUSEParameters.get_weekly_return('COM_ACC'), 0.005)
        
        with self.assertRaises(ValueError):
            ALLUSEParameters.get_weekly_return('INVALID_ACC')

if __name__ == '__main__':
    unittest.main()
