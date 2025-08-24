"""Tests for trading costs module (M6)"""

import numpy as np
import pandas as pd
import pytest

from bwv.costs import (
    apply_costs,
    compute_participation,
    linear_slippage_bps,
    impact_bps_above_threshold
)


class TestCostsModule:
    """Test suite for trading costs functionality"""

    def test_zero_trades_cost_is_zero(self):
        """Test that zero trades result in zero costs."""
        # Create empty trades
        trades = pd.DataFrame(columns=['date', 'ticker', 'notional_vnd'])
        adv = pd.DataFrame({'date': [], 'ticker': [], 'adv_vnd': []})
        
        costs = apply_costs(trades, adv)
        assert costs.empty
        assert isinstance(costs, pd.Series)
        assert costs.index.name == 'date'

    def test_fee_only_matches_when_slope_zero(self):
        """Test that with zero slippage slope, cost equals fee only."""
        # Create test data: 1M VND trade with 10M ADV
        trades = pd.DataFrame({
            'date': ['2023-01-03'],
            'ticker': ['VIC'],
            'notional_vnd': [1_000_000]
        })
        trades['date'] = pd.to_datetime(trades['date'])
        
        adv = pd.DataFrame({
            'date': ['2023-01-03'],
            'ticker': ['VIC'], 
            'adv_vnd': [10_000_000]
        })
        adv['date'] = pd.to_datetime(adv['date'])
        
        # Zero slippage, no impact
        slip_params = {'bps_per_1pct_participation': 0.0, 'cap_bps': 50.0}
        impact_params = {'threshold_participation': 0.10, 'impact_bps': 10.0}
        
        costs = apply_costs(
            trades, adv, fee_bps=25.0,
            slip_params=slip_params, impact_params=impact_params
        )
        
        # Expected: 1M * 25bps / 10000 = 2500 VND
        expected_cost = 1_000_000 * 25.0 / 10000
        assert abs(costs.iloc[0] - expected_cost) < 1e-10

    def test_linear_slippage_scales_with_participation(self):
        """Test that slippage scales linearly with participation."""
        # Same ADV, different notionals
        trades = pd.DataFrame({
            'date': ['2023-01-03', '2023-01-03', '2023-01-04'],  # Different dates for each trade
            'ticker': ['VIC', 'VIC', 'VIC'],
            'notional_vnd': [1_000_000, 2_000_000, 4_000_000]  # 10%, 20%, 40% participation
        })
        trades['date'] = pd.to_datetime(trades['date'])
        
        adv = pd.DataFrame({
            'date': ['2023-01-03', '2023-01-03', '2023-01-04'],
            'ticker': ['VIC', 'VIC', 'VIC'],
            'adv_vnd': [10_000_000, 10_000_000, 10_000_000]
        })
        adv['date'] = pd.to_datetime(adv['date'])
        
        # Linear slippage: 2 bps per 1% participation, cap at 50 bps
        slip_params = {'bps_per_1pct_participation': 2.0, 'cap_bps': 50.0}
        
        costs = apply_costs(trades, adv, fee_bps=25.0, slip_params=slip_params)
        
        # Calculate expected costs per trade
        participations = [0.10, 0.20, 0.40]
        slippages = [min(2.0 * (p * 100), 50.0) for p in participations]
        total_bps = [25.0 + s for s in slippages]
        expected_costs = [n * bps / 10000 for n, bps in zip([1e6, 2e6, 4e6], total_bps)]
        
        # Check each daily cost matches expected
        expected_total_day1 = expected_costs[0] + expected_costs[1]  # First two trades on same day
        expected_total_day2 = expected_costs[2]  # Third trade on different day
        
        assert len(costs) == 2  # Should have costs for two different dates
        assert abs(costs.loc[pd.Timestamp('2023-01-03')] - expected_total_day1) < 1e-10
        assert abs(costs.loc[pd.Timestamp('2023-01-04')] - expected_total_day2) < 1e-10

    def test_slippage_cap_applied(self):
        """Test that slippage cap is respected."""
        # Very large trade that should hit slippage cap
        trades = pd.DataFrame({
            'date': ['2023-01-03'],
            'ticker': ['VIC'],
            'notional_vnd': [50_000_000]  # 500% participation - way above cap
        })
        trades['date'] = pd.to_datetime(trades['date'])
        
        adv = pd.DataFrame({
            'date': ['2023-01-03'],
            'ticker': ['VIC'],
            'adv_vnd': [10_000_000]  # 10M ADV
        })
        adv['date'] = pd.to_datetime(adv['date'])
        
        # Linear slippage: 2 bps per 1% participation, cap at 50 bps
        slip_params = {'bps_per_1pct_participation': 2.0, 'cap_bps': 50.0}
        
        costs = apply_costs(trades, adv, fee_bps=25.0, slip_params=slip_params)
        
        # Should hit cap: 25 bps fee + 50 bps slippage = 75 bps total
        expected_cost = 50_000_000 * 75.0 / 10000
        assert abs(costs.iloc[0] - expected_cost) < 1e-10

    def test_impact_applies_above_threshold(self):
        """Test that impact cost is added above participation threshold."""
        trades = pd.DataFrame({
            'date': ['2023-01-03', '2023-01-04'],  # Different dates for each trade
            'ticker': ['VIC', 'VIC'],
            'notional_vnd': [500_000, 1_500_000]  # 5% and 15% participation
        })
        trades['date'] = pd.to_datetime(trades['date'])
        
        adv = pd.DataFrame({
            'date': ['2023-01-03', '2023-01-04'],
            'ticker': ['VIC', 'VIC'],
            'adv_vnd': [10_000_000, 10_000_000]
        })
        adv['date'] = pd.to_datetime(adv['date'])
        
        # Impact threshold at 10%, impact cost 10 bps
        impact_params = {'threshold_participation': 0.10, 'impact_bps': 10.0}
        
        costs = apply_costs(
            trades, adv, fee_bps=25.0, 
            slip_params={'bps_per_1pct_participation': 2.0, 'cap_bps': 50.0},
            impact_params=impact_params
        )
        
        # First trade: 5% participation -> no impact
        # Slippage: 2 * 5 = 10 bps, fee: 25 bps, total 35 bps
        expected1 = 500_000 * 35.0 / 10000
        
        # Second trade: 15% participation -> 10 bps impact
        # Slippage: 2 * 15 = 30 bps, fee: 25 bps, impact: 10 bps, total 65 bps
        expected2 = 1_500_000 * 65.0 / 10000
        
        assert len(costs) == 2  # Should have two separate daily costs
        assert abs(costs.loc[pd.Timestamp('2023-01-03')] - expected1) < 1e-10
        assert abs(costs.loc[pd.Timestamp('2023-01-04')] - expected2) < 1e-10

    def test_missing_or_zero_adv_penalized(self):
        """Test that missing or zero ADV results in maximum slippage penalty."""
        trades = pd.DataFrame({
            'date': ['2023-01-03', '2023-01-04'],  # Different dates for each trade
            'ticker': ['VIC', 'VNINDEX'],
            'notional_vnd': [1_000_000, 1_000_000]
        })
        trades['date'] = pd.to_datetime(trades['date'])
        
        # VIC has missing ADV, VNINDEX has zero ADV
        adv = pd.DataFrame({
            'date': ['2023-01-03', '2023-01-04'],
            'ticker': ['VIC', 'VNINDEX'],
            'adv_vnd': [np.nan, 0.0]
        })
        adv['date'] = pd.to_datetime(adv['date'])
        
        # Should get slippage cap for both
        slip_params = {'bps_per_1pct_participation': 2.0, 'cap_bps': 50.0}
        
        costs = apply_costs(trades, adv, fee_bps=25.0, slip_params=slip_params)
        
        # Both should get 25 bps fee + 50 bps slippage = 75 bps total
        expected_cost = 1_000_000 * 75.0 / 10000
        
        # Check each daily cost separately
        assert len(costs) == 2  # Should have two separate daily costs
        assert abs(costs.loc[pd.Timestamp('2023-01-03')] - expected_cost) < 1e-10
        assert abs(costs.loc[pd.Timestamp('2023-01-04')] - expected_cost) < 1e-10

    def test_participation_cap_enforced(self):
        """Test that participation cap limits slippage and impact costs."""
        trades = pd.DataFrame({
            'date': ['2023-01-03'],
            'ticker': ['VIC'],
            'notional_vnd': [20_000_000]  # 200% participation
        })
        trades['date'] = pd.to_datetime(trades['date'])
        
        adv = pd.DataFrame({
            'date': ['2023-01-03'],
            'ticker': ['VIC'],
            'adv_vnd': [10_000_000]  # 10M ADV
        })
        adv['date'] = pd.to_datetime(adv['date'])
        
        # Cap participation at 10% (should limit to 10% regardless of actual 200%)
        participation_cap = 0.10
        
        costs = apply_costs(
            trades, adv, fee_bps=25.0,
            slip_params={'bps_per_1pct_participation': 2.0, 'cap_bps': 50.0},
            impact_params={'threshold_participation': 0.10, 'impact_bps': 10.0},
            participation_cap=participation_cap
        )
        
        # With cap, effective participation is 10% (not 200%)
        # Slippage: 2 * 10 = 20 bps (not hitting cap of 50)
        # Impact: 0 bps (since 10% == threshold, not greater than)
        # Fee: 25 bps
        # Total: 25 + 20 + 0 = 45 bps
        expected_cost = 20_000_000 * 45.0 / 10000
        assert abs(costs.iloc[0] - expected_cost) < 1e-10

    def test_sum_across_tickers_and_days(self):
        """Test that costs are aggregated correctly across tickers and days."""
        trades = pd.DataFrame({
            'date': ['2023-01-03', '2023-01-03', '2023-01-04'],
            'ticker': ['VIC', 'VNM', 'VIC'],
            'notional_vnd': [1_000_000, 2_000_000, 3_000_000]
        })
        trades['date'] = pd.to_datetime(trades['date'])
        
        adv = pd.DataFrame({
            'date': ['2023-01-03', '2023-01-03', '2023-01-04'],
            'ticker': ['VIC', 'VNM', 'VIC'],
            'adv_vnd': [10_000_000, 20_000_000, 10_000_000]
        })
        adv['date'] = pd.to_datetime(adv['date'])
        
        costs = apply_costs(trades, adv, fee_bps=25.0)
        
        # Check we have costs for both days
        assert len(costs) == 2
        assert costs.index[0] == pd.Timestamp('2023-01-03')
        assert costs.index[1] == pd.Timestamp('2023-01-04')
        
        # Day 1: VIC (10% participation) + VNM (10% participation)
        # Slippage: 2 * 10 = 20 bps each
        # Total bps: 25 + 20 = 45 bps each
        expected_day1 = (1_000_000 * 45.0 / 10000) + (2_000_000 * 45.0 / 10000)
        
        # Day 2: VIC (30% participation)
        # Slippage: 2 * 30 = 60 bps (capped at 50)
        # Total bps: 25 + 50 = 75 bps
        expected_day2 = 3_000_000 * 75.0 / 10000
        
        assert abs(costs.iloc[0] - expected_day1) < 1e-10
        assert abs(costs.iloc[1] - expected_day2) < 1e-10

    def test_helper_functions(self):
        """Test the individual helper functions."""
        # Test compute_participation
        assert compute_participation(1_000_000, 10_000_000) == 0.10
        assert np.isnan(compute_participation(1_000_000, 0.0))
        assert np.isnan(compute_participation(1_000_000, np.nan))
        
        # Test linear_slippage_bps
        assert linear_slippage_bps(0.10, 2.0, 50.0) == 20.0  # 2 * 10
        assert linear_slippage_bps(0.50, 2.0, 50.0) == 50.0  # hits cap
        assert linear_slippage_bps(np.nan, 2.0, 50.0) == 50.0  # missing ADV
        
        # Test impact_bps_above_threshold
        assert impact_bps_above_threshold(0.05, 0.10, 10.0) == 0.0  # below threshold
        assert impact_bps_above_threshold(0.15, 0.10, 10.0) == 10.0  # above threshold
        assert impact_bps_above_threshold(np.nan, 0.10, 10.0) == 0.0  # missing ADV


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
