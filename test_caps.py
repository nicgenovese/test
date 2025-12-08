#!/usr/bin/env python3
"""
Test script to verify individual token capping logic works correctly
"""

import sys
sys.path.insert(0, '/home/user/test')

from app import IndexStrategy
import pandas as pd

def test_token_caps():
    """Test various capping scenarios"""

    print("=" * 70)
    print("TESTING TOKEN CAPPING LOGIC")
    print("=" * 70)

    # Test Case 1: max_any should apply to ALL tokens including BTC/ETH
    print("\n" + "=" * 70)
    print("TEST 1: max_any applies to BTC and ETH")
    print("=" * 70)

    config1 = {
        'name': 'Test1',
        'rebalance_freq': 'quarterly',
        'top_n': 20,
        'btc_cap': 1.0,      # 100% - no individual BTC cap
        'eth_cap': 1.0,      # 100% - no individual ETH cap
        'btc_eth_combined': 1.0,  # 100% - no combined cap
        'max_any': 0.15,     # 15% - should apply to ALL tokens
        'excluded': [],
        'min_weight': 0.0,
        'method': 'market_cap',
        'manual_weights': {}
    }

    strategy1 = IndexStrategy(config1)

    # Simulate weights before caps
    initial_weights = {
        'BTC': 0.40,   # 40% - should be capped to 15%
        'ETH': 0.30,   # 30% - should be capped to 15%
        'SOL': 0.15,   # 15% - at cap
        'ADA': 0.10,   # 10% - below cap
        'DOT': 0.05    # 5% - below cap
    }

    print(f"\nInitial weights:")
    for token, weight in sorted(initial_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {token}: {weight*100:.2f}%")

    capped_weights = strategy1._apply_caps(initial_weights.copy())

    print(f"\nAfter applying caps:")
    total = 0
    for token, weight in sorted(capped_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {token}: {weight*100:.2f}%")
        total += weight
    print(f"\nTotal: {total*100:.2f}%")

    # Verify all tokens are <= 15%
    all_capped = all(w <= 0.15 + 0.0001 for w in capped_weights.values())
    print(f"\n✓ All tokens <= 15%: {all_capped}")
    print(f"✓ Weights sum to 100%: {abs(total - 1.0) < 0.001}")

    # Test Case 2: Individual BTC/ETH caps take precedence if lower than max_any
    print("\n" + "=" * 70)
    print("TEST 2: Individual caps lower than max_any")
    print("=" * 70)

    config2 = {
        'name': 'Test2',
        'rebalance_freq': 'quarterly',
        'top_n': 20,
        'btc_cap': 0.25,     # 25% BTC cap
        'eth_cap': 0.20,     # 20% ETH cap
        'btc_eth_combined': 0.50,  # 50% combined
        'max_any': 0.30,     # 30% - higher than individual caps
        'excluded': [],
        'min_weight': 0.0,
        'method': 'market_cap',
        'manual_weights': {}
    }

    strategy2 = IndexStrategy(config2)

    initial_weights2 = {
        'BTC': 0.35,   # Should be capped to min(25%, 30%) = 25%
        'ETH': 0.30,   # Should be capped to min(20%, 30%) = 20%
        'SOL': 0.20,   # Should be capped to 30% (already below)
        'ADA': 0.10,
        'DOT': 0.05
    }

    print(f"\nInitial weights:")
    for token, weight in sorted(initial_weights2.items(), key=lambda x: x[1], reverse=True):
        print(f"  {token}: {weight*100:.2f}%")

    capped_weights2 = strategy2._apply_caps(initial_weights2.copy())

    print(f"\nAfter applying caps:")
    total2 = 0
    for token, weight in sorted(capped_weights2.items(), key=lambda x: x[1], reverse=True):
        print(f"  {token}: {weight*100:.2f}%")
        total2 += weight
    print(f"\nTotal: {total2*100:.2f}%")

    btc_ok = capped_weights2['BTC'] <= 0.25 + 0.0001
    eth_ok = capped_weights2['ETH'] <= 0.20 + 0.0001
    combined_ok = (capped_weights2['BTC'] + capped_weights2['ETH']) <= 0.50 + 0.0001

    print(f"\n✓ BTC <= 25%: {btc_ok} (actual: {capped_weights2['BTC']*100:.2f}%)")
    print(f"✓ ETH <= 20%: {eth_ok} (actual: {capped_weights2['ETH']*100:.2f}%)")
    print(f"✓ BTC+ETH <= 50%: {combined_ok} (actual: {(capped_weights2['BTC'] + capped_weights2['ETH'])*100:.2f}%)")
    print(f"✓ Weights sum to 100%: {abs(total2 - 1.0) < 0.001}")

    # Test Case 3: Combined cap interaction
    print("\n" + "=" * 70)
    print("TEST 3: Combined cap with individual caps")
    print("=" * 70)

    config3 = {
        'name': 'Test3',
        'rebalance_freq': 'quarterly',
        'top_n': 20,
        'btc_cap': 0.30,     # 30% BTC cap
        'eth_cap': 0.25,     # 25% ETH cap
        'btc_eth_combined': 0.40,  # 40% combined - lower than sum of individual caps
        'max_any': 0.15,     # 15% for other tokens
        'excluded': [],
        'min_weight': 0.0,
        'method': 'market_cap',
        'manual_weights': {}
    }

    strategy3 = IndexStrategy(config3)

    initial_weights3 = {
        'BTC': 0.35,   # Should be capped considering combined limit
        'ETH': 0.30,   # Should be capped considering combined limit
        'SOL': 0.20,   # Should be capped to 15%
        'ADA': 0.10,
        'DOT': 0.05
    }

    print(f"\nInitial weights:")
    for token, weight in sorted(initial_weights3.items(), key=lambda x: x[1], reverse=True):
        print(f"  {token}: {weight*100:.2f}%")

    capped_weights3 = strategy3._apply_caps(initial_weights3.copy())

    print(f"\nAfter applying caps:")
    total3 = 0
    for token, weight in sorted(capped_weights3.items(), key=lambda x: x[1], reverse=True):
        print(f"  {token}: {weight*100:.2f}%")
        total3 += weight
    print(f"\nTotal: {total3*100:.2f}%")

    # BTC and ETH should respect max_any (15%), which is lower than their individual caps
    btc_final = capped_weights3['BTC']
    eth_final = capped_weights3['ETH']
    combined_final = btc_final + eth_final

    print(f"\n✓ BTC <= 15% (max_any): {btc_final <= 0.15 + 0.0001} (actual: {btc_final*100:.2f}%)")
    print(f"✓ ETH <= 15% (max_any): {eth_final <= 0.15 + 0.0001} (actual: {eth_final*100:.2f}%)")
    print(f"✓ BTC+ETH <= 40%: {combined_final <= 0.40 + 0.0001} (actual: {combined_final*100:.2f}%)")
    print(f"✓ SOL <= 15%: {capped_weights3['SOL'] <= 0.15 + 0.0001} (actual: {capped_weights3['SOL']*100:.2f}%)")
    print(f"✓ Weights sum to 100%: {abs(total3 - 1.0) < 0.001}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    test_token_caps()
