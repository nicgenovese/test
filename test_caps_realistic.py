#!/usr/bin/env python3
"""
Realistic test to verify token capping works correctly in typical scenarios
"""

import sys
sys.path.insert(0, '/home/user/test')

from app import IndexStrategy

def test_realistic_caps():
    """Test realistic capping scenarios that an index fund would actually use"""

    print("=" * 70)
    print("REALISTIC TOKEN CAPPING TESTS")
    print("=" * 70)

    # Test Case 1: Typical index fund caps - should work perfectly
    print("\n" + "=" * 70)
    print("TEST 1: Realistic index fund caps")
    print("Scenario: BTC gets too large, should be redistributed to other tokens")
    print("=" * 70)

    config1 = {
        'name': 'Realistic',
        'rebalance_freq': 'quarterly',
        'top_n': 20,
        'btc_cap': 0.40,     # 40% BTC cap
        'eth_cap': 0.25,     # 25% ETH cap
        'btc_eth_combined': 0.60,  # 60% combined
        'max_any': 0.15,     # 15% for other tokens
        'excluded': [],
        'min_weight': 0.0,
        'method': 'market_cap',
        'manual_weights': {}
    }

    strategy1 = IndexStrategy(config1)

    # Simulate a scenario where BTC dominates market cap
    initial_weights = {
        'BTC': 0.60,   # BTC is 60% of market cap - should be capped to 40%
        'ETH': 0.20,   # ETH is 20% - ok
        'SOL': 0.10,   # SOL is 10% - should grow from redistribution
        'ADA': 0.05,   # ADA is 5% - should grow from redistribution
        'DOT': 0.03,   # DOT is 3% - should grow from redistribution
        'MATIC': 0.02  # MATIC is 2% - should grow from redistribution
    }

    print(f"\nInitial market cap weights:")
    for token, weight in sorted(initial_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {token}: {weight*100:.2f}%")

    capped_weights = strategy1._apply_caps(initial_weights.copy())

    print(f"\nAfter applying caps:")
    total = 0
    for token, weight in sorted(capped_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {token}: {weight*100:.2f}%")
        total += weight
    print(f"\nTotal: {total*100:.2f}%")

    # Verify caps are respected
    btc_ok = capped_weights['BTC'] <= 0.40 + 0.001
    eth_ok = capped_weights['ETH'] <= 0.25 + 0.001
    combined_ok = (capped_weights['BTC'] + capped_weights['ETH']) <= 0.60 + 0.001
    others_ok = all(capped_weights[t] <= 0.15 + 0.001 for t in capped_weights if t not in ['BTC', 'ETH'])
    sum_ok = abs(total - 1.0) < 0.001

    print(f"\n✓ BTC <= 40%: {btc_ok} (actual: {capped_weights['BTC']*100:.2f}%)")
    print(f"✓ ETH <= 25%: {eth_ok} (actual: {capped_weights['ETH']*100:.2f}%)")
    print(f"✓ BTC+ETH <= 60%: {combined_ok} (actual: {(capped_weights['BTC'] + capped_weights['ETH'])*100:.2f}%)")
    print(f"✓ Other tokens <= 15%: {others_ok}")
    print(f"✓ Sum = 100%: {sum_ok}")

    all_pass = btc_ok and eth_ok and combined_ok and others_ok and sum_ok
    print(f"\n{'✅ TEST 1 PASSED' if all_pass else '❌ TEST 1 FAILED'}")

    # Test Case 2: Verify redistribution doesn't violate max_any cap (realistic caps)
    print("\n" + "=" * 70)
    print("TEST 2: Redistribution respects max_any cap")
    print("Scenario: Ensure tokens don't exceed max_any during redistribution")
    print("=" * 70)

    config2 = {
        'name': 'Test2',
        'rebalance_freq': 'quarterly',
        'top_n': 20,
        'btc_cap': 0.30,     # 30% BTC cap
        'eth_cap': 0.25,     # 25% ETH cap
        'btc_eth_combined': 0.50,  # 50% combined (achievable)
        'max_any': 0.15,     # 15% for other tokens (achievable with 4+ tokens)
        'excluded': [],
        'min_weight': 0.0,
        'method': 'market_cap',
        'manual_weights': {}
    }

    strategy2 = IndexStrategy(config2)

    initial_weights2 = {
        'BTC': 0.35,   # Will be capped to 30%
        'ETH': 0.25,   # At cap
        'SOL': 0.20,   # Will be capped to 15%
        'ADA': 0.10,   # Below cap, should receive redistribution but not exceed 15%
        'AVAX': 0.06,  # Should receive redistribution
        'DOT': 0.04    # Should receive redistribution
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

    # Key test: All tokens should respect their caps
    btc_ok2 = capped_weights2['BTC'] <= 0.30 + 0.001
    eth_ok2 = capped_weights2['ETH'] <= 0.25 + 0.001
    all_others_ok = all(capped_weights2[t] <= 0.15 + 0.001 for t in capped_weights2 if t not in ['BTC', 'ETH'])
    sum_ok2 = abs(total2 - 1.0) < 0.001

    print(f"\n✓ BTC <= 30%: {btc_ok2} (actual: {capped_weights2['BTC']*100:.2f}%)")
    print(f"✓ ETH <= 25%: {eth_ok2} (actual: {capped_weights2['ETH']*100:.2f}%)")
    print(f"✓ All other tokens <= 15%: {all_others_ok}")
    if not all_others_ok:
        for t in capped_weights2:
            if t not in ['BTC', 'ETH'] and capped_weights2[t] > 0.15:
                print(f"    - {t}: {capped_weights2[t]*100:.2f}% > 15%")
    print(f"✓ Sum = 100%: {sum_ok2}")

    all_pass2 = btc_ok2 and eth_ok2 and all_others_ok and sum_ok2
    print(f"\n{'✅ TEST 2 PASSED' if all_pass2 else '❌ TEST 2 FAILED'}")

    # Test Case 3: Combined cap interaction
    print("\n" + "=" * 70)
    print("TEST 3: Combined cap properly constrains BTC and ETH")
    print("=" * 70)

    config3 = {
        'name': 'Test3',
        'rebalance_freq': 'quarterly',
        'top_n': 20,
        'btc_cap': 0.45,     # 45% BTC cap (high individual cap)
        'eth_cap': 0.35,     # 35% ETH cap (high individual cap)
        'btc_eth_combined': 0.50,  # 50% combined (more restrictive)
        'max_any': 0.20,     # 20% for others
        'excluded': [],
        'min_weight': 0.0,
        'method': 'market_cap',
        'manual_weights': {}
    }

    strategy3 = IndexStrategy(config3)

    initial_weights3 = {
        'BTC': 0.50,   # 50% - should be reduced by combined cap
        'ETH': 0.30,   # 30% - should be reduced by combined cap
        'SOL': 0.12,
        'ADA': 0.05,
        'DOT': 0.03
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

    combined_final = capped_weights3['BTC'] + capped_weights3['ETH']
    combined_ok3 = combined_final <= 0.50 + 0.001
    sum_ok3 = abs(total3 - 1.0) < 0.001

    print(f"\n✓ BTC+ETH <= 50%: {combined_ok3} (actual: {combined_final*100:.2f}%)")
    print(f"✓ Sum = 100%: {sum_ok3}")

    all_pass3 = combined_ok3 and sum_ok3
    print(f"\n{'✅ TEST 3 PASSED' if all_pass3 else '❌ TEST 3 FAILED'}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test 1 (Realistic caps): {'✅ PASSED' if all_pass else '❌ FAILED'}")
    print(f"Test 2 (Redistribution respects caps): {'✅ PASSED' if all_pass2 else '❌ FAILED'}")
    print(f"Test 3 (Combined cap): {'✅ PASSED' if all_pass3 else '❌ FAILED'}")

    if all_pass and all_pass2 and all_pass3:
        print("\n🎉 ALL TESTS PASSED! Token capping is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Review the output above.")

if __name__ == "__main__":
    test_realistic_caps()
