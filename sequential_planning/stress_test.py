import random
from generator import generate_scenario
from solver import solve

rng = random.Random(123)
n = 5000
cautious_count = 0
aggressive_count = 0
finite_count = 0
unknown_count = 0

for i in range(n):
    s = generate_scenario(rng=rng)

    # Structural invariants promised by the generator's docstring.
    assert s.r_high_cautious > s.r_low_cautious, s
    assert s.r_high_aggressive > s.r_low_aggressive, s
    assert s.r_low_aggressive > s.r_low_cautious, s
    assert s.r_high_aggressive > s.r_high_cautious, s
    assert s.p_high_low_cautious > s.p_high_low_aggressive, s
    assert s.p_high_high_cautious > s.p_high_high_aggressive, s
    assert 0 < s.discount_factor < 1, s
    assert s.start_state == "Low"

    result = solve(s)
    assert result["action"] in ("cautious", "aggressive")
    assert result["margin"] >= 0

    if s.horizon == "unknown":
        unknown_count += 1
    else:
        finite_count += 1
        assert isinstance(s.horizon, int) and s.horizon >= 2

    if result["action"] == "cautious":
        cautious_count += 1
    else:
        aggressive_count += 1

print(f"Total: {n}")
print(f"Finite horizon: {finite_count}   Unknown horizon: {unknown_count}")
print(f"Optimal action = cautious:   {cautious_count}")
print(f"Optimal action = aggressive: {aggressive_count}")

# Non-degeneracy check on the primary research question's premise: neither
# action should be optimal in (near-)100% of cases. If one action always
# wins regardless of horizon/discount, the "planning" tradeoff the
# generator is meant to encode isn't actually biting, and the module
# reduces to a rigged coin flip rather than a genuine DP problem.
assert 0.05 < cautious_count / n < 0.95, (
    "Cautious is optimal in a suspiciously lopsided fraction of cases; "
    "the generator's reward/probability tradeoff may not be creating "
    "genuine tension. Investigate before treating this as a valid module."
)

print("All assertions passed.")
