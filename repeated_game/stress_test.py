import random
from generator import generate_scenario
from solver import solve, critical_discount_factor

rng = random.Random(123)
n = 5000
coop_count = 0
defect_count = 0
finite_count = 0
unknown_count = 0

for i in range(n):
    s = generate_scenario(rng=rng)
    assert s.T > s.R > s.P > s.S, s
    assert 2*s.R > s.T + s.S, s
    assert 0 < s.discount_factor < 1, s

    result = solve(s)
    assert result['action'] in ('cooperate', 'defect')

    if s.horizon == 'unknown':
        unknown_count += 1
        ds = critical_discount_factor(s.T, s.R, s.P)
        assert 0 < ds < 1, (s, ds)
        if s.discount_factor >= ds:
            assert result['action'] == 'cooperate'
            coop_count += 1
        else:
            assert result['action'] == 'defect'
            defect_count += 1
    else:
        finite_count += 1
        assert result['action'] == 'defect'

print(f'Total: {n}')
print(f'Finite horizon (all defect): {finite_count}')
print(f'Unknown horizon: {unknown_count}  -> cooperate: {coop_count}, defect: {defect_count}')
print('All assertions passed.')