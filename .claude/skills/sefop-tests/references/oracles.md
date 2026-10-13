# Oracles for optimization and scientific code

A test oracle is whatever tells a test the expected behaviour. Code under test can never be its own oracle.
Tests that compare a function's output with that same function's earlier output are characterization tests, and
are labelled as such.

## Contents

1. Known oracle
2. Pseudo-oracle (brute force)
3. Differential oracle (several implementations agree)
4. Metamorphic relations
5. Regression against logged or published results
6. Heuristics and samplers
7. Exact solvers as oracles

## 1. Known oracle

Use a tiny instance whose answer you work out by hand, and put the derivation in the docstring so a reviewer can
check it:

```python
def test__build_q_matrix__given_one_edge__matches_hand_derived_matrix() -> None:
    """Known oracle: c = (1, 2), w = 0.5, s = 1, rho = 3 gives
    Q_aa = c_a - w + rho - 2 rho s = -2.5,  Q_bb = -1.5,  Q_ab = w/2 + rho = 3.25,  cQ = rho s^2 + w = 3.5."""
```

Boundary instances find the most bugs: a capacity filled exactly, a budget of 0 or n, an empty input, ties.

## 2. Pseudo-oracle (brute force)

A second implementation of the same contract, written from the problem's definition: it enumerates every
assignment and keeps the best feasible one. It is slow, so use it on at most ~20 binary variables. Because it
shares no code with production, a bug would have to occur twice in the same way to slip through. Template:
`assets/enumeration_solver.py`.

1. **Check the oracle itself first**, against a known answer. A pseudo-oracle is only useful if it is trusted.
2. **Then sweep** 100–200 random small instances with a fixed seed, and compare objective values, not argmins:
   ties make the argmin ambiguous.

```python
rng = random.Random(20261007)
for _ in range(SWEEP):
    instance = random_instance(rng)
    assert solve(instance).objective == pytest.approx(EnumerationSolver().run(instance).objective), f"{instance}"
```

## 3. Differential oracle (several implementations agree)

When the same objective exists in several encodings, check that they give the same value on every assignment of
a small instance. Typical encodings are a formula, a Q matrix, a dimod BQM, a Pyomo or Gurobi model, and a
Hamiltonian. Disagreement means one encoding is wrong, and the failure message names which.

```python
for bits in itertools.product((0, 1), repeat=n):
    reference = objective_from_definition(bits)
    assert x_Q_x_plus_c(Q, cQ, bits) == pytest.approx(reference), f"Q matrix, {bits}"
    assert bqm.energy(dict(enumerate(bits))) == pytest.approx(reference), f"BQM, {bits}"
```

## 4. Metamorphic relations

These relate two runs whose results must compare in a provable way, even when neither result is known:

| Relation | Expected effect on the optimum |
|---|---|
| Permute the variables or the input order | unchanged |
| Multiply the objective by k > 0 | multiplied by k (argmin unchanged) |
| Add a constant to the objective | shifted by that constant |
| Enlarge the feasible set (relax a bound, raise a capacity or budget for ≤) | never worse |
| Shrink the feasible set (tighten, remove an option) | never better |
| Raise one reward or lower one cost | never worse |
| Penalty methods: change ρ on a feasible point | energy unchanged |
| Penalty methods: raise ρ on an infeasible point | energy strictly larger |

Metamorphic relations catch bugs that no known answer covers, such as a model that silently ignores the first
item, or a variable bounded as 0/1 when it should be an integer.

## 5. Regression against logged or published results

When a repository stores solver outputs (energies, objective values) or results appear in a paper or poster,
re-derive them from today's code and data. If the formulation or the data changes, these tests say that the
published numbers no longer follow from the code.

- Mark them `integration`.
- Read the data location from an environment variable, defaulting to the repository, and skip with a reason when
  the data is absent.
- Compare with `pytest.approx(rel=1e-9, abs=1e-6)`, or with the precision the results were reported to.

## 6. Heuristics and samplers

Simulated annealing, tabu search, QAOA and annealers may legitimately miss the optimum, so assert properties
rather than exact answers. Fix seeds so failures reproduce.

- The best sample is feasible (when the penalty or encoding promises feasibility).
- Its objective value is never better than the true optimum: if it is, the energy function is wrong.
- Output formats hold: probabilities sum to one, counts match the number of shots.

## 7. Exact solvers as oracles

Gurobi, HiGHS, OR-Tools or SCIP make good oracles at sizes where enumeration is impossible. They need installed
binaries and sometimes licences, so mark those tests `integration` and skip when the solver is missing. Prefer
the enumeration oracle in unit tests.
