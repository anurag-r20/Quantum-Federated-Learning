# SEFOP test conventions

Adapted from SEFOP's unit-testing guide (sefop-python-starter, `.claude/skills/tdd/unit-testing-guide.md`) and the
training-testing-python exercises. Examples are short; adapt names to the repo.

## Contents

1. Anatomy and naming
2. File layout
3. What to test
4. Floats, errors and parametrization
5. Fixtures and isolation
6. Mocks
7. Markers, integration tests and data
8. Coverage

## 1. Anatomy and naming

```python
def test__solve__given_two_item_knapsack__picks_the_higher_value_item() -> None:
    # Arrange
    items = [Item(weight=2, value=10), Item(weight=1, value=6)]

    # Act
    solution = solve(items, capacity=2)

    # Assert
    assert solution.value == pytest.approx(10)
```

- **Name pattern:** `test__<unit>__<scenario>__<outcome>`, with double underscores between the sections.
- **One Act per test.** Two calls under test usually mean two behaviours. The exception is a relation between
  calls, such as commutativity or a metamorphic pair.
- **Keep Arrange short.** Move repeated setup into fixtures or small factory helpers.
- **Explain the expected value.** A test's docstring or comment says which promise it checks and where the
  expected value came from, for example "Known oracle: worked out by hand…".

## 2. File layout

```
src/pkg/network.py        tests/test_network.py
src/pkg/model.py          tests/test_model.py
                          tests/conftest.py            shared fixtures, Agg backend
                          tests/oracles/               brute-force solvers and oracle tests
                          tests/integration/           slow or external runs (marked)
                          tests/resources/             small copies of real data
```

- **One test file per production module**, mirroring its path. Do not combine modules into one file.
- **No test classes.** Pytest finds functions, and fixtures replace setUp and tearDown.
- **No `__init__.py` in test folders**, unless two test files share a basename. Then either add packages or rename.

## 3. What to test

Test the contract as callers see it: return values, raised errors and written files. Do not test internal
attributes or private helpers, because a refactor that keeps behaviour must not break a test.

For each public function, list its behaviours:

- the happy path;
- properties that must hold for many inputs, such as identity, symmetry, monotonicity or conservation (a
  probability table summing to one);
- the error contract (which bad inputs raise what);
- boundary cases (empty input, a single element, values exactly at a threshold).

Do not damage the design to make testing easier, for example by making helpers public or adding constructor
parameters only for mocks. Use the Humble Object split instead: keep I/O in a thin shell and test the pure logic
behind it.

## 4. Floats, errors and parametrization

```python
assert energy == pytest.approx(expected, rel=1e-9, abs=1e-12)   # never ==

with pytest.raises(Exception) as error:        # Act records what happened...
    calculate_ratio(1.0, 0.0)
assert isinstance(error.value, ValueError)     # ...Assert checks which error it was

@pytest.mark.parametrize("a,b", [(1.0, 2.0), (-1.5, 3.5)], ids=["positive", "mixed-sign"])
def test__add__given_reversed_operands__returns_same_result(a: float, b: float) -> None:
    ...
```

- Parametrize only when the arrange and assert steps are identical across cases; otherwise write separate tests.
- A `bool` passes as an `int` in Python. When a contract excludes it, test that explicitly.

## 5. Fixtures and isolation

- **Build the object under test inside the test**, or in a function-scoped fixture when the setup is long. Never
  share a module-level mutable instance between tests.
- **Use `tmp_path` for files.** A test must never write into the repository's own data or log folders.
- **Matplotlib:** call `matplotlib.use("Agg")` in `conftest.py` before anything imports pyplot. Add an autouse
  fixture that closes figures after each test.
- **Randomness:** fix the seeds (`random.Random(SEED)`, `numpy.random.default_rng(SEED)`, and samplers' `seed=`),
  so a failure can be reproduced.

## 6. Mocks

Mock only at boundaries: network, files, solver binaries, time, and plotting `show()`.

```python
def test__plot__given_show_false__does_not_call_show(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(plotting.plt, "show", lambda *a, **k: calls.append(1))
    plotting.plot_energies(sampleset, show=False)
    assert calls == []
```

- Prefer dependency injection to patching.
- When you do patch, patch where the object is used (`module_under_test.plt`), not where it is defined.
- Never mock domain objects or pure functions: call them for real.

## 7. Markers, integration tests and data

```ini
markers =
    integration: end-to-end runs through several functions, real solvers or large data (deselect with -m "not integration")
```

- **Unit tests run by default and stay fast.**
- **Integration tests are marked**, and CI runs them separately.
- **Tests that need data outside the repo** read its location from an environment variable, and skip with a
  reason naming that variable when the data is absent.

## 8. Coverage

Measure it with `pytest --cov=<package> --cov-report=term-missing`. Treat it as a signal, not a goal: lines that
nothing executes show what is untested, while a high percentage alone proves little. The mutation check is the
stronger test of whether the suite would notice a wrong answer.
