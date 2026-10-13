---
name: sefop-tests
description: Build SEFOP-style pytest suites for research, optimization and scientific Python code. Covers test__unit__scenario__outcome names, Arrange/Act/Assert, one test file per module, and oracle-based tests for solvers, MIP/QUBO models and heuristics (hand-derived answers, brute-force pseudo-oracles, metamorphic relations, cross-implementation checks). Also covers characterization tests for legacy or notebook-derived code, strict-xfail tests that pin confirmed bugs, pytest config, pinned requirements, CI and a mutation check. Use whenever the user asks to add, write or improve tests, set up pytest, test an optimization model or solver, "add sefop tests", audit test quality, or make a repo safe to refactor, even if SEFOP is not mentioned.
---

# SEFOP tests

The goal is a suite a skeptical reviewer would trust. Each test states one promised behaviour, gets its expected
value from an explicit oracle (never from running the code under test), and confirmed bugs are pinned in plain
sight rather than worked around.

The conventions come from SEFOP's training material ([training-testing-python](https://github.com/sefop/training-testing-python),
[sefop-python-starter](https://github.com/sefop/sefop-python-starter)). They are embedded here, so there is no need
to fetch those repos. If the user asks to refresh from upstream, read the files with
`gh api repos/sefop/<repo>/contents/<path>`.

## Workflow

1. **Survey before writing anything.**
   - **Layout:** a `src/` package, one big utils module, notebooks?
   - **Results:** how the code produces its results.
   - **Environment:** find one that imports the code (existing conda envs, venvs, requirements). If none does,
     build a scratch venv with uv and note the versions it gets.
   - **Existing tests and CI.**
   - **The contracts:** docstrings, the README, the paper's formulation, and how notebooks call the code. These
     define the behaviours to test.

2. **Lay out the suite** as in `references/conventions.md`:
   - `tests/test_<module>.py`, mirroring each production module;
   - `tests/conftest.py`, copied from `assets/conftest.py`;
   - `tests/oracles/` for solver oracles;
   - `tests/integration/` for slow or external runs, marked `integration`;
   - `tests/resources/` for small copies of real data.

   Configure pytest from `assets/pytest.ini`, or the `[tool.pytest.ini_options]` equivalent in pyproject. Keep
   `xfail_strict = true`.

3. **Write one test per behaviour.** Cover the happy path, properties that must hold for many inputs
   (parametrized), the error contract, and edge cases.
   - For legacy code, add characterization tests that pin current behaviour, with a comment saying that is what
     they are (`references/legacy-code.md`).
   - Start from `assets/test_module_template.py`.

4. **For optimization, QUBO and solver code, add oracles** (`references/oracles.md`):
   - a known answer worked out by hand;
   - a brute-force pseudo-oracle (`assets/enumeration_solver.py`) that shares no code with production;
   - cross-implementation agreement on every assignment;
   - metamorphic relations;
   - regression against logged or published results.

5. **When a test exposes a bug:**
   1. Reproduce it.
   2. Keep the test stating the correct behaviour, and mark it
      `@pytest.mark.xfail(reason="BUG-NN: ...", raises=SomeError)`.
   3. Record the bug in the repo's bug list (`docs/code_review.md` or `BUGS.md`) with its severity, location and
      failure scenario.

   Do not fix production code unless the user asks. With strict xfail, a later fix turns the test into an XPASS
   failure, which prompts removing the marker so the test guards the fix.

6. **Verify the suite.**
   - Everything passes except the declared xfails.
   - Each xfail fails for its stated reason (`raises=`).
   - The collected count is what you expect.
   - Run it twice to catch order dependence.

7. **Run a mutation check.** Run `scripts/mutation_check.py` with 5–10 mutations of the core logic: a sign flip, an
   off-by-one, a dropped term, a wrong constant. Every mutation should make at least one test fail; strengthen the
   tests wherever one survives. Coverage says which lines ran; mutations say whether the tests would notice a
   wrong answer.

8. **Deliver:**
   - requirements pinned to the versions the suite was verified with;
   - CI from `assets/ci-unit-tests.yml`;
   - a short how-to-run note.

   Report the counts (passed, xfailed, skipped), the bugs found, the mutation results, and coverage per module.

## Conventions at a glance

Details and examples are in `references/conventions.md`.

- **Names** read `test__<unit>__<scenario>__<expected outcome>`, so a failing test says in plain words which
  promise broke.
- **Layout:** Arrange / Act / Assert, marked with comments. Use one Act per test; the exception is a relation
  between two calls (commutativity, a metamorphic pair).
- **Structure:** flat test functions, no classes. One test file per production module, mirroring its path.
- **Expected values** come from an oracle: a hand calculation, an independent implementation, or a provable
  relation.
- **Floats** are compared with `pytest.approx` or `math.isclose`, never `==`.
- **Scope:** test public behaviour only. Mock only at I/O boundaries: files, network, solver binaries, plotting
  `show()`, time.
- **Isolation:** build the object under test inside the test or in a function-scoped fixture, with no
  module-level mutable state.
- **Parametrize** cases that share structure, with readable `ids`.
- **Speed:** unit tests stay fast; anything slow, external or needing large data is marked `integration`, and
  skips with a clear reason when its data is absent.
