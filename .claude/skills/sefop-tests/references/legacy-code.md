# Testing legacy and notebook-derived code

Research code is often untested, lives in one large module, or was copied out of notebooks. The aim is to make it
safe to change, following Feathers' *Working Effectively with Legacy Code*: pin what it does now, expose what is
wrong, then change it.

## Getting the code under test

- **Import seams.** If the module imports matplotlib at top level, set `matplotlib.use("Agg")` in `conftest.py`
  before importing it. If it reads data from an absolute path, give the function a `base_dir`, or point a fixture
  at a small copy under `tests/resources/`.
- **No working environment?** Find one that imports the module (conda envs, venvs), or build a scratch venv with
  uv. Pin the versions the suite passes with in `requirements.txt`.
- **Notebooks.** Test the functions that notebooks call, not the notebooks themselves. When a notebook relies on
  `from module import *`, list the names it actually uses before changing any import.

## Characterization tests

These pin the current behaviour when the true contract is unknown or disputed. Name them by the behaviour, and
say in a comment that they are characterization tests and why. For example: "the docstring claims 1.5; the
notebook plots this value as '1 − ratio'".

## Known bugs: pinned, not hidden

When a test exposes a bug, keep the test asserting the correct behaviour, and mark it so the suite stays green
while the bug stays visible:

```python
@pytest.mark.xfail(
    reason="BUG-03: MIQP() constrains sum x <= s, but the documented model uses = s",
    raises=AssertionError,
)
def test__miqp__given_a_sensor_budget__matches_the_documented_equality_model() -> None:
    ...
```

- **Set `xfail_strict = true`** in `pytest.ini`. A fixed bug then reports XPASS and fails the run, which prompts
  removing the marker so the test guards the fix from then on.
- **Always pass `raises=`.** It makes the xfail accept only the expected failure, so an unrelated crash still
  fails the run.
- **When the bug is that nothing is raised,** the test uses `pytest.raises` and fails with "DID NOT RAISE". Use
  `raises=pytest.fail.Exception` for that case.
- **Record every bug in one list**, such as `docs/code_review.md` or `BUGS.md`:

  | ID | Severity | Where | What happens | Test |
  |---|---|---|---|---|

  Severity: H means wrong results or a crash in a documented step; M means wrong output in some cases, or broken
  reproducibility; L means an edge case.
- **Do not fix bugs while writing tests** unless asked. The test commit should be reviewable as "tests only", and
  the fixes arrive later, one at a time.

## Refactoring under the suite

For a behaviour-preserving move, such as splitting a large module into a package, verify it three ways:

1. **The same tests are collected.** Compare `pytest --collect-only -q` before and after, by test name.
2. **The same outcome:** identical passed and xfailed counts.
3. **For a pure move, the code is unchanged.** Compare each function's syntax tree (`ast.dump`) with the
   original. Docstrings change if tabs become spaces, so normalise those first. The only differences left should
   be the edits you meant to make.
