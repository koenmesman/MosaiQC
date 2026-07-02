# opt-core-refactor

This directory is an installable refactored copy of the repository's `src`
project. It keeps the same runtime package name, `opt_core`, so existing tests
and scripts can import it without code changes.

Install from this directory with:

```bash
python -m pip install .
```

For an editable development install:

```bash
python -m pip install -e .
```
