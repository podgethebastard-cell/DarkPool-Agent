Here’s a **useful, modern Python repo folder** you can drop into GitHub and reuse for most projects (library *or* app). It’s opinionated in a good way: clean `src/` layout, tests, CI, formatting/linting, and a tiny CLI starter.

---

## Folder layout (copy this)

```txt
your-repo/
├─ .github/
│  ├─ workflows/
│  │  └─ ci.yml
│  ├─ ISSUE_TEMPLATE/
│  │  ├─ bug_report.md
│  │  └─ feature_request.md
│  └─ PULL_REQUEST_TEMPLATE.md
├─ src/
│  └─ your_package/
│     ├─ __init__.py
│     ├─ cli.py
│     └─ core.py
├─ tests/
│  └─ test_smoke.py
├─ examples/
│  └─ quickstart.py
├─ scripts/
│  └─ dev.sh
├─ .gitignore
├─ .editorconfig
├─ .env.example
├─ LICENSE
├─ README.md
├─ CONTRIBUTING.md
├─ pyproject.toml
├─ ruff.toml
└─ pytest.ini
```

> Replace `your_package` with your actual package name (e.g. `starforge`, `my_tool`, etc.).

---

## File contents (paste as-is)

### `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyd
*.so
*.egg-info/
dist/
build/

# Virtualenvs
.venv/
venv/
ENV/

# Test/coverage
.pytest_cache/
.coverage
htmlcov/
.coverage.*

# Editors/OS
.vscode/
.idea/
.DS_Store

# Env files
.env
```

### `.editorconfig`

```ini
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
indent_style = space
indent_size = 4
trim_trailing_whitespace = true

[*.md]
trim_trailing_whitespace = false
```

### `.env.example`

```bash
# Copy to .env and fill in values if needed
# EXAMPLE_API_KEY="..."
```

### `README.md`

````md
# your-repo

A clean, modern Python project template with:
- `src/` layout
- tests (pytest)
- lint/format (ruff)
- GitHub Actions CI

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
pytest
````

## Run the CLI

```bash
python -m your_package --help
python -m your_package hello
```

## Project structure

* `src/your_package/` — app/library code
* `tests/` — tests
* `examples/` — runnable examples
* `.github/workflows/ci.yml` — CI

````

### `LICENSE` (MIT)
```txt
MIT License

Copyright (c) YEAR YOUR NAME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
````

*(Replace `YEAR` + `YOUR NAME`.)*

### `CONTRIBUTING.md`

````md
# Contributing

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
````

## Checks

```bash
ruff check .
ruff format .
pytest
```

## Notes

* Keep changes small and focused
* Add/adjust tests for behavior changes

````

---

## Python package starter

### `pyproject.toml`
```toml
[build-system]
requires = ["hatchling>=1.25.0"]
build-backend = "hatchling.build"

[project]
name = "your-repo"
version = "0.1.0"
description = "A useful Python project template."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "YOUR NAME" }]
dependencies = []

[project.optional-dependencies]
dev = [
  "pytest>=8.0.0",
  "ruff>=0.6.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/your_package"]

[project.scripts]
your-tool = "your_package.cli:main"
````

### `ruff.toml`

```toml
line-length = 100
target-version = "py310"

[lint]
select = ["E", "F", "I", "UP", "B"]
ignore = []

[format]
quote-style = "double"
indent-style = "space"
```

### `pytest.ini`

```ini
[pytest]
testpaths = tests
```

### `src/your_package/__init__.py`

```python
__all__ = ["__version__"]
__version__ = "0.1.0"
```

### `src/your_package/core.py`

```python
def hello(name: str = "world") -> str:
    return f"Hello, {name}!"
```

### `src/your_package/cli.py`

```python
import argparse
from .core import hello


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="your-tool", description="A tiny, useful CLI starter.")
    sub = p.add_subparsers(dest="cmd", required=True)

    hello_p = sub.add_parser("hello", help="Print a friendly greeting")
    hello_p.add_argument("name", nargs="?", default="world")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "hello":
        print(hello(args.name))


if __name__ == "__main__":
    main()
```

---

## Tests + example

### `tests/test_smoke.py`

```python
from your_package.core import hello


def test_hello_default():
    assert hello() == "Hello, world!"


def test_hello_name():
    assert hello("Ada") == "Hello, Ada!"
```

### `examples/quickstart.py`

```python
from your_package.core import hello

print(hello("GitHub"))
```

---

## Dev helper script

### `scripts/dev.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

ruff check .
ruff format .
pytest
```

---

## GitHub goodies

### `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install
        run: |
          python -m pip install -U pip
          pip install -e ".[dev]"

      - name: Lint
        run: ruff check .

      - name: Format check
        run: ruff format --check .

      - name: Test
        run: pytest
```

### `.github/PULL_REQUEST_TEMPLATE.md`

```md
## What changed?

## Why?

## How to test
- [ ] `ruff check .`
- [ ] `ruff format --check .`
- [ ] `pytest`
```

### `.github/ISSUE_TEMPLATE/bug_report.md`

```md
---
name: Bug report
about: Create a report to help us improve
---

## What happened?

## Steps to reproduce

## Expected behavior

## Environment
- OS:
- Python:
```

### `.github/ISSUE_TEMPLATE/feature_request.md`

```md
---
name: Feature request
about: Suggest an idea
---

## What do you want to add?

## Why is it useful?

## Possible approach
```

---

