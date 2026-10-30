# Install

TOPS needs **Python 3.10 or newer**. Its only dependencies are numpy, scipy,
pandas, matplotlib and progressbar2, so it installs in seconds.

## From PyPI

=== "pip"

    ```bash
    pip install tops
    ```

=== "uv"

    ```bash
    uv add tops
    ```

Check that it works:

```bash
python -c "import tops.dynamic; print('TOPS is ready')"
```

## From source

TOPS is meant to be read and modified, so many users work from a clone.
An editable install picks up your changes to the source straight away:

```bash
git clone https://github.com/hallvar-h/TOPS.git
cd TOPS
pip install -e .
```

The repository uses [uv](https://docs.astral.sh/uv/) for development. With
uv, `uv sync` creates an environment with TOPS and the test tools:

```bash
uv sync
uv run pytest            # fast tests
uv run pytest -m slow    # full-length validation against PowerFactory
```

## Try it without installing

The example notebooks run in the browser on Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hallvar-h/TOPS/HEAD?filepath=examples%2Fnotebooks)

## Building this documentation

```bash
uv sync --group docs
uv run mkdocs serve
```

Then open <http://127.0.0.1:8000>. `uv run mkdocs build --strict` is what CI
runs; it fails on broken links and on code examples that raise an error.

Next: [run your first simulation](quickstart.md).
