#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

echo "Setting up field-exploration..."

# Install uv locally if it is not already available.
if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found. Installing it..."

    if ! command -v curl >/dev/null 2>&1; then
        echo "Error: curl is required to install uv." >&2
        exit 1
    fi

    curl -LsSf https://astral.sh/uv/install.sh | sh

    # uv normally installs its executable here.
    export PATH="${HOME}/.local/bin:${PATH}"

    if ! command -v uv >/dev/null 2>&1; then
        echo "Error: uv was installed but is not available on PATH." >&2
        echo "Add \$HOME/.local/bin to your PATH and rerun this script." >&2
        exit 1
    fi
fi

echo "Using $(uv --version)"

# Install a uv-managed Python 3.10 interpreter when needed.
uv python install 3.12

# Record the Python version expected by the project.
echo "3.10" > .python-version

# Create .venv, resolve dependencies, and generate/update uv.lock.
uv sync

echo
echo "Setup complete."
echo
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
echo
echo "Or run commands without activating it:"
echo "  uv run python your_script.py"