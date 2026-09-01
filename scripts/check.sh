#!/usr/bin/env bash

set -euo pipefail

echo "Running Black..."
black --check src tests

echo "Running isort..."
isort --check-only src tests

echo "Running Ruff..."
ruff check src tests

echo "Running tests..."
pytest -v

echo
echo "All tools and tests are passing."
