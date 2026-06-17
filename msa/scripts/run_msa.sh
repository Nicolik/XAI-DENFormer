#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper kept for users who prefer shell entry points.
# The canonical MSA workflow entry point is:
#   python -m msa.run

python -m msa.run "$@"
