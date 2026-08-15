#!/usr/bin/env bash
# Build the static site into site/ and run the same checks CI runs.
#
#   ./tools/build_docs.sh     then open site/index.html
#
# `use_directory_urls: false` in mkdocs.yml means site/index.html works when
# opened straight from disk -- no web server needed.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="${PYTHON:-python3}"

# --strict turns warnings into errors, so a broken internal link fails here
# rather than shipping.
"$PY" -m mkdocs build --strict

"$PY" tools/check_notebook_outputs.py
"$PY" tools/check_site_excludes.py

echo
echo "Built. Open it with:"
echo "    open site/index.html"
