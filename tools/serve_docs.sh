#!/usr/bin/env bash
# Preview the documentation site locally with live reload.
#
#   ./tools/serve_docs.sh          then open http://127.0.0.1:8000
#
# Edits to docs/*.md rebuild in about a second. Notebooks are not re-executed, so
# editing one shows the new source with its existing outputs.
#
# Uses `python3 -m mkdocs` rather than the `mkdocs` command. pip installs console
# scripts into a bin/ directory that is frequently not on PATH -- on a python.org
# framework build of macOS Python, for instance, it lands in
# /Library/Frameworks/Python.framework/Versions/3.X/bin and pip warns about it.
# The module form works regardless of PATH, so this script does not care.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="${PYTHON:-python3}"

if ! "$PY" -c "import mkdocs" >/dev/null 2>&1; then
    echo "MkDocs is not importable by $PY. Install the docs dependencies:"
    echo "    $PY -m pip install -r requirements-docs.txt"
    exit 1
fi

# Fail early and clearly if MkDocs is too old for exclude_docs, which is the only
# thing keeping the unfinished tutorials off the site.
"$PY" - <<'PYCHECK'
import sys, mkdocs
major, minor = (int(x) for x in mkdocs.__version__.split(".")[:2])
if (major, minor) < (1, 6):
    sys.exit("MkDocs %s is too old: `exclude_docs` is ignored below 1.6, which\n"
             "republishes the unfinished tutorials. Run:\n"
             "    python3 -m pip install -r requirements-docs.txt" % mkdocs.__version__)
PYCHECK

echo "Serving on http://127.0.0.1:8000  (Ctrl-C to stop)"
exec "$PY" -m mkdocs serve "$@"
