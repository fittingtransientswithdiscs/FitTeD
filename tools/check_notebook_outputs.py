#!/usr/bin/env python3
"""Fail if a *published* notebook was committed without its outputs.

The site renders notebooks rather than executing them (mkdocs.yml sets
`execute: false`), so a notebook whose outputs were stripped publishes as bare
source with no figures and no numbers. To a reader that is indistinguishable from
a tutorial that was written but never run -- which is exactly the state the
tutorials were rescued from, so it is worth a guard.

Which notebooks count as published is read from `nav` in mkdocs.yml, not
hardcoded here. docs/tutorials is a symlink to ../tutorials, which also contains
notebooks that are deliberately not shipped; those are excluded in mkdocs.yml and
must not be checked. Keeping one list avoids the two drifting.

Run from the repository root:  python tools/check_notebook_outputs.py
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "mkdocs.yml"


class _Loader(yaml.SafeLoader):
    """mkdocs.yml carries `!!python/name:` tags for the Material icon extension.
    We only want `nav`, so ignore every unknown tag rather than failing."""


_Loader.add_multi_constructor("", lambda loader, suffix, node: None)
_Loader.add_multi_constructor("tag:yaml.org,2002:python/name:",
                              lambda loader, suffix, node: None)


def notebooks_in_nav(nav):
    """Every .ipynb path reachable from the nav tree, in order."""
    found = []
    if isinstance(nav, list):
        for item in nav:
            found += notebooks_in_nav(item)
    elif isinstance(nav, dict):
        for value in nav.values():
            found += notebooks_in_nav(value)
    elif isinstance(nav, str) and nav.endswith(".ipynb"):
        found.append(nav)
    return found


config = yaml.load(CONFIG.read_text(), Loader=_Loader)
docs_dir = ROOT / config.get("docs_dir", "docs")
published = notebooks_in_nav(config.get("nav"))

if not published:
    print("check_notebook_outputs: no notebooks found in the mkdocs.yml nav.")
    sys.exit(1)

problems = []
for rel in published:
    path = docs_dir / rel
    if not path.exists():
        problems.append("%s: in nav but missing from disk" % rel)
        continue

    nb = json.loads(path.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    if not code:
        problems.append("%s: no code cells at all" % rel)
        continue

    unexecuted = [i for i, c in enumerate(code) if not c.get("execution_count")]
    errored = [(i, o["ename"]) for i, c in enumerate(code)
               for o in c["outputs"] if o["output_type"] == "error"]

    if unexecuted:
        problems.append("%s: %d of %d code cells have no output (cells %s)"
                        % (rel, len(unexecuted), len(code),
                           ", ".join(map(str, unexecuted[:8]))))
    if errored:
        problems.append("%s: %d cell(s) raised: %s"
                        % (rel, len(errored), ", ".join(n for _, n in errored[:4])))
    if not nb.get("metadata", {}).get("kernelspec"):
        problems.append("%s: no kernelspec in metadata" % rel)

if problems:
    print("Notebook output check FAILED:\n")
    for p in problems:
        print("  - %s" % p)
    print("\nRe-run tools/execute_tutorials.sh and commit the executed notebooks.")
    sys.exit(1)

print("check_notebook_outputs: %d published notebooks, all executed, no errors."
      % len(published))
for rel in published:
    print("  ok  %s" % rel)
