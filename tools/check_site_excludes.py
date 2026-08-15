#!/usr/bin/env python3
"""Fail if the BUILT site contains a page that mkdocs.yml excludes.

Why this exists as a separate check, rather than trusting `exclude_docs`:

`docs/tutorials` is a symlink to `../tutorials`, so every notebook in the
repository is visible to the build -- including the five that are deliberately
unfinished. They are kept out by `exclude_docs` in mkdocs.yml, which requires
**MkDocs >= 1.6**. An older MkDocs does not error on the unknown key; it ignores
it, and publishes all thirteen notebooks as live, search-indexed pages. Nothing in
the navigation changes, so the only symptom is unfinished tutorials quietly
appearing on the public site.

requirements-docs.txt pins a new enough MkDocs, but a pin is a statement of
intent. This checks the artefact.

Run after `mkdocs build`, from the repository root:

    python tools/check_site_excludes.py
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "mkdocs.yml"


class _Loader(yaml.SafeLoader):
    """Ignore mkdocs.yml's `!!python/name:` tags."""


_Loader.add_multi_constructor("", lambda loader, suffix, node: None)

config = yaml.load(CONFIG.read_text(), Loader=_Loader)
site_dir = ROOT / config.get("site_dir", "site")

if not site_dir.is_dir():
    print("check_site_excludes: %s does not exist -- run `mkdocs build` first."
          % site_dir)
    sys.exit(1)

patterns = [line.strip()
            for line in (config.get("exclude_docs") or "").splitlines()
            if line.strip() and not line.strip().startswith("#")]

if not patterns:
    print("check_site_excludes: mkdocs.yml declares no exclude_docs patterns.")
    sys.exit(1)

leaked = []
for pattern in patterns:
    # A notebook excluded as tutorials/06_x.ipynb must not appear in the site as
    # either the rendered page or the copied source.
    stem = Path(pattern)
    if stem.suffix == ".ipynb":
        candidates = [stem.with_suffix(".html"), stem]
    else:
        candidates = list(site_dir.glob(pattern))
        leaked += [c.relative_to(site_dir) for c in candidates if c.exists()]
        continue
    for rel in candidates:
        if (site_dir / rel).exists():
            leaked.append(rel)

print("check_site_excludes: %d exclusion pattern(s) declared." % len(patterns))

if leaked:
    print("\nFAILED -- these were excluded in mkdocs.yml but are published anyway:\n")
    for rel in sorted(set(map(str, leaked))):
        print("  - %s" % rel)
    print("\nAlmost certainly MkDocs is older than 1.6, which ignores exclude_docs.")
    try:
        import mkdocs
        print("Installed MkDocs: %s" % mkdocs.__version__)
    except Exception:
        pass
    print("Fix with:  python3 -m pip install -r requirements-docs.txt")
    sys.exit(1)

print("Nothing excluded has leaked into %s." % site_dir.name)
