#!/usr/bin/env python3
"""Fail if the BibTeX on the citing page has drifted from cite_fitted.bib.

docs/citing.md tells the reader that cite_fitted.bib is authoritative. That is
only true if something enforces it -- the two silently disagreed for months
(the page dropped two of the four authors, and the .bib was still the 2024
arXiv preprint entry rather than the published MNRAS record).

Run from the repository root. Exits non-zero on any mismatch.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
bib_path = ROOT / "cite_fitted.bib"
md_path = ROOT / "docs" / "citing.md"

for p in (bib_path, md_path):
    if not p.exists():
        print("check_citation_sync: missing %s" % p.relative_to(ROOT))
        sys.exit(1)

bib = bib_path.read_text().strip()
md = md_path.read_text()

blocks = re.findall(r"```bibtex\n(.*?)\n```", md, flags=re.S)
if len(blocks) != 1:
    print("check_citation_sync: expected exactly one ```bibtex block in "
          "docs/citing.md, found %d" % len(blocks))
    sys.exit(1)

if blocks[0].strip() != bib:
    print("check_citation_sync: docs/citing.md does not match cite_fitted.bib")
    print()
    import difflib
    diff = difflib.unified_diff(bib.splitlines(), blocks[0].strip().splitlines(),
                                fromfile="cite_fitted.bib",
                                tofile="docs/citing.md (bibtex block)", lineterm="")
    for line in diff:
        print("  " + line)
    print()
    print("Fix: paste cite_fitted.bib verbatim into the bibtex block.")
    sys.exit(1)

# Cheap sanity checks on the .bib itself: these are the fields people forget to
# update when a preprint becomes a paper.
missing = [f for f in ("doi", "volume", "pages", "journal") if not re.search(r"^\s*%s\s*=" % f, bib, re.M)]
if missing:
    print("check_citation_sync: cite_fitted.bib is missing field(s): %s" % ", ".join(missing))
    sys.exit(1)
if "arXiv e-prints" in bib:
    print("check_citation_sync: cite_fitted.bib still points at the arXiv preprint, "
          "not the published record.")
    sys.exit(1)

print("check_citation_sync: docs/citing.md matches cite_fitted.bib, and the entry "
      "is a published record.")
