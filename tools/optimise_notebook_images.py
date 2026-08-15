#!/usr/bin/env python3
"""Shrink the PNGs embedded in the executed tutorials, without re-running them.

Matplotlib writes notebook figures as RGBA PNGs with no palette optimisation.
For the corner and walker-trace plots that is over a megabyte each, and since the
notebooks are embedded in the documentation pages it is the dominant cost of
loading the site -- tutorials 05 and 10 came to about 2.9 MB of HTML apiece.

Two transformations, in order of aggression:

1. Flatten RGBA onto white and re-encode with `optimize=True`. Visually lossless:
   the figures have no transparency to preserve once they are on a white page.

2. Quantise to a 256-colour palette. For line plots and the greyscale corner
   plots this is also visually lossless, but it need not be in general, so it is
   applied ONLY if the measured maximum per-channel error is under
   --max-error (default 8/255). Anything worse keeps the result of step 1.

This changes only how the figures are encoded. It does not touch a single number,
a single line of code, or any cell output other than image/png payloads.

Idempotent: run it again and it reports no further saving.

    python tools/optimise_notebook_images.py                 # all published notebooks
    python tools/optimise_notebook_images.py --dry-run
    python tools/optimise_notebook_images.py --max-error 0    # step 1 only
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "mkdocs.yml"


class _Loader(yaml.SafeLoader):
    """Ignore mkdocs.yml's `!!python/name:` tags; we only want `nav`."""


_Loader.add_multi_constructor("", lambda loader, suffix, node: None)


def notebooks_in_nav(nav):
    if isinstance(nav, list):
        return [n for item in nav for n in notebooks_in_nav(item)]
    if isinstance(nav, dict):
        return [n for value in nav.values() for n in notebooks_in_nav(value)]
    if isinstance(nav, str) and nav.endswith(".ipynb"):
        return [nav]
    return []


def flatten(raw):
    """RGBA -> RGB on white, re-encoded with optimize=True."""
    im = Image.open(io.BytesIO(raw))
    if im.mode in ("RGBA", "LA"):
        flat = Image.new("RGB", im.size, (255, 255, 255))
        flat.paste(im, mask=im.split()[-1])
    else:
        flat = im.convert("RGB")
    buf = io.BytesIO()
    flat.save(buf, "PNG", optimize=True)
    return flat, buf.getvalue()


def quantise(flat):
    q = flat.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE)
    buf = io.BytesIO()
    q.save(buf, "PNG", optimize=True)
    err = int(np.abs(np.asarray(flat, dtype=np.int16)
                     - np.asarray(q.convert("RGB"), dtype=np.int16)).max())
    return buf.getvalue(), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-error", type=int, default=8,
                    help="reject quantisation above this max per-channel error "
                         "(0 disables quantisation entirely)")
    args = ap.parse_args()

    config = yaml.load(CONFIG.read_text(), Loader=_Loader)
    docs_dir = ROOT / config.get("docs_dir", "docs")
    published = notebooks_in_nav(config.get("nav"))
    if not published:
        print("No notebooks in the mkdocs.yml nav.")
        return 1

    total_before = total_after = 0

    for rel in published:
        path = (docs_dir / rel).resolve()
        nb = json.loads(path.read_text())
        before = after = 0
        changed = False
        notes = []

        for cell in nb["cells"]:
            for out in cell.get("outputs", []):
                data = out.get("data", {})
                if "image/png" not in data:
                    continue
                payload = data["image/png"]
                raw = base64.b64decode(payload)
                before += len(raw)

                flat, flat_png = flatten(raw)
                best, how = flat_png, "flattened"

                if args.max_error > 0:
                    q_png, err = quantise(flat)
                    if err <= args.max_error and len(q_png) < len(best):
                        best, how = q_png, "quantised (max err %d)" % err
                    elif err > args.max_error:
                        notes.append("kept full colour, quantisation error %d" % err)

                if len(best) < len(raw):
                    after += len(best)
                    changed = True
                    if not args.dry_run:
                        data["image/png"] = base64.b64encode(best).decode("ascii")
                    notes.append("%.0f -> %.0f KB, %s"
                                 % (len(raw) / 1024, len(best) / 1024, how))
                else:
                    after += len(raw)

        total_before += before
        total_after += after
        if before:
            print("%-46s %6.2f -> %6.2f MB%s"
                  % (rel.split("/")[-1], before / 1e6, after / 1e6,
                     "" if changed else "   (already optimal)"))
            for n in notes:
                print("      %s" % n)

        if changed and not args.dry_run:
            path.write_text(json.dumps(nb, indent=1) + "\n")

    if total_before:
        print("\nembedded images: %.2f -> %.2f MB  (%.0f%% saved)%s"
              % (total_before / 1e6, total_after / 1e6,
                 100 * (1 - total_after / total_before),
                 "   [dry run, nothing written]" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
