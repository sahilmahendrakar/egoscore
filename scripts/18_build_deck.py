"""Inline figures into the deck template, producing a self-contained slideshow file."""

import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "demo" / "deck.html"
FIGS = ROOT / "reports" / "figs"
OUT = ROOT / "demo" / "egoscore_deck.html"

FIGURES = {
    "__FIG_MANIFOLD__": FIGS / "manifold_compare.png",
    "__FIG_PAIRED__": FIGS / "paired.png",
}

html = TPL.read_text()
for token, path in FIGURES.items():
    if token not in html:
        sys.exit(f"template is missing {token}")
    if not path.exists():
        sys.exit(f"missing figure {path}")
    uri = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    html = html.replace(token, uri)

OUT.write_text(html)
print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")
