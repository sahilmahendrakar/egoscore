"""Inline the demo payload into the page, producing a single self-contained HTML file."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "demo" / "index.html"
DATA = ROOT / "reports" / "demo_data.json"
OUT = ROOT / "demo" / "egoscore_demo.html"

tpl = TPL.read_text()
data = DATA.read_text()

if "__DATA__" not in tpl:
    sys.exit("template has no __DATA__ placeholder")

# The payload sits inside <script type="application/json">, so the only sequence that can
# break out is a literal "</script>". Escape it defensively; JSON keeps the same value.
data = data.replace("</", "<\\/")

html = tpl.replace("__DATA__", data)
OUT.write_text(html)
print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")
