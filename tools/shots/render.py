"""Render ANSI terminal text to a PNG that looks like a terminal window.

python render.py in.ansi out.png "window title" [highlight phrase ...]
Highlight phrases get a translucent amber background (for quotes in plain logs).
"""
import html
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops

src, out, title, *highlights = sys.argv[1:]
text = Path(src).read_text()

STYLE = {"1": "font-weight:700", "2": "opacity:.62", "31": "color:#ff6b6b", "32": "color:#7ee787",
         "33": "color:#f2cc60", "36": "color:#56d4dd"}
parts, active = [], []
for chunk in re.split(r"(\x1b\[\d+m)", text):
    m = re.fullmatch(r"\x1b\[(\d+)m", chunk)
    if m:
        active = [] if m.group(1) == "0" else active + [STYLE.get(m.group(1), "")]
        continue
    esc = html.escape(chunk)
    for h in highlights:
        # Words may be split across a wrapped line, so match any whitespace between them.
        pattern = r"\s+".join(re.escape(html.escape(w)) for w in h.split())
        esc = re.sub(pattern, lambda m: f"<mark>{m.group(0)}</mark>", esc)
    parts.append(f'<span style="{";".join(active)}">{esc}</span>' if active else esc)

page = f"""<!doctype html><meta charset="utf-8"><style>
body{{margin:0;background:#0b0f14;padding:28px;display:inline-block}}
.win{{background:#0d1117;border:1px solid #30363d;border-radius:10px;box-shadow:0 12px 40px #0008;width:860px}}
.bar{{height:34px;border-bottom:1px solid #21262d;display:flex;align-items:center;padding:0 14px;gap:8px}}
.dot{{width:12px;height:12px;border-radius:50%}}
.t{{color:#8b949e;font:13px system-ui,sans-serif;margin-left:10px}}
pre{{margin:0;padding:18px 22px 22px;color:#c9d1d9;font:14.5px/1.5 'DejaVu Sans Mono','Liberation Mono',monospace;white-space:pre-wrap}}
mark{{background:#f2cc6033;color:#f2cc60;border-radius:3px;padding:0 2px}}
</style><div class="win"><div class="bar"><span class="dot" style="background:#ff5f57"></span>
<span class="dot" style="background:#febc2e"></span><span class="dot" style="background:#28c840"></span>
<span class="t">{html.escape(title)}</span></div><pre>{"".join(parts)}</pre></div>"""
page_path = Path(src).with_suffix(".html")
page_path.write_text(page)
subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                f"--screenshot={Path(out).resolve()}", "--window-size=916,4000", page_path.resolve().as_uri()],
               check=True, capture_output=True)
img = Image.open(out).convert("RGB")
bg = Image.new("RGB", img.size, img.getpixel((0, img.height - 1)))
box = ImageChops.difference(img, bg).getbbox()
pad = 56
img.crop((0, 0, img.width, min(img.height, box[3] + pad))).save(out)
print(out, img.width, min(img.height, box[3] + pad))
