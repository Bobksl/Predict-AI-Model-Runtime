#!/usr/bin/env python
"""
Build a nicely formatted PDF from a Markdown source.

Pipeline:  Markdown  --(python-markdown + pymdown-extensions)-->  HTML
           HTML  --(headless Chrome --print-to-pdf, MathJax for math)-->  PDF

Math is written in LaTeX inside the Markdown ($...$ inline, $$...$$ display) and
rendered by MathJax v3. We use pymdownx.arithmatex in *generic* mode so the math
survives Markdown processing untouched and is typeset in the browser.

Usage:
    python build_pdf.py <input.md> <output.pdf> ["Document Title"] ["Subtitle"]
"""
from __future__ import annotations
import sys, os, subprocess, datetime, tempfile, shutil

import markdown

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if os.path.exists(c):
            return c
    raise SystemExit("No Chrome/Edge found for PDF rendering.")

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']] }},
  options: {{ ignoreHtmlClass: 'tex2jax_ignore', processHtmlClass: 'arithmatex' }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
:root {{ --accent:#1f4e79; --accent2:#2e7d32; --ink:#1a1a1a; --muted:#5a6470;
        --code-bg:#f5f7fa; --border:#d0d7de; }}
@page {{ size: A4; margin: 18mm 16mm 20mm 16mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: var(--ink);
        font-size: 10.6pt; line-height: 1.5; margin:0; }}
.cover {{ padding: 24mm 0 10mm 0; border-bottom: 3px solid var(--accent); margin-bottom: 8mm; }}
.cover h1 {{ font-size: 26pt; color: var(--accent); margin: 0 0 4mm 0; border:none; padding:0; }}
.cover .sub {{ font-size: 13pt; color: var(--muted); margin:0 0 6mm 0; }}
.cover .meta {{ font-size: 9pt; color: var(--muted); }}
h1 {{ font-size: 18pt; color: var(--accent); border-bottom: 2px solid var(--accent);
      padding-bottom: 2mm; margin-top: 10mm; break-before: page; }}
h1:first-of-type {{ break-before: avoid; }}
h2 {{ font-size: 14pt; color: var(--accent); margin-top: 7mm; border-bottom:1px solid var(--border); padding-bottom:1mm; }}
h3 {{ font-size: 12pt; color: #2c3e50; margin-top: 5mm; }}
h4 {{ font-size: 10.8pt; color: #2c3e50; margin-top: 4mm; }}
p {{ margin: 2.2mm 0; }}
a {{ color: var(--accent); text-decoration: none; }}
code {{ font-family: "Cascadia Code","Consolas",monospace; font-size: 9pt;
        background: var(--code-bg); padding: 0.5mm 1.2mm; border-radius: 3px; }}
pre {{ background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px;
       padding: 3mm 4mm; overflow-x: auto; break-inside: avoid; font-size: 8.8pt; line-height:1.4; }}
pre code {{ background: none; padding: 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 3mm 0; font-size: 9.2pt; break-inside: avoid; }}
th, td {{ border: 1px solid var(--border); padding: 1.6mm 2.4mm; text-align: left; vertical-align: top; }}
th {{ background: #eef2f7; color: var(--accent); }}
tr:nth-child(even) td {{ background: #fafbfc; }}
blockquote {{ border-left: 3px solid var(--accent2); background:#f3f8f3; margin: 3mm 0;
              padding: 1.5mm 4mm; color:#33402f; }}
ul, ol {{ margin: 2mm 0 2mm 6mm; }}
li {{ margin: 1mm 0; }}
hr {{ border:none; border-top:1px solid var(--border); margin: 6mm 0; }}
.toc {{ background:#f8fafc; border:1px solid var(--border); border-radius:6px; padding: 3mm 6mm;
        font-size: 9.4pt; break-inside: avoid; }}
.toc ul {{ list-style: none; margin-left: 3mm; }}
.toc > ul {{ margin-left: 0; }}
.arithmatex {{ font-size: 1.0em; }}
mjx-container[display="true"] {{ margin: 2mm 0; }}
img {{ max-width: 100%; }}
.katex, mjx-container {{ break-inside: avoid; }}
</style>
</head>
<body>
<div class="cover">
  <h1>{title}</h1>
  <div class="sub">{subtitle}</div>
  <div class="meta">{meta}</div>
</div>
{body}
</body>
</html>
"""

def md_to_html(md_text: str) -> str:
    md = markdown.Markdown(extensions=[
        "pymdownx.arithmatex",
        "tables", "fenced_code", "codehilite", "toc", "admonition",
        "pymdownx.superfences", "pymdownx.tilde", "pymdownx.caret",
        "sane_lists", "attr_list", "md_in_html",
    ], extension_configs={
        "pymdownx.arithmatex": {"generic": True},
        "codehilite": {"guess_lang": False, "noclasses": True},
        "toc": {"permalink": False, "toc_depth": "2-3"},
    })
    return md.convert(md_text)

def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(os.path.basename(src))[0]
    subtitle = sys.argv[4] if len(sys.argv) > 4 else ""
    meta = ("Google – Fast or Slow? Predict AI Model Runtime  ·  "
            "Generated " + datetime.date.today().isoformat())

    with open(src, "r", encoding="utf-8") as f:
        md_text = f.read()
    body = md_to_html(md_text)
    html = HTML_TEMPLATE.format(title=title, subtitle=subtitle, meta=meta, body=body)

    out_abs = os.path.abspath(out)
    work = tempfile.mkdtemp(prefix="pdfbuild_")
    try:
        html_path = os.path.join(work, "doc.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        chrome = find_chrome()
        user_data = os.path.join(work, "cud")
        cmd = [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
               "--no-first-run", "--no-default-browser-check",
               f"--user-data-dir={user_data}",
               "--run-all-compositor-stages-before-draw",
               "--virtual-time-budget=60000",
               "--no-pdf-header-footer",
               f"--print-to-pdf={out_abs}", html_path]
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not os.path.exists(out_abs):
            raise SystemExit(f"PDF was not produced at {out_abs}")
        print(f"Wrote {out_abs} ({os.path.getsize(out_abs):,} bytes)")
    finally:
        shutil.rmtree(work, ignore_errors=True)

if __name__ == "__main__":
    main()
