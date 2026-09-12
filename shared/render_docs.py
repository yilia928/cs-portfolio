# -*- coding: utf-8 -*-
"""
通用文档渲染器：Markdown -> 精美 HTML -> PDF（Edge headless 打印）
用法：
    python shared/render_docs.py a.md b.md --outdir output
    python shared/render_docs.py a.md --no-pdf      # 只出 HTML
所有项目复用同一套专业排版样式。
"""
import sys
import subprocess
from pathlib import Path

import markdown

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
:root{--brand:#2e5bff;--ink:#1f2937;--muted:#6b7280;--line:#e5e7eb;--bg:#f6f8fc}
*{box-sizing:border-box}
body{font-family:"Microsoft YaHei","PingFang SC","Source Han Sans SC",sans-serif;color:var(--ink);
 line-height:1.75;margin:0;background:var(--bg);font-size:14px}
.page{max-width:880px;margin:0 auto;padding:48px 56px;background:#fff;min-height:100vh}
h1{font-size:27px;color:var(--brand);border-bottom:3px solid var(--brand);padding-bottom:14px;margin:0 0 6px}
h2{font-size:19px;margin:34px 0 12px;padding-left:11px;border-left:5px solid var(--brand);color:#1e3a8a}
h3{font-size:16px;margin:22px 0 8px;color:#374151}
p{margin:9px 0}
blockquote{margin:14px 0;padding:10px 16px;background:#eef3ff;border-left:4px solid var(--brand);
 color:#374151;font-size:13px;border-radius:0 6px 6px 0}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:12.8px}
th{background:var(--brand);color:#fff;padding:8px 10px;text-align:left;font-weight:600}
td{border:1px solid var(--line);padding:7px 10px;vertical-align:top}
tr:nth-child(even) td{background:#f8fafc}
code{background:#eef2f7;padding:1px 6px;border-radius:4px;font-size:12.5px;color:#b5179e}
pre{background:#0f172a;color:#e2e8f0;padding:14px 16px;border-radius:8px;overflow-x:auto;font-size:12.5px;line-height:1.6}
pre code{background:none;color:inherit;padding:0}
ul,ol{padding-left:24px} li{margin:5px 0}
hr{border:none;border-top:1px dashed #cbd5e1;margin:26px 0}
strong{color:#111827}
a{color:var(--brand)}
@page{size:A4;margin:16mm 14mm}
@media print{
 body{background:#fff;font-size:12px}
 .page{max-width:none;padding:0}
 h2{page-break-after:avoid} table,blockquote,pre{page-break-inside:avoid}
 h2,h3{page-break-after:avoid}
}
"""

HTML_TPL = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{title}</title><style>{css}</style></head>
<body><div class="page">{body}</div></body></html>"""


def md_to_html(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "nl2br"])
    return HTML_TPL.format(title=md_path.stem, css=CSS, body=body)


def to_pdf(html_path: Path, pdf_path: Path) -> bool:
    edge = next((p for p in EDGE_CANDIDATES if Path(p).exists()), None)
    if not edge:
        print("[WARN] Edge not found, skip PDF:", pdf_path.name)
        return False
    cmd = [edge, "--headless", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    ok = pdf_path.exists()
    print(("[OK] PDF  -> " if ok else "[FAIL] PDF ") + str(pdf_path))
    if not ok:
        print(r.stderr[:300])
    return ok


def render(md_path: Path, outdir: Path, make_pdf: bool = True):
    outdir.mkdir(parents=True, exist_ok=True)
    html = md_to_html(md_path)
    html_path = outdir / (md_path.stem + ".html")
    html_path.write_text(html, encoding="utf-8")
    print("[OK] HTML -> " + str(html_path))
    if make_pdf:
        to_pdf(html_path, outdir / (md_path.stem + ".pdf"))


def main():
    raw = sys.argv[1:]
    outdir = None
    if "--outdir" in raw:
        i = raw.index("--outdir")
        outdir = Path(raw[i + 1]).resolve()
        del raw[i:i + 2]
    make_pdf = "--no-pdf" not in raw
    files = [Path(a).resolve() for a in raw if not a.startswith("--")]
    for md in files:
        render(md, outdir or md.parent.parent / "output", make_pdf=make_pdf)


if __name__ == "__main__":
    main()
