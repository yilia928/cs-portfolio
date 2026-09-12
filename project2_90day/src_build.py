# -*- coding: utf-8 -*-
"""项目②构建：合并8个模板为《模板包合集》并渲染主文档与合集的 HTML/PDF"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "shared"))
from render_docs import render  # noqa: E402

TPL_DIR = ROOT / "templates"
OUT = ROOT / "output"

order = [
    "模板1_欢迎邮件.md", "模板2_Onboarding检查清单.md", "模板3_首月回访问题清单.md",
    "模板4_QBR会议议程.md", "模板5_价值回顾PPT骨架.md", "模板6_续约邀约邮件.md",
    "模板7_异议处理FAQ.md", "模板8_30天激活任务表.md",
]

parts = ["# 90 天客户成功作战地图 · 模板包（8 份可直接复用）\n"]
for i, name in enumerate(order):
    text = (TPL_DIR / name).read_text(encoding="utf-8")
    if i > 0:
        parts.append('\n\n<div style="page-break-before:always"></div>\n\n')
    parts.append(text)
combined = OUT / "模板包合集.md"
combined.write_text("".join(parts), encoding="utf-8")

render(ROOT / "docs" / "90天客户成功作战地图.md", OUT)
render(combined, OUT)
combined.unlink()  # 中间文件不保留
print("[OK] project2 docs built ->", OUT)
