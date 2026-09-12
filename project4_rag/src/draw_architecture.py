# -*- coding: utf-8 -*-
"""绘制 RAG 架构图 -> docs/rag_architecture.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(12, 7.2), dpi=150)
ax.set_xlim(0, 12); ax.set_ylim(0, 7.2); ax.axis("off")

C_BLUE, C_GREEN, C_ORANGE, C_GRAY, C_RED = "#2e5bff", "#0ea466", "#f59e0b", "#64748b", "#ef4444"


def box(x, y, w, h, text, fc, tc="white", fs=10.5, lw=0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                       fc=fc, ec="white", lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
            fontsize=fs, zorder=3, linespacing=1.5)


def arrow(x1, y1, x2, y2, color=C_GRAY, style="-|>", ls="-", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=15,
                                 color=color, lw=lw, linestyle=ls, zorder=1))


ax.text(6, 6.95, "云帆ERP 智能客服 · RAG 架构（检索增强生成）", ha="center", fontsize=15, weight="bold", color="#0f172a")

# 离线建库链路（上排）
box(0.3, 5.55, 2.1, 0.85, "帮助中心文档\n15 模块 · 公开资料", C_GRAY, fs=10)
box(2.9, 5.55, 2.1, 0.85, "切块与元数据\n202 块 · 200-500字\n标题/模块/URL", C_BLUE, fs=9.5)
box(5.5, 5.55, 2.4, 0.85, "索引构建\n词级TF-IDF + 字符ngram\n(可切换embedding)", C_BLUE, fs=9.5)
box(8.4, 5.55, 3.3, 0.85, "知识索引\nkb_chunks.jsonl\n（本地可复现）", C_GREEN, fs=10)
arrow(2.4, 5.97, 2.9, 5.97); arrow(5.0, 5.97, 5.5, 5.97); arrow(7.9, 5.97, 8.4, 5.97)
ax.text(1.35, 6.52, "离线建库", fontsize=10, color=C_GRAY)

# 在线问答链路（下排）
box(0.3, 3.55, 1.9, 0.85, "用户提问\nWeb / CLI", C_GRAY, fs=10)
box(2.65, 3.55, 2.0, 0.85, "问题改写\n短追问拼接\n多轮上下文", C_ORANGE, tc="#1f2937", fs=9.5)
box(5.1, 3.55, 2.1, 0.85, "混合检索 Top-5\n词0.6 + 字0.4\n标题命中加权", C_BLUE, fs=9.5)
box(7.65, 3.55, 1.95, 0.85, "置信度判断\n阈值 0.11", C_ORANGE, tc="#1f2937", fs=10)
arrow(2.2, 3.97, 2.65, 3.97); arrow(4.65, 3.97, 5.1, 3.97); arrow(7.2, 3.97, 7.65, 3.97)
arrow(9.55, 5.55, 6.15, 4.42, color=C_GREEN, ls="--")
ax.text(8.15, 5.1, "召回", fontsize=9.5, color=C_GREEN)

# 置信度分支
box(7.65, 1.75, 1.95, 0.8, "Prompt 拼接\n资料+历史+指令", C_BLUE, fs=9.5)
box(4.55, 1.75, 2.35, 0.8, "生成回答\nLLM（有Key）\n抽取式（离线降级）", C_BLUE, fs=9.5)
box(0.9, 1.75, 2.6, 0.8, "带来源引用的回答\n[1][2] 编号 + URL", C_GREEN, fs=10)
box(9.9, 1.75, 1.85, 0.8, "兜底转人工\n工单/在线客服", C_RED, fs=9.5)
arrow(8.05, 3.55, 5.75, 2.57, color=C_GRAY)   # 高置信 -> prompt
ax.text(7.3, 3.12, "高置信", fontsize=9, color=C_GRAY)
arrow(9.25, 3.55, 10.6, 2.57, color=C_RED)     # 低置信 -> handoff
ax.text(10.05, 3.12, "低置信", fontsize=9, color=C_RED)
arrow(7.65, 2.15, 6.9, 2.15); arrow(4.55, 2.15, 3.5, 2.15)

# 旁路
box(0.3, 0.45, 3.6, 0.78, "多轮记忆：保留最近两轮问答，短追问自动改写", C_ORANGE, tc="#1f2937", fs=9.5)
box(4.35, 0.45, 3.6, 0.78, "追问建议：根据命中块的相邻小节标题生成", C_ORANGE, tc="#1f2937", fs=9.5)
box(8.35, 0.45, 3.4, 0.78, "评估闭环：100题测试集 Recall/MRR/引用/兜底", C_GRAY, fs=9.5)
arrow(2.1, 1.23, 3.6, 1.73, color=C_ORANGE, style="-|>", lw=1.2)
arrow(6.15, 1.23, 2.2, 1.73, color=C_ORANGE, style="-|>", lw=1.2)
arrow(10.05, 1.23, 10.05, 1.73, color=C_GRAY, style="-|>", lw=1.2)

plt.tight_layout()
out = "docs/rag_architecture.png"
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("[OK] ->", out)
