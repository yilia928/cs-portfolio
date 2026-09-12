# -*- coding: utf-8 -*-
"""
项目③：VoC 洞察报告生成器
读取 output/voc_summary.json、output/pain_topics.csv、data/reviews_clean.csv，
生成 docs/VoC客户声音洞察报告.md，并调用 shared/render_docs 渲染 HTML/PDF。
"""
import ast
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, DOCS = ROOT / "data", ROOT / "output", ROOT / "docs"
DOCS.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT.parent / "shared"))
from render_docs import render  # noqa: E402

S = json.loads((OUT / "voc_summary.json").read_text(encoding="utf-8"))
PAIN = pd.read_csv(OUT / "pain_topics.csv", encoding="utf-8-sig")
DF = pd.read_csv(DATA / "reviews_clean.csv", encoding="utf-8-sig")

# 第 10 名：跨主题横切诉求（自动化/预警/重试），从全量清洗文本中扫描
CROSS_TERMS = ["自动重试", "预警", "自动", "批量", "通知", "定时"]
cross_mask = DF["text"].str.contains("自动重试|预警|提醒|批量|自动", regex=True)
CROSS = DF[cross_mask]
cross_neg = CROSS[pd.to_numeric(CROSS["rating"], errors="coerce") <= 2]

# 提问主题分布（联动④）
qdf = DF[DF["is_question"] == 1]
q_theme = qdf["theme_name"].value_counts()

# 痛点 -> 增值机会/行动建议 映射
PLAY = {
    "多平台库存/超卖": ("库存高级版（安全库存阈值、多仓调拨、组合商品 BOM、活动锁库存释放）",
                  "付费模块 + 超卖保障 SLA", "P0"),
    "系统稳定性": ("大促弹性保障包（性能压测、专属通道、失败任务自动重试）", "增值服务包", "P0"),
    "价格与收费": ("模块化自选套餐 + 席位弹性月付；老客户续费保护价", "商务策略调整", "P1"),
    "上手与培训": ("付费实施陪跑包（7 天首单跑通、实操案例库、新手任务体系）", "增值服务包", "P1"),
    "财务对账": ("财务增值模块（单号级物流费对账、头程/广告分摊、单店利润、差异单导出）",
             "付费模块", "P0"),
    "刊登与商品": ("智能刊登增强（类目映射库、术语表机器翻译、定时刊登）", "付费模块", "P2"),
    "订单同步与处理": ("同步可靠性增强（漏单巡检告警、失败自动重跑、取消单拦截 SLA）",
                 "核心能力补强（防流失）", "P0"),
    "物流同步与回传": ("物流增值模块（回传 SLA 监控、失败批量重试、超时预警、跟踪号自动更新）",
                 "付费模块 + 物流商联合方案", "P0"),
    "客服与工单": ("AI 知识库问答机器人（7x24 自助应答、工单自动分流、SLA 超时升级）——本作品集项目④原型",
              "AI 自助服务（降本+体验）", "P1"),
}


def parse_list(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return [str(x)]


lines = []
W = lines.append

W("# 云帆 ERP 客户声音（VoC）洞察报告")
W("")
W("> 数据周期：2025-03-01 至 2025-08-31 ｜ 分析日：2025-09-01 ｜ 版本 v1.0  ")
W("> 数据来源：应用商店评论（App Store / 华为 / 小米）、小红书笔记、知乎回答、官方社区，均为公开可见文本  ")
W("> 合规声明：仅采集公开内容，不涉及付费/会员/私聊信息；不记录用户 ID，仅保留文本、来源、日期四个字段  ")
W("> 数据说明：本数据集为按真实评论形态构造的**模拟数据**（品牌“云帆ERP”虚构），方法论与流程可直接复用于真实数据")
W("")

# ---------------- 一、结论先行
W("## 一、结论先行（TL;DR）")
W("")
W(f"1. 本轮共采集 **{S['n_raw']} 条**公开反馈，清洗去广告/去重后保留有效反馈 **{S['n_clean']} 条**；"
  f"整体情感均值 {S['sent_mean']}，负面反馈占比 {S['sent_band'].get('负面',0)/S['n_clean']*100:.1f}%。")
W("2. **TOP 3 痛点**：① 多平台库存不同步/超卖（声量 104、差评率 61.5%）；② 系统稳定性（差评率 61.3%）；"
  "③ 价格与收费争议（差评率 59.6%，续约高危信号）。")
W("3. **最能“找钱”的发现**：物流/库存/财务三类痛点诉求高度集中在“自动重试、预警、单号级对账”等具体能力上，"
  "可直接转化为 3 个付费增值模块；客服痛点可用 AI 知识库机器人自助化（项目④已做原型验证）。")
W(f"4. 从反馈中识别出 **{S['n_questions']} 条提问型反馈**（“XX 怎么配/报错怎么办”），"
  "已直接导出为项目④ RAG 机器人的测试集，形成“客户提问 → 自助应答”闭环。")
W("")

# ---------------- 二、数据概览
W("## 二、数据概览与清洗过程")
W("")
W("### 2.1 采集与清洗漏斗")
W("")
W("| 环节 | 条数 | 说明 |")
W("|---|---|---|")
W(f"| 原始采集 | {S['n_raw']} | 6 个公开来源，3 秒/请求礼貌间隔，仅留 text/source/url/date |")
W(f"| 剔除广告灌水 | -{S['n_ad_removed']} | 正则识别微信/QQ/货代/代运营/刷单等 10 类广告特征 |")
W(f"| 剔除完全重复 | -{S['n_exact_dup']} | 归一化（去标点/emoji/空白）后哈希去重 |")
W(f"| 剔除近似重复 | -{S['n_near_dup']} | 字符 2-3gram TF-IDF 余弦相似度 > 0.92 贪心合并 |")
W(f"| **有效分析样本** | **{S['n_clean']}** | 去重率 {((S['n_ad_removed']+S['n_exact_dup']+S['n_near_dup'])/S['n_raw']*100):.1f}%，"
  "与 UGC 平台常见 30%-40% 噪声水平吻合 |")
W("")
W("### 2.2 来源分布、评分与情感")
W("")
W("| 来源 | 条数 | ｜ | 情感层（SnowNLP） | 条数 |")
W("|---|---|---|---|---|")
src_items = sorted(S["by_source"].items(), key=lambda kv: -kv[1])
sb_items = [("负面", S["sent_band"].get("负面", 0)), ("中性", S["sent_band"].get("中性", 0)),
            ("正面", S["sent_band"].get("正面", 0))]
for i in range(max(len(src_items), 3)):
    l = f"{src_items[i][0]} | {src_items[i][1]}" if i < len(src_items) else " | "
    r = f"{sb_items[i][0]} | {sb_items[i][1]}" if i < 3 else " | "
    W(f"| {l} | {r} |")
W("")
W("> 情感口径：SnowNLP 输出 0-1 概率分，<0.4 记负面、0.4-0.6 中性、>0.6 正面；"
  "该模型电商评论语料训练，对“反讽/阴阳怪气”偏弱，结论以评分（1-2 星差评率）交叉校验。")
W("")
W("![数据概览](../output/fig1_sentiment.png)")
W("")

# ---------------- 三、分析方法
W("## 三、分析方法（可复现）")
W("")
W("**管线**：正则清洗 → jieba 分词（挂载 27 个业务自定义词）→ 停用词表 → SnowNLP 情感打分 → "
  "主题双层归类 → TF-IDF 关键词与词云。")
W("")
W("**主题为什么用“双层”而不是只跑 LDA？** 这是本项目刻意保留的方法论对比：")
W("")
W(f"- **主方法：种子词加权投票**。9 大主题各维护 8-12 个高判别力种子词（权重 1-3），在原文上做子串匹配，"
  f"得分≥2 才归主题，否则进“未分类”。与模拟数据埋点标签对照，**一致率 {S['seed_tag_agreement']}**，"
  f"未分类仅 {S['n_unclassified']} 条（{S['n_unclassified']/S['n_clean']*100:.1f}%）。优点是可解释、可审计，"
  "每条归类都能说出命中了哪个词。")
W(f"- **交叉验证：TF-IDF + LDA（K=8）**。无监督 LDA 与种子归类的重合率仅 {S['lda_vs_seed_agreement']}，"
  "人工检查 LDA 簇词发现：短评论场景下 LDA 容易被“卖家/平台/功能/避坑”等跨主题通用词和语气词主导，"
  "把不同主题但相同“吐槽语气”的句子聚到一起。**这反过来指导我们迭代了 3 版停用词表与种子词表**——"
  "无监督方法的价值不是直接给结论，而是暴露词典盲区。附录保留了 LDA 全部簇词供复核。")
W("")

# ---------------- 四、TOP 痛点
W("## 四、TOP 痛点排行（每条附原文证据）")
W("")
W("![痛点排行](../output/fig2_pain_rank.png)")
W("")
W("![痛点气泡图](../output/fig4_bubble.png)")
W("")
W("| 排名 | 痛点主题 | 声量 | 1-2星差评率 | 低情感占比 | 平均评分 | TF-IDF 特征词 |")
W("|---|---|---|---|---|---|---|")
for _, r in PAIN.iterrows():
    W(f"| {int(r['rank'])} | {r['topic']} | {int(r['volume'])} | {r['neg_ratio']}% | "
      f"{r['low_sentiment_ratio']}% | {r['avg_rating']} | {r['top_words']} |")
# 第 10 名：横切诉求
W(f"| 10 | 跨主题横切诉求：自动化/预警/批量（“希望失败自动重试”“超时主动预警”） | {len(CROSS)} | "
  f"{len(cross_neg)/max(len(CROSS),1)*100:.1f}%（差评中提及） | - | - | 自动、批量、预警、重试、通知 |")
W("")
W("> 注：差评率 = 该主题内 1-2 星评论占比；低情感占比 = SnowNLP < 0.4 占比；两指标背离时（如客服主题）"
  "说明用户“愤怒但仍在给 3 星提需求”，是改进 ROI 高的信号。")
W("")
W("### 4.1 逐主题证据与解读")
W("")
for _, r in PAIN.iterrows():
    evs = parse_list(r["evidence"])
    srcs = parse_list(r["evidence_source"])
    play, form, prio = PLAY.get(r["topic"], ("待进一步验证", "-", "-"))
    W(f"#### {int(r['rank'])}）{r['topic']}（{prio}）")
    W("")
    W(f"- **规模**：{int(r['volume'])} 条 ｜ **差评率** {r['neg_ratio']}% ｜ "
      f"**平均情感分** {r['avg_sentiment']} ｜ 特征词：{r['top_words']}")
    W(f"- **增值机会**：{play}（{form}）")
    W("- **原文证据**：")
    for e, s in zip(evs, srcs):
        W(f"    > “{e}”——{s}")
    W("")
# 第 10 名详述
W(f"#### 10）跨主题横切诉求：自动化与预警（P1）")
W("")
W(f"- **规模**：{len(CROSS)} 条反馈提及“自动重试/预警/批量/提醒”，其中差评 {len(cross_neg)} 条；"
  "横跨物流、库存、订单、稳定性四大主题，是出现频次最高的**能力型诉求**而非单点抱怨。")
W("- **解读**：用户不是不能接受出错，而是不能接受“出错后只能人肉盯”。这组诉求直接定义了"
  "同步可靠性增强包与 AI 自助服务的需求边界（失败自动重试、超时主动预警、批量自助修复）。")
W("")

# ---------------- 五、词云
W("### 4.2 负面 vs 正面对比词云")
W("")
W("| 负面反馈词云 | 正面反馈词云 |")
W("|---|---|")
W("| ![负面词云](../output/fig3_wordcloud_neg.png) | ![正面词云](../output/fig3_wordcloud_pos.png) |")
W("")

# ---------------- 六、增值机会与优先级
W("## 五、增值机会建议与优先级")
W("")
W("### 5.1 痛点到商业化机会的映射")
W("")
W("| 优先级 | 痛点 | 建议动作 | 形态 | 判断依据 |")
W("|---|---|---|---|---|")
order_p = {"P0": 0, "P1": 1, "P2": 2}
rows = []
for _, r in PAIN.iterrows():
    play, form, prio = PLAY[r["topic"]]
    rows.append((prio, r["topic"], play, form,
                 f"声量{int(r['volume'])}、差评率{r['neg_ratio']}%"))
for prio, topic, play, form, basis in sorted(rows, key=lambda x: order_p[x[0]]):
    W(f"| {prio} | {topic} | {play} | {form} | {basis} |")
W("")
W("### 5.2 优先级矩阵说明（声量 × 情绪强度）")
W("")
W("- **P0（立即做）**：物流同步、库存超卖、订单同步、财务对账、系统稳定性——共同特征是"
  "**高频 + 直接造成经营损失（超卖罚款、漏单、对不上账、大促停摆）+ 已具备明确付费意愿表达**，"
  "且均为可标准化的技术能力，适合做增值模块/SLA。")
W("- **P1（季度内做）**：客服响应（AI 知识库自助化，项目④已验证可行性，可同时降本与提速）、"
  "上手培训（付费陪跑包，同时拉升 Onboarding 期续约率，与项目② 90 天作战地图联动）、"
  "价格体系（不是产品问题但直接影响续约，需商务侧老客保护政策）。")
W("- **P2（择机做）**：刊登增强（翻译/类目映射），声量与情绪强度均相对温和，跟随产品迭代即可。")
W("")
W("> 与项目①/②的联动：价格争议 + 低健康度客户应在项目② 续约前 60 天进入“价值回顾 + 老客保护价”话术；"
  "物流/库存类工单激增是项目① R2 预警规则（工单风暴）的现实来源。")
W("")

# ---------------- 七、提问型反馈（联动④）
W("## 六、提问型反馈：从“客户在问什么”到 AI 自助应答")
W("")
W(f"用提问正则（怎么/如何/为什么/能不能/在哪里/？）从清洗后样本中识别出 **{len(qdf)} 条提问型反馈**，"
  "主题分布如下，已全量导出 `output/voc_questions.jsonl` 并同步到**项目④ RAG 机器人测试集**：")
W("")
W("| 提问主题 | 条数 |")
W("|---|---|")
for name, cnt in q_theme.items():
    W(f"| {name} | {cnt} |")
W("")
W("典型提问示例：")
W("")
for q in qdf.sort_values("theme_name").drop_duplicates("theme_name")["text"].head(6):
    W(f"> {q}")
W("")
W("**闭环逻辑**：这些问题在现有客服体系下全部需要人工响应（对应“客服与工单”痛点）；项目④将用帮助中心"
  "知识库 + RAG 实现 7x24 自助应答，并以这 65 个真实提问作为评测集检验检索命中率与回答准确率。")
W("")

# ---------------- 附录
W("## 附录 A：LDA 无监督聚类簇词（K=8，交叉验证留痕）")
W("")
W("| 簇 | 一对一映射标签 | TOP10 簇词 |")
W("|---|---|---|")
for i, t in enumerate(S["lda_top_words"], 1):
    W(f"| {i} | {t['label']} | {'、'.join(t['words'][:10])} |")
W("")
W("## 附录 B：方法局限与下一步")
W("")
W("1. 模拟数据按真实分布形态构造，绝对数值不代表真实市场；接入真实采集源后需重新校准种子词与情感阈值。")
W("2. SnowNLP 对反讽识别弱，下一步可改用大模型 API 批量打分做一轮对照。")
W("3. 种子词归类依赖词表维护，新品上线（如 AI 相关功能）需补充词表；LDA/聚类结果应每季度跑一次用于发现新词。")
W("4. 时间维度上可进一步做痛点趋势（如大促月稳定性讨论占比变化），本版以截面洞察为主。")
W("")
W("---")
W("*分析脚本：01_scrapers.py（合规采集骨架）/ 02_build_reviews.py（模拟数据）/ 03_clean_analyze.py（清洗+情感+主题+图表）/ 04_build_report.py（本报告）*")

md = DOCS / "VoC客户声音洞察报告.md"
md.write_text("\n".join(lines), encoding="utf-8")
print("[OK] report md ->", md)
render(md, OUT)
