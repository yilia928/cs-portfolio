# -*- coding: utf-8 -*-
"""
项目③ W4-W5：清洗 / 分词 / 情感打分(SnowNLP) / TF-IDF+LDA 主题聚类 / 图表 / 测试集导出
输入：data/reviews_raw.csv
输出：
  data/reviews_clean.csv            清洗后明细（含情感分、LDA主题、挖掘标签）
  output/pain_topics.csv            TOP 痛点主题（含原文证据）
  output/voc_questions.jsonl        提问型反馈 -> 项目④ RAG 测试集
  output/voc_summary.json           汇总指标（供报告生成器读取）
  output/*.png                      5 张分析图
"""
import json
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import jieba
from snownlp import SnowNLP
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "output"
OUT.mkdir(exist_ok=True)
P4_TESTS = ROOT.parent / "project4_rag" / "tests"
P4_TESTS.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------ 词典与停用词
USER_WORDS = ["跨境电商", "物流单号", "单号回传", "回传失败", "库存同步", "多平台库存", "超卖", "安全库存",
              "面单", "打单", "发货", "站内信", "店铺", "平台", "对账", "刊登", "采购单", "海外仓",
              "FBA", "Shopee", "虾皮", "TikTok", "Temu", "亚马逊", "速卖通", "eBay", "客服", "工单",
              "卡顿", "闪退", "大促", "漏单", "合单", "拆单", "头程", "运费", "利润报表", "批量改价",
              "库存预警", "物流轨迹", "订单同步", "自动审单", "角色权限", "套餐", "年费", "席位"]
for w in USER_WORDS:
    jieba.add_word(w)

STOP = set("""的 了 和 是 我 你 他 她 它 们 也 都 就 还 在 有 没 不 很 太 真的 一个 一下 这个 那个 什么 怎么
为什么 如何 可以 能 会 要 让 给 把 被 跟 与 及 等 着 过 吗 呢 吧 啊 呀 哦 嗯 哈哈 然后 现在 已经 还是
就是 不是 觉得 自己 他们 我们 你们 这 那 里 个 多 少 上 下 中 后 时 时候 东西 样子 问题 次 天
有人 大家 请问 谢谢 求助 吐槽 客观 一句 真的 而且 但是 感觉 比较 非常 特别 一直 经常 完全 根本
有点 有些 某个 这种 那种 之类 目前 最近 之前 以后 出来 下去 进去 起来 应该 可能 知道 看到 发现
卖家 买家 老板 平台 店铺 系统 功能 使用 用 时候 手动 自动 小时 大促 旺季 之后 只能 确实 建议
优化 整体 继续 加油 满意 小毛病 迭代 实测 用户 公道话 有用 点个赞 注意 避坑 同款 情况 研究 手机
当天 不敢 放量 成功 专业 团队 季度 消失 规则 配置 两天 半夜 每月 损失 不少 直接 活动 扣分 扣到
钱 一个月 一两 半天 晚上 中午 凌晨 下午 早上 今天 昨天 每天 当月 月 个 次 单 笔 句 下""".split())

AD_PAT = re.compile(r"(?:微信|vx|VX|Vx|加我|QQ|Q群|q群|联系电话|电话见|私聊|代运营|货代|刷单|测评资源|返现渠道|回收店铺|招跨境)")
URL_PAT = re.compile(r"https?://\S+|www\.\S+")
NOISE_PAT = re.compile(r"[\s\u3000]+")


def clean_text(t: str) -> str:
    t = URL_PAT.sub(" ", t)
    t = t.replace("…", " ").replace("　", " ")
    t = re.sub(r"[#＃][^\s#＃]+", " ", t)  # 话题标签
    return NOISE_PAT.sub(" ", t).strip()


def tokens(t: str):
    toks = jieba.lcut(t)
    return [x for x in toks if len(x) > 1 and x not in STOP and not re.fullmatch(r"[\W_\d]+", x)]


def main():
    raw = pd.read_csv(DATA / "reviews_raw.csv", encoding="utf-8-sig")
    n_raw = len(raw)
    df = raw.copy()
    df["text"] = df["text"].astype(str).map(clean_text)

    # 1) 去广告
    ad_mask = df["text"].str.contains(AD_PAT, regex=True) | (df["theme"] == "ad")
    n_ad = int(ad_mask.sum())
    df = df[~ad_mask].copy()

    # 2) 完全去重（标准化文本哈希）
    df["norm"] = df["text"].str.replace(r"[\s，。！？、,.!?…\.]+", "", regex=True)
    dup_mask = df.duplicated("norm")
    n_exact = int(dup_mask.sum())
    df = df[~dup_mask].copy()

    # 3) 近似去重（字符 2-gram TF-IDF 余弦 > 0.88 贪心保留）
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2)
    X = vec.fit_transform(df["text"])
    keep = np.ones(len(df), dtype=bool)
    idx = df.index.to_numpy()
    for i in range(len(df)):
        if not keep[i]:
            continue
        lo = i + 1
        if lo >= len(df):
            break
        sims = cosine_similarity(X[i], X[lo:])[0]
        for j in np.where(sims > 0.92)[0]:
            keep[lo + j] = False
    n_near = int((~keep).sum())
    df = df[keep].reset_index(drop=True)

    # 4) 分词
    df["tokens"] = df["text"].map(tokens)
    df["token_text"] = df["tokens"].map(lambda xs: " ".join(xs))
    df = df[df["tokens"].map(len) >= 2].reset_index(drop=True)

    # 5) 情感打分
    def senti(t):
        try:
            return round(float(SnowNLP(t).sentiments), 3)
        except Exception:
            return 0.5
    df["sentiment"] = df["text"].map(senti)
    df["sent_band"] = pd.cut(df["sentiment"], [-0.01, 0.4, 0.6, 1.01], labels=["负面", "中性", "正面"])

    # 6) LDA 无监督主题聚类（8 个主题，作为交叉验证手段）
    cv = CountVectorizer(max_df=0.8, min_df=5, max_features=1000)
    Xc = cv.fit_transform(df["token_text"])
    K = 8
    lda = LatentDirichletAllocation(n_components=K, random_state=42, max_iter=30, learning_method="batch")
    doc_topic = lda.fit_transform(Xc)
    terms = np.array(cv.get_feature_names_out())
    top_words = []
    for k in range(K):
        top = terms[np.argsort(-lda.components_[k])[:10]]
        top_words.append(top.tolist())

    # 主题种子词 -> 自动命名
    SEEDS = {"物流同步与回传": ["物流", "单号", "回传", "轨迹", "发货", "面单", "物流商"],
             "多平台库存/超卖": ["库存", "超卖", "安全库存", "可售", "盘点", "在途", "扣减"],
             "订单同步与处理": ["订单", "同步", "漏单", "合单", "拆单", "审单", "延迟"],
             "系统稳定性": ["卡顿", "崩", "闪退", "卡死", "转圈", "加载", "超时", "稳定"],
             "客服与工单": ["客服", "工单", "响应", "排队", "没人理"],
             "价格与收费": ["涨价", "收费", "价格", "席位", "套餐", "年费", "贵", "费用"],
             "上手与培训": ["上手", "新手", "培训", "引导", "复杂", "文档", "教程"],
             "财务对账": ["对账", "利润", "财务", "运费", "头程", "账单", "扣费"],
             "刊登与商品": ["刊登", "改价", "listing", "Listing", "类目", "图片", "铺货"]}
    def topic_seed_scores(words):
        scores = {}
        for name, seeds in SEEDS.items():
            scores[name] = sum(1 for w in words for s in seeds
                               if s.lower() in w.lower() or w.lower() in s.lower())
        return scores
    # 一对一贪心分配：按"最佳匹配-次佳匹配"优势度排序，避免多个簇抢同一标签
    score_mat = [topic_seed_scores(w) for w in top_words]
    names = list(SEEDS.keys())
    order = sorted(range(K), key=lambda k: -(max(score_mat[k].values())
                 - sorted(score_mat[k].values(), reverse=True)[1]), reverse=False)
    assigned, used = {}, set()
    for k in sorted(order, key=lambda k: -max(score_mat[k].values())):
        for name in sorted(names, key=lambda nm: -score_mat[k][nm]):
            if name not in used:
                assigned[k] = name if score_mat[k][name] > 0 else "其他使用体验"
                used.add(name)
                break
        else:
            assigned[k] = "其他使用体验"
    topic_labels = [assigned[k] for k in range(K)]
    df["lda_topic"] = [topic_labels[x] for x in doc_topic.argmax(1)]

    # 7) 可解释主题归类：种子词加权投票（主方法，短文本上比纯 LDA 更稳更可解释）
    LEX = {
        "物流同步与回传": {"物流": 3, "单号": 3, "回传": 3, "轨迹": 3, "面单": 3, "跟踪号": 3,
                      "补传": 2, "打单": 2, "物流商": 2, "揽收": 2, "云途": 1, "燕文": 1, "极兔": 1},
        "多平台库存/超卖": {"库存": 3, "超卖": 3, "卖超": 3, "安全库存": 3, "断货": 3, "可售": 2, "盘点": 2,
                      "在途": 2, "调拨": 2, "组合商品": 2, "子品": 2, "缺货": 2, "FBA": 2},
        "订单同步与处理": {"漏单": 3, "订单": 3, "合单": 2, "拆单": 2, "审单": 2, "异常单": 2,
                      "取消": 1, "备注": 1},
        "系统稳定性": {"卡顿": 3, "崩": 3, "闪退": 3, "卡死": 3, "转圈": 2, "加载": 2,
                  "超时": 2, "宕机": 2, "服务器": 2, "bug": 2, "Bug": 2, "失灵": 2,
                  "错位": 2, "重跑": 1, "任务失败": 2},
        "客服与工单": {"客服": 3, "工单": 3, "响应": 2, "排队": 2, "回电": 2, "踢皮球": 2, "技术支持": 1},
        "价格与收费": {"涨价": 3, "收费": 3, "价格": 2, "席位": 2, "套餐": 2, "年费": 3,
                  "贵": 2, "费用": 2, "月付": 2, "年付": 2, "扣费": 1},
        "上手与培训": {"上手": 3, "新手": 3, "培训": 3, "劝退": 2, "陪跑": 2, "引导": 2, "复杂": 2,
                  "文档": 2, "教程": 2, "帮助中心": 2, "实操案例": 1, "授权": 2},
        "财务对账": {"对账": 3, "利润": 3, "财务": 3, "头程": 2, "账单": 2, "会计": 2,
                "月结": 2, "报表": 2, "广告费": 2, "分摊": 2},
        "刊登与商品": {"刊登": 3, "改价": 2, "类目": 2, "铺货": 2, "采集": 2, "翻译": 2,
                  "跟卖": 2, "差评提醒": 1, "详情页": 2, "listing": 3, "图片": 1},
    }
    POS_LEX = {"推荐": 2, "好评": 2, "回本": 2, "省心": 2, "靠谱": 2, "福音": 2, "安家": 2}
    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")

    def vote_label(toks, tag, rating):
        # 在原文上做子串匹配，避免 jieba 切分歧义（如“物流轨迹”整词）导致漏判
        raw = df["text"][vote_label.i] if hasattr(vote_label, "i") else ""
        low = raw.lower()
        scores = {lab: sum(w for t, w in lex.items() if (t.lower() in low if t.isascii() else t in raw))
                  for lab, lex in LEX.items()}
        lab, sc = max(scores.items(), key=lambda kv: kv[1])
        pos = sum(w for t, w in POS_LEX.items() if t in raw)
        if sc >= 2:
            return lab
        if tag == "praise" or pos >= 2 or (pd.notna(rating) and rating >= 5 and sc == 0):
            return "正向评价"
        return "未分类"

    labels_voted = []
    for i, (toks, tag, rating) in enumerate(zip(df["tokens"], df["theme"], df["rating_num"])):
        vote_label.i = i
        labels_voted.append(vote_label(toks, tag, rating))
    df["mined_theme"] = labels_voted

    # 各类 TF-IDF 高频词（用于报告解释主题）
    tv = TfidfVectorizer(max_df=0.9, min_df=3)
    Tv = tv.fit_transform(df["token_text"])
    tv_terms = np.array(tv.get_feature_names_out())

    def theme_keywords(sub_df, topn=8):
        cent = Tv[sub_df.index].mean(axis=0).A1
        return tv_terms[np.argsort(-cent)[:topn]].tolist()

    LAB2TAG = {"物流同步与回传": "logistics", "多平台库存/超卖": "inventory", "订单同步与处理": "order_sync",
               "系统稳定性": "stability", "客服与工单": "support", "价格与收费": "pricing",
               "上手与培训": "onboarding", "财务对账": "finance", "刊登与商品": "listing"}
    topics = []
    for lab in list(LEX.keys()):
        sub = df[df["mined_theme"] == lab]
        if len(sub) < 8:
            continue
        ev_all = sub[(sub["rating_num"] <= 2) & (sub["is_question"] == 0)]
        ev = ev_all[ev_all["theme"] == LAB2TAG[lab]]
        if len(ev) < 2:  # 兜底：双标签一致的证据不足时再放宽
            ev = ev_all
        ev = ev.sort_values("sentiment")
        topics.append(dict(
            topic=lab, top_words="、".join(theme_keywords(sub)),
            volume=int(len(sub)),
            neg_ratio=round(float((sub["rating_num"] <= 2).mean()) * 100, 1),
            low_sentiment_ratio=round(float((sub["sentiment"] < 0.4).mean()) * 100, 1),
            avg_rating=round(float(sub["rating_num"].mean()), 2),
            avg_sentiment=round(float(sub["sentiment"].mean()), 3),
            evidence=ev["text"].head(2).tolist(),
            evidence_source=ev["source"].head(2).tolist()))
    topics.sort(key=lambda r: (-r["neg_ratio"], -r["volume"]))
    pain = pd.DataFrame(topics)
    pain.insert(0, "rank", range(1, len(pain) + 1))
    pain.to_csv(OUT / "pain_topics.csv", index=False, encoding="utf-8-sig")

    # LDA 与种子词归类的一致性（无监督验证）
    cmp_df = df[df["mined_theme"].isin(LEX.keys())]
    lda_agree = round(float((cmp_df["lda_topic"] == cmp_df["mined_theme"]).mean()), 3)

    # 8) 提问型反馈 -> 项目④测试集
    q_pat = re.compile(r"[?？]|怎么|如何|为什么|能不能|在哪里|求教|有人知道|请问")
    qmask = df["text"].str.contains(q_pat, regex=True)
    qs = df[qmask].copy()
    with (OUT / "voc_questions.jsonl").open("w", encoding="utf-8") as f, \
         (P4_TESTS / "voc_questions.jsonl").open("w", encoding="utf-8") as f2:
        for _, r in qs.iterrows():
            rec = dict(qid=r["review_id"], question=r["text"].lstrip("，。！？ "),
                       theme=r["theme_name"], source=r["source"])
            line = json.dumps(rec, ensure_ascii=False)
            f.write(line + "\n"); f2.write(line + "\n")

    # 9) 保存明细
    df_out = df.drop(columns=["norm"])
    df_out.to_csv(DATA / "reviews_clean.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------- 校验
    # 种子词归类 vs 埋点标签（模拟数据的“真值”代理，用于验证归类质量）
    tag_map = {"logistics": "物流同步与回传", "inventory": "多平台库存/超卖", "order_sync": "订单同步与处理",
               "stability": "系统稳定性", "support": "客服与工单", "pricing": "价格与收费",
               "onboarding": "上手与培训", "finance": "财务对账", "listing": "刊登与商品"}
    tagged = df[df["theme"].isin(tag_map)]
    seed_agree = round(float(np.mean([tag_map[t] == m for t, m in zip(tagged["theme"], tagged["mined_theme"])])), 3)
    n_unclassified = int((df["mined_theme"] == "未分类").sum())

    summary = dict(
        n_raw=n_raw, n_ad_removed=n_ad, n_exact_dup=n_exact, n_near_dup=n_near,
        n_clean=len(df), n_questions=int(len(qs)),
        rating_dist={str(k): int(v) for k, v in df["rating_num"].value_counts().sort_index().items()},
        sent_band={k: int(v) for k, v in df["sent_band"].value_counts().items()},
        by_source={k: int(v) for k, v in df["source"].value_counts().items()},
        sent_mean=round(float(df["sentiment"].mean()), 3),
        seed_tag_agreement=seed_agree,
        lda_vs_seed_agreement=lda_agree,
        n_unclassified=n_unclassified,
        topics=topics,
        lda_top_words=[dict(label=topic_labels[k], words=top_words[k]) for k in range(K)],
    )
    json.dump(summary, open(OUT / "voc_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # --------------------------------------------------------- 图表
    # 图1 评分分布 + 情感分层
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rc = df["rating_num"].value_counts().sort_index()
    axes[0].bar(rc.index.astype(str), rc.values, color=["#e53935", "#fb8c00", "#fdd835", "#9ccc65", "#43a047"])
    for i, v in enumerate(rc.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontsize=10)
    axes[0].set_title("评分分布（1-5 星）"); axes[0].set_ylabel("条数")
    sb = df["sent_band"].value_counts().reindex(["负面", "中性", "正面"])
    axes[1].pie(sb.values, labels=[f"{k} {v}条" for k, v in sb.items()], autopct="%1.1f%%",
                colors=["#e53935", "#b0bec5", "#43a047"], startangle=90)
    axes[1].set_title("SnowNLP 情感分层")
    plt.tight_layout(); plt.savefig(OUT / "fig1_sentiment.png", dpi=150); plt.close()

    # 图2 TOP 痛点条形（负面占比）
    fig, ax = plt.subplots(figsize=(10, 4.8))
    p2 = pain.sort_values("volume")
    colors = ["#e53935" if x >= 60 else "#fb8c00" if x >= 40 else "#fbc02d" for x in p2["neg_ratio"]]
    bars = ax.barh(p2["topic"], p2["volume"], color=colors)
    for b, nr, vol in zip(bars, p2["neg_ratio"], p2["volume"]):
        ax.text(b.get_width() + 1.5, b.get_y() + b.get_height()/2, f"{vol}条 / 差评率{nr}%", va="center", fontsize=10)
    ax.set_title("痛点主题排行（颜色=差评率：红>=60%，橙>=40%，黄其他）")
    ax.set_xlim(0, p2["volume"].max() * 1.28)
    plt.tight_layout(); plt.savefig(OUT / "fig2_pain_rank.png", dpi=150); plt.close()

    # 图3 词云（负面 vs 正面）
    neg_words = " ".join(df[df["sentiment"] < 0.4]["token_text"])
    pos_words = " ".join(df[df["sentiment"] > 0.6]["token_text"])
    font = r"C:\Windows\Fonts\msyh.ttc"
    for name, words, cmap in [("fig3_wordcloud_neg.png", neg_words, "Reds"),
                              ("fig3_wordcloud_pos.png", pos_words, "Greens")]:
        wc = WordCloud(font_path=font, width=900, height=500, background_color="white",
                       max_words=120, colormap=cmap).generate(words if words.strip() else "暂无")
        wc.to_file(OUT / name)

    # 图4 主题气泡（x 平均情感, y 量级, 大小=量）
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    label_offset = {"刊登与商品": (-78, 16), "财务对账": (-84, -20), "系统稳定性": (46, -6),
                    "客服与工单": (0, 12), "上手与培训": (0, 12)}
    for t in topics:
        x, y = t["avg_sentiment"], t["volume"]
        c = "#e53935" if t["neg_ratio"] >= 60 else "#fb8c00" if t["neg_ratio"] >= 40 else "#43a047"
        ax.scatter(x, y, s=max(t["volume"] * 9, 120), color=c, alpha=0.45, edgecolors="white", zorder=3)
        dx, dy = label_offset.get(t["topic"], (0, 12))
        ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
        ax.annotate(t["topic"], (x, y), fontsize=9.5, ha=ha, va="center",
                    xytext=(dx, dy), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.8) if dx else None,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=c, alpha=0.9))
    ax.axvline(0.4, ls="--", c="#e53935", lw=1); ax.axvline(0.6, ls="--", c="#43a047", lw=1)
    ax.set_xlabel("平均情感分（左低右高）"); ax.set_ylabel("提及量")
    ax.set_title("痛点气泡图：越靠左越痛，越靠上声量越大（红=差评率>=60%，橙>=40%）")
    ax.set_xlim(-0.03, 0.55); ax.set_ylim(0, max(t["volume"] for t in topics) * 1.25)
    plt.tight_layout(); plt.savefig(OUT / "fig4_bubble.png", dpi=150); plt.close()

    print(f"[OK] raw {n_raw} -> clean {len(df)} | ads {n_ad} exactDup {n_exact} nearDup {n_near}")
    print(f"[OK] questions for RAG: {len(qs)} | seed agreement: {summary['seed_tag_agreement']} "
          f"| LDA-vs-seed: {summary['lda_vs_seed_agreement']} | unclassified: {summary['n_unclassified']}")
    print(pain[["rank", "topic", "volume", "neg_ratio", "avg_sentiment"]].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
