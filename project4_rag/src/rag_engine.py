# -*- coding: utf-8 -*-
"""
项目④：RAG 引擎核心
====================
检索：jieba 词级 TF-IDF + 字符 2-3gram TF-IDF 混合余弦（离线可用，默认）；
      配置环境变量后可切换 OpenAI 兼容 embedding 接口（豆包/DeepSeek/通义等）。
生成：有 LLM_API_KEY 走大模型生成（带来源引用）；无 key 走抽取式模板回答（现场可 demo）。
能力：多轮记忆（追问改写）、追问建议、低置信度兜底转人工。
"""
import json
import os
import re
from pathlib import Path

import jieba
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DOMAIN_WORDS = ["跟踪号", "面单", "物流商", "云途", "燕文", "极兔", "令牌", "授权", "SKU", "SPU", "FBA",
                "安全库存", "可售", "可用库存", "在途", "调拨", "盘点", "超卖", "组合商品", "BOM",
                "审单", "合单", "拆单", "异常单", "回传", "批量补传", "跟踪号", "揽收", "轨迹",
                "对账", "利润", "头程", "广告费", "结算单", "刊登", "类目", "跟卖", "采集",
                "预占", "断货", "库龄", "PDA", "库位", "移动加权", "工单", "席位", "套餐", "年费",
                "亚马逊", "Shopee", "TikTok", "Temu", "速卖通", "eBay", "Wish", "Lazada",
                "二次验证", "IP白名单", "角色权限", "打印控件", "错误码", "健康度"]
for w in DOMAIN_WORDS:
    jieba.add_word(w)

STOP = set("""的 了 和 是 我 你 他 她 它 们 也 都 就 还 在 有 没 不 很 太 一个 一下 这个 那个 什么 怎么
为什么 如何 吗 呢 吧 啊 呀 请问 有人 知道 求教 能不能 可以 能 会 要 让 给 把 被 跟 吗呢
些 样 时候 哪里 哪个 怎样 如何 为什么 还是 或者 以及 而且 但是 因为 所以 如果 就是 不是
怎么办 该 咋 如何是好""".split())

CONFIDENCE_THRESHOLD = 0.11  # hybrid 分数低于该值 -> 转人工；依据评估网格：域内最低≈0.094、越界最高≈0.092
EXTRACT_COVER_THRESHOLD = 0.25  # 抽取式答案句子入选门槛（词覆盖率）；可按新知识库效果回归调整，见 tests/eval_rag.py
EXPANSION_WEIGHT = 0.5  # 查询扩展词在词级 TF-IDF 查询向量中的权重（低于原词，避免带偏召回）

# ---------------------------------------------------------------- 查询扩展
# 客服口语 -> 帮助文档术语。词条归纳自项目③ VoC 真实提问的高频口语模式，
# 在【原文】上做子串匹配（不依赖 jieba 切词），扩展词以 EXPANSION_WEIGHT 低权重
# 只注入词级 TF-IDF 检索与抽取覆盖计算，不改变字符 ngram 一路。
SYNONYM_PHRASES = {
    # 物流回传类（③痛点：物流单号同步慢/回传失败）
    "不回传": ["回传失败", "批量补传", "预回传", "已发货", "平台后台待发货"],
    "没回传": ["回传失败", "批量补传"],
    "回传不了": ["回传失败", "批量补传"],
    "回传失败": ["批量补传", "重试队列"],
    "发货状态没更新": ["回传失败", "已发货"],
    "单号没更新": ["回传失败", "跟踪号"],
    # 订单同步/漏单类
    "不同步": ["同步失败", "同步延迟"],
    "同步不了": ["同步失败", "手动拉单"],
    "同步失败": ["重试", "同步日志"],
    "漏单": ["手动拉单", "同步日志"],
    "没有订单": ["手动拉单"],
    "看不到订单": ["手动拉单"],
    "没同步": ["手动拉单", "同步延迟"],
    # 库存/超卖类
    "卖超": ["超卖", "零库存", "自动下架"],
    "超卖": ["可售库存", "预占"],
    "库存不对": ["盘点", "库存同步"],
    "库存对不上": ["盘点", "库存同步"],
    # 稳定性/性能类
    "卡顿": ["超时", "异步"],
    "打不开": ["503", "故障"],
    "一直转圈": ["超时", "503"],
    "加载不出来": ["超时", "异步"],
    "批量导出": ["异步导出", "任务中心"],
    # 打印/面单类
    "打不出单": ["打印控件", "面单"],
    "打印偏移": ["偏移", "校准"],
    "打印机连不上": ["打印控件", "端口"],
    # 授权类
    "授权失败": ["令牌", "Token", "Cookie"],
    "授权不了": ["令牌", "重新授权"],
    "连不上店铺": ["授权", "令牌过期"],
    "店铺掉线": ["令牌过期", "重新授权"],
    # 财务/对账类
    "对不上账": ["对账", "结算报告", "口径"],
    "算利润": ["利润报表", "头程", "分摊"],
    "成本怎么算": ["头程", "分摊", "移动加权"],
    # 套餐/计费类
    "加席位": ["临时席位", "套餐"],
    "多少钱": ["套餐", "计费", "价格"],
    "怎么收费": ["套餐", "计费"],
    "续费多少": ["套餐", "年费"],
    "超了套餐": ["套餐", "超额", "计费", "额度"],
    "超出额度": ["套餐", "超额", "额度"],
    "订单量超": ["套餐", "超额", "计费"],
    # 新手入门类
    "新手": ["新用户", "新手任务", "入门"],
    "第一次用": ["新用户", "新手任务", "入门"],
    "刚开始用": ["新用户", "新手任务", "入门"],
}
for _w in {w for vs in SYNONYM_PHRASES.values() for w in vs}:
    jieba.add_word(_w)


def expand_query(raw_query):
    """返回该问题命中的文档术语扩展词列表（原文子串匹配）。"""
    hits = []
    for phrase, terms in SYNONYM_PHRASES.items():
        if phrase in raw_query:
            for t in terms:
                if t not in hits:
                    hits.append(t)
    return hits


def tokenize(text):
    return [t for t in jieba.lcut(text) if t.strip() and t not in STOP and not re.match(r"^[\W_]+$", t)]


def load_kb():
    rows = [json.loads(line) for line in (DATA / "kb_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    return rows


# ---------------------------------------------------------------- 检索
class HybridRetriever:
    """词级 TF-IDF（默认0.6，含同义词扩展低权重向量）+ 字符 ngram（0.4）+ 标题命中加权。"""

    def __init__(self, docs, word_weight=0.6, expand=True, expansion_weight=EXPANSION_WEIGHT):
        self.docs = docs
        self.word_weight = word_weight
        self.expand = expand
        self.expansion_weight = expansion_weight
        corpus = [" ".join(tokenize(d["text"])) for d in docs]
        self.tf_word = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 2))
        self.Mw = self.tf_word.fit_transform(corpus)
        self.tf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
        self.Mc = self.tf_char.fit_transform([d["text"] for d in docs])

    def vectorize_query(self, q):
        qt = tokenize(q)
        qw = self.tf_word.transform([" ".join(qt)])
        if self.expand:
            exp = expand_query(q)
            if exp:
                # 扩展向量低权重叠加：让含文档术语的块上浮，但不压过用户原词
                qw = qw + self.expansion_weight * self.tf_word.transform(
                    [" ".join(tokenize(" ".join(exp)))])
        return qw, self.tf_char.transform([q])

    def search(self, query, top_k=5):
        qw, qc = self.vectorize_query(query)
        sw = cosine_similarity(qw, self.Mw)[0]
        sc = cosine_similarity(qc, self.Mc)[0]
        scores = self.word_weight * sw + (1 - self.word_weight) * sc
        # 标题/小节命中加权：多字符实词命中标题才算（单字"号/单"噪声太大）
        qterms = {t for t in tokenize(query) if len(t) >= 2}
        if self.expand:
            qterms |= set(tokenize(" ".join(expand_query(query))))
        for i, d in enumerate(self.docs):
            title_terms = set(tokenize(d["title"] + " " + d.get("heading", "")))
            hit = len(qterms & title_terms)
            if hit:
                scores[i] += 0.02 * hit
        idx = np.argsort(-scores)[:top_k]
        return [{"doc": self.docs[i], "score": float(scores[i])} for i in idx]


class ApiEmbedRetriever:
    """OpenAI 兼容 embedding 接口检索（设置 EMBEDDING_API_KEY + EMBEDDING_BASE_URL +
    EMBEDDING_MODEL 后启用；用于演示“真正的向量检索”升级路径）。"""

    def __init__(self, docs):
        self.docs = docs
        self.key = os.environ["EMBEDDING_API_KEY"]
        self.base = os.environ.get("EMBEDDING_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        cache = DATA / "embedding_cache.json"
        emb = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
        missing = [d["chunk_id"] for d in docs if d["chunk_id"] not in emb]
        if missing:
            for start in range(0, len(missing), 64):
                batch = missing[start:start + 64]
                texts = [next(d["text"] for d in docs if d["chunk_id"] == cid) for cid in batch]
                r = requests.post(f"{self.base}/embeddings",
                                  headers={"Authorization": f"Bearer {self.key}"},
                                  json={"model": self.model, "input": texts}, timeout=60)
                r.raise_for_status()
                for cid, item in zip(batch, r.json()["data"]):
                    emb[cid] = item["embedding"]
            cache.write_text(json.dumps(emb), encoding="utf-8")
        self.M = np.array([emb[d["chunk_id"]] for d in docs])

    def _embed(self, q):
        r = requests.post(f"{self.base}/embeddings",
                          headers={"Authorization": f"Bearer {self.key}"},
                          json={"model": self.model, "input": [q]}, timeout=30)
        r.raise_for_status()
        return np.array(r.json()["data"][0]["embedding"])

    def search(self, query, top_k=5):
        v = self._embed(query)
        sims = self.M @ v / (np.linalg.norm(self.M, axis=1) * np.linalg.norm(v) + 1e-9)
        idx = np.argsort(-sims)[:top_k]
        return [{"doc": self.docs[i], "score": float(sims[i])} for i in idx]


# ---------------------------------------------------------------- 生成
SYS_PROMPT = (
    "你是云帆ERP（跨境电商ERP）的官方智能客服。请严格依据提供的帮助中心资料回答用户问题。\n"
    "规则：1）答案中的步骤和数字必须来自资料，不得编造；2）引用资料内容时在句末标注来源编号如[1][2]；"
    "3）资料不足以回答时明确说“帮助中心暂时没有相关说明”，并建议转人工客服，不要硬答；"
    "4）用简洁中文分点作答，先给结论再给步骤；5）不讨论竞品、不承诺平台政策以外的赔付。"
)


def call_llm(messages, temperature=0.2):
    key = os.environ.get("LLM_API_KEY")
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    r = requests.post(f"{base}/chat/completions",
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      json={"model": model, "messages": messages, "temperature": temperature,
                            "max_tokens": 800}, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def split_sentences(text):
    # 去掉块内小标题标记后切句；只在句末标点/换行切，不在"；"切（保留"排查顺序：…；…；…"完整步骤序列）
    t = re.sub(r"【[^】]+】", "", text)
    parts = re.split(r"(?<=[。！？\n])", t)
    return [p.strip() for p in parts if len(p.strip()) >= 8]


def extractive_answer(query, hits, threshold=EXTRACT_COVER_THRESHOLD, expansion=None):
    """无 LLM 时的降级方案：从召回块中按词覆盖抽取最相关句子组织回答。

    threshold: 句子入选的最低词覆盖率（默认 0.25，EXTRACT_COVER_THRESHOLD 可配置）；
    expansion: expand_query() 产出的文档术语，计入覆盖（"不回传"也能命中"回传失败"句）；
    选择策略：每个召回块最多贡献 2 句，避免答案被单个块主导，总数上限 5 句。
    """
    raw_terms = [t for t in tokenize(query) if len(t) >= 2]  # 只用多字符实词做覆盖
    qterms = set(raw_terms) or set(tokenize(query))
    exp_terms = set(tokenize(" ".join(expansion or [])))
    wanted = qterms | exp_terms
    titles = {t for h in hits for t in [h["doc"]["title"]] + h["doc"].get("alt_titles", [])}
    cands = []
    for rank, h in enumerate(hits):
        for sent in split_sentences(h["doc"]["text"]):
            if sent in titles or len(sent) < 12:  # 过滤标题残句和碎片
                continue
            sterms = set(tokenize(sent))
            cover = len(wanted & sterms) / max(len(wanted), 1)
            if cover < threshold:
                continue
            cands.append((cover - rank * 0.02, h["doc"], sent, rank + 1))
    cands.sort(key=lambda x: -x[0])
    picked, seen, per_chunk = [], set(), {}
    for _, doc, sent, ref in cands:
        key = sent[:18]
        if key in seen or per_chunk.get(doc["chunk_id"], 0) >= 2:  # 每块最多2句
            continue
        seen.add(key)
        per_chunk[doc["chunk_id"]] = per_chunk.get(doc["chunk_id"], 0) + 1
        picked.append((doc, sent, ref))
        if len(picked) >= 5:
            break
    picked.sort(key=lambda x: x[2])
    if not picked:
        return None
    lines = ["根据云帆ERP帮助中心的说明："]
    for _, sent, ref in picked:
        sent = sent.rstrip("。；;\n") + "。"
        lines.append(f"- {sent}[{ref}]")
    return "\n".join(lines)


def followup_suggestions(hits, n=3):
    """基于召回块所在模块/文章的相邻主题生成追问建议（从小节标题改写）。"""
    titles, out = [], []
    for h in hits:
        for hdg in h["doc"].get("headings", []):
            t = re.sub(r"（\d+/\d+）", "", hdg)
            if t not in titles and len(t) <= 14:
                titles.append(t)
    for t in titles:
        if any(k in t for k in ["怎么", "如何", "流程", "步骤", "排查", "处理", "设置", "配置", "说明"]):
            out.append(f"继续了解：{t}")
        else:
            out.append(f"想问下：{t}具体怎么操作？")
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- 对话
FOLLOWUP_PTR = re.compile(r"^(这个|那个|它|他|她|这|那|还是|然后|接着|继续|上面|刚才|还是说)")


class RAGChatBot:
    def __init__(self, top_k=5, confidence=CONFIDENCE_THRESHOLD,
                 extract_threshold=EXTRACT_COVER_THRESHOLD):
        self.docs = load_kb()
        self.retriever = ApiEmbedRetriever(self.docs) if os.environ.get("EMBEDDING_API_KEY") \
            else HybridRetriever(self.docs)
        self.top_k = top_k
        self.confidence = confidence
        self.extract_threshold = extract_threshold
        self.history = []  # [{"role","content"}]
        self.use_llm = bool(os.environ.get("LLM_API_KEY"))

    def reset(self):
        self.history = []

    def _rewrite_query(self, question):
        """短追问：拼接上一轮问题中的实体，提升召回。"""
        if self.history and (FOLLOWUP_PTR.search(question.strip()) or len(question) <= 8):
            prev_user = next((m["content"] for m in reversed(self.history) if m["role"] == "user"), "")
            if prev_user:
                return f"{prev_user} {question}"
        return question

    def answer(self, question):
        retrieval_q = self._rewrite_query(question)
        hits = self.retriever.search(retrieval_q, top_k=self.top_k)
        expansion = expand_query(retrieval_q)
        top_score = hits[0]["score"] if hits else 0.0
        result = {"question": question, "retrieval_query": retrieval_q,
                  "top_score": round(top_score, 4), "hits": hits,
                  "mode": "llm" if self.use_llm else "extractive", "handoff": False}

        if top_score < self.confidence:
            result["handoff"] = True
            result["answer"] = ("这个问题我在云帆ERP帮助中心没有找到足够可靠的说明，为避免误导，"
                                "已为你准备转人工：你可以提交工单（帮助中心-提交工单）或联系在线客服，"
                                "并附上店铺和操作截图。")
            result["sources"], result["suggestions"] = [], []
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": result["answer"]})
            return result

        context, sources = [], []
        for i, h in enumerate(hits, 1):
            context.append(f"[资料{i}]（模块：{h['doc']['module']}｜文章：{h['doc']['title']}）\n{h['doc']['text']}")
            sources.append({"index": i, "module": h["doc"]["module"], "title": h["doc"]["title"],
                            "heading": h["doc"]["heading"], "url": h["doc"]["url"],
                            "score": round(h["score"], 4)})

        if self.use_llm:
            # 多轮：系统提示 + 最近两轮历史 + 本轮（带最新检索资料）
            msgs = [{"role": "system", "content": SYS_PROMPT}] + self.history[-4:] + [
                {"role": "user", "content": "帮助中心资料：\n\n" + "\n\n".join(context)
                 + f"\n\n用户问题：{question}"}]
            try:
                result["answer"] = call_llm(msgs)
            except Exception as e:
                result["answer"] = extractive_answer(
                    question, hits, self.extract_threshold, expansion) or \
                    "生成服务暂时不可用，建议稍后重试或转人工。"
                result["mode"] = "extractive(llm失败降级)"
        else:
            result["answer"] = extractive_answer(question, hits,
                                                 self.extract_threshold, expansion)
            if result["answer"] is None:
                result["answer"] = "我找到了相关资料但没能组织出准确答案，建议查看下方来源文档或转人工确认。"
                result["handoff"] = True

        result["sources"] = sources
        result["suggestions"] = followup_suggestions(hits)
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": result["answer"]})
        return result


if __name__ == "__main__":
    # 简易自检
    bot = RAGChatBot()
    for q in ["物流单号一直不回传怎么办？", "安全库存预警在哪里设置？", "你们老板叫什么名字？"]:
        r = bot.answer(q)
        print("Q:", q)
        print("score:", r["top_score"], "| handoff:", r["handoff"], "| top:",
              r["hits"][0]["doc"]["title"] if r["hits"] else "-")
        print(r["answer"][:220])
        print("-" * 70)
