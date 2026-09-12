# -*- coding: utf-8 -*-
"""
项目④：RAG 离线评估
====================
指标：
  Recall@K   Top-K 召回块中包含期望模块的比例
  MRR        首个期望模块块的倒数排名（K=10 内）
  ModPrec@K  Top-K 中期望模块块占比（引用正确率的代理）
  Handoff    转人工率（置信阈值下）
  AnsCov     抽取式答案包含期望关键词的比例（curated 题）
迭代网格：切块大小（标准 200-500 / 小窗口约 240）× TopK(3,5) × 词权重(0.5,0.6,0.7)
消融实验：查询扩展 开/关（选定配置 std / ww0.6 / K5 上对比）
输出：tests/eval_results.json、tests/iteration_log.csv
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rag_engine import (load_kb, HybridRetriever, extractive_answer, expand_query,  # noqa
                        CONFIDENCE_THRESHOLD, EXTRACT_COVER_THRESHOLD)

TESTS = ROOT / "tests"
KB = load_kb()
QUESTIONS = [json.loads(l) for l in (TESTS / "test_questions.jsonl").read_text(encoding="utf-8").splitlines()]


def small_windows(docs, target=240):
    """把标准块进一步切成约 240 字的句子窗口（模拟更小块的切块策略）。"""
    out = []
    for d in docs:
        buf, sub = "", 0
        for s in re.split(r"(?<=[。\n])", d["text"]):
            if buf and len(buf) + len(s) > target:
                sub += 1
                nd = dict(d); nd["chunk_id"] = d["chunk_id"] + f"-{sub}"
                nd["text"] = buf.strip(); nd["n_chars"] = len(buf)
                out.append(nd); buf = s
            else:
                buf += s
        if buf.strip():
            sub += 1
            nd = dict(d); nd["chunk_id"] = d["chunk_id"] + f"-{sub}"
            nd["text"] = buf.strip(); nd["n_chars"] = len(buf)
            out.append(nd)
    return out


def eval_config(docs, ww, top_k, threshold, expand, label=None):
    idx = HybridRetriever(docs, word_weight=ww, expand=expand)
    n, rec_list, mrr_list, prec_list, handoff = len(QUESTIONS), [], [], [], 0
    for q in QUESTIONS:
        hits = idx.search(q["question"], top_k=max(top_k, 10))
        scores = [h["score"] for h in hits]
        mods = [h["doc"]["module"] for h in hits]
        if scores[0] < threshold:
            handoff += 1
        hit_mods = [j for j, m in enumerate(mods[:top_k]) if m == q["expected_module"]]
        rec_list.append(1.0 if hit_mods else 0.0)
        prec_list.append(len(hit_mods) / top_k)
        all_rank = [j for j, m in enumerate(mods) if m == q["expected_module"]]
        mrr_list.append(1.0 / (all_rank[0] + 1) if all_rank else 0.0)
    return {"chunk_variant": label or ("std-200-500" if docs is KB else "small-240"),
            "top_k": top_k, "word_weight": ww, "expansion": int(expand),
            "threshold": threshold,
            "recall": round(float(np.mean(rec_list)), 4),
            "mrr": round(float(np.mean(mrr_list)), 4),
            "module_precision": round(float(np.mean(prec_list)), 4),
            "handoff_rate": round(handoff / n, 4)}


def answer_quality(expand=True):
    """抽取式答案质量：关键词覆盖 + 转人工准确率（curated 题子集）。"""
    idx = HybridRetriever(KB, word_weight=0.6, expand=expand)
    cov, ans_n, wrong_hand, no_answer = [], 0, 0, 0
    for q in QUESTIONS:
        hits = idx.search(q["question"], top_k=5)
        if q["source"] != "curated":
            continue
        ans_n += 1
        if hits[0]["score"] < CONFIDENCE_THRESHOLD:
            mods = [h["doc"]["module"] for h in idx.search(q["question"], top_k=10)]
            if q["expected_module"] in mods:
                wrong_hand += 1
            continue
        ans = extractive_answer(q["question"], hits, EXTRACT_COVER_THRESHOLD,
                                expand_query(q["question"]) if expand else None)
        if ans is None:
            no_answer += 1
        elif q["keywords"]:
            hit = sum(1 for k in q["keywords"] if k.lower() in ans.lower())
            cov.append(hit / len(q["keywords"]))
    return {"curated_n": ans_n,
            "answer_keyword_coverage": round(float(np.mean(cov)), 4) if cov else 0,
            "answer_rate": round(len(cov) / ans_n, 4),
            "no_answer_handoff": no_answer, "wrong_handoff": wrong_hand}


def main():
    rows = []
    docs_variants = {"std": KB, "small": small_windows(KB)}
    # 基础网格：统一关闭查询扩展，研究切块/TopK/权重本身的影响
    for variant_name, docs in docs_variants.items():
        for ww in (0.5, 0.6, 0.7):
            for k in (3, 5):
                rows.append(eval_config(docs, ww, k, CONFIDENCE_THRESHOLD, expand=False))
    # 消融：选定配置（std/ww0.6/K5）开启查询扩展
    ab_off = next(r for r in rows if r["chunk_variant"] == "std-200-500"
                  and r["top_k"] == 5 and r["word_weight"] == 0.6)
    ab_on = eval_config(KB, 0.6, 5, CONFIDENCE_THRESHOLD, expand=True,
                        label="std-200-500+扩展(选定)")
    rows.append(ab_on)
    rows.sort(key=lambda r: (-r["recall"], -r["mrr"]))
    with (TESTS / "iteration_log.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    aq = answer_quality(expand=True)
    aq_off = answer_quality(expand=False)
    summary = {"n_questions": len(QUESTIONS),
               "n_chunks_std": len(KB), "n_chunks_small": len(docs_variants["small"]),
               "selected_retrieval": ab_on,
               "ablation_expand_off": ab_off, "ablation_expand_on": ab_on,
               "all_configs": rows,
               "answer_quality_on": aq, "answer_quality_off": aq_off}
    (TESTS / "eval_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding="utf-8")

    print(f"[OK] questions={len(QUESTIONS)} chunks(std/small)={len(KB)}/{len(docs_variants['small'])}")
    print("Ablation (std/ww0.6/K5):")
    print(f"  expand=OFF | Recall={ab_off['recall']} MRR={ab_off['mrr']} "
          f"ModPrec={ab_off['module_precision']} Handoff={ab_off['handoff_rate']}")
    print(f"  expand=ON  | Recall={ab_on['recall']} MRR={ab_on['mrr']} "
          f"ModPrec={ab_on['module_precision']} Handoff={ab_on['handoff_rate']}")
    print("Top 5 configs:")
    for r in rows[:5]:
        print(f"  {r['chunk_variant']:>20} K={r['top_k']} ww={r['word_weight']} ex={r['expansion']} "
              f"| Recall={r['recall']} MRR={r['mrr']} ModPrec={r['module_precision']} "
              f"Handoff={r['handoff_rate']}")
    print("Answer quality (expand on/off):", aq, "/", aq_off)


if __name__ == "__main__":
    main()
