# -*- coding: utf-8 -*-
"""
项目④：知识库构建器
将 kb_data_part1/2/3 中的帮助中心内容按小节切块（200-500 字/块），
写入 data/kb_chunks.jsonl（每行一个切块，含元数据），
并输出知识库统计与每模块一份 Markdown 合订本（便于人工浏览核对）。
"""
import json
from pathlib import Path

from kb_data_part1 import CONTENT_P1
from kb_data_part2 import CONTENT_P2
from kb_data_part3 import CONTENT_P3
from kb_data_part4 import CONTENT_P4
from kb_data_part5 import CONTENT_P5
from kb_data_part6 import CONTENT_P6
from kb_data_part7 import CONTENT_P7

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw_docs"
RAW.mkdir(parents=True, exist_ok=True)

TARGET_MIN, HARD_MAX = 200, 500

# 按模块合并第 1-4 批（同模块文章追加在一起）
MODULE_ORDER, grouped = [], {}
for group in CONTENT_P1 + CONTENT_P2 + CONTENT_P3 + CONTENT_P4 + CONTENT_P5 + CONTENT_P6 + CONTENT_P7:
    mod = group["module"]
    if mod not in grouped:
        grouped[mod] = []
        MODULE_ORDER.append(mod)
    grouped[mod].extend(group["articles"])
CONTENT = [{"module": m, "articles": grouped[m]} for m in MODULE_ORDER]


def split_long(text, limit=HARD_MAX):
    """超长正文按句号切分为不超过 limit 的多段（尽量保留语义）。"""
    if len(text) <= limit:
        return [text]
    parts, buf = [], ""
    for sent in text.replace("。", "。\n").splitlines():
        if len(buf) + len(sent) > limit and buf:
            parts.append(buf.strip())
            buf = sent
        else:
            buf += sent
    if buf.strip():
        parts.append(buf.strip())
    return parts


def module_units(articles):
    """把模块内所有文章展平为带文章来源的小节单元：(title,url,heading,body)。"""
    units = []
    for art in articles:
        for heading, body in art["sections"]:
            segs = split_long(body)
            for k, seg in enumerate(segs):
                units.append((art["title"], art["url"],
                              heading if len(segs) == 1 else f"{heading}（{k+1}/{len(segs)}）", seg))
    return units


def greedy_pack_module(units):
    """模块级贪心打包：相邻小节（允许跨文章）合并为 200-500 字块。
    返回 [{"titles","urls","headings","text"}]。"""
    packs, cur = [], None
    for title, url, h, b in units:
        piece = f"【{h}】{b}"
        head = f"{title}\n"
        if cur is None:
            cur = {"titles": [title], "urls": [url], "headings": [h],
                   "text": head + piece, "_len": len(head) + len(piece)}
            continue
        same_art = title in cur["titles"]
        add = (("" if same_art else f"\n{title}\n") + piece)
        if cur["_len"] + len(add) <= HARD_MAX and (same_art or cur["_len"] < TARGET_MIN):
            cur["text"] += add
            cur["_len"] += len(add)
            if title not in cur["titles"]:
                cur["titles"].append(title)
                cur["urls"].append(url)
            cur["headings"].append(h)
        else:
            # 当前块还短但新小节会超长：先尝试把短块并入更早的块
            if cur["_len"] < TARGET_MIN and packs:
                p = packs[-1]
                new_lines = [ln for ln in cur["text"].splitlines() if ln not in p["titles"]]
                add_text = "\n".join(new_lines)
                if p["_len"] + 1 + len(add_text) <= HARD_MAX:
                    p["text"] += "\n" + add_text
                    p["_len"] += 1 + len(add_text)
                    p["titles"].extend(t for t in cur["titles"] if t not in p["titles"])
                    p["urls"] = list(dict.fromkeys(p["urls"] + cur["urls"]))
                    p["headings"].extend(cur["headings"])
                else:
                    packs.append(cur)
            else:
                packs.append(cur)
            cur = {"titles": [title], "urls": [url], "headings": [h],
                   "text": head + piece, "_len": len(head) + len(piece)}
    if cur:
        # 收尾块仍不足 200：尝试并入上一块（不超长时）
        merged = False
        if cur["_len"] < TARGET_MIN and packs:
            p = packs[-1]
            new_lines = [ln for ln in cur["text"].splitlines() if ln not in p["titles"]]
            add_text = "\n".join(new_lines)
            if p["_len"] + 1 + len(add_text) <= HARD_MAX:
                p["text"] += "\n" + add_text
                p["_len"] += 1 + len(add_text)
                p["titles"].extend(t for t in cur["titles"] if t not in p["titles"])
                p["urls"] = list(dict.fromkeys(p["urls"] + cur["urls"]))
                p["headings"].extend(cur["headings"])
                merged = True
        if not merged:
            packs.append(cur)
    return packs


def main():
    chunks, short_hits, long_hits = [], 0, 0
    md_by_module = {}
    cid = 0
    for group in CONTENT:
        mod = group["module"]
        md_by_module.setdefault(mod, [f"# {mod}\n"])
        for art in group["articles"]:
            md_by_module[mod].append(f"\n## {art['title']}\n")
            md_by_module[mod].append(f"来源：云帆ERP帮助中心 `{art['url']}`\n")
            for heading, body in art["sections"]:
                md_by_module[mod].append(f"\n### {heading}\n\n{body}\n")
        for pack in greedy_pack_module(module_units(group["articles"])):
            cid += 1
            chunk_text = pack["text"]
            if len(chunk_text) < TARGET_MIN:
                short_hits += 1
            if len(chunk_text) > HARD_MAX:
                long_hits += 1
            headings = pack["headings"]
            chunks.append({
                "chunk_id": f"KB{cid:04d}",
                "module": mod,
                "title": pack["titles"][0],
                "alt_titles": pack["titles"][1:],
                "headings": headings,
                "heading": headings[0] if len(headings) == 1 else f"{headings[0]} 等{len(headings)}节",
                "url": pack["urls"][0],
                "alt_urls": pack["urls"][1:],
                "updated": "2025-08-20",
                "text": chunk_text,
                "n_chars": len(chunk_text),
            })

    out = DATA / "kb_chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 模块合订 Markdown（人工核对用，不参与检索）
    for mod, lines in md_by_module.items():
        safe = mod.replace("/", "_")
        (RAW / f"{safe}.md").write_text("\n".join(lines), encoding="utf-8")

    stats = {}
    for c in chunks:
        stats.setdefault(c["module"], 0)
        stats[c["module"]] += 1
    lens = [c["n_chars"] for c in chunks]
    print(f"[OK] chunks: {len(chunks)} | modules: {len(stats)}")
    print(f"[OK] char length min/avg/max = {min(lens)}/{sum(lens)//len(lens)}/{max(lens)} "
          f"| <200: {short_hits} | split(>520): {long_hits}")
    for m, n in stats.items():
        print(f"  - {m}: {n}")
    print("[OK] ->", out)


if __name__ == "__main__":
    main()
