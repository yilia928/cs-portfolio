# -*- coding: utf-8 -*-
"""
项目③ W3：公开客户反馈采集脚本（合规版样板）
=====================================================================
合规红线（务必遵守）：
1. 只采集"未登录即可见"的公开内容；不绕过登录、验证码、付费墙、会员权限；
2. 遵守目标站 robots.txt 与服务条款；知乎/小红书/七麦/蝉大师均有较强反爬与
   登录要求，本脚本只提供"人工确认合规后"的请求骨架，默认不执行；
3. 控制频率（≥3 秒/请求）、标识真实 UA、仅做研究用途、不二次传播原文全文；
4. 采集后仅保留：正文、时间、来源平台、URL（不采集用户 ID 等个人信息）。

推荐落地方式（面试可讲）：
- 应用商店评论：七麦/蝉大师公开页 + 人工导出，或厂商开放的评论 API；
- 知乎/小红书：以公开检索页为线索，人工/半自动摘录，遵守平台条款；
- 官方社区：Discuz/Flarum 等标准论坛可在站长授权后用 sitemap/API 采集。

本项目默认使用 02_build_reviews.py 生成的"带标注模拟数据"跑通全流程（850 条，
含广告/重复等真实噪声），真实数据就位后，保持相同字段即可无缝替换。

用法：
    python 01_scrapers.py --demo https://某公开帮助社区/topic/123
    # --demo 仅对单个你有权访问的公开 URL 做一次请求演示，不做批量
"""
import argparse
import csv
import time
import urllib.request
from pathlib import Path
from datetime import datetime

OUT = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
MIN_INTERVAL = 3.0  # 秒/请求


def fetch(url: str, timeout: int = 15) -> str:
    """单次公开页面请求骨架（不解析、不翻页、不绕登录）"""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" not in ctype and "application/json" not in ctype:
            raise RuntimeError(f"unexpected content-type: {ctype}")
        return resp.read().decode("utf-8", errors="ignore")


def save_rows(rows, name="public_feedback.csv"):
    """统一字段：text/source/url/date（不含任何用户标识）"""
    p = OUT / name
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["text", "source", "url", "date"])
        w.writeheader()
        w.writerows(rows)
    print("[OK] saved", len(rows), "rows ->", p)


def demo(url: str):
    print("[INFO] single public-page fetch demo:", url)
    html = fetch(url)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (OUT / f"demo_page_{ts}.html").write_text(html[:200000], encoding="utf-8")
    print("[OK] html length:", len(html), "| saved first 200KB for manual inspection")
    print("[NEXT] 人工检查页面结构后再写解析规则；批量请求请保持 >=%ss 间隔并获得授权" % MIN_INTERVAL)
    time.sleep(MIN_INTERVAL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", help="单个公开 URL 的一次请求演示")
    a = ap.parse_args()
    if a.demo:
        demo(a.demo)
    else:
        print(__doc__)
