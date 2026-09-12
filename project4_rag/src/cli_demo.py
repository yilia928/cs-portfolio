# -*- coding: utf-8 -*-
"""
项目④：命令行 demo
用法：python src/cli_demo.py
命令：/reset 清空对话  /quit 退出
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_engine import RAGChatBot


def main():
    print("=" * 72)
    print("云帆ERP 智能客服 RAG Demo （离线抽取式模式；配置 LLM_API_KEY 后自动切换大模型）")
    print("示例问题：物流单号一直不回传怎么办 / 安全库存预警在哪里开 / 合单规则在哪里配置")
    print("=" * 72)
    bot = RAGChatBot()
    while True:
        try:
            q = input("\n我：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/reset":
            bot.reset()
            print("（对话已清空）")
            continue
        r = bot.answer(q)
        print(f"\n助手（{r['mode']}，置信度 {r['top_score']}）：")
        print(r["answer"])
        if r["sources"]:
            print("\n来源：")
            for s in r["sources"][:3]:
                print(f"  [{s['index']}] {s['module']}｜{s['title']}｜{s['heading']}｜{s['url']}")
        if r["suggestions"]:
            print("\n你可能还想问：")
            for sg in r["suggestions"]:
                print("  -", sg)
        if r["handoff"]:
            print("\n[系统] 本条已触发兜底转人工。")
    print("\n再见。")


if __name__ == "__main__":
    main()
