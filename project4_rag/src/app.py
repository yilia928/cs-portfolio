# -*- coding: utf-8 -*-
"""
项目④：Flask Web Demo
启动：python src/app.py  ->  http://127.0.0.1:5000
无 API key 时使用离线混合检索 + 抽取式回答，现场可直接演示完整链路。
"""
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_engine import RAGChatBot

app = Flask(__name__, static_folder=None)
bot = RAGChatBot()

HTML = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>云帆ERP 智能客服 · RAG Demo</title>
<style>
:root{--brand:#2e5bff;--ink:#1f2937;--muted:#6b7280;--line:#e5e7eb;--bg:#f4f6fb}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink);height:100vh;display:flex;flex-direction:column}
header{background:#fff;border-bottom:1px solid var(--line);padding:14px 22px;display:flex;align-items:center;gap:12px}
.logo{width:34px;height:34px;border-radius:8px;background:var(--brand);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700}
header h1{font-size:17px;font-weight:600}
header .mode{margin-left:auto;font-size:12px;color:var(--muted);background:#eef2ff;padding:4px 10px;border-radius:20px}
main{flex:1;overflow-y:auto;padding:22px 0}
.wrap{max-width:820px;margin:0 auto;padding:0 18px;display:flex;flex-direction:column;gap:16px}
.msg{display:flex;gap:10px;max-width:88%}
.msg.user{align-self:flex-end;flex-direction:row-reverse}
.avatar{width:32px;height:32px;border-radius:50%;flex:none;display:flex;align-items:center;justify-content:center;font-size:13px;color:#fff}
.msg.bot .avatar{background:var(--brand)} .msg.user .avatar{background:#0ea466}
.bubble{background:#fff;border:1px solid var(--line);border-radius:10px;padding:11px 14px;font-size:14px;line-height:1.8;white-space:pre-wrap;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.msg.user .bubble{background:var(--brand);color:#fff;border-color:var(--brand)}
.meta{font-size:12px;color:var(--muted);margin-top:8px}
.src{margin-top:8px;border-top:1px dashed var(--line);padding-top:8px}
.src a{color:var(--brand);text-decoration:none}
.src li{margin:3px 0;font-size:12.5px;list-style:none}
.handoff{background:#fff7ed;border:1px solid #fdba74;color:#9a3412;border-radius:8px;padding:8px 12px;font-size:12.5px;margin-top:8px}
.sugg{margin-top:8px;display:flex;flex-wrap:wrap;gap:8px}
.sugg button{font-size:12.5px;border:1px solid var(--brand);color:var(--brand);background:#fff;border-radius:20px;padding:4px 12px;cursor:pointer}
.sugg button:hover{background:#eef2ff}
footer{background:#fff;border-top:1px solid var(--line);padding:12px 18px}
.inputbar{max-width:820px;margin:0 auto;display:flex;gap:10px}
.inputbar input{flex:1;border:1px solid var(--line);border-radius:10px;padding:11px 14px;font-size:14px;outline:none}
.inputbar input:focus{border-color:var(--brand)}
.inputbar button{background:var(--brand);color:#fff;border:none;border-radius:10px;padding:0 22px;font-size:14px;cursor:pointer}
.chips{max-width:820px;margin:8px auto 0;display:flex;gap:8px;flex-wrap:wrap}
.chips span{font-size:12px;color:var(--muted);border:1px solid var(--line);border-radius:20px;padding:3px 10px;cursor:pointer;background:#fff}
.typing{color:var(--muted);font-size:13px;padding:4px 2px}
</style></head><body>
<header>
 <div class="logo">帆</div>
 <h1>云帆ERP 智能客服</h1>
 <span class="mode" id="mode">初始化中…</span>
</header>
<main id="chat"><div class="wrap" id="msgs"></div></main>
<footer>
 <div class="chips" id="chips"></div>
 <div class="inputbar">
  <input id="q" placeholder="请输入问题，例如：物流单号一直不回传怎么办？" autocomplete="off">
  <button id="send">发送</button>
 </div>
</footer>
<script>
const msgs=document.getElementById('msgs'), modeEl=document.getElementById('mode');
const CHIPS=['安全库存预警在哪里开？','合单规则在哪里配置？','批量导出报表总是超时怎么办？',
 '临时加席位怎么收费？','打印控件连不上怎么办？','头程运费怎么分摊？'];
CHIPS.forEach(c=>{const s=document.createElement('span');s.textContent=c;s.onclick=()=>{document.getElementById('q').value=c;send()};document.getElementById('chips').appendChild(s)});
function add(role,text){const d=document.createElement('div');d.className='msg '+role;
 d.innerHTML=`<div class="avatar">${role==='bot'?'帆':'我'}</div><div class="bubble">${text}</div>`;
 msgs.appendChild(d);window.scrollTo(0,document.body.scrollHeight);return d.querySelector('.bubble')}
async function send(){
 const q=document.getElementById('q').value.trim();if(!q)return;
 document.getElementById('q').value='';
 add('user',q.replace(/</g,'&lt;'));
 const tip=add('bot','<span class="typing">正在检索知识库并组织回答…</span>');
 const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
 const d=await r.json();tip.innerHTML=render(d);
 window.scrollTo(0,document.body.scrollHeight);
}
function render(d){
 let h=d.answer.replace(/</g,'&lt;').replace(/\\n/g,'<br>');
 h+=`<div class="meta">回答模式：${d.mode} ｜ 检索置信度：${d.top_score} ｜ 命中块：${d.hits.length}</div>`;
 if(d.sources&&d.sources.length){
  h+='<div class="src"><b>来源引用：</b><ul>';
  d.sources.slice(0,4).forEach(s=>{h+=`<li>[${s.index}] ${s.module}｜${s.title}｜${s.heading}｜<a href="${s.url}">${s.url}</a>（相似度 ${s.score}）</li>`});
  h+='</ul></div>';
 }
 if(d.suggestions&&d.suggestions.length){
  h+='<div class="sugg">';d.suggestions.forEach(s=>{h+=`<button onclick="document.getElementById('q').value='${s.replace(/'/g,"\\'").replace(/想问下：|继续了解：/g,'')}怎么操作？';send()">${s}</button>`});
  h+='</div>';
 }
 if(d.handoff){h+='<div class="handoff">未找到可靠依据，已触发兜底策略：建议转人工客服并提交工单（附店铺与截图）。</div>';}
 return h;
}
document.getElementById('send').onclick=send;
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')send()});
fetch('/api/health').then(r=>r.json()).then(d=>{modeEl.textContent=d.mode});
add('bot','你好，我是云帆ERP智能客服，可以回答功能配置与报错处理类问题。回答均基于官方帮助中心并标注来源，试试下方示例问题。');
</script></body></html>"""


@app.route("/")
def index():
    return HTML


@app.route("/api/health")
def health():
    return jsonify({"mode": ("LLM 大模型" if bot.use_llm else "离线检索 + 抽取式回答"),
                    "retriever": type(bot.retriever).__name__, "chunks": len(bot.docs)})


@app.route("/api/chat", methods=["POST"])
def chat():
    q = (request.json or {}).get("question", "").strip()
    if not q:
        return jsonify({"error": "empty question"}), 400
    r = bot.answer(q)
    r["hits"] = [{"chunk_id": h["doc"]["chunk_id"], "module": h["doc"]["module"],
                  "title": h["doc"]["title"], "score": round(h["score"], 4)} for h in r["hits"]]
    return jsonify(r)


@app.route("/api/reset", methods=["POST"])
def reset():
    bot.reset()
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Web demo: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
