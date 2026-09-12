# -*- coding: utf-8 -*-
"""
项目① W3：可视化看板
1) dashboard.html —— 单文件 ECharts 看板（KPI/续约趋势/健康分层/行业风险/预警TOP榜/案例曲线）
2) 客户成功看板.xlsx —— Excel 原生图表 + 全量数据 + 条件格式 + 指标口径
"""
import json
import pandas as pd
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import CellIsRule
from openpyxl.chart import PieChart, LineChart, BarChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "output"

S = json.load(open(OUT / "metrics_summary.json", encoding="utf-8"))
CASES = json.load(open(OUT / "case_timeline.json", encoding="utf-8"))
INTERV = pd.read_csv(OUT / "intervention_list.csv", encoding="utf-8-sig")
K = S["kpi"]

# ================================================================== HTML
SEV_STYLE = {"高": ("#e53935", "红色预警"), "中": ("#fb8c00", "橙色预警"), "低": ("#fbc02d", "黄色预警")}

top_rows = ""
for r in S["top_warnings"]:
    color, label = SEV_STYLE[r["severity"]]
    acts = r["actions"].split("；")
    act_txt = "；".join(acts[:2])
    top_rows += f"""
    <tr>
      <td class="rk">{r['rank']}</td>
      <td><b>{r['customer_id']}</b><br><span class="sub">{r['industry']} · {r['plan']}</span></td>
      <td><span class="hscore" style="color:{'#e53935' if r['health']<60 else '#fb8c00'}">{r['health']}</span></td>
      <td><span class="badge" style="background:{color}">{r['top_rule']} {label}</span></td>
      <td class="ev">{r['evidence']}</td>
      <td class="act">{act_txt}</td>
    </tr>"""

case_meta = {
    "C0007": ("案例① 信号→预警→高层干预→挽回成功", "美妆个护 · 专业版 · 13席位",
              "2月登录断崖，3月R1红色预警（102→31→20次）。48h内CSM外呼诊断为核心运营离职，"
              "随即启动高层回访+新运营1v1培训+30天激活任务；8月健康度回升至83，7月按期续约并增购5席位。"),
    "C0023": ("案例② 工单风暴→技术介入→稳住续约", "服饰鞋包 · 基础版 · 36席位",
              "12月切换新物流渠道后工单激增，1月R2红色预警（工单+175%、解决28h、CSAT 3.3）。"
              "技术专家48h介入、开通4小时SLA通道并输出《物流对账操作手册》；5月CSAT回到4.1，7月续约。"),
    "C0150": ("案例③ 漏警复盘→规则迭代", "服饰鞋包 · 基础版 · 50席位",
              "崩前9个月健康度长期停在69~72（关注层），无任何规则命中；6月数据崩塌（453→70→10），"
              "7月才触发R1，距到期仅50天，老板已签约竞品，8月20日流失。复盘后新增R5『关注层停滞』规则。"),
}

case_cards = ""
for cid, (title, sub, story) in case_meta.items():
    case_cards += f"""
    <div class="case-card">
      <div class="case-h"><b>{title}</b><span class="sub">{sub}</span></div>
      <div id="chart_case_{cid}" style="height:170px"></div>
      <p class="story">{story}</p>
    </div>"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>客户成功指标体系与流失预警看板</title>
<script src="../assets/echarts.min.js"></script>  <!-- 本地引用，断网可用；重新生成前需保证 assets/echarts.min.js 存在（从 echarts@5.5.0 CDN 下载一次即可） -->
<style>
 *{{box-sizing:border-box;margin:0;padding:0}}
 body{{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f2f4f8;color:#263238;padding:24px}}
 h1{{font-size:24px}} .head{{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:18px}}
 .head .sub{{color:#78909c;margin-top:6px;font-size:13px}}
 .kpis{{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-bottom:16px}}
 .kpi{{background:#fff;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 .kpi .v{{font-size:28px;font-weight:700;margin:6px 0}} .kpi .l{{font-size:13px;color:#78909c}}
 .kpi .d{{font-size:12px;margin-top:4px}}
 .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
 .card{{background:#fff;border-radius:10px;padding:16px 18px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 .card h3{{font-size:15px;margin-bottom:8px;border-left:4px solid #2e7dff;padding-left:8px}}
 table{{width:100%;border-collapse:collapse;font-size:12.5px}}
 th,td{{padding:8px 8px;border-bottom:1px solid #eceff1;text-align:left;vertical-align:top}}
 th{{background:#f8fafc;color:#546e7a;font-weight:600;white-space:nowrap}}
 .rk{{color:#90a4ae;font-weight:700}} .sub{{color:#90a4ae;font-size:11.5px}}
 .hscore{{font-weight:700;font-size:15px}}
 .badge{{color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;white-space:nowrap}}
 .ev{{max-width:300px;color:#5d4037}} .act{{max-width:260px;color:#37474f}}
 .cases{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:14px}}
 .case-card{{background:#fff;border-radius:10px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 .case-h{{display:flex;flex-direction:column;gap:3px;margin-bottom:4px}}
 .story{{font-size:12px;color:#546e7a;line-height:1.65;margin-top:6px}}
 .foot{{color:#90a4ae;font-size:12px;text-align:center;margin:10px 0 4px}}
 @media print{{body{{background:#fff}} .card,.kpi,.case-card{{box-shadow:none;border:1px solid #e0e0e0}}}}
</style></head><body>
<div class="head">
 <div><h1>客户成功指标体系与流失预警看板</h1>
 <div class="sub">跨境电商 SaaS · 500家客户 · 数据窗口 2024-09 ~ 2025-08 · 分析基准日 2025-09-01（模拟数据，种子42可复现）</div></div>
 <div class="sub">健康度 = 使用率40% + 登录趋势25% + 工单趋势25% + CSAT10%</div>
</div>
<div class="kpis">
 <div class="kpi"><div class="l">整体续约率</div><div class="v" style="color:#2e7dff">{K['renewal_rate']}%</div><div class="d sub">到期{K['due']}家 / 续约{K['renewed']}家</div></div>
 <div class="kpi"><div class="l">平均健康度</div><div class="v" style="color:#00897b">{K['avg_health']}</div><div class="d sub">在服{K['active']}家</div></div>
 <div class="kpi"><div class="l">高风险客户</div><div class="v" style="color:#e53935">{K['tier_risk']}</div><div class="d sub">关注{K['tier_watch']} / 健康{K['tier_healthy']}</div></div>
 <div class="kpi"><div class="l">当月预警客户</div><div class="v" style="color:#fb8c00">{K['warning_red']+K['warning_orange']+K['warning_yellow']}</div><div class="d sub">红{K['warning_red']} 橙{K['warning_orange']} 黄{K['warning_yellow']}</div></div>
 <div class="kpi"><div class="l">风险 ARR</div><div class="v" style="color:#8e24aa">¥{K['arr_at_risk']:,}</div><div class="d sub">红/橙客户年合同额</div></div>
 <div class="kpi"><div class="l">已流失客户</div><div class="v" style="color:#616161">{K['churned']}</div><div class="d sub">待续约{K['pending']}家</div></div>
</div>
<div class="grid2">
 <div class="card"><h3>续约率趋势（按到期月份队列）</h3><div id="c_renew" style="height:290px"></div></div>
 <div class="card"><h3>健康度分层占比（当前在服客户）</h3><div id="c_tier" style="height:290px"></div></div>
</div>
<div class="grid2">
 <div class="card"><h3>在服客户平均健康度趋势（12个月）</h3><div id="c_htrend" style="height:280px"></div></div>
 <div class="card"><h3>行业 × 健康度分层</h3><div id="c_ind" style="height:280px"></div></div>
</div>
<div class="card" style="margin-bottom:14px"><h3>流失预警 TOP 15（按严重度 × 健康度排序，附触发证据与干预动作）</h3>
<table><thead><tr><th>#</th><th>客户</th><th>健康度</th><th>预警</th><th>触发证据</th><th>建议干预动作</th></tr></thead>
<tbody>{top_rows}</tbody></table></div>
<div class="cases">{case_cards}</div>
<div class="foot">数据源：customers.csv / monthly_usage.csv（pandas 生成） · 评分：02_metrics_scoring.py · 预警规则 R1-R4 全部可解释</div>
<script>
const ZH = {{tooltip:'#fff'}};
const renewMonths = {json.dumps([r['month'] for r in S['renewal_trend']], ensure_ascii=False)};
echarts.init(document.getElementById('c_renew')).setOption({{
 tooltip:{{trigger:'axis'}}, legend:{{data:['到期数','续约率'],top:0}}, grid:{{left:50,right:55,top:40,bottom:30}},
 xAxis:{{type:'category',data:renewMonths}},
 yAxis:[{{type:'value',name:'家'}},{{type:'value',name:'%',min:50,max:100}}],
 series:[{{name:'到期数',type:'bar',data:{json.dumps([r['due'] for r in S['renewal_trend']])},
   itemStyle:{{color:'#bbdefb'}},label:{{show:true,position:'top',fontSize:11}}}},
  {{name:'续约率',type:'line',yAxisIndex:1,smooth:true,data:{json.dumps([r['rate'] for r in S['renewal_trend']])},
   itemStyle:{{color:'#2e7dff'}},lineStyle:{{width:3}},label:{{show:true,formatter:'{{c}}%',fontSize:11}}}}]
}});
const total = {K['tier_healthy']+K['tier_watch']+K['tier_risk']};
echarts.init(document.getElementById('c_tier')).setOption({{
 tooltip:{{trigger:'item',formatter:'{{b}}: {{c}}家 ({{d}}%)'}}, legend:{{bottom:0}},
 series:[{{type:'pie',radius:['45%','72%'],center:['50%','46%'],
  label:{{formatter:'{{b}}\\n{{c}}家 {{d}}%',fontSize:12}},
  data:[{{value:{K['tier_healthy']},name:'健康 ≥80',itemStyle:{{color:'#43a047'}}}},
        {{value:{K['tier_watch']},name:'关注 60-79',itemStyle:{{color:'#fb8c00'}}}},
        {{value:{K['tier_risk']},name:'高风险 <60',itemStyle:{{color:'#e53935'}}}}]}}]
}});
echarts.init(document.getElementById('c_htrend')).setOption({{
 tooltip:{{trigger:'axis'}},grid:{{left:45,right:20,top:30,bottom:30}},
 xAxis:{{type:'category',data:{json.dumps([r['month'] for r in S['health_trend']])}}},
 yAxis:{{type:'value',min:50,max:90}},
 series:[{{type:'line',smooth:true,data:{json.dumps([r['health'] for r in S['health_trend']])},
  lineStyle:{{width:3,color:'#00897b'}},itemStyle:{{color:'#00897b'}},areaStyle:{{color:'rgba(0,137,123,.12)'}},
  markLine:{{silent:true,data:[{{yAxis:80,lineStyle:{{color:'#43a047',type:'dashed'}},label:{{formatter:'健康线80'}}}},
   {{yAxis:60,lineStyle:{{color:'#e53935',type:'dashed'}},label:{{formatter:'风险线60'}}}}]}}}}]
}});
const ind = {json.dumps(S['industry'], ensure_ascii=False)};
echarts.init(document.getElementById('c_ind')).setOption({{
 tooltip:{{trigger:'axis',axisPointer:{{type:'shadow'}}}},legend:{{top:0}},
 grid:{{left:70,right:20,top:34,bottom:20}},
 xAxis:{{type:'value'}},yAxis:{{type:'category',data:ind.map(d=>d.industry)}},
 series:[
  {{name:'健康',type:'bar',stack:'t',data:ind.map(d=>d.healthy),itemStyle:{{color:'#43a047'}}}},
  {{name:'关注',type:'bar',stack:'t',data:ind.map(d=>d.watch),itemStyle:{{color:'#fb8c00'}}}},
  {{name:'高风险',type:'bar',stack:'t',data:ind.map(d=>d.risk),itemStyle:{{color:'#e53935'}}}}]
}});
const caseData = {json.dumps(CASES, ensure_ascii=False)};
const eventPt = {{C0007:['2025-03',42.1],C0023:['2025-01',61.3],C0150:['2025-07',20.9]}};
Object.keys(caseData).forEach(cid=>{{
 const d=caseData[cid];
 echarts.init(document.getElementById('chart_case_'+cid)).setOption({{
  tooltip:{{trigger:'axis'}},grid:{{left:38,right:14,top:18,bottom:24}},
  xAxis:{{type:'category',data:d.map(r=>r.month),axisLabel:{{fontSize:10,interval:1}}}},
  yAxis:{{type:'value',min:0,max:100,axisLabel:{{fontSize:10}}}},
  series:[{{type:'line',smooth:true,showSymbol:false,data:d.map(r=>r.health),
   lineStyle:{{width:2.5,color:'#5c6bc0'}},areaStyle:{{color:'rgba(92,107,192,.10)'}},
   markLine:{{silent:true,symbol:'none',data:[{{yAxis:60,lineStyle:{{color:'#e53935',type:'dashed'}}}}]}},
   markPoint:{{symbolSize:46,data:[{{coord:eventPt[cid],itemStyle:{{color:'#e53935'}},
     label:{{formatter:'预警',fontSize:10,color:'#fff'}}}}]}}}}]
 }});
}});
window.addEventListener('resize',()=>document.querySelectorAll('div[id^=c_],[id^=chart_case]').forEach(el=>{{
 const ins=echarts.getInstanceByDom(el); if(ins)ins.resize();}}));
</script></body></html>"""

(OUT / "dashboard.html").write_text(html, encoding="utf-8")
print("[OK] dashboard.html ->", OUT / "dashboard.html")

# ================================================================== Excel
wb = Workbook()
thin = Side(style="thin", color="CFD8DC")
border = Border(left=thin, right=thin, top=thin, bottom=thin)
head_fill = PatternFill("solid", fgColor="2E7DFF")
head_font = Font(color="FFFFFF", bold=True, size=11)
title_font = Font(bold=True, size=16, color="263238")


def style_header(ws, row, ncol):
    for c in range(1, ncol + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill, cell.font = head_fill, head_font
        cell.alignment = Alignment(horizontal="center", vertical="center")


def autosize(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


# ---------- Sheet 图表数据
wsd = wb.active
wsd.title = "图表数据"
wsd["A1"] = "健康度分层"
wsd.append(["分层", "客户数"])
for name, val in [("健康>=80", K["tier_healthy"]), ("关注60-79", K["tier_watch"]), ("高风险<60", K["tier_risk"])]:
    wsd.append([name, val])
wsd["E1"] = "月份"; wsd["F1"] = "平均健康度"; wsd["H1"] = "月份"; wsd["I1"] = "到期数"; wsd["J1"] = "续约率%"
for i, r in enumerate(S["health_trend"], 2):
    wsd.cell(i, 5, r["month"]); wsd.cell(i, 6, r["health"])
for i, r in enumerate(S["renewal_trend"], 2):
    wsd.cell(i, 8, r["month"]); wsd.cell(i, 9, r["due"]); wsd.cell(i, 10, r["rate"])
wsd["L1"] = "行业"; wsd["M1"] = "健康"; wsd["N1"] = "关注"; wsd["O1"] = "高风险"
for i, r in enumerate(S["industry"], 2):
    wsd.cell(i, 12, r["industry"]); wsd.cell(i, 13, r["healthy"]); wsd.cell(i, 14, r["watch"]); wsd.cell(i, 15, r["risk"])
for col in "AFHIJLMNO":
    wsd.column_dimensions[col].width = 13
style_header(wsd, 1, 15)
for c in ["A1", "E1", "H1", "L1"]:
    wsd[c].font = Font(bold=True)

# ---------- Sheet 看板
ws = wb.create_sheet("看板", 0)
ws.sheet_view.showGridLines = False
ws["B2"] = "客户成功指标体系与流失预警看板"; ws["B2"].font = title_font
ws["B3"] = "数据窗口 2024-09~2025-08 · 分析日 2025-09-01 · 500家客户（模拟数据）"; ws["B3"].font = Font(size=10, color="78909C")
kpi_defs = [("整体续约率", f"{K['renewal_rate']}%", f"到期{K['due']}/续约{K['renewed']}"),
            ("平均健康度", K["avg_health"], f"在服{K['active']}家"),
            ("高风险", K["tier_risk"], f"关注{K['tier_watch']} 健康{K['tier_healthy']}"),
            ("当月预警", K["warning_red"] + K["warning_orange"] + K["warning_yellow"],
             f"红{K['warning_red']} 橙{K['warning_orange']} 黄{K['warning_yellow']}"),
            ("风险ARR(元)", K["arr_at_risk"], "红/橙客户合同额"),
            ("已流失", K["churned"], f"待续约{K['pending']}")]
for i, (lab, val, note) in enumerate(kpi_defs):
    col = 2 + i * 2
    ws.cell(5, col, lab).font = Font(size=10, color="78909C", bold=True)
    ws.cell(6, col, val).font = Font(size=20, bold=True, color="2E7DFF")
    ws.cell(7, col, note).font = Font(size=9, color="90A4AE")

pie = PieChart(); pie.title = "健康度分层占比"
pie.add_data(Reference(wsd, min_col=2, min_row=2, max_row=4), titles_from_data=False)
pie.set_categories(Reference(wsd, min_col=1, min_row=2, max_row=4))
pie.dataLabels = DataLabelList(); pie.dataLabels.showPercent = True
pie.height, pie.width = 8, 11
ws.add_chart(pie, "B9")

ln = LineChart(); ln.title = "在服客户平均健康度月度趋势"
ln.add_data(Reference(wsd, min_col=6, min_row=1, max_row=13), titles_from_data=True)
ln.set_categories(Reference(wsd, min_col=5, min_row=2, max_row=13))
ln.height, ln.width = 8, 11; ln.y_axis.scaling.min = 50; ln.y_axis.scaling.max = 90
ws.add_chart(ln, "H9")

bar = BarChart(); bar.type = "bar"; bar.grouping = "stacked"; bar.overlap = 100
bar.title = "行业 x 健康度分层"
bar.add_data(Reference(wsd, min_col=13, max_col=15, min_row=1, max_row=9), titles_from_data=True)
bar.set_categories(Reference(wsd, min_col=12, min_row=2, max_row=9))
bar.height, bar.width = 9, 11
ws.add_chart(bar, "B26")

bar2 = BarChart(); bar2.title = "月度到期数与续约率"
bar2.add_data(Reference(wsd, min_col=9, min_row=1, max_row=7), titles_from_data=True)
line2 = LineChart(); line2.add_data(Reference(wsd, min_col=10, min_row=1, max_row=7), titles_from_data=True)
line2.y_axis.axId = 200; line2.y_axis.title = "续约率%"; line2.y_axis.scaling.min = 50
bar2.set_categories(Reference(wsd, min_col=8, min_row=2, max_row=7))
bar2 += line2
bar2.height, bar2.width = 9, 11
ws.add_chart(bar2, "H26")
ws["B45"] = "说明：本页所有图表均为 Excel 原生图表，数据源在『图表数据』页；完整名单见『预警名单』页，口径见『指标口径』页。"
ws["B45"].font = Font(size=10, color="78909C")

# ---------- 预警名单
wsi = wb.create_sheet("预警名单")
cols = ["rank", "customer_id", "industry", "plan", "account_manager", "health", "tier", "top_rule",
        "severity", "evidence", "all_evidence", "actions", "renewal_status", "contract_end", "annual_amount"]
headers = ["排名", "客户ID", "行业", "版本", "客户经理", "健康度", "分层", "主规则", "严重度",
           "触发证据", "全部证据", "建议干预动作", "续约状态", "合同到期", "年合同额"]
wsi.append(headers); style_header(wsi, 1, len(headers))
for _, r in INTERV.iterrows():
    wsi.append([r[c] for c in cols])
red_fill = PatternFill("solid", fgColor="FFEBEE")
orange_fill = PatternFill("solid", fgColor="FFF3E0")
yellow_fill = PatternFill("solid", fgColor="FFFDE7")
wsi.conditional_formatting.add(f"F2:F{len(INTERV)+1}", CellIsRule(operator="lessThan", formula=["60"], fill=red_fill))
for row in range(2, len(INTERV) + 2):
    sev = wsi.cell(row, 9).value
    fill = red_fill if sev == "高" else orange_fill if sev == "中" else yellow_fill
    for c in (8, 9):
        wsi.cell(row, c).fill = fill
autosize(wsi, [6, 10, 10, 8, 10, 8, 8, 8, 8, 42, 60, 55, 10, 12, 11])
wsi.freeze_panes = "A2"

# ---------- 客户健康度快照
wss = wb.create_sheet("客户健康度快照")
snap = pd.read_csv(OUT / "customer_health_current.csv", encoding="utf-8-sig")
keep = ["customer_id", "industry", "company_size", "plan", "seats", "annual_amount", "signup_date",
        "contract_end", "renewal_status", "account_manager", "archetype",
        "logins", "active_accounts", "tickets", "avg_resolve_hours", "csat", "active_modules", "health", "tier"]
snap = snap[keep]
wss.append(list(snap.columns)); style_header(wss, 1, len(snap.columns))
for row in snap.itertuples(index=False):
    wss.append(list(row))
wss.conditional_formatting.add(f"R2:R{len(snap)+1}", CellIsRule(operator="lessThan", formula=["60"], fill=red_fill))
autosize(wss, [10, 10, 10, 8, 7, 10, 12, 12, 9, 10, 14, 8, 9, 8, 9, 7, 9, 8, 8])
wss.freeze_panes = "A2"

# ---------- 月度健康度
wsh = wb.create_sheet("月度健康度")
hist = pd.read_csv(OUT / "health_history.csv", encoding="utf-8-sig")
wsh.append(list(hist.columns)); style_header(wsh, 1, hist.shape[1])
for row in hist.itertuples(index=False):
    wsh.append(list(row))
autosize(wsh, [11, 12, 10, 11, 11, 11, 10, 9, 8]); wsh.freeze_panes = "A2"

# ---------- 明细数据
for name, fname in [("客户主数据", "customers.csv"), ("月度明细", "monthly_usage.csv")]:
    d = pd.read_csv(DATA / fname, encoding="utf-8-sig")
    wsx = wb.create_sheet(name)
    wsx.append(list(d.columns)); style_header(wsx, 1, d.shape[1])
    for row in d.itertuples(index=False):
        wsx.append(list(row))
    autosize(wsx, [12] * min(d.shape[1], 16)); wsx.freeze_panes = "A2"

# ---------- 指标口径
wsm = wb.create_sheet("指标口径")
metric_rows = [
    ("指标", "公式/口径", "权重/阈值", "数据来源"),
    ("续约率", "窗口内已续约客户数 / 已到期客户数（按合同到期月分队列）", "目标 ≥85%", "customers.renewal_status/contract_end"),
    ("月活使用率(席位)", "当月 active_accounts / 购买 seats；客户群口径取在服客户均值", "目标 ≥70%", "monthly_usage"),
    ("模块采用率", "active_modules / 12（全量12个模块）", "用满10个记满分", "monthly_usage"),
    ("使用率得分", "0.6×席位利用率分(80%利用率满分)+0.4×模块采用分", "健康度权重40%", "派生"),
    ("登录趋势得分", "近3月登录均值/前3月均值分段映射：≥1.1→100，1.0→90，0.7→55，0.5→30", "权重25%", "monthly_usage.logins"),
    ("工单趋势得分", "近3月/前3月工单量分段映射(持平85、+50%=60、翻倍40)×解决时效系数(24h=0.9)", "权重25%", "tickets/avg_resolve_hours"),
    ("CSAT得分", "近3月CSAT均值/5×100；无工单月份按80中性分", "权重10%", "monthly_usage.csat"),
    ("健康度", "0.40×使用率+0.25×登录趋势+0.25×工单趋势+0.10×CSAT", "0-100", "派生"),
    ("分层", "健康 ≥80 健康；60-79 关注；<60 高风险", "三档", "派生"),
    ("R1 登录骤降(红)", "连续2月登录环比下降>30%（基期≥20次）且健康度<60", "高", "派生"),
    ("R2 工单风暴(红)", "近2月工单较前2月+≥50%，平均解决>24h，CSAT<3.5（三项同时满足）", "高", "派生"),
    ("R3 活跃萎缩(黄)", "连续3月席位利用率<50% 或 使用模块≤2", "低", "派生"),
    ("R4 续约窗口(橙)", "评估当月起60天内合同到期 且 健康度<70", "中", "派生"),
    ("风险ARR", "当月红/橙预警客户的年合同金额合计", "经营兜底指标", "customers.annual_amount"),
]
for r in metric_rows:
    wsm.append(r)
style_header(wsm, 1, 4)
autosize(wsm, [18, 72, 30, 34])
for row in wsm.iter_rows(min_row=2, max_row=wsm.max_row, max_col=4):
    for cell in row:
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        cell.border = border

wb.save(OUT / "客户成功看板.xlsx")
print("[OK] 客户成功看板.xlsx saved, sheets:", wb.sheetnames)
