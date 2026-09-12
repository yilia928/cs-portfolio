# -*- coding: utf-8 -*-
"""
项目① W2：指标计算 / 健康度评分 / 可解释预警规则
=====================================================================
健康度（0-100）= 使用率 40% + 登录趋势 25% + 工单趋势 25% + CSAT 10%
分层：>=80 健康 / 60-79 关注 / <60 高风险
预警规则（全部可解释、带触发证据）：
  R1 登录骤降(红)：连续2个月登录环比下降>30%（基期>=20次）且健康度<60
  R2 工单风暴(红)：近2月工单较前2月增长>=50% 且 平均解决时长>24h 且 CSAT<3.5
  R3 活跃萎缩(黄)：连续3个月席位使用率<50% 或 使用模块数<=2
  R4 续约窗口(橙)：合同60天内到期 且 健康度<70
输出：健康度历史、当前快照、预警事件、干预动作清单、指标汇总JSON、案例时间线
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
ANALYSIS_DATE = pd.Timestamp("2025-09-01")

WEIGHTS = {"usage": 0.40, "login_trend": 0.25, "ticket_trend": 0.25, "csat": 0.10}


# ------------------------------------------------------------ 打分原语
def interp(x, xp, fp):
    return float(np.clip(np.interp(x, xp, fp), fp[0], fp[-1]))


def usage_score(util, modules, total_modules=12):
    """使用率得分：席位利用率(60%权重，80%利用率即满分) + 模块采用(40%，用满10个即满分)"""
    util_c = np.clip(util / 0.8, 0, 1) * 100
    adopt_c = np.clip((modules - 1) / 9, 0, 1) * 100
    return 0.6 * util_c + 0.4 * adopt_c


def trend_ratio(curr, prev):
    curr_m, prev_m = np.mean(curr), np.mean(prev)
    if prev_m < 5:
        return None  # 基期过小，趋势不具解释性 -> 中性
    return curr_m / prev_m


def login_trend_score(ratio):
    if ratio is None:
        return 85.0
    # 环比>=10%满分；持平90；-30%(0.7)55；-50%(0.5)30；-70% 10
    return interp(ratio, [0.3, 0.5, 0.7, 1.0, 1.1], [10, 30, 55, 90, 100])


def ticket_trend_score(ratio, resolve_h):
    if ratio is None:
        base = 88.0
    else:
        # 工单下降/持平高分；增长50%=60；翻倍=40；3倍=25
        base = interp(ratio, [0.5, 0.8, 1.0, 1.5, 2.0, 3.0], [100, 95, 85, 60, 40, 25])
    # 解决时效修正：<=12h不扣分，24h*0.9，48h*0.78，72h*0.6
    factor = interp(resolve_h, [0, 12, 24, 48, 72], [1.0, 1.0, 0.9, 0.78, 0.6])
    return base * factor


def csat_score(v):
    return 80.0 if pd.isna(v) else float(v) / 5 * 100


def tier_of(h):
    return "健康" if h >= 80 else ("关注" if h >= 60 else "高风险")


# ------------------------------------------------------------ 主流程
def main():
    cust = pd.read_csv(DATA / "customers.csv", encoding="utf-8-sig", parse_dates=["signup_date", "contract_end"])
    panel = pd.read_csv(DATA / "monthly_usage.csv", encoding="utf-8-sig", parse_dates=["month"])
    panel = panel.sort_values(["customer_id", "month"]).reset_index(drop=True)
    months = sorted(panel["month"].unique())

    hist_rows, warn_rows = [], []
    case_ids = ["C0007", "C0023", "C0150"]
    case_timeline = {cid: [] for cid in case_ids}

    for cid, g in panel.groupby("customer_id"):
        g = g.sort_values("month").reset_index(drop=True)
        end_month = cust.loc[cust["customer_id"] == cid, "contract_end"].iloc[0]
        end_idx = (pd.Timestamp(end_month).to_period("M") - pd.Timestamp(months[0]).to_period("M")).n
        renewed = cust.loc[cust["customer_id"] == cid, "renewal_status"].iloc[0]
        logins, tickets, resolves, csats = (g["logins"].to_numpy(float), g["tickets"].to_numpy(float),
                                            g["avg_resolve_hours"].to_numpy(float), g["csat"].to_numpy(float))
        for j in range(len(g)):
            r = g.iloc[j]
            in_service = not (renewed == "未续约" and j > end_idx)
            util = r["active_accounts"] / r["seats"] if r["seats"] else 0
            s_use = usage_score(util, r["active_modules"])
            # 近3月 vs 前3月
            if j >= 5:
                cur, prev = slice(j - 2, j + 1), slice(j - 5, j - 2)
            else:
                cur, prev = slice(0, j + 1), slice(0, max(1, j - 1))
            lr = trend_ratio(logins[cur], logins[prev])
            tr = trend_ratio(tickets[cur], tickets[prev])
            res_in = resolves[cur][tickets[cur] > 0]
            resolve_h = float(np.mean(res_in)) if len(res_in) else 0.0
            csat_in = csats[cur][~np.isnan(csats[cur])]
            csat_v = float(np.mean(csat_in)) if len(csat_in) else np.nan
            s_login = login_trend_score(lr)
            s_ticket = ticket_trend_score(tr, resolve_h)
            s_csat = csat_score(csat_v)
            health = round(WEIGHTS["usage"] * s_use + WEIGHTS["login_trend"] * s_login
                           + WEIGHTS["ticket_trend"] * s_ticket + WEIGHTS["csat"] * s_csat, 1)
            tier = tier_of(health)
            hist_rows.append(dict(customer_id=cid, month=pd.Timestamp(months[j]).date(),
                                  in_service=in_service, usage_score=round(s_use, 1),
                                  login_score=round(s_login, 1), ticket_score=round(s_ticket, 1),
                                  csat_score=round(s_csat, 1), health=health, tier=tier))

            # -------- 预警判定（仅在服务期内；各规则按自身回溯窗口独立判定）
            if in_service and j >= 2:
                ev = []
                # R1 连续2月环比下降>30%
                if (logins[j - 2] >= 20 and logins[j - 1] / logins[j - 2] < 0.7
                        and logins[j] / logins[j - 1] < 0.7 and health < 60):
                    d1, d2 = 1 - logins[j - 1] / logins[j - 2], 1 - logins[j] / logins[j - 1]
                    ev.append(("R1", "高",
                               f"登录连续两月环比下降 {d1*100:.0f}%/{d2*100:.0f}%（{int(logins[j-2])}→{int(logins[j-1])}→{int(logins[j])}次），健康度{health}"))
                # R2 工单风暴（工单/时长/CSAT 统一使用近2月口径；j>=3 才有完整前2月窗口）
                if j >= 3:
                    tk_cur, tk_prev = np.mean(tickets[j - 1:j + 1]), np.mean(tickets[j - 3:j - 1])
                    res2 = resolves[j - 1:j + 1][tickets[j - 1:j + 1] > 0]
                    res2 = float(np.mean(res2)) if len(res2) else 0.0
                    cs_recent = csats[j - 1:j + 1]
                    cs_recent = cs_recent[~np.isnan(cs_recent)]
                    if tk_prev >= 1 and tk_cur / tk_prev >= 1.5 and res2 > 24 and len(cs_recent) and np.mean(cs_recent) < 3.5:
                        ev.append(("R2", "高",
                                   f"近2月工单较前2月增长 {(tk_cur/tk_prev-1)*100:.0f}%（{tk_prev:.1f}→{tk_cur:.1f}张/月），平均解决{res2:.0f}h，CSAT {np.mean(cs_recent):.1f}"))
                # R3 连续3月活跃萎缩
                u3 = g["active_accounts"].to_numpy(float)[j - 2:j + 1] / r["seats"]
                m3 = g["active_modules"].to_numpy(float)[j - 2:j + 1]
                if (np.all(u3 < 0.5) or np.all(m3 <= 2)):
                    ev.append(("R3", "低",
                               f"连续3月席位利用率 {u3.min()*100:.0f}%~{u3.max()*100:.0f}%、使用模块 {int(m3.min())}~{int(m3.max())} 个"))
                # R4 续约窗口：相对评估当月，60天内到期
                days_to_end = (pd.Timestamp(end_month) - pd.Timestamp(months[j])).days
                if 0 <= days_to_end <= 60 and health < 70:
                    ev.append(("R4", "中",
                               f"合同 {pd.Timestamp(end_month).date()} 到期（剩{days_to_end}天），健康度{health}"))
                for rid, sev, desc in ev:
                    warn_rows.append(dict(customer_id=cid, month=pd.Timestamp(months[j]).date(),
                                          rule_id=rid, severity=sev, evidence=desc, health=health))
            if cid in case_timeline:
                case_timeline[cid].append(dict(
                    month=pd.Timestamp(months[j]).strftime("%Y-%m"), logins=int(logins[j]),
                    tickets=int(tickets[j]), resolve=round(float(resolve_h), 1),
                    csat=None if np.isnan(csat_v) else round(csat_v, 1),
                    active_modules=int(r["active_modules"]),
                    util=round(util, 3), health=health, tier=tier, in_service=in_service))

    hist = pd.DataFrame(hist_rows)
    warns = pd.DataFrame(warn_rows)
    hist.to_csv(OUT / "health_history.csv", index=False, encoding="utf-8-sig")
    warns.to_csv(OUT / "warning_events.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------ 当前快照
    last_month = pd.Timestamp(months[-1])
    cur_hist = hist[hist["month"] == last_month.date()][["customer_id", "health", "tier"]]
    snap = cust.merge(cur_hist, on="customer_id", how="left")
    latest = panel[panel["month"] == last_month].drop(columns=["month"])
    snap = snap.merge(latest, on="customer_id", how="left", suffixes=("", "_m"))
    snap.to_csv(OUT / "customer_health_current.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------ 干预动作清单
    ACTIONS = {
        "R1": ["48小时内客户成功总监+销售VP高层回访", "CSM 24小时内电话诊断流失根因",
               "定制挽救方案（赠送服务期/专属培训包）", "每周跟进直至健康度回升至70"],
        "R2": ["技术支持专家48小时内介入排查根因", "开通应急通道（响应SLA提升至4小时）",
               "针对高频报错安排1v1操作培训", "2周内复盘工单解决率与CSAT"],
        "R3": ["下发30天激活任务（每周2个核心功能）", "报名直播培训并布置课后实操任务",
               "双周巡检使用数据并电话辅导", "识别并培养客户内部推动人(champion)"],
        "R4": ["立即生成《客户价值回顾报告》并预约续约沟通", "销售+CSM联合续约拜访，提前处理异议",
               "匹配增值方案/老客续约权益", "按异议处理FAQ应对价格与竞品说辞"],
    }
    sev_rank = {"高": 0, "中": 1, "低": 2}
    cur_warn = warns[warns["month"] == last_month.date()].copy()
    rows = []
    for cid, w in cur_warn.groupby("customer_id"):
        top = w.sort_values("severity", key=lambda s: s.map(sev_rank)).iloc[0]
        srow = snap[snap["customer_id"] == cid].iloc[0]
        rules = "、".join(sorted(w["rule_id"].unique()))
        actions = []
        for rid in w.sort_values("severity", key=lambda s: s.map(sev_rank))["rule_id"].unique():
            actions += ACTIONS[rid][:2]
        evidence = " | ".join(w.sort_values("severity", key=lambda s: s.map(sev_rank))["evidence"].tolist())
        rows.append(dict(customer_id=cid, industry=srow["industry"], plan=srow["plan"],
                         account_manager=srow["account_manager"], health=float(top["health"]),
                         tier=srow["tier"], rules=rules, top_rule=top["rule_id"],
                         severity=top["severity"], evidence=top["evidence"], all_evidence=evidence,
                         actions="；".join(dict.fromkeys(actions)),
                         renewal_status=srow["renewal_status"],
                         contract_end=str(srow["contract_end"].date()),
                         annual_amount=int(srow["annual_amount"])))
    interv = pd.DataFrame(rows)
    interv["sev_order"] = interv["severity"].map(sev_rank)
    interv = interv.sort_values(["sev_order", "health"]).drop(columns="sev_order").reset_index(drop=True)
    interv.insert(0, "rank", range(1, len(interv) + 1))
    interv.to_csv(OUT / "intervention_list.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------ 指标汇总
    active = snap[snap["renewal_status"] != "未续约"]
    due = snap[snap["renewal_status"] != "待续约"]
    tier_counts = active["tier"].value_counts().to_dict()
    # 月度续约率（按到期队列）
    renew_trend = []
    for m in pd.date_range("2025-03-01", "2025-08-01", freq="MS"):
        d = due[due["contract_end"].dt.to_period("M") == m.to_period("M")]
        renew_trend.append(dict(month=m.strftime("%Y-%m"), due=int(len(d)),
                                renewed=int((d["renewal_status"] == "已续约").sum()),
                                rate=round(100 * (d["renewal_status"] == "已续约").mean(), 1) if len(d) else None))
    # 健康度月度趋势（在服客户均值）
    htrend = (hist[hist["in_service"]].groupby("month")["health"].mean().round(1).reset_index())
    health_trend = [dict(month=pd.Timestamp(m).strftime("%Y-%m"), health=float(h))
                    for m, h in zip(htrend["month"], htrend["health"])]
    # 行业 x 分层
    ind = []
    for name, ig in active.groupby("industry"):
        tc = ig["tier"].value_counts().to_dict()
        ind.append(dict(industry=name, total=int(len(ig)),
                        healthy=int(tc.get("健康", 0)), watch=int(tc.get("关注", 0)),
                        risk=int(tc.get("高风险", 0)), avg_health=round(float(ig["health"].mean()), 1)))
    ind = sorted(ind, key=lambda x: -x["risk"])
    arr_at_risk = int(interv[interv["severity"].isin(["高", "中"])]["annual_amount"].sum())

    summary = dict(
        kpi=dict(total_customers=500,
                 active=int(len(active)),
                 due=int(len(due)), renewed=int((due["renewal_status"] == "已续约").sum()),
                 churned=int((due["renewal_status"] == "未续约").sum()),
                 pending=int((snap["renewal_status"] == "待续约").sum()),
                 renewal_rate=round(100 * (due["renewal_status"] == "已续约").mean(), 1),
                 avg_health=round(float(active["health"].mean()), 1),
                 tier_healthy=int(tier_counts.get("健康", 0)),
                 tier_watch=int(tier_counts.get("关注", 0)),
                 tier_risk=int(tier_counts.get("高风险", 0)),
                 warning_red=int((interv["severity"] == "高").sum()),
                 warning_orange=int((interv["severity"] == "中").sum()),
                 warning_yellow=int((interv["severity"] == "低").sum()),
                 arr_at_risk=arr_at_risk),
        renewal_trend=renew_trend, health_trend=health_trend, industry=ind,
        weights=WEIGHTS,
        top_warnings=interv.head(15)[["rank", "customer_id", "industry", "plan", "health", "tier",
                                      "top_rule", "severity", "evidence", "actions"]].to_dict("records"),
    )
    with open(OUT / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    with open(OUT / "case_timeline.json", "w", encoding="utf-8") as f:
        json.dump(case_timeline, f, ensure_ascii=False, indent=2, default=str)

    print("[OK] health history rows:", len(hist), "| warning events:", len(warns),
          "| current intervention list:", len(interv))
    print("[OK] tier(active):", summary["kpi"]["tier_healthy"], summary["kpi"]["tier_watch"], summary["kpi"]["tier_risk"])
    print("[OK] warnings red/orange/yellow:", summary["kpi"]["warning_red"],
          summary["kpi"]["warning_orange"], summary["kpi"]["warning_yellow"])
    print("[OK] renewal rate: %.1f%% | avg health: %.1f | ARR at risk: %d"
          % (summary["kpi"]["renewal_rate"], summary["kpi"]["avg_health"], arr_at_risk))


if __name__ == "__main__":
    main()
