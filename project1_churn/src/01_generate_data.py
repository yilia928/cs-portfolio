# -*- coding: utf-8 -*-
"""
项目① W1：模拟数据集生成
=====================================================================
- 500 家跨境电商卖家客户 x 12 个月（2024-09 ~ 2025-08）面板数据
- 客户主数据：客户ID/行业/规模/版本/签约日期/席位数/合同金额/合同到期日/续约状态
- 月度行为：登录次数、活跃账号数、工单数、工单解决时长、CSAT、使用模块数
- 埋入 6 类可讲述的行为模式（健康稳定/登录骤降/工单风暴/温水衰退/突发流失/预警挽回）
- 固定随机种子，结果可复现；输出 UTF-8 CSV，控制台仅打印 ASCII
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------- 配置
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

N = 500
MONTHS = pd.date_range("2024-09-01", "2025-08-01", freq="MS")
M = len(MONTHS)
ANALYSIS_DATE = pd.Timestamp("2025-09-01")
TOTAL_MODULES = 12  # ERP 全部 12 个功能模块

INDUSTRIES = ["3C电子", "服饰鞋包", "家居园艺", "美妆个护", "户外运动", "母婴玩具", "汽摩配件", "工业品"]
IND_P = [0.20, 0.24, 0.14, 0.12, 0.10, 0.08, 0.07, 0.05]
SIZES = ["小微卖家", "中型卖家", "大卖/品牌方"]
PLANS = ["基础版", "专业版", "旗舰版"]
AMS = ["林一", "陈璐", "周衡", "苏芮", "黄炬"]

# 六种行为模式及占比
ARCH_LABELS = (
    ["healthy"] * 350      # 健康稳定/增长
    + ["login_cliff"] * 40  # 登录连续骤降
    + ["ticket_storm"] * 35 # 工单激增、体验恶化
    + ["gradual_decline"] * 40  # 温水煮青蛙式衰退
    + ["sudden_churn"] * 20 # 末期突发流失
    + ["recovered"] * 15    # 中途遇阻、被预警后挽回（案例素材）
)
assert len(ARCH_LABELS) == N


def build_archetypes():
    labels = np.array(ARCH_LABELS)
    rng.shuffle(labels)
    # 强制 3 个案例客户的模式（同时保证整体数量不变）
    def force(idx, target):
        old = labels[idx]
        labels[idx] = target
        swap = np.where(labels == target)[0]
        swap = swap[swap != idx]
        if len(swap):
            labels[swap[0]] = old
    force(6, "recovered")       # C0007 高风险被挽回
    force(22, "ticket_storm")   # C0023 工单风暴后稳住续约
    force(149, "sudden_churn")  # C0150 预警过晚、最终流失
    return labels


def gen_customers(labels):
    """客户主数据"""
    rows = []
    # 到期队列：2025-03 ~ 2025-08 到期的客户在分析日(09-01)前续约结论已定
    due_months = pd.date_range("2025-03-01", "2025-08-01", freq="MS")
    for i in range(N):
        cid = f"C{i+1:04d}"
        industry = rng.choice(INDUSTRIES, p=IND_P)
        size = rng.choice(SIZES, p=[0.45, 0.40, 0.15])
        plan = rng.choice(PLANS, p=[0.45, 0.40, 0.15])
        # 席位数与版本/规模挂钩
        seats = int({"小微卖家": (5, 20), "中型卖家": (20, 60), "大卖/品牌方": (60, 160)}[size][0]
                    + rng.integers(0, 15 if size == "小微卖家" else 40 if size == "中型卖家" else 100))
        unit_price = {"基础版": 120, "专业版": 220, "旗舰版": 360}[plan]
        annual_amount = seats * unit_price
        signup = pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 900)))
        # 合同起始日 2024 年某月，到期日落在 2025-03~08（到期队列）或 2025-09~12（未到期）
        if i < 380:
            end_month = due_months[rng.integers(0, len(due_months))]
        else:
            end_month = pd.date_range("2025-09-01", "2025-12-01", freq="MS")[rng.integers(0, 4)]
        end_day = int(rng.choice([28, 28, 28, 15, 20]))
        end_date = (end_month + pd.offsets.MonthEnd(0)).replace(day=end_day) \
            if end_day == 28 else end_month + pd.Timedelta(days=end_day - 1)
        # 固定三个案例客户的到期日（保证复盘故事时间线成立）
        case_end = {"C0007": "2025-07-28", "C0023": "2025-07-20", "C0150": "2025-08-20"}
        if cid in case_end:
            end_date = pd.Timestamp(case_end[cid])
        rows.append(dict(
            customer_id=cid, industry=industry, company_size=size, plan=plan,
            seats=seats, annual_amount=annual_amount, signup_date=signup.date(),
            contract_end=pd.Timestamp(end_date).date(), account_manager=rng.choice(AMS),
            archetype=labels[i],
        ))
    df = pd.DataFrame(rows)

    # ---- 续约结论：依据末期健康度 + 模式（在月度数据生成后最终回填，这里先置空）
    df["renewal_status"] = "待续约"
    return df


def trajectory_multipliers(arch, cid=None):
    """返回 12 个月上的行为乘数列：util/adoption/login_noise/ticket/csat/resolve"""
    # 案例② C0023：工单风暴致体验恶化 -> 技术介入 -> 稳住续约（专属轨迹）
    if cid == "C0023":
        return (
            np.array([0.82, 0.82, 0.80, 0.72, 0.62, 0.55, 0.58, 0.62, 0.65, 0.67, 0.69, 0.70]),
            np.full(M, 0.66),
            np.array([1.0, 1.0, 1.0, 0.85, 0.85, 0.85, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0]),
            np.array([0.8, 0.9, 0.9, 2.6, 3.1, 3.2, 1.6, 1.1, 0.9, 0.9, 0.85, 0.85]),
            np.array([1.0, 1.0, 1.0, 1.9, 2.4, 2.0, 1.3, 1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([4.4, 4.4, 4.3, 3.6, 3.1, 2.9, 3.2, 3.6, 3.9, 4.0, 4.1, 4.1]),
        )
    t = np.arange(M)
    util = np.full(M, 0.75)          # 席位使用率基线
    adoption = np.full(M, 0.62)      # 模块采用率基线
    login_pu = np.full(M, 1.0)       # 人均登录次数波动
    ticket_m = np.full(M, 1.0)       # 工单量乘子
    resolve_m = np.full(M, 1.0)      # 解决时长乘子
    csat = np.full(M, 4.3)           # CSAT 基线

    if arch == "healthy":
        trend = rng.choice([0.0, 0.004, 0.008])
        util = 0.72 + trend * t + rng.normal(0, 0.03, M)
        adoption = np.clip(0.60 + 0.012 * t + rng.normal(0, 0.02, M), 0, 1)
        login_pu = rng.normal(1.0, 0.06, M)
        ticket_m = np.clip(rng.normal(0.9, 0.12, M), 0.5, 1.3)
        csat = np.clip(rng.normal(4.4, 0.25, M), 3.6, 5.0)
    elif arch == "login_cliff":
        onset = int(rng.integers(5, 10))
        util = 0.75 + rng.normal(0, 0.03, M)
        adoption = 0.6 + rng.normal(0, 0.02, M)
        drop = rng.uniform(0.38, 0.5)
        for k in range(onset, M):
            util[k] *= drop ** (k - onset + 1) * (1 / 0.55)
            adoption[k] -= 0.08 * (k - onset + 1)
            login_pu[k] *= (0.62 if k == onset else 0.55)
        ticket_m[onset:] = [1.3, 1.8, 2.2, 2.4, 2.4, 2.4, 2.4][: M - onset]
        resolve_m[onset:] = 1.6
        csat[onset:] = np.linspace(3.9, 2.6, M - onset)
    elif arch == "ticket_storm":
        onset = int(rng.integers(4, 9))
        util[:onset] = 0.78 + rng.normal(0, 0.03, onset)
        util[onset:] = np.linspace(0.78, 0.5, M - onset)
        adoption = np.full(M, 0.66)
        ticket_m[onset:] = rng.uniform(2.2, 3.2, M - onset)
        resolve_m[onset:] = np.linspace(1.0, 2.6, M - onset)
        csat[onset:] = np.linspace(4.2, 2.4, M - onset)
        login_pu[onset:] = 0.8
    elif arch == "gradual_decline":
        util = np.linspace(0.8, 0.28, M) + rng.normal(0, 0.02, M)
        adoption = np.linspace(0.66, 0.18, M)
        login_pu = np.linspace(1.0, 0.55, M)
        ticket_m = np.linspace(0.9, 1.6, M)
        csat = np.linspace(4.3, 3.1, M)
    elif arch == "sudden_churn":
        util[:9] = 0.8 + rng.normal(0, 0.03, 9)
        util[9:] = [0.3, 0.12, 0.02]
        adoption[:9] = 0.64
        adoption[9:] = [0.35, 0.18, 0.08]
        login_pu[9:] = [0.4, 0.15, 0.05]
        ticket_m[9:] = [2.6, 1.8, 1.0]
        resolve_m[9:] = 2.2
        csat[9:] = [3.0, 2.3, 2.0]
    elif arch == "recovered":
        # 第 5~7 月骤降，第 8 月起被干预挽回（登录/使用回升、工单回落）
        util = 0.8 + rng.normal(0, 0.025, M)
        util[5:8] = [0.42, 0.3, 0.38]
        util[8:] = [0.55, 0.68, 0.78, 0.82]
        adoption = 0.64 + rng.normal(0, 0.02, M)
        adoption[5:8] = [0.4, 0.32, 0.36]
        login_pu[5:8] = [0.6, 0.5, 0.6]
        ticket_m[5:9] = [2.0, 2.6, 1.8, 1.2]
        resolve_m[5:9] = [1.5, 1.8, 1.3, 1.0]
        csat[5:9] = [3.4, 2.8, 3.3, 4.0]
        csat[:5] = rng.normal(4.4, 0.2, 5)
        csat[9:] = rng.normal(4.5, 0.15, 3)

    util = np.clip(util, 0.01, 1.0)
    adoption = np.clip(adoption, 0.05, 1.0)
    login_pu = np.clip(login_pu, 0.02, 1.4)
    ticket_m = np.clip(ticket_m, 0.2, 3.5)
    csat = np.clip(csat, 1.5, 5.0)
    return util, adoption, login_pu, ticket_m, resolve_m, csat


def gen_monthly(customers):
    """生成 500 x 12 = 6000 行月度面板"""
    records = []
    # 续约概率（依据模式与末期使用），在循环后统一回填主表
    renewal_map = {}
    for _, c in customers.iterrows():
        cid = c["customer_id"]
        seats = int(c["seats"])
        util, adoption, login_pu, ticket_m, resolve_m, csat_line = trajectory_multipliers(c["archetype"], cid)
        # 人均登录基线 6~16 次/月（大卖流程化、人均偏低）
        base_logins_pu = rng.uniform(7, 15)
        # 每席位月工单基线率
        ticket_rate = rng.uniform(0.08, 0.22)
        end_idx = None
        if c["contract_end"] < ANALYSIS_DATE.date():
            end_idx = max(0, (pd.Timestamp(c["contract_end"]).to_period("M") - MONTHS[0].to_period("M")).n)

        # ---------------- 续约结论（先生成、再决定到期后行为）
        if c["contract_end"] >= ANALYSIS_DATE.date():
            renewal = "待续约"
        elif cid in ("C0007", "C0023"):
            renewal = "已续约"
        elif cid == "C0150":
            renewal = "未续约"
        else:
            last_health_proxy = util[-1] * 0.5 + adoption[-1] * 0.3 + (csat_line[-1] / 5) * 0.2
            if c["archetype"] == "sudden_churn":
                p = 0.05
            elif c["archetype"] == "recovered":
                p = 0.92
            elif c["archetype"] == "login_cliff":
                p = 0.25
            elif c["archetype"] == "gradual_decline":
                p = 0.35
            elif c["archetype"] == "ticket_storm":
                p = 0.45
            else:
                p = float(np.clip(0.55 + last_health_proxy * 0.5, 0.6, 0.98))
            renewal = "已续约" if rng.random() < p else "未续约"
        renewal_map[cid] = renewal
        churned = renewal == "未续约"

        for j in range(M):
            month = MONTHS[j]
            # 合同到期且未续约：到期次月起访问回收、使用归零（仅保留极少量残留登录）
            if churned and j > end_idx:
                active_acc = max(1, int(seats * rng.uniform(0.0, 0.04))) if j == end_idx + 1 else 0
                records.append(dict(
                    customer_id=cid, month=month.date(), logins=0, active_accounts=active_acc,
                    seats=seats, tickets=0, avg_resolve_hours=0.0, csat=np.nan,
                    active_modules=0,
                ))
                continue
            active_acc = int(round(seats * util[j]))
            logins = int(round(active_acc * base_logins_pu * login_pu[j]))
            modules = int(round(np.clip(adoption[j], 0.03, 1) * TOTAL_MODULES))
            # 业务旺季(10-12 月)工单略增
            season = 1.15 if month.month in (10, 11, 12) else 1.0
            tickets = int(rng.poisson(seats * ticket_rate * ticket_m[j] * season))
            if tickets > 0:
                resolve_h = float(np.clip(rng.lognormal(mean=2.3, sigma=0.45) * resolve_m[j], 0.5, 160))
                # CSAT 仅在有已关闭工单的月份出现，约 85% 回收
                csat_v = float(round(csat_line[j] + rng.normal(0, 0.12), 1)) if rng.random() < 0.85 else np.nan
            else:
                resolve_h, csat_v = 0.0, np.nan
            records.append(dict(
                customer_id=cid, month=month.date(), logins=logins, active_accounts=active_acc,
                seats=seats, tickets=tickets,
                avg_resolve_hours=round(resolve_h, 1), csat=csat_v, active_modules=modules,
            ))

    panel = pd.DataFrame(records)
    return panel, renewal_map


def main():
    labels = build_archetypes()
    customers = gen_customers(labels)
    panel, renewal_map = gen_monthly(customers)
    customers["renewal_status"] = customers["customer_id"].map(renewal_map)

    customers.to_csv(DATA_DIR / "customers.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(DATA_DIR / "monthly_usage.csv", index=False, encoding="utf-8-sig")

    due = customers[customers["renewal_status"] != "待续约"]
    print("[OK] customers:", len(customers), "| panel rows:", len(panel))
    print("[OK] renewal cohort:", len(due),
          "| renewed:", (due["renewal_status"] == "已续约").sum(),
          "| renewal rate: %.1f%%" % (100 * (due["renewal_status"] == "已续约").mean()))
    print("[OK] archetype counts:")
    print(customers["archetype"].value_counts().to_string())
    print("[OK] ->", DATA_DIR)


if __name__ == "__main__":
    main()
