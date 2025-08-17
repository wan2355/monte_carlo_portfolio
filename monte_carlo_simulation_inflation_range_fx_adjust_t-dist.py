#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## 解説
#{{{
"""
Monte Carlo simulation with:
- FX sensitivity
- Inflation (fixed / per_path / per_year)
- Withdrawals (inflation-indexed)
- Yearly snapshots (hist option)
- Sweep across inflation rates
- Random seed for reproducibility
- Writes 95% CI (Wilson) and SE of ruin probability into sweep_summary.csv
- Optional auto overlay plot after sweep (ruin + median + mode), with error bars if available
- Python 3.7+ 互換

Quick examples
--------------
# 1) Sweep + auto overlay plot.
# inflation_min ~ max の間を inflation_sweep_stepで分割して、その値のinflationで固定して最後まで計算.
# 1年ごとのhistgram の snapshotなし。
#
# eg) step 5 ならば,以下のinふlationで計算。
#  0.02 0.0225 0.025 0.0275 0.03

## t分布 : for conte_carlo
##  t_df : jsonで設定した各資産の標準偏差を使った月初の資産のゆらぎ
##  fx_t_df : 為替のゆらぎ。
##   use 5~8 , lower num is strong.

## add comment
##  -z : comment

sample)
----------
## intial asset : 10_000_000
## running 30yrs
## asset info : portfolio.json_custom, (defalt : portfolio.json)
## start withdraw 3years later
## annual_withdraw=50万/year
## set inflation rate 0.02, 0.03/yr, and calc & compare.
## use t-dist for asset & fx for monte carlo. 
##  --t_df : 5~8 , lower num is strong.
##
python xx3_monte_carlo_simulation_inflation_range_fx_adjust_t-dist.py \
  -i 10_000_000 -y 30 --withdraw_start_years 3  -p portfolio.json_custom \
  --inflation_min 0.02 --inflation_max 0.03 --inflation_sweep_steps 2 \
  -w 2_000_000 \
  --t_df 6 \
  --fx_t_df 6 \
  -z wd3_comp_if2-3_wd50man_i1000man_外貨補正0.10


# 2) Single run with yearly snapshot histograms every year, using random num(t-dist).
##  --inflation_min 0.02 --inflation_max 0.03 は「0.02〜0.03の乱数」で1回/yrの実行を行う.
##
## intial asset : 10_000_000
## running 30yrs
## asset info : portfolio.json_custom, (defalt : portfolio.json)
## start withdraw 3years later
## annual_withdraw=50万/year
## set inflation rate 0.02, 0.03/yr, but 「0.02〜0.03の乱数」で1回/yrの実行を行う.
## use t-dist for asset & fx for monte carlo.
##  --t_df : 5~8 , lower num is strong.
##

python xx3_monte_carlo_simulation_inflation_range_fx_adjust_t-dist.py \
  -i 10_000_000 -y 30 -p portfolio.json \
  --inflation_mode per_year --inflation_min 0.02 --inflation_max 0.03 \
  --annual_withdrawal 500_000 \
  --save_snapshot_hists --mode_interval_years 1 \
  -z wd3_if3.0_wd50man \
  --t_df 6


# 3) fix_inflation rate and save hist every year.
##  --inflation_mode fixed --inflation_rate X で Xで固定、実行。
##
## intial asset : 10_000_000
## running 30yrs
## asset info : portfolio.json_custom, (defalt : portfolio.json)
## start withdraw 3years later
## annual_withdraw=50万/year
## set inflation rate 0.03/yr
## use t-dist for asset & fx for monte carlo.
##  --t_df : 5~8 , lower num is strong.

python xx3_monte_carlo_simulation_inflation_range_fx_adjust_t-dist.py \
  -i 10_000_000 -y 30 --withdraw_start_years 3 -p portfolio.json_hatano_fundwrap \
  --inflation_rate 0.03 \
  --annual_withdrawal 500_000 \
  --save_snapshot_hists \
  -z wd3_if3.0_wd50man \
  --t_df 6
  --fx_t_df 6

## 以下は　default : 正規分布.
## t_df : jsonで設定した各資産の標準偏差を使った月初の資産のゆらぎ
## fx_t_df : 為替のゆらぎ。

python xx3_monte_carlo_simulation_inflation_range_fx_adjust_t-dist.py \
        -i 10_000_000 -y 30 --withdraw_start_years 3 -p portfolio.json \
        --save_snapshot_hists --inflation_rate 0.03 \
        -z wd3_comp_if3_wd50man_i1000man_外貨補正0.10_t6 \
        -w 500_000 \
        --t_df 6 \
        --fx_t_df 6

///////////////////////////////////////////////////////////////////////
引数の説明

## (2-1) コア・実行制御

--simulation, -s      : int（例 10000） モンテカルロ試行数
--initial_asset, -i   : float           初期資産（JPY）
--years, -y           : int             運用年数
--portfolio, -p       : str             ポートフォリオJSONのパス
--seed                : int or None     乱数シード（再現性）
--early_terminate_if_ruined : flag      破綻（資産<=0）でその経路を打ち切り
---------------------------------------------------------------

## (2-2) 取り崩し・スナップショット

--annual_withdrawal, -w : float   初年度取り崩し額（名目, JPY）
--withdraw_start_years  : int     取り崩し開始年（0=初年）
--save_snapshot_hists   : flag    途中年ヒストグラムを年次出力
--mode_interval_years   : int     ヒスト/モード抽出間隔（年, 既定=1）
--------------------------------------------------------

## (2-3) インフレ（inflation）

--inflation_rate        : float   単発実行時の年率インフレ
--inflation_min/max     : float   スイープの下限/上限（年率）
--inflation_sweep_steps : int     スイープ分割数（等間隔）
 考: 取り崩し額は毎年「前年取り崩し×(1+inflation)」で増額。
 ポート上の資産は実質換算（インフレ影響を調整）で比較。
---------------------------

## (2-4) 資産リターンの分布（t分布）

--t_df : float or None  df>2 で Student-t を使用（df 小ほど裾が厚い）。
 指定/<=2 は正規分布。
-------------

# (2-5) 為替（FX：毎月変動＋初期レベル差）

--base_fx_rate          : float   基準レート（JPY/USD想定, 例 147）
--fx_min / --fx_max     : float   経路ごとに初期FXレートを一様分布で1回サンプル
--fx_annual_vol         : float   年率ボラ（月次は /√12）
--fx_drift_annual       : float   年率ドリフト（月次は 12乗根で換算）
--fx_t_df               : float   月次FXにも t分布を使用（df>2）。未指定は正規
 考: 外貨判定は資産名に "foreign" または "usd" を含む場合に適用。
-------------------------------------------

## (2-6) 結果保存・後処理

（自動）結果ディレクトリ : results/res_{years}yr_{YYMMDD_HHMM} 形式で作成
-z                        : str     保存ディレクトリ名に任意タグ付与
--post_plot_multi         : flag    スイープ後に外部プロッタを自動起動（既定ON）
--no-post-plot-multi      : flag    上記の無効化
--post_plot_call_multi    : str     図スクリプトのパス（既定: yy1_plot_sweep_multi_v5.py）
--post_plot_errorbars     : flag    エラーバー描画をプロッタへ指示（既定ON）
-----------------------------------------------------------

"""
# }}}

## module
#{{{
import os
import re
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import sys
import random
import subprocess
from typing import Optional, List, Tuple
# }}}

# ----------------- Helpers -----------------

def approx_mode(values: np.ndarray, bins: int = 100) -> Tuple[float, int]:
    freq, edges = np.histogram(values, bins=bins)
    i = int(np.argmax(freq))
    return float(edges[i]), int(freq[i])

def save_histogram(values: np.ndarray, title: str, out_path: str,
                   x_min: Optional[float], x_max: Optional[float],
                   annotate_text: Optional[str]) -> None:
#{{{
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.grid(True)
    ax.set_title(title, fontsize=16)
    ax.hist(values, bins=100, edgecolor="black")
    ax.set_xlabel("Asset Value (real JPY, base-year)")
    ax.set_ylabel("Frequency")
    if (x_min is not None) or (x_max is not None):
        xmin = x_min if x_min is not None else ax.get_xlim()[0]
        xmax = x_max if x_max is not None else ax.get_xlim()[1]
        if xmin < xmax:
            ax.set_xlim(xmin, xmax)
    if annotate_text:
        ax.text(0.985, 0.985, annotate_text, fontsize=12, ha="right", va="top",
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
# }}}

def summarize_snapshots(snapshots: np.ndarray, capture_months: List[int]) -> pd.DataFrame:
#{{{
    rows = []
    for j, m in enumerate(capture_months):
        arr = snapshots[:, j]
        mode_val, _ = approx_mode(arr, bins=100)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        q05 = float(np.quantile(arr, 0.05))
        q25 = float(np.quantile(arr, 0.25))
        q75 = float(np.quantile(arr, 0.75))
        q95 = float(np.quantile(arr, 0.95))
        share_negative = float((arr < 0).mean())
        year = (m + 1) // 12
        rows.append({
            "year": year,
            "mode": mode_val, "mean": mean_val, "median": median_val,
            "p05": q05, "p25": q25, "p75": q75, "p95": q95,
            "share_negative": round(share_negative, 3)
        })
    df = pd.DataFrame(rows)
    for col in ["mode", "mean", "median", "p05", "p25", "p75", "p95"]:
        df[col] = np.rint(df[col]).astype(np.int64)
    return df
# }}}

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z / denom * np.sqrt(max(p * (1 - p) / n + z2 / (4 * n * n), 0.0))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return float(lo), float(hi)

def try_run_external_multi_plotter(results_dir: str,
                                   percent: bool,
                                   y1max: Optional[float],
                                   y2max: Optional[float],
                                   asset_divisor: float,
                                   asset_label: str,
                                   right_margin: float,
                                   legend_anchor_x: float,
                                   legend_anchor_y: float,
                                   plotter_path: Optional[str],
                                   errorbars: bool) -> bool:
    """
    Try to run plot_sweep_multi_v5.py (or v4) via subprocess. Return True if succeeded.
    Prefers v5 (supports --errorbars). Falls back to v4 if v5 not found.
    """
    candidates = []
    if plotter_path:
        candidates.append(plotter_path)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.extend([
            os.path.join(script_dir, "plot_sweep_multi_v5.py"),
            os.path.join(script_dir, "plot_sweep_multi_v4.py"),
        ])
    except Exception:
        pass
    candidates.extend([
        os.path.join(os.getcwd(), "plot_sweep_multi_v5.py"),
        os.path.join(os.getcwd(), "plot_sweep_multi_v4.py"),
    ])

    for candidate in candidates:
        if os.path.isfile(candidate):
            cmd = [sys.executable, candidate, "--sum_res_d", results_dir,
                   "--plot_median", "--plot_mode",
                   "--asset_divisor", str(asset_divisor),
                   "--asset_label", asset_label,
                   "--right_margin", str(right_margin),
                   "--legend_anchor_x", str(legend_anchor_x),
                   "--legend_anchor_y", str(legend_anchor_y)]
            if percent:
                cmd.append("--percent")
            if y1max is not None:
                cmd.extend(["--y1max", str(y1max)])
            if y2max is not None:
                cmd.extend(["--y2max", str(y2max)])
            if errorbars and "v5" in os.path.basename(candidate):
                cmd.append("--errorbars")
            try:
                print("Invoking overlay plotter:", " ".join(cmd))
                subprocess.check_call(cmd)
                return True
            except Exception as e:
                print("Failed running overlay plotter:", candidate, "->", repr(e))
    return False


## 資産の外貨判定
## portfolio.json中の資産名にusd, foreignを含めば外貨と判定。
def is_foreign_asset(name: str, data: dict) -> bool:
    """
    外貨判定:
      - 資産名に 'foreign' を含む
      - 資産名に 'usd' という単語を含む（'used' に誤反応しないよう単語境界で判定）
      - （任意）portfolio側で ccy='USD' が指定されていれば外貨とみなす
    """
    name_l = name.lower()
    if "foreign" in name_l:
        return True
    if re.search(r"\busd\b", name, flags=re.IGNORECASE):
        return True
    return False

# ----------------- Simulation core -----------------

def run_simulation(
    n_sim: int,
    initial_asset: float,
    years: int,
    base_fx_rate: float,
    fx_min: float,
    fx_max: float,
    portfolio: dict,
    annual_withdrawal: float,
    mode_interval_years: int,
    withdraw_start_years: int,
    withdraw_stop_min_asset: Optional[float],
    early_terminate_if_ruined: bool,
    inflation_mode: str,
    inflation_rate_fixed: float,
    inflation_min: Optional[float],
    inflation_max: Optional[float],
    ## 8/16 t-dist
    t_df: Optional[float],
    ## 8/16 fx option
    fx_annual_vol: float,
    fx_drift_annual: float,
    fx_t_df: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:

    n_months = years * 12

    # Capture months for snapshots
    capture_months: List[int] = []
    for k in range(1, years + 1):
        if k % mode_interval_years == 0:
            capture_months.append(k * 12 - 1)

    final_assets = np.empty(n_sim, dtype=float)
    ruin_flags = np.zeros(n_sim, dtype=bool)
    snapshots = np.empty((n_sim, len(capture_months)), dtype=float)

    start_month = max(0, int(withdraw_start_years) * 12)

    # Yearly inflation path
    def sample_yearly_inflation_rates():
        if inflation_mode == "fixed":
            return np.full(years, inflation_rate_fixed, dtype=float)
        elif inflation_mode == "per_path":
            r = np.random.uniform(inflation_min, inflation_max)
            return np.full(years, r, dtype=float)
        elif inflation_mode == "per_year":
            return np.random.uniform(inflation_min, inflation_max, size=years).astype(float)
        else:
            return np.full(years, inflation_rate_fixed, dtype=float)

    for sim in range(n_sim):
        asset = float(initial_asset)
        fx_rate = np.random.uniform(fx_min, fx_max)

        fx_level_factor = fx_rate / base_fx_rate  # ← レベル差（倍率）

        # Monthly FX parameters from annuals
        mu_fx_m = (1.0 + fx_drift_annual) ** (1.0/12.0) - 1.0
        sigma_fx_m = fx_annual_vol / np.sqrt(12.0)

        ruined = False
        withdrawing = True

        yr_rates = sample_yearly_inflation_rates()
        cum_factors = np.ones(years + 1, dtype=float)
        for y in range(1, years + 1):
            cum_factors[y] = cum_factors[y - 1] * (1.0 + yr_rates[y - 1])

        snap_idx = 0

        for m in range(n_months):
            yidx = m // 12
            monthly_inflation = (1.0 + yr_rates[yidx]) ** (1.0 / 12.0) - 1.0

            # Portfolio monthly return
            monthly_return = 0.0
            for name, data in portfolio.items():
                weight = float(data["weight"])
                mean = float(data["mean_return"]) / 12.0
                std = float(data["std_dev"]) / np.sqrt(12.0)

                # Student-t heavy-tail option (falls back to Normal if t_df is None or <=2)
                if (t_df is not None) and (t_df > 2.0):
                    z = np.random.standard_t(t_df) * np.sqrt((t_df - 2.0) / t_df)  # unit variance scaling
                else:
                    z = np.random.normal(0.0, 1.0)
                ret = mean + std * z

                ## 外貨判定（foreign / usd）
                ## --- FX integration (monthly) ---
                if is_foreign_asset(name, data):
                    # 月次FXリターン（0なら“動かない”）: Normal or t
                    if sigma_fx_m > 0.0:
                        if (fx_t_df is not None) and (fx_t_df > 2.0):
                            z_fx = np.random.standard_t(fx_t_df) * np.sqrt((fx_t_df - 2.0) / fx_t_df)  # unit variance
                        else:
                            z_fx = np.random.normal(0.0, 1.0)
                        r_fx = mu_fx_m + sigma_fx_m * z_fx
                    else:
                        r_fx = 0.0

                    # 初月は「初期レベル差 × 当月リターン」を一発で反映
                    if m == 0:
                        fx_r_eff = fx_level_factor * (1.0 + r_fx) - 1.0
                    else:
                        fx_r_eff = r_fx

                    # 資産と為替の合成リターン： (1+r_asset) * (1+fx_r_eff) - 1
                    ret = (1.0 + ret) * (1.0 + fx_r_eff) - 1.0


                monthly_return += weight * ret

            # Apply nominal return
            asset *= (1.0 + monthly_return)

            # Withdrawal (nominal), escalated yearly by inflation
            if annual_withdrawal > 0 and m >= start_month and withdrawing:
                base_monthly_w = annual_withdrawal / 12.0
                monthly_w = base_monthly_w * cum_factors[yidx]
                asset -= monthly_w

            # Deflate to real terms
            asset /= (1.0 + monthly_inflation)

            if (asset <= 0) and (not ruined):
                ruined = True
                if early_terminate_if_ruined:
                    while snap_idx < len(capture_months):
                        if m <= capture_months[snap_idx]:
                            snapshots[sim, snap_idx] = asset
                        snap_idx += 1
                    break

            if (withdraw_stop_min_asset is not None) and withdrawing and (asset < withdraw_stop_min_asset):
                withdrawing = False

            if (snap_idx < len(capture_months)) and (m == capture_months[snap_idx]):
                snapshots[sim, snap_idx] = asset
                snap_idx += 1

        final_assets[sim] = asset
        ruin_flags[sim] = ruined
        while snap_idx < len(capture_months):
            snapshots[sim, snap_idx] = asset
            snap_idx += 1

    return final_assets, ruin_flags, snapshots, capture_months

# ----------------- Main -----------------

def main():
    parser = argparse.ArgumentParser(description="MC with FX, inflation, withdrawals, snapshots, seed, CI export, and auto overlay plotting (py37+)")
    # Core args
    parser.add_argument('-s', "--simulation", type=int, default=10000)
    parser.add_argument('-i', "--initial_asset", type=float, default=10_000_000)
    parser.add_argument('-y', "--years", type=int, default=20)
    parser.add_argument('-p', "--portfolio_file", type=str, default='portfolio.json')
    parser.add_argument('-w', "--annual_withdrawal", type=float, default=500_000)
    parser.add_argument('-f', "--base_fx_rate", type=float, default=147.0)

    ## fx 
    parser.add_argument("--fx_min", type=float, default=120.0)
    parser.add_argument("--fx_max", type=float, default=160.0)
    # FX monthly process (optional)
    parser.add_argument("--fx_annual_vol", type=float, default=0.10,
                    help="FX annualized volatility (e.g., 0.10 = 10%/yr). 0 disables monthly FX moves.")
    parser.add_argument("--fx_drift_annual", type=float, default=0.0,
                    help="FX annual drift (e.g., 0.0). Typically 0 for long-run neutrality.")
    parser.add_argument("--fx_t_df", type=float, default=None,
                    help="If >2, use Student-t for FX monthly returns; else Normal.")

    # Inflation control
    parser.add_argument("--inflation_rate", type=float, default=0.03, help="Fixed annual inflation (used when mode=fixed)")
    parser.add_argument("--inflation_min", type=float, default=None, help="Lower bound for inflation range")
    parser.add_argument("--inflation_max", type=float, default=None, help="Upper bound for inflation range")
    parser.add_argument("--inflation_mode", type=str, choices=["fixed", "per_path", "per_year"], default="fixed")
    parser.add_argument("--inflation_sweep_steps", type=int, default=0, help="If >0, run fixed-rate sweeps from min..max (inclusive)")

    # Snapshots & plotting
    parser.add_argument("--save_snapshot_hists", action="store_true")
    parser.add_argument("--mode_interval_years", type=int, default=1)
    parser.add_argument("--x_min", type=float, default=-1000000)
    ## make bigger
    #parser.add_argument("--x_min", type=float, default=-100000000)
    parser.add_argument("--x_max", type=float, default=50_000_000)

    # Withdrawal timing & stopping
    parser.add_argument("--withdraw_start_years", type=int, default=3)
    parser.add_argument("--withdraw_stop_min_asset", type=float, default=None)
    parser.add_argument("--early_terminate_if_ruined", action="store_true")

    # Seed & comment
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('-z', "--comment", type=str, default='_', help="add commnet")

    # Heavy tail (Student-t) option
    parser.add_argument("--t_df", type=float, default=None,
                    help="Degrees of freedom for Student-t (>2). If not set or <=2, Normal is used.")

    ## Post-sweep overlay plotting
    ## action="store_true", default=True,  # ← これでフラグ省略でも True になる`
    parser.add_argument("--post_plot_multi", action="store_true", default=True, help="After sweep, auto-run overlay plotter (ruin + median + mode)")
    parser.add_argument("--post_plot_percent", action="store_true", help="Left axis in percent")
    parser.add_argument("--post_plot_y1max", type=float, default=None, help="Left y-axis max")
    parser.add_argument("--post_plot_y2max", type=float, default=None, help="Right y-axis max")
    parser.add_argument("--post_plot_asset_divisor", type=float, default=1_000_000.0, help="Right axis divisor (e.g., 1e6 for million JPY)")
    parser.add_argument("--post_plot_asset_label", type=str, default="Asset (million JPY)", help="Right axis label")
    parser.add_argument("--post_plot_right_margin", type=float, default=0.78, help="Right margin")
    parser.add_argument("--post_plot_legend_anchor_x", type=float, default=1.18, help="Legend bbox x")
    parser.add_argument("--post_plot_legend_anchor_y", type=float, default=0.5, help="Legend bbox y")
    parser.add_argument("--post_plot_call_multi", type=str, default='yy1_plot_sweep_multi.py', help="Path to plot_sweep_multi.py")
    parser.add_argument("--post_plot_errorbars", action="store_true", default=True, help="Ask overlay plotter to draw error bars (if supported)")


    args = parser.parse_args()

    # Seed
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Load portfolio
    with open(args.portfolio_file, "r", encoding="utf-8") as f:
        portfolio = json.load(f)

    # Results dir
    now = dt.datetime.now().strftime("%y%m%d_%H%M")
    base_res_dir = f"results/res_{args.years}yr_{args.comment}_{now}"
    os.makedirs(base_res_dir, exist_ok=True)

    print(now)
    print("res_dir =", base_res_dir)
    print("inflation_mode=", args.inflation_mode, " fixed=", args.inflation_rate,
          " min=", args.inflation_min, " max=", args.inflation_max, " sweep_steps=", args.inflation_sweep_steps)
    if args.seed is not None:
        print("seed =", args.seed)
    print()

    # Save run snapshot
    args_used = {k: getattr(args, k) for k in vars(args)}
    env_info = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }
    with open(os.path.join(base_res_dir, "args_used.json"), "w", encoding="utf-8") as fout:
        json.dump(args_used, fout, ensure_ascii=False, indent=2)
    with open(os.path.join(base_res_dir, "env_info.json"), "w", encoding="utf-8") as fout:
        json.dump(env_info, fout, ensure_ascii=False, indent=2)
    with open(os.path.join(base_res_dir, "portfolio_snapshot.json"), "w", encoding="utf-8") as fout:
        json.dump(portfolio, fout, ensure_ascii=False, indent=2)

    # --------------- Sweep mode ---------------
    if args.inflation_sweep_steps and args.inflation_sweep_steps > 0 and args.inflation_min is not None and args.inflation_max is not None:
        rates = np.linspace(args.inflation_min, args.inflation_max, args.inflation_sweep_steps)
        summary_rows = []
        for r in rates:
            ## Re-seed per step for apples-to-apples comparison
            if args.seed is not None:
                np.random.seed(args.seed)
                random.seed(args.seed)

            sub_dir = os.path.join(base_res_dir, "sweep_inf_{:.4f}".format(r))
            os.makedirs(sub_dir, exist_ok=True)

            ## Persist per-run
            with open(os.path.join(sub_dir, "args_used.json"), "w", encoding="utf-8") as fout:
                args_for_step = dict(args_used)
                args_for_step["inflation_rate"] = float(r)
                args_for_step["inflation_mode"] = "fixed"
                json.dump(args_for_step, fout, ensure_ascii=False, indent=2)
            with open(os.path.join(sub_dir, "env_info.json"), "w", encoding="utf-8") as fout:
                json.dump(env_info, fout, ensure_ascii=False, indent=2)
            with open(os.path.join(sub_dir, "portfolio_snapshot.json"), "w", encoding="utf-8") as fout:
                json.dump(portfolio, fout, ensure_ascii=False, indent=2)

            final_assets, ruin_flags, snapshots, capture_months = run_simulation(
                n_sim=args.simulation,
                initial_asset=args.initial_asset,
                years=args.years,
                base_fx_rate=args.base_fx_rate,
                fx_min=args.fx_min,
                fx_max=args.fx_max,
                portfolio=portfolio,
                annual_withdrawal=args.annual_withdrawal,
                mode_interval_years=max(1, args.mode_interval_years),
                withdraw_start_years=args.withdraw_start_years,
                withdraw_stop_min_asset=args.withdraw_stop_min_asset,
                early_terminate_if_ruined=args.early_terminate_if_ruined,
                inflation_mode="fixed",
                inflation_rate_fixed=float(r),
                inflation_min=args.inflation_min,
                inflation_max=args.inflation_max,
                t_df=args.t_df,
                fx_annual_vol=args.fx_annual_vol,
                fx_drift_annual=args.fx_drift_annual,
                fx_t_df=args.fx_t_df,
            )

            stats = pd.DataFrame(final_assets, columns=["final_asset"]).describe().T
            k_ruin = int(ruin_flags.sum())
            n_sim = int(args.simulation)
            p_hat = k_ruin / n_sim
            se = float(np.sqrt(max(p_hat * (1 - p_hat) / n_sim, 0.0)))
            lo, hi = wilson_ci(k_ruin, n_sim, z=1.96)

            share_final_negative = float((final_assets < 0).mean())
            mode_val, _ = approx_mode(final_assets, bins=100)

            # Save artifacts per rate
            pd.DataFrame({"final_asset": final_assets, "ruined": ruin_flags.astype(int)}).to_csv(
                os.path.join(sub_dir, "asset_ruin.csv"), index=False
            )
            summarize_snapshots(snapshots, capture_months).to_csv(
                os.path.join(sub_dir, "yearly_modes.csv"), index=False
            )
            with open(os.path.join(sub_dir, "summary.txt"), "w", encoding="utf-8") as fout:
                fout.write(stats.to_string()); fout.write("\n")
                fout.write("ruin_probability={:.4f}\n".format(p_hat))
                fout.write("ruin_se={:.6f}\n".format(se))
                fout.write("ruin_ci_wilson_low={:.4f}\n".format(lo))
                fout.write("ruin_ci_wilson_high={:.4f}\n".format(hi))
                fout.write("share_final_negative={:.4f}\n".format(share_final_negative))
                fout.write("mode_approx={:d}\n".format(int(np.rint(mode_val))))

            # Final distribution plot per rate
            title = "Asset sim: {}*10^4 JPY, {}yr, inf={:.4f}".format(int(args.initial_asset/10000), args.years, r)
            annotate = "mode approx {}\\nruin_prob = {:.3f}".format(int(np.rint(mode_val)), p_hat)
            save_histogram(final_assets, title, os.path.join(sub_dir, "final_dist_inf{:.4f}.png".format(r)),
                           args.x_min, args.x_max, annotate)

            # Collect for sweep summary
            summary_rows.append({
                "inflation_rate": float(r),
                "ruin_probability": round(p_hat, 5),
                "ruin_se": se,
                "ruin_ci_low": lo,
                "ruin_ci_high": hi,
                "n_sim": n_sim,
                "share_final_negative": round(share_final_negative, 5),
                "mode": int(np.rint(mode_val)),
                "mean_final": int(np.rint(np.mean(final_assets))),
                "median_final": int(np.rint(np.median(final_assets)))
            })

        # Save sweep summary and run config
        sweep_csv = os.path.join(base_res_dir, "sweep_summary.csv")
        pd.DataFrame(summary_rows).to_csv(sweep_csv, index=False)
        with open(os.path.join(base_res_dir, "run_config.json"), "w", encoding="utf-8") as fout:
            payload = dict(args_used)
            payload.update({
                "timestamp": now,
                "base_res_dir": base_res_dir,
                "env": env_info,
                "portfolio_snapshot": portfolio,
                "sweep_summary": os.path.basename(sweep_csv)
            })
            json.dump(payload, fout, ensure_ascii=False, indent=2)

        print("Saved sweep runs under:", base_res_dir)
        print("Created sweep_summary.csv (with ruin_ci_low/high)")

        # Auto overlay plotting (optional)
        if args.post_plot_multi:
            print("\n[Post-plot] Attempting overlay chart (ruin + median + mode)...")
            ok = try_run_external_multi_plotter(
                base_res_dir,
                percent=args.post_plot_percent,
                y1max=args.post_plot_y1max,
                y2max=args.post_plot_y2max,
                asset_divisor=args.post_plot_asset_divisor,
                asset_label=args.post_plot_asset_label,
                right_margin=args.post_plot_right_margin,
                legend_anchor_x=args.post_plot_legend_anchor_x,
                legend_anchor_y=args.post_plot_legend_anchor_y,
                plotter_path=args.post_plot_call_multi,
                errorbars=args.post_plot_errorbars
            )
            if not ok:
                print("[Post-plot] External overlay plotter not found or failed. Using internal fallback (no error bars).")
                # Internal simple overlay (no error bars)
                df = pd.read_csv(sweep_csv)
                x = df["inflation_rate"].values
                y_prob = df["ruin_probability"].values * (100.0 if args.post_plot_percent else 1.0)
                y_median = df["median_final"].values / args.post_plot_asset_divisor
                y_mode = df["mode"].values / args.post_plot_asset_divisor
                fig = plt.figure(figsize=(9.8, 5.6))
                ax1 = plt.gca()
                line_prob, = ax1.plot(x, y_prob, marker="o", label=("Ruin probability (%)" if args.post_plot_percent else "Ruin probability"))
                ax1.grid(True)
                ax1.set_xlabel("Inflation rate")
                ax1.set_ylabel("Ruin probability" + (" (%)" if args.post_plot_percent else ""))
                if args.post_plot_y1max is not None:
                    ax1.set_ylim(0, args.post_plot_y1max)
                ax2 = ax1.twinx()
                line_median, = ax2.plot(x, y_median, marker="s", label="Median final (scaled)")
                line_mode, = ax2.plot(x, y_mode, marker="^", label="Mode (scaled)")
                ax2.set_ylabel(args.post_plot_asset_label)
                if args.post_plot_y2max is not None:
                    ymin, _ = ax2.get_ylim()
                    ax2.set_ylim(ymin, args.post_plot_y2max)
                handles = [line_prob, line_median, line_mode]
                labels = [h.get_label() for h in handles]
                leg = ax1.legend(handles, labels, loc="center left", bbox_to_anchor=(1.18, 0.5), fontsize=9, frameon=True, borderaxespad=0.0)
                plt.subplots_adjust(right=0.78)
                ax1.set_title("Inflation rate vs Ruin probability (with asset overlays)")
                out_fallback = os.path.join(base_res_dir, f"sweep_multi_{dt.datetime.now().strftime('%y%m%d_%H%M')}.png")
                plt.savefig(out_fallback, bbox_inches="tight", bbox_extra_artists=(leg,))
                plt.close()
                print("[Post-plot] Saved:", out_fallback)
        return

    # --------------- Single-run mode ---------------
    if args.inflation_mode in ("per_path", "per_year"):
        if args.inflation_min is None or args.inflation_max is None:
            raise SystemExit("When using per_path/per_year, set --inflation_min and --inflation_max (e.g., 0.02 0.03).")

    final_assets, ruin_flags, snapshots, capture_months = run_simulation(
        n_sim=args.simulation,
        initial_asset=args.initial_asset,
        years=args.years,
        base_fx_rate=args.base_fx_rate,
        fx_min=args.fx_min,
        fx_max=args.fx_max,
        portfolio=portfolio,
        annual_withdrawal=args.annual_withdrawal,
        mode_interval_years=max(1, args.mode_interval_years),
        withdraw_start_years=args.withdraw_start_years,
        withdraw_stop_min_asset=args.withdraw_stop_min_asset,
        early_terminate_if_ruined=args.early_terminate_if_ruined,
        inflation_mode=args.inflation_mode,
        inflation_rate_fixed=args.inflation_rate,
        inflation_min=args.inflation_min,
        inflation_max=args.inflation_max,
        t_df=args.t_df,
        fx_annual_vol=args.fx_annual_vol,
        fx_drift_annual=args.fx_drift_annual,
        fx_t_df=args.fx_t_df,

    )

    stats = pd.DataFrame(final_assets, columns=["final_asset"]).describe().T
    ruin_prob = float(ruin_flags.mean())
    share_final_negative = float((final_assets < 0).mean())
    mode_val, mode_freq = approx_mode(final_assets, bins=100)

    res_dir = os.path.join(base_res_dir, "mode_{}".format(args.inflation_mode))
    os.makedirs(res_dir, exist_ok=True)

    print(stats)
    print()
    print("ruin_probability ; {:.4f}".format(ruin_prob))
    print("share_final_negative ; {:.4f}".format(share_final_negative))
    print("mode approx ; {} (freq={})".format(int(np.rint(mode_val)), mode_freq))
    print()

    pd.DataFrame({"final_asset": final_assets, "ruined": ruin_flags.astype(int)}).to_csv(
        os.path.join(res_dir, "asset_ruin.csv"), index=False
    )
    summarize_snapshots(snapshots, capture_months).to_csv(
        os.path.join(res_dir, "yearly_modes.csv"), index=False
    )

    with open(os.path.join(res_dir, "summary.txt"), "w", encoding="utf-8") as fout:
        fout.write(stats.to_string()); fout.write("\n")
        fout.write("ruin_probability={:.4f}\n".format(ruin_prob))
        fout.write("share_final_negative={:.4f}\n".format(share_final_negative))
        fout.write("mode_approx={:d}\n".format(int(np.rint(mode_val))))

    title = "Asset sim: {}*10^4 JPY, {}yr, inf_mode={}".format(int(args.initial_asset/10000), args.years, args.inflation_mode)
    annotate = "mode approx {}\nruin_prob = {:.3f}".format(int(np.rint(mode_val)), ruin_prob)
    save_histogram(final_assets, title, os.path.join(res_dir, "final_dist.png"),
                   args.x_min, args.x_max, annotate)

    if args.save_snapshot_hists and len(capture_months) > 0:
        for j, m in enumerate(capture_months):
            yr = (m + 1) // 12
            vals = snapshots[:, j]
            mode_y, _ = approx_mode(vals, bins=100)
            share_neg = float((vals < 0).mean())
            annotate_y = "mode approx {}\nshare_negative = {:.3f}".format(int(np.rint(mode_y)), share_neg)
            save_histogram(vals, "Snapshot Y{}".format(yr), os.path.join(res_dir, "snapshot_Y{}.png".format(yr)),
                           args.x_min, args.x_max, annotate_y)

    with open(os.path.join(res_dir, "run_config.json"), "w", encoding="utf-8") as fout:
        json.dump({
            **{k: getattr(args, k) for k in vars(args)},
            "timestamp": now,
            "res_dir": res_dir,
            "env": env_info,
            "portfolio_snapshot": portfolio
        }, fout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()


## program 概論 250816
#{{{
'''

================================================================
取扱いメモ
対象:
・メイン: xx3_monte_carlo_simulation_inflation_range_fx_adjust_t-dist.py
・図作成: yy1_plot_sweep_multi.py

1. 概略
10,000本の独立パスを用意し、各パスをxy年×12ヶ月ぶんシミュレートしていきます。
各パスは互いに干渉しません.
（分布の中央値・平均は“集計用”に使うだけで、次年の初期値に影響しません）。

年末ごとに全パスの分布を集計して yearly_modes.csv やヒストPNGに記録だけします。計算自体は各パスが自分の資産額を持って進みます。

====================================

[1] 使い方（Usage）

A) インフレ率スイープ実行（例）
python xx3_monte_carlo_simulation_inflation_range_fx_adjust_t-dist.py
-s 10000 -i 20000000 -y 20 ^
-p portfolio.json -w 2300000 --withdraw_start_years 3 
--inflation_min 0.02 --inflation_max 0.03 --inflation_sweep_steps 6 
--t_df 6 
--fx_annual_vol 0.10 --fx_drift_annual 0.00 --fx_t_df 6 
--seed 123

B) 単発（インフレ固定）の実行（例）
python xx3_monte_carlo_simulation_inflation_range_fx_adjust_t-dist.py
-s 10000 -i 20000000 -y 20 ^
-p portfolio.json -w 2300000 --withdraw_start_years 3 
--inflation_rate 0.025 ^
--t_df 6 ^
--fx_annual_vol 0.10 --fx_drift_annual 0.00 --fx_t_df 6 
--seed 123

C) スイープ結果のオーバーレイ図（v5）（例）
python xx3_monte_carlo_simulation_inflation_range_fx_adjust_t-dist.py
--sum_res_d results/res_20yr_YYMMDD_HHMM 
--percent ^
--plot_median --plot_mode 
--asset_divisor 1000000 --asset_label "Asset (million JPY)" 
--right_margin 0.78 --legend_anchor_x 1.18 --legend_anchor_y 0.5 
--errorbars --capsize 3

ref:
メイン実行後に v5 を自動起動する設定にしておくと、--post_plot_multi 等の指定を省略できます（無効化は --no-post-plot-multi）。

[2] 引数の説明（メイン 21b）

## (2-1) コア・実行制御

--simulation, -s      : int（例 10000） モンテカルロ試行数
--initial_asset, -i   : float           初期資産（JPY）
--years, -y           : int             運用年数
--portfolio, -p       : str             ポートフォリオJSONのパス
--seed                : int or None     乱数シード（再現性）
--early_terminate_if_ruined : flag      破綻（資産<=0）でその経路を打ち切り
---------------------------------------------------------------

## (2-2) 取り崩し・スナップショット

--annual_withdrawal, -w : float   初年度取り崩し額（名目, JPY）
--withdraw_start_years  : int     取り崩し開始年（0=初年）
--save_snapshot_hists   : flag    途中年ヒストグラムを年次出力
--mode_interval_years   : int     ヒスト/モード抽出間隔（年, 既定=1）
--------------------------------------------------------

## (2-3) インフレ（inflation）

--inflation_rate        : float   単発実行時の年率インフレ
--inflation_min/max     : float   スイープの下限/上限（年率）
--inflation_sweep_steps : int     スイープ分割数（等間隔）
 考: 取り崩し額は毎年「前年取り崩し×(1+inflation)」で増額。
 ポート上の資産は実質換算（インフレ影響を調整）で比較。
---------------------------

## (2-4) 資産リターンの分布（t分布）

--t_df : float or None  df>2 で Student-t を使用（df 小ほど裾が厚い）。
 指定/<=2 は正規分布。
-------------

# (2-5) 為替（FX：毎月変動＋初期レベル差）

--base_fx_rate          : float   基準レート（JPY/USD想定, 例 147）
--fx_min / --fx_max     : float   経路ごとに初期FXレートを一様分布で1回サンプル
--fx_annual_vol         : float   年率ボラ（月次は /√12）
--fx_drift_annual       : float   年率ドリフト（月次は 12乗根で換算）
--fx_t_df               : float   月次FXにも t分布を使用（df>2）。未指定は正規
 考: 外貨判定は資産名に "foreign" または "usd" を含む場合に適用。
-------------------------------------------

## (2-6) 結果保存・後処理

（自動）結果ディレクトリ : results/res_{years}yr_{YYMMDD_HHMM} 形式で作成
-z                        : str     保存ディレクトリ名に任意タグ付与
--post_plot_multi         : flag    スイープ後に外部プロッタを自動起動（既定ON）
--no-post-plot-multi      : flag    上記の無効化
--post_plot_call_multi    : str     図スクリプトのパス（既定: yy1_plot_sweep_multi_v5.py）
--post_plot_errorbars     : flag    エラーバー描画をプロッタへ指示（既定ON）
-----------------------------------------------------------

[3] モデルの概略（ポイント解説）

(3-1) モンテカルロの走らせ方
・試行数が N のとき、N 本の経路（パス）を用意し、各経路を月次で years×12 ステップ更新。
・各経路は相互に独立（他の経路とは干渉しない）。
・--early_terminate_if_ruined 指定時は、その経路の資産が0以下になった時点で打ち切り。

(3-2) インフレの補正（parser の使い方含む）
・取り崩しは名目で増額：AnnualWithdraw(y) = AnnualWithdraw(y-1) * (1 + inflation)。
inflation は --inflation_rate（単発）または --inflation_min/max/steps（スイープ）。
・レポート/集計は実質ベース（インフレ影響を除いて比較）になるように設計。

(3-3) 為替（FX）の補正（parser の使い方含む）
・経路開始時に fx_rate0 ~ U(fx_min, fx_max) を1回サンプル。
初期レベル差 L0 = fx_rate0 / base_fx_rate を計算。
・月0（最初の1か月）の外貨資産には、(1 + r_asset0) \* L0 \* (1 + r_fx0) - 1 を適用。
・月1以降は、(1 + r_asset_t) * (1 + r_fx_t) - 1 を毎月適用。
r_fx_t は、μ_m, σ_m を用いて生成（μ_m=(1+fx_drift_annual)^(1/12)-1、σ_m=fx_annual_vol/√12）。
・外貨判定: 資産名に "foreign" または "usd" を含むと外貨扱い（大小無視）。
例: "usd: US Equity", "Foreign Bond" など。
・平均回帰（特定レベルへ戻る力）は入れていません。必要なら対数レートに AR(1)/半減期で拡張可能。

(3-4) ret（資産リターン）の計算方法
・資産 i の月次リターン:
r_i_month = (mean_return_i / 12) + (std_dev_i / √12) \* z
（mean_return_i, std_dev_i は年率パラメータを想定）
・z は標準正規または t 乱数（--t_df>2 で t を使用）。
・外貨資産は上記に r_fx を掛け合わせて合成:
r_yen_t = (1 + r_local_t) \* (1 + r_fx_t_eff) - 1
※ 月0のみ r_fx_0_eff = L0 \* (1 + r_fx0) - 1、月1以降は r_fx_t_eff = r_fx_t。

(3-5) t分布を使う理由、どこで使っているか
・金融リターンは実務的に重尾（極端値が正規より多い）→正規だと下落リスク過少評価になりがち。
・--t_df を 5～8 程度に設定すると、ショック頻度が現実に近づき、破綻確率の見積もりが保守的になる。
・適用箇所は「資産の月次乱数生成」。オプションで「為替の月次乱数」にも t（--fx_t_df）を適用可能。
・t 乱数は分散が df/(df-2) なので sqrt((df-2)/df) を掛けて「単位分散」に正規化してから使用（平均・分散のキャリブを維持）。

(3-6) 出力の読み方
・各経路の最終資産（実質JPY）と破綻フラグ（最終時点で資産<=0）を集計。
・スイープ時は sweep_summary.csv に以下を出力:
inflation_rate, ruin_probability, ruin_se, ruin_ci_low, ruin_ci_high, median_final, mode など。
・yy1_plot_sweep_multi_v5.py では、ruin に 95%CI エラーバーを重ね、右軸に median/mode を同時表示可能。
・CI は二項推定の近似（例: Wilson）を利用。試行数を増やすと CI が狭くなる（精度向上）。

[4] 簡易インデックス（引数分類サマリ）

コア:         -s -i -y -p --seed --early_terminate_if_ruined
取り崩し:     -w --withdraw_start_years --save_snapshot_hists --mode_interval_years
インフレ:      --inflation_rate  --inflation_min --inflation_max --inflation_sweep_steps
t分布:        --t_df
為替:         --base_fx_rate --fx_min --fx_max --fx_annual_vol --fx_drift_annual --fx_t_df
保存/後処理:   -z（任意タグ） --post_plot_multi（既定ON） --no-post-plot-multi
--post_plot_call_multi（既定: v5） --post_plot_errorbars（既定ON）

 （v5）:     --sum_res_d --percent --plot_median --plot_mode
--asset_divisor --asset_label --y1max --y2max
--errorbars --capsize --right_margin --legend_anchor_x --legend_anchor_y
'''
#}}}

# mode line
# vim:set foldmethod=marker:
