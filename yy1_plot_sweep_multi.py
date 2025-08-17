#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay helper for sweep_summary.csv (v4: legend vs right-axis overlap fix).
- Reads {dir}/sweep_summary.csv when --sum_res_d is provided.
- Places legend OUTSIDE on the right with adjustable anchor and right margin,
  so it won't collide with the right y-axis ticks/label.

Usage:
  python plot_sweep_multi_v4.py \
    --sum_res_d results/res_20yr_YYMMDD_HHMM \
    --percent \
    --asset_divisor 1000000 \
    --asset_label "Asset (million JPY)" \
    --plot_median --plot_mode \
    --y1max 100 --y2max 6000 \
    --legend_fontsize 9 \
    --legend_anchor_x 1.18 \
    --right_margin 0.78

Notes:
- Increase --legend_anchor_x (e.g., 1.25) or decrease --right_margin (e.g., 0.75)
  if the legend still looks close to the axis. Default values usually suffice.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

def main():
    ap = argparse.ArgumentParser(description="Overlay median/mode on Rate vs Ruin-Probability chart (non-overlapping legend)")
    ##
    ap.add_argument("--sum_res_d", type=str, default=None, help="Directory that contains sweep_summary.csv")
    ap.add_argument("--summary", type=str, default=None, help="(legacy) Path to sweep_summary.csv")
    ap.add_argument("--out", type=str, default=None, help="(legacy) Output PNG path")
    ## y-axis
    ap.add_argument("--percent", action="store_true", help="Plot ruin_probability as percent")
    ap.add_argument("--asset_divisor", type=float, default=1_000_000.0, help="Divide assets by this value")
    ap.add_argument("--asset_label", type=str, default="Asset (million JPY)", help="Right y-axis label")
    ap.add_argument("--y1max", type=float, default=None, help="Left y-axis max")
    ap.add_argument("--y2max", type=float, default=None, help="Right y-axis max")
    ## mode, media もplotするか、どうか。
    ap.add_argument("--plot_median", action="store_true", help="Overlay median_final on right y-axis")
    ap.add_argument("--plot_mode", action="store_true", help="Overlay mode on right y-axis")
    ## legend の位置
    ap.add_argument("--legend_fontsize", type=float, default=9, help="Legend font size")
    ap.add_argument("--legend_inside", action="store_true", help="Place legend inside the axes")
    ap.add_argument("--legend_anchor_x", type=float, default=1.18, help="Legend bbox_to_anchor x (outside right)")
    ap.add_argument("--legend_anchor_y", type=float, default=0.5, help="Legend bbox_to_anchor y")
    ap.add_argument("--right_margin", type=float, default=0.78, help="Figure right margin for axes (0-1). Smaller = more space for legend")
    ap.add_argument("--errorbars", action="store_true",
                    help="Draw 95% CI error bars for ruin probability if columns ruin_ci_low/high exist")
    ap.add_argument("--capsize", type=float, default=3.0,
                    help="Error bar cap size (points)")
    args = ap.parse_args()

    # Resolve input/output paths
    if args.sum_res_d:
        csv_path = os.path.join(args.sum_res_d, "sweep_summary.csv")
        if not os.path.isfile(csv_path):
            raise SystemExit(f"sweep_summary.csv not found in: {args.sum_res_d}")
        now = dt.datetime.now().strftime("%y%m%d_%H%M")
        out_path = os.path.join(args.sum_res_d, f"sweep_multi_{now}.png")
    else:
        if not args.summary:
            raise SystemExit("Please specify --sum_res_d DIR (recommended) or legacy --summary CSV_PATH")
        csv_path = args.summary
        out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(csv_path)), "sweep_multi.png")

    # Load data
    df = pd.read_csv(csv_path)
    required_cols = {"inflation_rate", "ruin_probability"}
    if not required_cols.issubset(df.columns):
        raise SystemExit("sweep_summary.csv must include: inflation_rate, ruin_probability")
    has_median = "median_final" in df.columns
    has_mode = "mode" in df.columns
    has_ci = ("ruin_ci_low" in df.columns) and ("ruin_ci_high" in df.columns)

    x = df["inflation_rate"].values
    y_prob = df["ruin_probability"].values
    if args.percent:
        y_prob = y_prob * 100.0

    # Plot
    fig = plt.figure(figsize=(9.8, 5.6))
    ax1 = plt.gca()
    label_prob = "Ruin probability" + (" (%)" if args.percent else "")

    if args.errorbars and has_ci:
        y_low  = df["ruin_ci_low"].values  * (100.0 if args.percent else 1.0)
        y_high = df["ruin_ci_high"].values * (100.0 if args.percent else 1.0)
        yerr = np.vstack([y_prob - y_low, y_high - y_prob])
        _err = ax1.errorbar(x, y_prob, yerr=yerr, fmt="-o", capsize=args.capsize, label=label_prob)
        if hasattr(_err, "lines") and len(_err.lines) > 0:
            main_handle = _err.lines[0]
        else:
            (main_handle,) = ax1.plot(x, y_prob, marker="o", label=label_prob)
    else:
        (main_handle,) = ax1.plot(x, y_prob, marker="o", label=label_prob)
    ax1.grid(True)
    ax1.set_xlabel("Inflation rate")
    ax1.set_ylabel("Ruin probability" + (" (%)" if args.percent else ""))
    if args.y1max is not None:
        ax1.set_ylim(0, args.y1max)

    ax2 = ax1.twinx()
    handles = [main_handle]
    labels = [label_prob]

    if args.plot_median:
        if "median_final" not in df.columns:
            raise SystemExit("median_final column not found in sweep_summary.csv")
        y_median = (df["median_final"].values / args.asset_divisor)
        line_median, = ax2.plot(x, y_median, marker="s", label="Median final (scaled)")
        handles.append(line_median); labels.append(line_median.get_label())

    if args.plot_mode:
        if "mode" not in df.columns:
            raise SystemExit("mode column not found in sweep_summary.csv")
        y_mode = (df["mode"].values / args.asset_divisor)
        line_mode, = ax2.plot(x, y_mode, marker="^", label="Mode (scaled)")
        handles.append(line_mode); labels.append(line_mode.get_label())

    ax2.set_ylabel(args.asset_label)
    if args.y2max is not None:
        ymin, _ = ax2.get_ylim()
        ax2.set_ylim(ymin, args.y2max)

    # Adjust right margin to create spacing between right axis and legend
    plt.subplots_adjust(right=args.right_margin)

    # Legend
    if args.legend_inside:
        leg = ax1.legend(handles, labels, loc="best", fontsize=args.legend_fontsize)
        extra_artists = ()
    else:
        leg = ax1.legend(handles, labels, loc="center left",
                         bbox_to_anchor=(args.legend_anchor_x, args.legend_anchor_y),
                         fontsize=args.legend_fontsize, frameon=True, borderaxespad=0.0)
        extra_artists = (leg,)

    ax1.set_title("Inflation rate vs Ruin probability (with asset overlays)")

    plt.savefig(out_path, bbox_inches="tight", bbox_extra_artists=extra_artists)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
