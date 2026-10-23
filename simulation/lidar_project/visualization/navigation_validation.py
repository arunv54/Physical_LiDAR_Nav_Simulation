"""Near-threshold detection figure used by the PDF and offline report.

Read saved Monte Carlo summaries without rerunning sensor measurements.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


_POSE_COLORS = ("#0072B2", "#D55E00", "#009E73")


def _axes_style(ax, axis="both"):
    ax.tick_params(labelsize=8, width=.6, length=2.7)
    ax.grid(axis=axis, color="#E6E8EB", lw=.5)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(.6)


def _threshold(summary, destination):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    settings = summary["settings"]
    pose_ids = list(dict.fromkeys(row["pose_id"] for row in settings))
    threshold_estimates = [row["mean_snr"]-NormalDist().inv_cdf(row["expected_pd"])
                           for row in settings if .05 < row["expected_pd"] < .95]
    gamma = float(summary.get("snr_threshold", np.median(threshold_estimates)))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2), gridspec_kw={"width_ratios": (1.65, 1.)})
    fig.subplots_adjust(left=.08, right=.985, bottom=.24, top=.80, wspace=.33)
    x = np.linspace(min(r["mean_snr"] for r in settings), max(r["mean_snr"] for r in settings), 300)
    probability = np.array([.5*math.erfc((gamma-value)/math.sqrt(2)) for value in x])
    axes[0].plot(x, probability, color="#222222", lw=1.1)
    axes[0].axvspan(gamma-1.645, gamma+1.645, color="#EDF1F4", zorder=-1)
    axes[0].axvline(gamma, color="#68788B", lw=.7, ls="--")
    handles = [Line2D([], [], color="#222222", lw=1.1, label="Gaussian prediction")]
    for index, pose_id in enumerate(pose_ids):
        rows = sorted((r for r in settings if r["pose_id"] == pose_id), key=lambda r: r["mean_snr"])
        xx = np.array([r["mean_snr"] for r in rows])
        yy = np.array([r["observed_pd"] for r in rows])
        expected = np.array([r["expected_pd"] for r in rows])
        error = np.array([[max(0., r["observed_pd"]-r["wilson_low"]) for r in rows],
                          [max(0., r["wilson_high"]-r["observed_pd"]) for r in rows]])
        color = _POSE_COLORS[index % len(_POSE_COLORS)]
        marker = ("o", "s", "^")[index % 3]
        offset = (index-(len(pose_ids)-1)/2)*.075
        for ax, values in zip(axes, (yy, yy-expected)):
            ax.errorbar(xx+offset, values, yerr=error, fmt=marker, ms=3.3, color=color,
                        mew=.6, mfc="white", lw=.7, capsize=1.8, capthick=.6)
        handles.append(Line2D([], [], marker=marker, ms=3.5, color=color, mfc="white", ls="",
                              label=rows[0].get("pose_label", rows[0].get("label", str(pose_id)))))
    axes[0].set(xlim=(x[0]-.3, x[-1]+.3), ylim=(-.025, 1.025),
                yticks=(0, .25, .5, .75, 1), ylabel="Detection probability", xlabel="Mean received SNR")
    axes[1].axhline(0, color="#222222", lw=.8)
    axes[1].set(xlim=(x[0]-.3, x[-1]+.3), ylabel="Observed − predicted", xlabel="Mean received SNR")
    axes[1].yaxis.label.set_fontsize(8)
    for ax in axes:
        ax.set_xticks((6, 8, 10, 12, 14))
        ax.xaxis.label.set_fontsize(8.2)
        ax.yaxis.label.set_fontsize(8.2)
        _axes_style(ax)
    axes[0].set_title("(a) Detection across the threshold", loc="left", fontsize=9, pad=7)
    axes[1].set_title("(b) Calibration residual", loc="left", fontsize=9, pad=7)
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7.8, frameon=False,
               columnspacing=1., handletextpad=.4, bbox_to_anchor=(.5, 1.01))
    fig.savefig(destination, dpi=300, facecolor="white")
    plt.close(fig)
    return {"png": str(destination), "pose_ids": pose_ids, "settings": len(settings),
            "trials": sum(row["trials"] for row in settings),
            "intervals": "Pointwise 95% Wilson binomial intervals; horizontal pose offsets are display only",
            "prediction": "Phi(mean_SNR - SNR_threshold); shaded band is predicted 5–95% transition"}
