#!/usr/bin/env python3
"""
CSE 240B - Plot benchmark results: TFLOPs/s vs sequence length.
Reads results.csv, produces figures/throughput_b1.pdf and figures/throughput_b8.pdf.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reference lines: FA3 published H100 numbers ────────────────────
FA3_H100_FP16 = 740   # TFLOPs/s
FA3_H100_BF16 = 840   # TFLOPs/s

# ── Styling ─────────────────────────────────────────────────────────
KERNEL_STYLES = {
    "SDPA_math":  {"color": "#636363", "marker": "s", "label": "PyTorch SDPA (math)"},
    "FlashAttn2": {"color": "#2171b5", "marker": "o", "label": "FlashAttention-2"},
    "SageAttn2":  {"color": "#cb181d", "marker": "^", "label": "SageAttention 2.2"},
}


def plot_batch(df_batch, batch_size, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for kernel, style in KERNEL_STYLES.items():
        sub = df_batch[(df_batch["kernel"] == kernel) & (df_batch["status"] == "OK")]
        if sub.empty:
            continue
        ax.plot(
            sub["seqlen"], sub["tflops"],
            marker=style["marker"], color=style["color"],
            linewidth=2, markersize=7, label=style["label"],
        )

    # FA3 H100 reference lines
    ax.axhline(FA3_H100_FP16, color="#31a354", linestyle="--", linewidth=1.2,
               label=f"FA3 H100 FP16 ({FA3_H100_FP16} TF/s)")
    ax.axhline(FA3_H100_BF16, color="#006d2c", linestyle="--", linewidth=1.2,
               label=f"FA3 H100 BF16 ({FA3_H100_BF16} TF/s)")

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("TFLOPs/s", fontsize=12)
    ax.set_title(f"Attention Kernel Throughput — RTX 5060 (SM120)\nBatch Size = {batch_size}",
                 fontsize=13, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df_batch["seqlen"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    csv_path = "results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run benchmark.py first.")
        return

    df = pd.read_csv(csv_path)
    df["tflops"] = pd.to_numeric(df["tflops"], errors="coerce")
    df["seqlen"] = pd.to_numeric(df["seqlen"], errors="coerce")

    os.makedirs("figures", exist_ok=True)

    for batch_size, out_name in [(1, "throughput_b1.pdf"), (8, "throughput_b8.pdf")]:
        df_batch = df[df["batch"] == batch_size]
        if df_batch.empty:
            print(f"No data for batch_size={batch_size}, skipping.")
            continue
        plot_batch(df_batch, batch_size, os.path.join("figures", out_name))


if __name__ == "__main__":
    main()
