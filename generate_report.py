#!/usr/bin/env python3
"""
Generate CSE 240B course project report as a 4-page PDF.
Uses reportlab for text layout, regenerates figures as PNG for embedding.
"""

import csv
import os
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.lib import colors


def regenerate_figure_as_png(csv_path, batch_size, out_png_path):
    """Regenerate a throughput figure directly as PNG for embedding."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FA3_H100_FP16 = 740
    FA3_H100_BF16 = 840
    KERNEL_STYLES = {
        "SDPA_math":  {"color": "#636363", "marker": "s", "label": "SDPA (math)"},
        "FlashAttn2": {"color": "#2171b5", "marker": "o", "label": "FlashAttn-2"},
        "SageAttn2":  {"color": "#cb181d", "marker": "^", "label": "SageAttn"},
    }

    df = pd.read_csv(csv_path)
    df["tflops"] = pd.to_numeric(df["tflops"], errors="coerce")
    df_batch = df[df["batch"] == batch_size]

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    for kernel, style in KERNEL_STYLES.items():
        sub = df_batch[(df_batch["kernel"] == kernel) & (df_batch["status"] == "OK")]
        if sub.empty:
            continue
        ax.plot(sub["seqlen"], sub["tflops"], marker=style["marker"],
                color=style["color"], linewidth=2, markersize=7, label=style["label"])

    ax.axhline(FA3_H100_FP16, color="#31a354", linestyle="--", linewidth=1.2,
               label=f"FA3 H100 FP16 ({FA3_H100_FP16})")
    ax.axhline(FA3_H100_BF16, color="#006d2c", linestyle="--", linewidth=1.2,
               label=f"FA3 H100 BF16 ({FA3_H100_BF16})")

    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("TFLOPs/s", fontsize=11)
    ax.set_title(f"Attention Kernel Throughput — RTX 5060 (SM120), Batch Size = {batch_size}",
                 fontsize=12, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df_batch["seqlen"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png_path


def load_results(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_report(csv_path, output_path):
    results = load_results(csv_path)

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.5*inch, bottomMargin=0.5*inch,
        leftMargin=0.72*inch, rightMargin=0.72*inch,
    )

    styles = getSampleStyleSheet()

    ts = ParagraphStyle("T", parent=styles["Title"], fontSize=14, leading=17,
                        spaceAfter=2, alignment=TA_CENTER)
    author = ParagraphStyle("A", parent=styles["Normal"], fontSize=10, leading=13,
                            alignment=TA_CENTER, spaceAfter=1)
    email = ParagraphStyle("Em", parent=styles["Normal"], fontSize=9, leading=11,
                           alignment=TA_CENTER, spaceAfter=1,
                           textColor=colors.HexColor("#2171b5"))
    affil = ParagraphStyle("Af", parent=styles["Normal"], fontSize=9, leading=11,
                           alignment=TA_CENTER, spaceAfter=8, textColor=colors.HexColor("#444"))
    abst_t = ParagraphStyle("AbsT", parent=styles["Normal"], fontSize=10, leading=13,
                            fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=3)
    abst = ParagraphStyle("Abs", parent=styles["Normal"], fontSize=9, leading=11.5,
                          alignment=TA_JUSTIFY, leftIndent=20, rightIndent=20, spaceAfter=8)
    h1 = ParagraphStyle("H1", parent=styles["Heading2"], fontSize=11, leading=13,
                        spaceBefore=6, spaceAfter=2, fontName="Helvetica-Bold")
    h2 = ParagraphStyle("H2", parent=styles["Heading3"], fontSize=9.5, leading=11,
                        spaceBefore=4, spaceAfter=1, fontName="Helvetica-BoldOblique")
    body = ParagraphStyle("B", parent=styles["Normal"], fontSize=9.5, leading=12.2,
                          alignment=TA_JUSTIFY, spaceAfter=4)
    cap = ParagraphStyle("Cap", parent=styles["Normal"], fontSize=8, leading=10,
                         alignment=TA_CENTER, spaceAfter=5, textColor=colors.HexColor("#333"))
    refstyle = ParagraphStyle("Ref", parent=styles["Normal"], fontSize=8, leading=10,
                         spaceAfter=1, leftIndent=14, firstLineIndent=-14)

    tbl_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#ccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ])

    story = []

    # ══════════ Title block ══════════

    story.append(Paragraph(
        "Attention Kernel Performance on Consumer Blackwell SM120:<br/>"
        "Benchmarking What Runs on the RTX 5060", ts))
    story.append(Paragraph(
        "Shantanu Nigam &nbsp;&nbsp;&nbsp;&nbsp; Ananya Kapoor", author))
    story.append(Paragraph(
        "{shnigam, a4kapoor}@ucsd.edu", email))
    story.append(Paragraph(
        "CSE 240B — University of California, San Diego", affil))

    # ══════════ Abstract ══════════

    story.append(Paragraph("Abstract", abst_t))
    story.append(Paragraph(
        "FlashAttention-3 and FlashAttention-4 achieve record attention throughput on datacenter GPUs "
        "(Hopper SM90, Blackwell SM100), but neither runs on NVIDIA's consumer Blackwell GPUs (SM120). "
        "We test three kernels that <i>do</i> work on the RTX 5060 (SM120, 8 GB GDDR7): PyTorch's "
        "SDPA math backend, FlashAttention-2 v2.7.4 installed from a community-maintained wheel, "
        "and SageAttention 1.0.6 running through Triton. At sequence length 8192 with BF16 causal "
        "attention, SageAttention reaches 110 TFLOPs/s and FA2 reaches 65 TFLOPs/s. For context, "
        "FA3 on an H100 hits 840 TFLOPs/s in BF16 — roughly 7.6x and 12.9x faster, respectively.",
        abst))

    # ══════════ 1. Introduction + Motivation ══════════

    story.append(Paragraph("1. Introduction", h1))
    story.append(Paragraph(
        "When we bought an RTX 5060 laptop, the reasoning was straightforward: a GPU that could "
        "handle gaming but also be useful for ML coursework and experimentation. NVIDIA markets the "
        "RTX 50-series under the Blackwell brand — the same name as the B200 datacenter accelerator — "
        "so we assumed that software built for \"Blackwell\" would mostly just work. That turned out "
        "to be wrong. The consumer SM120 chip is architecturally different from the datacenter SM100 "
        "in ways that matter: no multicast TMA, a different tensor core ISA, and clusters locked to "
        "1x1x1. FlashAttention-3 checks your GPU's compute capability at import time and refuses to "
        "run on SM120. FA4 is SM100-only. We found this out the hard way when our first benchmark "
        "script crashed immediately.", body))
    story.append(Paragraph(
        "This is probably a common experience for students and hobbyists who buy consumer GPUs "
        "expecting them to work for both gaming and ML. There is not much published data on what "
        "attention kernels actually work on SM120 or how fast they are, so we decided to find out. "
        "The goal is simple: test every attention kernel we can get running on the RTX 5060, measure "
        "throughput, and compare against published FA3 numbers on the H100 to quantify the gap.", body))
    story.append(Paragraph(
        "For background: attention's O(N<super>2</super>) cost has driven a series of increasingly "
        "specialized kernels. FlashAttention [1] introduced IO-aware tiling for Ampere, FA2 [2] "
        "improved occupancy and warp specialization, FA3 [3] added Hopper-specific WGMMA and TMA "
        "support, and FA4 [4] targets SM100 with tcgen05/UMMA. Each generation is more tightly "
        "coupled to specific hardware, which is great for datacenter users but leaves consumer "
        "GPUs behind.", body))

    # ══════════ 2. Background ══════════

    story.append(Paragraph("2. Background", h1))
    story.append(Paragraph("2.1 Architecture Comparison", h2))

    arch_data = [
        ["Feature", "SM90 (H100)", "SM100 (B200)", "SM120 (RTX 5060)"],
        ["Tensor Core ISA", "WGMMA", "tcgen05/UMMA", "SM120-specific"],
        ["TMA", "Yes", "Yes + multicast", "No multicast"],
        ["Cluster shape", "Up to 16 SMs", "Up to 16 SMs", "1x1x1"],
        ["FA3 / FA4", "FA3 only", "FA4 only", "Neither"],
    ]
    t = Table(arch_data, colWidths=[1.2*inch, 1.3*inch, 1.3*inch, 1.4*inch])
    t.setStyle(tbl_style)
    story.append(t)
    story.append(Paragraph("<b>Table 1.</b> Architecture comparison relevant to attention kernels.", cap))

    story.append(Paragraph("2.2 Kernels Under Test", h2))
    story.append(Paragraph(
        "<b>PyTorch SDPA (math backend)</b> — the simplest baseline, materializes the full "
        "N x N score matrix. We expected it to be slow and memory-hungry, and it was. "
        "<b>FlashAttention-2 v2.7.4</b> — the official pip package does not ship SM120 binaries, "
        "so we used a community wheel (Zarrac on GitHub). It installs fine but runs the SM80 code "
        "path, essentially treating our Blackwell GPU as Ampere. "
        "<b>SageAttention 1.0.6</b> — INT8-quantized QK matmul. The PyPI version uses "
        "Triton-generated kernels; since Triton 3.6 added SM120 code generation, these actually "
        "compile natively on our hardware, unlike FA2's SM80 fallback.", body))

    # ══════════ 3. Methodology ══════════

    story.append(Paragraph("3. Methodology", h1))
    story.append(Paragraph(
        "All experiments run on an RTX 5060 Laptop GPU (SM120, 8 GB GDDR7) under Windows 11 "
        "with WSL2. Software: PyTorch 2.7.0+cu128 nightly (stable does not support SM120), "
        "Triton 3.6.0, FA2 v2.7.4.post1 (Zarrac community wheel — the official one segfaults "
        "on SM120), SageAttention 1.0.6 from PyPI. Config: 32 heads, d=128, BF16, causal mask. "
        "Sequence lengths 512-8192, batch sizes 1 and 8. Each config gets 10 warmup + 100 timed "
        "iterations using torch.cuda.Event. TFLOPs/s = "
        "4 x B x H x S<super>2</super> x D / time / 10<super>12</super>. "
        "Reference: FA3 on H100 at 740 (FP16) and 840 (BF16) TFLOPs/s [3].", body))

    # ══════════ 4. Results ══════════
    story.append(PageBreak())

    story.append(Paragraph("4. Results", h1))

    # Combined results table
    header = ["Kernel", "B", "S=512", "S=1K", "S=2K", "S=4K", "S=8K"]
    kmap = {"SDPA_math": "SDPA math", "FlashAttn2": "FlashAttn-2", "SageAttn2": "SageAttn"}
    data_rows = [header]
    for batch in ["1", "8"]:
        for kname in ["SDPA_math", "FlashAttn2", "SageAttn2"]:
            row = [kmap[kname], batch]
            for sl in [512, 1024, 2048, 4096, 8192]:
                m = [r for r in results if r["kernel"] == kname and r["batch"] == batch and r["seqlen"] == str(sl)]
                if m and m[0]["status"] == "OK":
                    row.append(f"{float(m[0]['tflops']):.1f}")
                else:
                    row.append("OOM")
            data_rows.append(row)

    cw = [0.9*inch, 0.35*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch]
    rt = Table(data_rows, colWidths=cw)
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#ccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LINEABOVE", (0, 4), (-1, 4), 1.0, colors.HexColor("#2c3e50")),
    ]))
    story.append(rt)
    story.append(Paragraph(
        "<b>Table 2.</b> Throughput (TFLOPs/s) for all kernels. BF16, causal, 32 heads, d=128.", cap))

    story.append(Paragraph(
        "The most interesting result is that SageAttention consistently beats FA2 at longer "
        "sequences, reaching 110 TFLOPs/s at S=8192 compared to FA2's ~65 TFLOPs/s. We did "
        "not expect this going in — FA2 is the more established kernel. But FA2 seems to hit a "
        "ceiling around 65 TFLOPs/s regardless of batch size or sequence length, which we think "
        "reflects the SM80 compatibility path not being able to use the hardware efficiently. "
        "SDPA math is predictably bad: 2-2.5 TFLOPs/s, and it OOMs once the full attention "
        "matrix no longer fits in 8 GB.", body))

    # Figures — stacked vertically for larger display
    story.append(Paragraph("4.1 Throughput vs. Sequence Length", h2))

    fig1_png = os.path.join(tempfile.gettempdir(), "fig_b1.png")
    fig2_png = os.path.join(tempfile.gettempdir(), "fig_b8.png")
    regenerate_figure_as_png(csv_path, 1, fig1_png)
    regenerate_figure_as_png(csv_path, 8, fig2_png)

    img1 = Image(fig1_png, width=6.3*inch, height=2.85*inch)
    story.append(img1)
    story.append(Paragraph(
        "<b>Figure 1.</b> Throughput vs. sequence length at batch size 1. "
        "Dashed lines: published FA3 H100 reference numbers.", cap))

    img2 = Image(fig2_png, width=6.3*inch, height=2.85*inch)
    story.append(img2)
    story.append(Paragraph(
        "<b>Figure 2.</b> Throughput vs. sequence length at batch size 8. "
        "Dashed lines: published FA3 H100 reference numbers.", cap))

    story.append(Paragraph(
        "The figures make the FA2 plateau pretty clear — the line essentially goes flat after "
        "S=2048 while SageAttention keeps climbing. The gap between our best number (110 TFLOPs/s) "
        "and the FA3 H100 lines at 740-840 is striking. Even accounting for the H100 being a "
        "much larger chip with 3.35 TB/s of HBM3 bandwidth (vs. roughly 336 GB/s on our "
        "GDDR7 card, based on the 5060's 168-bit bus at 28 Gbps), the software gap is real too.", body))

    # Gap summary table
    story.append(PageBreak())
    story.append(Paragraph("4.2 Gap vs. Datacenter", h2))
    gap_data = [
        ["Kernel", "Peak TF/s", "vs FA3 FP16 (740)", "vs FA3 BF16 (840)"],
        ["SageAttention", "110.2", "6.7x", "7.6x"],
        ["FlashAttn-2", "65.6", "11.3x", "12.8x"],
        ["SDPA math", "2.5", "296x", "336x"],
    ]
    gt = Table(gap_data, colWidths=[1.2*inch, 0.9*inch, 1.4*inch, 1.4*inch])
    gt.setStyle(tbl_style)
    story.append(gt)
    story.append(Paragraph(
        "<b>Table 3.</b> Peak throughput on RTX 5060 vs. published FA3 H100 numbers.", cap))

    # ══════════ 5. Analysis ══════════

    story.append(Paragraph("5. Analysis", h1))

    story.append(Paragraph("5.1 Sources of the Performance Gap", h2))
    story.append(Paragraph(
        "The 7-13x gap is not one thing — it is several problems stacked on top of each other. "
        "The most obvious is memory bandwidth: the H100 SXM has 3.35 TB/s of HBM3, while the "
        "RTX 5060 has roughly 336 GB/s of GDDR7 (168-bit bus, 28 Gbps). That alone is about a "
        "10x difference, and attention is heavily memory-bound at the sequence lengths we tested. "
        "On top of that, FA2 on SM120 runs SM80-targeted code — it literally cannot issue SM120 "
        "tensor core instructions. And the instructions that make FA3 fast on Hopper (WGMMA for "
        "warp-group matmuls, TMA for async data movement) do not exist in the same form on SM120. "
        "The H100 also just has more raw compute: 989 TFLOPS BF16 peak, which our laptop GPU "
        "does not come close to.", body))

    story.append(Paragraph("5.2 Comparison with Published SageAttention Results", h2))
    story.append(Paragraph(
        "One thing we wanted to check was whether the SageAttention-over-FA2 speedup we measured "
        "lines up with what the original paper reports. The SageAttention v1 paper [5] shows about "
        "2x average speedup over FA2 on the RTX 4090 (SM89), peaking at 341 TOPS vs. FA2's 165 TOPS. "
        "We only see 1.4-1.7x on SM120, which is noticeably lower.", body))
    story.append(Paragraph(
        "We think this difference is mostly a software maturity issue, not an algorithmic one. The "
        "INT8 quantization trick in SageAttention should give roughly the same benefit regardless of "
        "hardware — halving the bandwidth for the QK matmul is halving the bandwidth. But the "
        "version we are running (1.0.6 from PyPI) uses Triton-generated kernels, not the hand-tuned "
        "CUDA kernels that were benchmarked in the paper. Triton's SM120 code generation only "
        "landed in version 3.6 and is probably not as mature as its SM89 path. SageAttention 2.0 "
        "claims ~3x over FA2 on the 4090 with 4-bit quantization; on SM120 we would expect the "
        "ratio to be similarly lower until someone builds native CUDA kernels for this chip.", body))

    story.append(Paragraph("5.3 Why SageAttention Beats FA2 Here", h2))
    story.append(Paragraph(
        "It is a bit surprising that a Triton-JIT kernel beats FlashAttention-2, which is "
        "hand-written CUDA. But the explanation is simple once you think about it: FA2's CUDA "
        "kernels were compiled for SM80 and cannot issue SM120 tensor core instructions at all. "
        "SageAttention's Triton kernels get JIT-compiled by Triton 3.6, which actually has an "
        "SM120 backend. So even though Triton-generated code is generally less optimized than "
        "hand-tuned CUDA, it wins here because it is at least targeting the right hardware. "
        "On top of that, the INT8 QK matmul uses half the bandwidth of FA2's BF16 path, which "
        "helps a lot on a bandwidth-constrained card.", body))

    story.append(Paragraph("5.4 Practical Takeaways", h2))
    story.append(Paragraph(
        "If you have an RTX 50-series and want to run attention workloads, use SageAttention. "
        "It is about 1.7x faster than FA2 at longer sequences. That said, even with SageAttention, "
        "you are getting about 13% of what an H100 can do with FA3 — and a big chunk of that gap "
        "is just the memory bandwidth difference (roughly 10x), which no amount of kernel "
        "optimization will fix. You can run attention up to sequence length 8192 at batch 8 "
        "before hitting the 8 GB VRAM wall, which is enough for experimentation and coursework "
        "but not for anything resembling production inference.", body))

    story.append(Paragraph("5.5 Limitations and Future Work", h2))
    story.append(Paragraph(
        "There are a few things we did not get to. We ran SageAttention 1.0.6 (the Triton version), "
        "not v2.2 which has native CUDA kernels — building that from source with the right CUDA "
        "toolkit turned out to be more setup than we had time for, and would be a good follow-up. "
        "We also only tested causal attention with d=128; GQA or non-causal attention might shift "
        "the relative numbers. Our RTX 5060 is a laptop variant, so a desktop 5060 might clock "
        "higher and show slightly different results. A roofline analysis would help separate how "
        "much of the gap is bandwidth-bound vs. compute-bound. And Triton's SM120 codegen will "
        "presumably get better — it would be worth re-running these benchmarks in six months.", body))

    # ══════════ 6. Conclusion + References ══════════

    story.append(Paragraph("6. Conclusion", h1))
    story.append(Paragraph(
        "We set out to answer a simple question — can a consumer Blackwell GPU run attention "
        "kernels, and if so, how fast? The answer: yes, but with caveats. SageAttention hits "
        "110 TFLOPs/s, FA2 hits 65, and the SDPA math backend is essentially unusable at "
        "2.5 TFLOPs/s before it OOMs. The gap to FA3 on an H100 (840 TFLOPs/s BF16) is 7-13x, "
        "driven by both hardware differences (the 10x bandwidth gap) and software immaturity "
        "(everything running in compatibility mode or through immature JIT paths).", body))
    story.append(Paragraph(
        "The SageAttention speedup we measured (1.4-1.7x over FA2) is lower than the 2x reported "
        "in the original paper on RTX 4090, which we attribute to Triton's relatively new SM120 "
        "backend rather than anything algorithmic. This will probably improve as the tooling "
        "matures. For now, if you are a student or hobbyist with a consumer Blackwell GPU and "
        "you want to run attention workloads, SageAttention is the way to go. Just know that "
        "you are leaving about an order of magnitude of performance on the table compared to "
        "datacenter hardware — and most of that is physics (bandwidth), not software.", body))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Acknowledgements", h1))
    story.append(Paragraph(
        "We owe a lot to Zarrac on GitHub for maintaining the SM120-compatible FA2 wheel — "
        "without that, we would not have been able to test FA2 at all. Thanks also to the "
        "SageAttention team for making 1.0.6 available on PyPI with Triton support, and to "
        "the PyTorch nightly team for getting SM120 detection working.", body))

    story.append(Spacer(1, 8))
    story.append(Paragraph("References", h1))
    refs = [
        "[1] Dao, T., Fu, D.Y., Ermon, S., Rudra, A., and Re, C. "
        "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. "
        "Advances in Neural Information Processing Systems (NeurIPS), 2022.",
        "[2] Dao, T. FlashAttention-2: Faster Attention with Better Parallelism and Work "
        "Partitioning. International Conference on Learning Representations (ICLR), 2024.",
        "[3] Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T. "
        "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. "
        "Advances in Neural Information Processing Systems (NeurIPS), 2024.",
        "[4] Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T. "
        "FlashAttention-4: Hardware-Friendly Attention with Blackwell Optimizations. arXiv preprint, 2025.",
        "[5] Zhang, J., Huang, H., et al. SageAttention: Accurate 8-Bit Attention for Plug-and-play "
        "Inference Acceleration. International Conference on Learning Representations (ICLR), 2025.",
        "[6] NVIDIA Corporation. NVIDIA Blackwell Architecture Technical Brief. 2024.",
        "[7] Tillet, P., Kung, H.T., and Cox, D. Triton: An Intermediate Language and Compiler "
        "for Tiled Neural Network Computations. MLSys, 2019.",
    ]
    for r in refs:
        story.append(Paragraph(r, refstyle))

    doc.build(story)
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "results.csv")
    out_path = os.path.join(base, "report", "cse240b_report_v6.pdf")
    os.makedirs(os.path.join(base, "report"), exist_ok=True)
    build_report(csv_path, out_path)
