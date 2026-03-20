#!/usr/bin/env python3
"""
CSE 240B - Attention Kernel Benchmark on SM120 (RTX 5060)
Tests: PyTorch SDPA (math), FlashAttention-2, SageAttention 2.2
"""

import csv
import gc
import sys
import torch
import torch.nn.functional as F

# ── Config ──────────────────────────────────────────────────────────
NUM_HEADS = 32
HEAD_DIM = 128
BATCH_SIZES = [1, 8]
SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192]
DTYPE = torch.bfloat16
CAUSAL = True
WARMUP = 10
REPEATS = 100
OUTPUT_CSV = "results.csv"


def tflops(batch, heads, seqlen, head_dim, time_s):
    """Standard attention FLOP formula: 4 * B * H * S^2 * D / time / 1e12."""
    return 4 * batch * heads * seqlen ** 2 * head_dim / time_s / 1e12


def make_qkv(batch, seqlen, num_heads, head_dim, dtype, layout="bhsd"):
    """Create random Q, K, V tensors on GPU.
    layout='bhsd' -> (B, H, S, D)  — for SDPA and SageAttention
    layout='bshd' -> (B, S, H, D)  — for FlashAttention-2
    """
    if layout == "bhsd":
        shape = (batch, num_heads, seqlen, head_dim)
    else:  # bshd
        shape = (batch, seqlen, num_heads, head_dim)
    q = torch.randn(shape, dtype=dtype, device="cuda")
    k = torch.randn(shape, dtype=dtype, device="cuda")
    v = torch.randn(shape, dtype=dtype, device="cuda")
    return q, k, v


def time_fn(fn, warmup=WARMUP, repeats=REPEATS):
    """Time a function using CUDA events. Returns average time in seconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_ms = sum(times_ms) / len(times_ms)
    return avg_ms / 1000.0  # seconds


def bench_sdpa_math(batch, seqlen):
    """PyTorch SDPA with math backend only (no flash/efficient)."""
    q, k, v = make_qkv(batch, seqlen, NUM_HEADS, HEAD_DIM, DTYPE, layout="bhsd")

    def fn():
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            F.scaled_dot_product_attention(q, k, v, is_causal=CAUSAL)

    # Verify it runs on GPU
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=CAUSAL)
    assert out.device.type == "cuda", "SDPA math output not on GPU!"

    t = time_fn(fn)
    del q, k, v, out
    return t


def bench_flash_attn2(batch, seqlen):
    """FlashAttention-2 via flash_attn.flash_attn_func."""
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        return None

    q, k, v = make_qkv(batch, seqlen, NUM_HEADS, HEAD_DIM, DTYPE, layout="bshd")

    def fn():
        flash_attn_func(q, k, v, causal=CAUSAL)

    # Verify GPU execution
    out = flash_attn_func(q, k, v, causal=CAUSAL)
    assert out.device.type == "cuda", "FA2 output not on GPU — silent CPU fallback!"

    t = time_fn(fn)
    del q, k, v, out
    return t


def bench_sage_attention(batch, seqlen):
    """SageAttention 2.2 via sageattention.sageattn."""
    try:
        from sageattention import sageattn
    except ImportError:
        return None

    q, k, v = make_qkv(batch, seqlen, NUM_HEADS, HEAD_DIM, DTYPE, layout="bhsd")

    def fn():
        sageattn(q, k, v, is_causal=CAUSAL)

    # Verify GPU execution
    out = sageattn(q, k, v, is_causal=CAUSAL)
    assert out.device.type == "cuda", "SageAttention output not on GPU — silent CPU fallback!"

    t = time_fn(fn)
    del q, k, v, out
    return t


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Config: heads={NUM_HEADS}, head_dim={HEAD_DIM}, dtype={DTYPE}, causal={CAUSAL}")
    print(f"Timing: {WARMUP} warmup, {REPEATS} repeats\n")

    kernels = [
        ("SDPA_math", bench_sdpa_math),
        ("FlashAttn2", bench_flash_attn2),
        ("SageAttn2", bench_sage_attention),
    ]

    rows = []

    for batch in BATCH_SIZES:
        for seqlen in SEQ_LENGTHS:
            for kname, kfunc in kernels:
                gc.collect()
                torch.cuda.empty_cache()

                try:
                    t = kfunc(batch, seqlen)
                except torch.cuda.OutOfMemoryError:
                    print(f"  {kname:15s}  B={batch} S={seqlen:>5d}  OOM")
                    rows.append({
                        "kernel": kname, "batch": batch, "seqlen": seqlen,
                        "time_s": None, "tflops": None, "status": "OOM",
                    })
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    print(f"  {kname:15s}  B={batch} S={seqlen:>5d}  ERROR: {e}")
                    rows.append({
                        "kernel": kname, "batch": batch, "seqlen": seqlen,
                        "time_s": None, "tflops": None, "status": f"ERROR: {e}",
                    })
                    continue

                if t is None:
                    print(f"  {kname:15s}  B={batch} S={seqlen:>5d}  NOT INSTALLED")
                    rows.append({
                        "kernel": kname, "batch": batch, "seqlen": seqlen,
                        "time_s": None, "tflops": None, "status": "NOT_INSTALLED",
                    })
                    continue

                tf = tflops(batch, NUM_HEADS, seqlen, HEAD_DIM, t)
                print(f"  {kname:15s}  B={batch} S={seqlen:>5d}  {t*1000:8.2f} ms  {tf:7.1f} TFLOPs/s")
                rows.append({
                    "kernel": kname, "batch": batch, "seqlen": seqlen,
                    "time_s": f"{t:.6f}", "tflops": f"{tf:.2f}", "status": "OK",
                })

        print()  # blank line between batch sizes

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kernel", "batch", "seqlen", "time_s", "tflops", "status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
