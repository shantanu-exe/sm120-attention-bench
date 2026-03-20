#!/usr/bin/env python3
"""
CSE 240B - Environment Sanity Check
Verifies: GPU detection, compute capability, FA2 forward pass, SageAttention forward pass.
"""

import sys

def check_pytorch_gpu():
    """Check 1: GPU detection and compute capability."""
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "CUDA not available — PyTorch cannot see the GPU"

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"Device name:     {name}")
    print(f"Compute cap:     sm_{cap[0]}{cap[1]}")
    assert cap == (12, 0), f"Expected SM120 (12,0), got {cap}"
    assert "5060" in name, f"Expected RTX 5060, got {name}"
    print("[PASS] GPU detected as RTX 5060, SM120\n")


def check_flash_attention():
    """Check 2: FlashAttention-2 forward pass on GPU."""
    import torch
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        print("[SKIP] flash_attn not installed")
        return False

    B, H, S, D = 1, 4, 128, 128
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")

    out = flash_attn_func(q, k, v, causal=True)
    assert out.device.type == "cuda", "FA2 output is NOT on GPU — silent CPU fallback detected!"
    assert out.shape == (B, S, H, D), f"Unexpected output shape: {out.shape}"
    print(f"FA2 output shape: {out.shape}, device: {out.device}")
    print("[PASS] FlashAttention-2 forward pass on GPU\n")
    return True


def check_sage_attention():
    """Check 3: SageAttention forward pass on GPU."""
    import torch
    try:
        from sageattention import sageattn
    except ImportError:
        print("[SKIP] sageattention not installed")
        return False

    B, H, S, D = 1, 4, 128, 128
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    out = sageattn(q, k, v)
    assert out.device.type == "cuda", "SageAttention output is NOT on GPU — silent CPU fallback!"
    assert out.shape == (B, H, S, D), f"Unexpected output shape: {out.shape}"
    print(f"SageAttn output shape: {out.shape}, device: {out.device}")
    print("[PASS] SageAttention forward pass on GPU\n")
    return True


def main():
    print("=" * 60)
    print("CSE 240B — Environment Sanity Check")
    print("=" * 60 + "\n")

    results = {}

    # Check 1: PyTorch + GPU
    try:
        check_pytorch_gpu()
        results["PyTorch + GPU"] = "PASS"
    except Exception as e:
        print(f"[FAIL] PyTorch GPU check: {e}\n")
        results["PyTorch + GPU"] = "FAIL"

    # Check 2: FlashAttention-2
    try:
        ok = check_flash_attention()
        results["FlashAttention-2"] = "PASS" if ok else "SKIP"
    except Exception as e:
        print(f"[FAIL] FlashAttention-2: {e}\n")
        results["FlashAttention-2"] = "FAIL"

    # Check 3: SageAttention
    try:
        ok = check_sage_attention()
        results["SageAttention"] = "PASS" if ok else "SKIP"
    except Exception as e:
        print(f"[FAIL] SageAttention: {e}\n")
        results["SageAttention"] = "FAIL"

    # Summary
    print("=" * 60)
    print("Summary:")
    for name, status in results.items():
        print(f"  {name:25s} {status}")
    print("=" * 60)

    if any(v == "FAIL" for v in results.values()):
        print("\nSome checks FAILED. Fix issues before running benchmark.")
        sys.exit(1)
    elif any(v == "SKIP" for v in results.values()):
        print("\nSome checks SKIPPED (packages not yet installed).")
        sys.exit(0)
    else:
        print("\nAll checks PASSED. Ready to benchmark.")
        sys.exit(0)


if __name__ == "__main__":
    main()
