"""Quick test: does TOPK=64 trigger 561002?

Runs GatherSelectionKvCache with production-scale dimensions to isolate
whether the vector path (topk>32) causes the validation failure.
"""
import torch
import torch_npu
import sys

DEVICE = "npu:0"
DTYPE = torch.bfloat16
BLOCK_SIZE = 64
KV_DIM = 512
K_ROPE_DIM = 64
TOPK_BLOCK_SIZE = 64

try:
    import gather_wrapper
except ImportError:
    print("Run: bash build.sh")
    sys.exit(1)


def run_test(batch, topk, num_full_blocks, label):
    total_sel = batch * topk
    seq_len = num_full_blocks * BLOCK_SIZE

    sel_k_rope = torch.zeros(total_sel, BLOCK_SIZE, K_ROPE_DIM, dtype=DTYPE, device=DEVICE)
    sel_kv = torch.zeros(total_sel, BLOCK_SIZE, KV_DIM, dtype=DTYPE, device=DEVICE)
    sel_bt = torch.arange(total_sel, dtype=torch.int32, device=DEVICE).view(batch, topk)
    sel_bs = -torch.ones(batch, 1, 1, topk + 1, dtype=torch.int32, device=DEVICE)
    topk_idx = torch.zeros(batch, 1, 1, topk, dtype=torch.int32, device=DEVICE)
    for b in range(batch):
        for k in range(topk):
            topk_idx[b, 0, 0, k] = (b * topk + k) % num_full_blocks

    full_k_rope = torch.zeros(num_full_blocks, BLOCK_SIZE, K_ROPE_DIM, dtype=DTYPE, device=DEVICE)
    full_kv = torch.zeros(num_full_blocks, BLOCK_SIZE, KV_DIM, dtype=DTYPE, device=DEVICE)
    full_bt = torch.arange(num_full_blocks, dtype=torch.int32, device=DEVICE).unsqueeze(0).expand(batch, -1).contiguous()
    full_seq = torch.full((batch,), seq_len, dtype=torch.int32, device=DEVICE)
    full_q = torch.ones(batch, dtype=torch.int32, device=DEVICE)

    print(f"\n--- {label} ---")
    print(f"  B={batch}, topk={topk}, full_blocks={num_full_blocks}")
    print(f"  sel_kv: {sel_kv.shape}, full_kv: {full_kv.shape}")
    print(f"  sel_bt: {sel_bt.shape}, full_bt: {full_bt.shape}")
    print(f"  topk_idx: {topk_idx.shape}, sel_bs: {sel_bs.shape}")

    try:
        result = gather_wrapper.npu_gather_selection_kv_cache(
            sel_k_rope, sel_kv, sel_bt, sel_bs, topk_idx,
            full_k_rope, full_kv, full_bt, full_seq, full_q,
            TOPK_BLOCK_SIZE)
        torch.npu.synchronize()
        print(f"  PASS: result={result.cpu().tolist()[:4]}...")
        return True
    except RuntimeError as e:
        print(f"  FAIL: {e}")
        return False


results = {}

# Control: same as original test (known PASS)
results["B=2,topk=4"] = run_test(2, 4, 16, "Control (B=2, topk=4)")

# Increase topk to 32 boundary
results["B=2,topk=32"] = run_test(2, 32, 64, "topk=32 (scalar boundary)")

# Cross to vector path
results["B=2,topk=33"] = run_test(2, 33, 66, "topk=33 (vector path entry)")

# Production topk
results["B=2,topk=64"] = run_test(2, 64, 128, "topk=64 (production topk)")

# Production batch
results["B=16,topk=64"] = run_test(16, 64, 260, "Production dims (B=16, topk=64)")

# Large full blocks (production-scale)
results["B=16,topk=64,18k"] = run_test(16, 64, 18324, "Full production (B=16, topk=64, 18k blocks)")

print("\n=== Summary ===")
for k, v in results.items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")
