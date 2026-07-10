"""
Test script for GatherSelectionKvCache operator.

Tests:
  1. Basic gather: fill full KV cache with known data, trigger gathering,
     verify selection cache gets correct values.
  2. Cache reuse: call again with overlapping top-k indices, verify reuse
     (some entries should already be in selection cache).

Tensor shapes and semantics (from op source analysis):
  - full_k_rope:      [f_blk, f_blk_size, k_rope_dim]     Host pinned
  - full_kv_cache:    [f_blk, f_blk_size, kv_cache_dim]    Host pinned
  - full_kv_block_table: [B, f_max_blk]                    INT32, Device
  - full_kv_actual_seq:  [B]                                INT32, Device
  - full_q_actual_seq:   [B]                                INT32, Device
  - selection_k_rope:    [s_blk, s_blk_size, k_rope_dim]   Device
  - selection_kv_cache:  [s_blk, s_blk_size, kv_cache_dim] Device
  - selection_kv_block_table: [B*S*H, s_max_blk]           INT32, Device
  - selection_kv_block_status:[B, S, H, topk+1]            INT32, Device
  - selection_topk_indices:   [B, S, H, topk]              INT32, Device
  - Output:
    selection_kv_actual_seq: [B*S*H]                        INT32, Device

Key constants:
  - selTopKBlockSize = 64 (each topk index = a group of 64 tokens)
  - f_blk_size is typically 128 (full block size, page-attention style)
  - kv_cache_dim = 512 (MLA latent dim, c^KV)
  - k_rope_dim = 64 (RoPE position dim, k^R)
"""
import sys
import torch
import numpy as np

try:
    import torch_npu
except ImportError:
    print("ERROR: torch_npu not available. Run on NPU machine.")
    sys.exit(1)

try:
    import gather_kv_wrapper
except ImportError:
    print("ERROR: gather_kv_wrapper not found. Build first with:")
    print("  python setup.py build_ext --inplace")
    sys.exit(1)


def create_test_data(
    batch_size=1,
    num_heads=1,  # S*H combined, simplified
    topk=4,       # small for testing (tilingKey=1 path: topk<=32)
    sel_topk_block_size=64,
    full_block_size=128,
    kv_cache_dim=512,
    k_rope_dim=64,
    full_seq_len=1024,   # total tokens in full cache
    dtype=torch.float16,
):
    """Create test tensors with known data patterns for verification."""

    device = torch.device("npu:0")

    # --- Full KV Cache (Host pinned memory) ---
    # Calculate number of full blocks needed
    num_full_blocks = (full_seq_len + full_block_size - 1) // full_block_size
    # Add some extra blocks for padding
    total_full_blocks = num_full_blocks + 4

    # Fill with identifiable pattern
    # For initial test, put full cache on Device too (Host offload tested separately)
    full_k_rope = torch.zeros(total_full_blocks, full_block_size, k_rope_dim,
                              dtype=dtype)
    full_kv_cache = torch.zeros(total_full_blocks, full_block_size, kv_cache_dim,
                                dtype=dtype)

    for blk in range(num_full_blocks):
        for tok in range(full_block_size):
            global_tok_id = blk * full_block_size + tok
            if global_tok_id < full_seq_len:
                val = float(global_tok_id + 1)
                full_k_rope[blk, tok, :] = val
                full_kv_cache[blk, tok, :] = val

    full_k_rope = full_k_rope.to(device)
    full_kv_cache = full_kv_cache.to(device)

    # Block table: maps logical block index -> physical block index
    # Simple 1:1 mapping for test
    max_full_blocks_per_seq = num_full_blocks
    full_kv_block_table = torch.arange(
        num_full_blocks, dtype=torch.int32
    ).unsqueeze(0).expand(batch_size, -1).contiguous().to(device)

    # Actual sequence lengths
    full_kv_actual_seq = torch.tensor([full_seq_len], dtype=torch.int32).to(device)
    full_q_actual_seq = torch.tensor([1], dtype=torch.int32).to(device)  # decode step

    # --- Selection Cache (Device) ---
    # Number of selection blocks needed: topk groups, each group = sel_topk_block_size tokens
    # But stored in pages. Let's assume selection block size = sel_topk_block_size for simplicity
    sel_block_size = sel_topk_block_size  # each selection block = 64 tokens
    num_sel_blocks = topk + 4  # extra padding
    total_sel_blocks = num_sel_blocks * batch_size * num_heads + 4

    selection_k_rope = torch.zeros(total_sel_blocks, sel_block_size, k_rope_dim,
                                   dtype=dtype, device=device)
    selection_kv_cache = torch.zeros(total_sel_blocks, sel_block_size, kv_cache_dim,
                                     dtype=dtype, device=device)

    # Selection block table: [B*S*H, s_max_blk]
    # Maps each head's selection blocks to physical block indices
    bsh = batch_size * num_heads
    s_max_blk = topk  # each topk entry maps to one selection block
    selection_kv_block_table = torch.zeros(bsh, s_max_blk, dtype=torch.int32, device=device)
    for i in range(bsh):
        for j in range(s_max_blk):
            selection_kv_block_table[i, j] = i * s_max_blk + j

    # Block status: [B, S, H, topk+1] - initially all -1 (no cache)
    # -1 means the slot is empty/invalid
    S = 1  # num_seqs = 1
    H = num_heads
    selection_kv_block_status = torch.full(
        (batch_size, S, H, topk + 1), -1, dtype=torch.int32, device=device)

    # --- Top-k indices ---
    # Each index is a "block group index" into the full sequence
    # block_group_index = token_offset // sel_topk_block_size
    # E.g., topk_indices=[0, 2, 5, 8] means we want groups of tokens:
    #   group 0: tokens 0-63
    #   group 2: tokens 128-191
    #   group 5: tokens 320-383
    #   group 8: tokens 512-575
    max_group = full_seq_len // sel_topk_block_size
    # Pick some groups spread across the sequence
    topk_indices_values = sorted(np.random.choice(range(max_group), size=topk, replace=False).tolist())
    selection_topk_indices = torch.tensor(
        topk_indices_values, dtype=torch.int32
    ).reshape(batch_size, S, H, topk).to(device)

    print(f"Test configuration:")
    print(f"  batch_size={batch_size}, num_heads={H}, topk={topk}")
    print(f"  sel_topk_block_size={sel_topk_block_size}")
    print(f"  full_seq_len={full_seq_len}, full_block_size={full_block_size}")
    print(f"  kv_cache_dim={kv_cache_dim}, k_rope_dim={k_rope_dim}")
    print(f"  num_full_blocks={num_full_blocks}, total_sel_blocks={total_sel_blocks}")
    print(f"  topk_indices={topk_indices_values}")
    print(f"  full_k_rope: shape={full_k_rope.shape}, device={full_k_rope.device}")
    print(f"  full_kv_cache: shape={full_kv_cache.shape}, device={full_kv_cache.device}")
    print(f"  selection_k_rope: shape={selection_k_rope.shape}, device={device}")
    print(f"  selection_kv_cache: shape={selection_kv_cache.shape}, device={device}")
    print(f"  selection_kv_block_table: shape={selection_kv_block_table.shape}")
    print(f"  selection_kv_block_status: shape={selection_kv_block_status.shape}")
    print(f"  selection_topk_indices: shape={selection_topk_indices.shape}")

    return {
        "full_k_rope": full_k_rope,
        "full_kv_cache": full_kv_cache,
        "full_kv_block_table": full_kv_block_table,
        "full_kv_actual_seq": full_kv_actual_seq,
        "full_q_actual_seq": full_q_actual_seq,
        "selection_k_rope": selection_k_rope,
        "selection_kv_cache": selection_kv_cache,
        "selection_kv_block_table": selection_kv_block_table,
        "selection_kv_block_status": selection_kv_block_status,
        "selection_topk_indices": selection_topk_indices,
        "sel_topk_block_size": sel_topk_block_size,
        "topk_indices_values": topk_indices_values,
        "topk": topk,
    }


def test_basic_gather():
    """Test 1: Basic gather from full cache to selection cache."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Gather")
    print("=" * 60)

    data = create_test_data(topk=4)

    print("\nCalling GatherSelectionKvCache...")
    selection_kv_actual_seq = gather_kv_wrapper.npu_gather_selection_kv_cache(
        data["selection_k_rope"],
        data["selection_kv_cache"],
        data["selection_kv_block_table"],
        data["selection_kv_block_status"],
        data["selection_topk_indices"],
        data["full_k_rope"],
        data["full_kv_cache"],
        data["full_kv_block_table"],
        data["full_kv_actual_seq"],
        data["full_q_actual_seq"],
        data["sel_topk_block_size"],
    )

    torch.npu.synchronize()

    print(f"\nselection_kv_actual_seq: {selection_kv_actual_seq.cpu()}")
    print(f"selection_kv_block_status:\n{data['selection_kv_block_status'].cpu()}")

    # Verify: check that selection cache has the right data
    sel_cache = data["selection_kv_cache"].cpu()
    sel_rope = data["selection_k_rope"].cpu()
    sel_block_table = data["selection_kv_block_table"].cpu()

    passed = True
    for i, group_idx in enumerate(data["topk_indices_values"]):
        sel_blk = sel_block_table[0, i].item()
        # First token in this group
        first_token = group_idx * data["sel_topk_block_size"]
        expected_val = float(first_token + 1)  # 1-indexed
        actual_val = sel_cache[sel_blk, 0, 0].item()

        if abs(actual_val - expected_val) > 0.1:
            print(f"  FAIL: group {group_idx}, sel_blk {sel_blk}, "
                  f"expected {expected_val}, got {actual_val}")
            passed = False
        else:
            print(f"  OK: group {group_idx}, sel_blk {sel_blk}, "
                  f"value={actual_val} (expected {expected_val})")

    if passed:
        print("\nTEST 1 PASSED")
    else:
        print("\nTEST 1 FAILED")
    return passed


def test_cache_reuse():
    """Test 2: Cache reuse - call twice with overlapping indices."""
    print("\n" + "=" * 60)
    print("TEST 2: Cache Reuse")
    print("=" * 60)

    data = create_test_data(topk=4)

    # First call
    print("\nFirst gather call...")
    selection_kv_actual_seq = gather_kv_wrapper.npu_gather_selection_kv_cache(
        data["selection_k_rope"],
        data["selection_kv_cache"],
        data["selection_kv_block_table"],
        data["selection_kv_block_status"],
        data["selection_topk_indices"],
        data["full_k_rope"],
        data["full_kv_cache"],
        data["full_kv_block_table"],
        data["full_kv_actual_seq"],
        data["full_q_actual_seq"],
        data["sel_topk_block_size"],
    )
    torch.npu.synchronize()

    print(f"After first call - block_status:\n{data['selection_kv_block_status'].cpu()}")

    # Second call with overlapping indices
    # Keep 2 of the original groups, replace 2 with new ones
    old_indices = data["topk_indices_values"]
    max_group = 1024 // data["sel_topk_block_size"]
    # Pick 2 new groups not in old_indices
    new_groups = []
    for g in range(max_group):
        if g not in old_indices and len(new_groups) < 2:
            new_groups.append(g)

    new_indices = sorted(old_indices[:2] + new_groups)
    print(f"\nSecond call: old_indices={old_indices}, new_indices={new_indices}")
    print(f"  Overlapping: {old_indices[:2]} (should be reused)")
    print(f"  New: {new_groups} (should be fetched from host)")

    new_topk_indices = torch.tensor(
        new_indices, dtype=torch.int32
    ).reshape(1, 1, 1, 4).to(torch.device("npu:0"))
    data["selection_topk_indices"] = new_topk_indices

    print("\nSecond gather call...")
    selection_kv_actual_seq2 = gather_kv_wrapper.npu_gather_selection_kv_cache(
        data["selection_k_rope"],
        data["selection_kv_cache"],
        data["selection_kv_block_table"],
        data["selection_kv_block_status"],
        data["selection_topk_indices"],
        data["full_k_rope"],
        data["full_kv_cache"],
        data["full_kv_block_table"],
        data["full_kv_actual_seq"],
        data["full_q_actual_seq"],
        data["sel_topk_block_size"],
    )
    torch.npu.synchronize()

    print(f"After second call - block_status:\n{data['selection_kv_block_status'].cpu()}")

    # Verify using block_status to find which slot holds which group
    # block_status stores the group index for each slot (not in topk order)
    sel_cache = data["selection_kv_cache"].cpu()
    sel_block_table = data["selection_kv_block_table"].cpu()
    block_status = data["selection_kv_block_status"].cpu()

    passed = True
    for slot_idx in range(data["topk"]):
        group_idx = block_status[0, 0, 0, slot_idx].item()
        sel_blk = sel_block_table[0, slot_idx].item()
        first_token = group_idx * data["sel_topk_block_size"]
        expected_val = float(first_token + 1)
        actual_val = sel_cache[sel_blk, 0, 0].item()

        reused = "REUSED" if group_idx in old_indices[:2] else "NEW"
        if abs(actual_val - expected_val) > 0.1:
            print(f"  FAIL [{reused}]: slot {slot_idx} -> group {group_idx}, "
                  f"expected {expected_val}, got {actual_val}")
            passed = False
        else:
            print(f"  OK [{reused}]: slot {slot_idx} -> group {group_idx}, value={actual_val}")

    if passed:
        print("\nTEST 2 PASSED")
    else:
        print("\nTEST 2 FAILED")
    return passed


if __name__ == "__main__":
    print("GatherSelectionKvCache Operator Test")
    print(f"torch_npu version: {torch_npu.__version__}")
    print(f"Device: {torch.npu.get_device_name(0)}")

    results = []
    results.append(("Basic Gather", test_basic_gather()))
    results.append(("Cache Reuse", test_cache_reuse()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    sys.exit(0 if all_passed else 1)
