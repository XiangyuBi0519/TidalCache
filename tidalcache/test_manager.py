"""
TidalCacheManager integration test.

Simulates the vllm-ascend decode flow:
  1. Init TidalCacheManager
  2. Allocate layers (like _allocate_kv_cache_tensors)
  3. Write data to Host Full KV (simulating compressor scatter)
  4. Call gather() (simulating _forward_decode insertion point)
  5. Verify gathered data matches Host source

Run on NPU machine:
  cd test_gather && bash build.sh  # if not already built
  cd .. && python3 -m tidalcache.test_manager
"""

import sys
import os

# Add test_gather to path for gather_wrapper and zero_copy_npu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test_gather"))

import torch
import torch_npu

DEVICE = torch.device("npu:0")
DTYPE = torch.bfloat16

# DSA parameters (scaled down for testing)
NUM_BLOCKS = 32
BLOCK_SIZE = 64
KV_DIM = 512
ROPE_DIM = 64
INDEX_TOPK = 4
MAX_BATCH = 4
NUM_LAYERS = 2  # test with 2 layers


def test_manager_lifecycle():
    """Test full TidalCacheManager lifecycle."""
    print("=" * 60)
    print("Test: TidalCacheManager Lifecycle")
    print("=" * 60)

    from tidalcache.offload_manager import TidalCacheManager

    mgr = TidalCacheManager(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        kv_dim=KV_DIM,
        rope_dim=ROPE_DIM,
        index_topk=INDEX_TOPK,
        max_batch_size=MAX_BATCH,
        dtype=DTYPE,
        device=DEVICE,
    )

    # Step 1: Allocate layers
    layer_names = [f"model.layers.{i}.self_attn" for i in range(NUM_LAYERS)]
    for name in layer_names:
        state = mgr.alloc_layer(name)
        print(f"  Allocated {name}:")
        print(f"    host_kv: shape={state.host_kv_cache.shape}, "
              f"ptr=0x{state.host_kv_cache.data_ptr():x}")
        print(f"    npu_kv:  shape={state.npu_kv_cache.shape}, "
              f"device={state.npu_kv_cache.device}, "
              f"ptr=0x{state.npu_kv_cache.data_ptr():x}")
        print(f"    sel_kv:  shape={state.sel_kv_cache.shape}, "
              f"device={state.sel_kv_cache.device}")
        assert state.host_kv_cache.data_ptr() != state.npu_kv_cache.data_ptr(), \
            "host_ptr should differ from dev_ptr"

    # Step 2: Simulate compressor scatter → write to Host Full KV
    print("\n  Writing marker data to Host Full KV Cache...")
    for name in layer_names:
        state = mgr.layers[name]
        for b in range(NUM_BLOCKS):
            state.host_kv_cache[b, 0, 0] = float(b + 1000)
            state.host_k_rope[b, 0, 0] = float(b + 2000)

    # Step 3: Simulate decode — call gather
    print("\n  Simulating decode gather...")
    batch_size = 2
    topk_indices = torch.zeros(
        batch_size, 1, 1, INDEX_TOPK,
        dtype=torch.int32, device=DEVICE,
    )
    # Batch 0 selects blocks [0, 3, 7, 15]
    topk_indices[0, 0, 0] = torch.tensor([0, 3, 7, 15], dtype=torch.int32)
    # Batch 1 selects blocks [1, 5, 10, 20]
    topk_indices[1, 0, 0] = torch.tensor([1, 5, 10, 20], dtype=torch.int32)

    full_block_table = torch.arange(
        NUM_BLOCKS, dtype=torch.int32, device=DEVICE,
    ).unsqueeze(0).expand(batch_size, -1).contiguous()
    full_actual_seq = torch.full(
        (batch_size,), NUM_BLOCKS * BLOCK_SIZE,
        dtype=torch.int32, device=DEVICE,
    )
    full_q_actual_seq = torch.ones(
        batch_size, dtype=torch.int32, device=DEVICE,
    )

    passed = True
    for name in layer_names:
        sel_kv, sel_rope, sel_actual = mgr.gather(
            name, topk_indices,
            full_block_table, full_actual_seq, full_q_actual_seq,
        )
        torch.npu.synchronize()

        state = mgr.layers[name]
        status = state.sel_block_status[:batch_size]

        print(f"\n  Layer {name}:")
        print(f"    sel_actual_seq = {sel_actual.cpu().tolist()}")

        # Verify: check each batch's gathered data
        for b in range(batch_size):
            bs = status[b, 0, 0, :INDEX_TOPK].cpu()
            for slot in range(INDEX_TOPK):
                group = bs[slot].item()
                if group < 0:
                    continue
                sel_block_idx = b * INDEX_TOPK + slot
                actual_kv = sel_kv[sel_block_idx, 0, 0].cpu().item()
                expected_kv = float(group + 1000)
                actual_rope = sel_rope[sel_block_idx, 0, 0].cpu().item()
                expected_rope = float(group + 2000)
                kv_ok = abs(actual_kv - expected_kv) < 1e-2
                rope_ok = abs(actual_rope - expected_rope) < 1e-2
                if not kv_ok or not rope_ok:
                    print(f"    MISMATCH b={b} slot={slot} group={group}: "
                          f"kv={actual_kv:.0f}(exp {expected_kv:.0f}) "
                          f"rope={actual_rope:.0f}(exp {expected_rope:.0f})")
                    passed = False

    # Step 4: Test cache reuse — second call with overlapping topk
    print("\n  Testing cache reuse (second gather with overlap)...")
    topk_indices_2 = topk_indices.clone()
    # Keep first 2 the same, change last 2
    topk_indices_2[0, 0, 0, 2] = 8   # was 7
    topk_indices_2[0, 0, 0, 3] = 12  # was 15
    topk_indices_2[1, 0, 0, 2] = 11  # was 10
    topk_indices_2[1, 0, 0, 3] = 25  # was 20

    for name in layer_names:
        sel_kv, sel_rope, sel_actual = mgr.gather(
            name, topk_indices_2,
            full_block_table, full_actual_seq, full_q_actual_seq,
        )
        torch.npu.synchronize()

        state = mgr.layers[name]
        status = state.sel_block_status[:batch_size]
        print(f"  Layer {name} round 2 status: "
              f"{status[:, 0, 0, :INDEX_TOPK].cpu().tolist()}")

        for b in range(batch_size):
            bs = status[b, 0, 0, :INDEX_TOPK].cpu()
            for slot in range(INDEX_TOPK):
                group = bs[slot].item()
                if group < 0:
                    continue
                sel_block_idx = b * INDEX_TOPK + slot
                actual_kv = sel_kv[sel_block_idx, 0, 0].cpu().item()
                expected_kv = float(group + 1000)
                if abs(actual_kv - expected_kv) > 1e-2:
                    print(f"    REUSE MISMATCH b={b} slot={slot} "
                          f"group={group}: {actual_kv:.0f} != {expected_kv:.0f}")
                    passed = False

    # Step 5: Test reset
    print("\n  Testing batch reset...")
    reset_idx = torch.tensor([0], dtype=torch.long, device=DEVICE)
    for name in layer_names:
        mgr.reset_requests(name, reset_idx)
        state = mgr.layers[name]
        b0_status = state.sel_block_status[0, 0, 0, :INDEX_TOPK].cpu()
        assert (b0_status == -1).all(), \
            f"Batch 0 should be reset to -1, got {b0_status}"
        b1_status = state.sel_block_status[1, 0, 0, :INDEX_TOPK].cpu()
        assert not (b1_status == -1).all(), \
            "Batch 1 should NOT be reset"
    print("  Reset OK: batch 0 cleared, batch 1 preserved")

    # Cleanup
    mgr.cleanup()
    print(f"\n  Test: {'PASS' if passed else 'FAIL'}")
    return passed


def test_multi_layer_independence():
    """Verify different layers have independent state."""
    print("\n" + "=" * 60)
    print("Test: Multi-Layer Independence")
    print("=" * 60)

    from tidalcache.offload_manager import TidalCacheManager

    mgr = TidalCacheManager(
        num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE,
        kv_dim=KV_DIM, rope_dim=ROPE_DIM,
        index_topk=INDEX_TOPK, max_batch_size=MAX_BATCH,
        dtype=DTYPE, device=DEVICE,
    )

    mgr.alloc_layer("layer_0")
    mgr.alloc_layer("layer_1")

    # Write different data per layer
    mgr.layers["layer_0"].host_kv_cache[:, 0, 0] = 100.0
    mgr.layers["layer_1"].host_kv_cache[:, 0, 0] = 200.0

    batch_size = 1
    topk = torch.tensor([[[[0, 1, 2, 3]]]], dtype=torch.int32, device=DEVICE)
    bt = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=DEVICE).unsqueeze(0)
    seq = torch.tensor([NUM_BLOCKS * BLOCK_SIZE], dtype=torch.int32, device=DEVICE)
    qseq = torch.ones(1, dtype=torch.int32, device=DEVICE)

    sel0, _, _ = mgr.gather("layer_0", topk, bt, seq, qseq)
    sel1, _, _ = mgr.gather("layer_1", topk, bt, seq, qseq)
    torch.npu.synchronize()

    val0 = sel0[0, 0, 0].cpu().item()
    val1 = sel1[0, 0, 0].cpu().item()
    passed = abs(val0 - 100.0) < 1e-2 and abs(val1 - 200.0) < 1e-2
    print(f"  Layer 0 gathered: {val0:.0f} (expect 100)")
    print(f"  Layer 1 gathered: {val1:.0f} (expect 200)")
    print(f"  Test: {'PASS' if passed else 'FAIL'}")

    mgr.cleanup()
    return passed


if __name__ == "__main__":
    print("TidalCacheManager Integration Test")
    print(f"torch_npu: {torch_npu.__version__}")
    print(f"Device: {torch.npu.get_device_name(0)}")
    print()

    results = {
        "Lifecycle": test_manager_lifecycle(),
        "Multi-Layer Independence": test_multi_layer_independence(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, result in results.items():
        print(f"  {name}: {'PASS' if result else 'FAIL'}")

    if all(results.values()):
        print("\nAll tests passed!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)
