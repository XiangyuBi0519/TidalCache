"""
GatherSelectionKvCache 完整验证测试

Test 1: Basic Gather (Device→Device) — 验证算子基本功能
Test 2: Cache Reuse (Device→Device) — 验证缓存复用机制
Test 3: Host Memory Registration — 验证 hugepage + NPU MMU 注册
Test 4: Host→Device Gather — 完整链路: Host hugepage 上的 Full KV → Device Selection Cache
"""

import os
import sys
import mmap
import torch
import torch_npu

# ─── 配置 ───
DEVICE = "npu:0"
DTYPE = torch.bfloat16
BLOCK_SIZE = 64
KV_DIM = 512
K_ROPE_DIM = 64
TOPK = 4
TOPK_BLOCK_SIZE = 64
BATCH = 2
NUM_FULL_BLOCKS = 16
NUM_SEL_BLOCKS_PER_BATCH = TOPK
TOTAL_SEL_BLOCKS = NUM_SEL_BLOCKS_PER_BATCH * BATCH
SEQ_LEN = NUM_FULL_BLOCKS * BLOCK_SIZE  # 1024 tokens

# ─── 导入 ───
try:
    import gather_wrapper
    print("[OK] gather_wrapper loaded")
except ImportError as e:
    print(f"[FAIL] Cannot import gather_wrapper: {e}")
    print("Run: bash build.sh")
    sys.exit(1)


def make_full_data_on_device():
    """创建 Device 上的测试数据（Test 1/2 使用）"""
    full_k_rope = torch.randn(
        NUM_FULL_BLOCKS, BLOCK_SIZE, K_ROPE_DIM,
        dtype=DTYPE, device=DEVICE)
    full_kv_cache = torch.randn(
        NUM_FULL_BLOCKS, BLOCK_SIZE, KV_DIM,
        dtype=DTYPE, device=DEVICE)
    for b in range(NUM_FULL_BLOCKS):
        full_k_rope[b, 0, 0] = float(b + 100)
        full_kv_cache[b, 0, 0] = float(b + 200)
    return full_k_rope, full_kv_cache


def make_common_tensors(full_k_rope, full_kv_cache):
    """创建算子调用需要的公共张量"""
    sel_k_rope = torch.zeros(
        TOTAL_SEL_BLOCKS, BLOCK_SIZE, K_ROPE_DIM,
        dtype=DTYPE, device=DEVICE)
    sel_kv_cache = torch.zeros(
        TOTAL_SEL_BLOCKS, BLOCK_SIZE, KV_DIM,
        dtype=DTYPE, device=DEVICE)
    sel_block_table = torch.arange(
        TOTAL_SEL_BLOCKS, dtype=torch.int32, device=DEVICE
    ).view(BATCH, NUM_SEL_BLOCKS_PER_BATCH)
    sel_block_status = -torch.ones(
        BATCH, 1, 1, TOPK + 1, dtype=torch.int32, device=DEVICE)

    topk_indices = torch.zeros(
        BATCH, 1, 1, TOPK, dtype=torch.int32, device=DEVICE)
    for b in range(BATCH):
        for k in range(TOPK):
            topk_indices[b, 0, 0, k] = b * TOPK + k

    full_block_table = torch.arange(
        NUM_FULL_BLOCKS, dtype=torch.int32, device=DEVICE
    ).unsqueeze(0).expand(BATCH, -1).contiguous()

    full_kv_actual_seq = torch.full(
        (BATCH,), SEQ_LEN, dtype=torch.int32, device=DEVICE)
    full_q_actual_seq = torch.ones(
        BATCH, dtype=torch.int32, device=DEVICE)

    return (sel_k_rope, sel_kv_cache, sel_block_table, sel_block_status,
            topk_indices, full_block_table, full_kv_actual_seq, full_q_actual_seq)


def verify_gather(sel_kv_cache, sel_block_status, topk_indices,
                  full_kv_cache, test_name):
    """验证 gather 结果：通过 block_status 找 slot→group 映射"""
    passed = True
    for b in range(BATCH):
        status = sel_block_status[b, 0, 0, :TOPK].cpu()
        for slot_idx in range(TOPK):
            group_idx = status[slot_idx].item()
            if group_idx < 0:
                print(f"  [{test_name}] batch={b} slot={slot_idx}: empty (group=-1)")
                continue
            full_block_idx = group_idx
            sel_block_idx = b * NUM_SEL_BLOCKS_PER_BATCH + slot_idx
            expected = full_kv_cache[full_block_idx, 0, 0].item()
            actual = sel_kv_cache[sel_block_idx, 0, 0].item()
            match = abs(expected - actual) < 1e-2
            if not match:
                print(f"  [{test_name}] MISMATCH batch={b} slot={slot_idx} "
                      f"group={group_idx}: expected={expected:.1f} actual={actual:.1f}")
                passed = False
    return passed


# ═══════════════════════════════════════════════════════════
# Test 1: Basic Gather (全 Device)
# ═══════════════════════════════════════════════════════════
def test_basic_gather():
    print("\n" + "=" * 60)
    print("Test 1: Basic Gather (Device → Device)")
    print("=" * 60)

    full_k_rope, full_kv_cache = make_full_data_on_device()
    (sel_k_rope, sel_kv_cache, sel_block_table, sel_block_status,
     topk_indices, full_block_table, full_kv_actual_seq,
     full_q_actual_seq) = make_common_tensors(full_k_rope, full_kv_cache)

    result = gather_wrapper.npu_gather_selection_kv_cache(
        sel_k_rope, sel_kv_cache, sel_block_table, sel_block_status,
        topk_indices, full_k_rope, full_kv_cache, full_block_table,
        full_kv_actual_seq, full_q_actual_seq, TOPK_BLOCK_SIZE)
    torch.npu.synchronize()

    passed = verify_gather(
        sel_kv_cache, sel_block_status, topk_indices,
        full_kv_cache, "Basic")

    print(f"  selection_kv_actual_seq = {result.cpu().tolist()}")
    print(f"  block_status = {sel_block_status.cpu().numpy()}")
    print(f"  Test 1: {'PASS' if passed else 'FAIL'}")
    return passed


# ═══════════════════════════════════════════════════════════
# Test 2: Cache Reuse (重叠 topk 测试复用)
# ═══════════════════════════════════════════════════════════
def test_cache_reuse():
    print("\n" + "=" * 60)
    print("Test 2: Cache Reuse (overlapping topk)")
    print("=" * 60)

    full_k_rope, full_kv_cache = make_full_data_on_device()
    (sel_k_rope, sel_kv_cache, sel_block_table, sel_block_status,
     topk_indices, full_block_table, full_kv_actual_seq,
     full_q_actual_seq) = make_common_tensors(full_k_rope, full_kv_cache)

    # 第一次调用
    gather_wrapper.npu_gather_selection_kv_cache(
        sel_k_rope, sel_kv_cache, sel_block_table, sel_block_status,
        topk_indices, full_k_rope, full_kv_cache, full_block_table,
        full_kv_actual_seq, full_q_actual_seq, TOPK_BLOCK_SIZE)
    torch.npu.synchronize()
    print("  Round 1 status:", sel_block_status.cpu().numpy())

    # 第二次调用: 修改 topk，部分重叠
    new_topk = topk_indices.clone()
    for b in range(BATCH):
        old = topk_indices[b, 0, 0].cpu().tolist()
        new_topk[b, 0, 0, 0] = old[0]          # 保留
        new_topk[b, 0, 0, 1] = old[1]          # 保留
        new_topk[b, 0, 0, 2] = old[2] + TOPK   # 新的
        new_topk[b, 0, 0, 3] = old[3] + TOPK   # 新的

    result2 = gather_wrapper.npu_gather_selection_kv_cache(
        sel_k_rope, sel_kv_cache, sel_block_table, sel_block_status,
        new_topk, full_k_rope, full_kv_cache, full_block_table,
        full_kv_actual_seq, full_q_actual_seq, TOPK_BLOCK_SIZE)
    torch.npu.synchronize()
    print("  Round 2 status:", sel_block_status.cpu().numpy())

    passed = verify_gather(
        sel_kv_cache, sel_block_status, new_topk,
        full_kv_cache, "Reuse")

    print(f"  Test 2: {'PASS' if passed else 'FAIL'}")
    return passed


# ═══════════════════════════════════════════════════════════
# Test 3: Host Memory Registration (hugepage + NPU MMU)
# ═══════════════════════════════════════════════════════════
def test_host_memory_registration():
    print("\n" + "=" * 60)
    print("Test 3: Host Memory Registration (hugepage + NPU MMU)")
    print("=" * 60)

    hugepage_path = "/dev/hugepages/test_dsa_offload"
    HUGEPAGE_SIZE = 2 * 1024 * 1024  # 2MB

    if not os.path.isdir("/dev/hugepages"):
        print("  SKIP: /dev/hugepages not available")
        return None

    try:
        import zero_copy_npu
        print("[OK] zero_copy_npu loaded")
    except ImportError as e:
        print(f"  SKIP: Cannot import zero_copy_npu: {e}")
        return None

    num_elements = NUM_FULL_BLOCKS * BLOCK_SIZE * KV_DIM
    data_bytes = num_elements * 2  # BF16 = 2 bytes
    aligned_size = ((data_bytes + HUGEPAGE_SIZE - 1) // HUGEPAGE_SIZE) * HUGEPAGE_SIZE

    print(f"  Data size: {data_bytes} bytes, aligned: {aligned_size} bytes")

    try:
        fd = os.open(hugepage_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(fd, aligned_size)
        mmap_obj = mmap.mmap(fd, aligned_size,
                             flags=mmap.MAP_SHARED,
                             prot=mmap.PROT_READ | mmap.PROT_WRITE)

        host_tensor = torch.frombuffer(
            mmap_obj, dtype=DTYPE, count=num_elements
        ).view(NUM_FULL_BLOCKS, BLOCK_SIZE, KV_DIM)

        for b in range(NUM_FULL_BLOCKS):
            host_tensor[b, 0, 0] = float(b + 300)

        print(f"  Host tensor: shape={host_tensor.shape}, "
              f"ptr=0x{host_tensor.data_ptr():x}")

        host_out, npu_tensor = zero_copy_npu.register_hugepage_as_npu_tensor(
            host_tensor, 0)

        print(f"  NPU tensor: shape={npu_tensor.shape}, "
              f"device={npu_tensor.device}, "
              f"ptr=0x{npu_tensor.data_ptr():x}")

        assert npu_tensor.device.type in ("npu", "privateuseone"), \
            f"npu_tensor should be on NPU device, got {npu_tensor.device}"

        print(f"  Host ptr:  0x{host_tensor.data_ptr():x}")
        print(f"  NPU ptr:   0x{npu_tensor.data_ptr():x}")
        print(f"  Same ptr:  {host_tensor.data_ptr() == npu_tensor.data_ptr()}")
        print("  Test 3: PASS")
        return True

    except Exception as e:
        print(f"  Test 3: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            mmap_obj.close()
            os.close(fd)
            os.unlink(hugepage_path)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# Test 4: Host→Device Gather (完整链路)
# ═══════════════════════════════════════════════════════════
def test_host_to_device_gather():
    print("\n" + "=" * 60)
    print("Test 4: Host → Device Gather (full pipeline)")
    print("=" * 60)

    hugepage_path_kv = "/dev/hugepages/test_dsa_full_kv"
    hugepage_path_rope = "/dev/hugepages/test_dsa_full_rope"
    HUGEPAGE_SIZE = 2 * 1024 * 1024

    if not os.path.isdir("/dev/hugepages"):
        print("  SKIP: /dev/hugepages not available")
        return None

    try:
        import zero_copy_npu
    except ImportError as e:
        print(f"  SKIP: Cannot import zero_copy_npu: {e}")
        return None

    fds = []
    mmaps = []

    try:
        # ── 分配 full_kv_cache 在 hugepage 上 ──
        kv_elements = NUM_FULL_BLOCKS * BLOCK_SIZE * KV_DIM
        kv_bytes = kv_elements * 2
        kv_aligned = ((kv_bytes + HUGEPAGE_SIZE - 1) // HUGEPAGE_SIZE) * HUGEPAGE_SIZE

        fd_kv = os.open(hugepage_path_kv, os.O_CREAT | os.O_RDWR, 0o600)
        fds.append(fd_kv)
        os.ftruncate(fd_kv, kv_aligned)
        mmap_kv = mmap.mmap(fd_kv, kv_aligned, flags=mmap.MAP_SHARED,
                            prot=mmap.PROT_READ | mmap.PROT_WRITE)
        mmaps.append(mmap_kv)

        host_kv = torch.frombuffer(mmap_kv, dtype=DTYPE, count=kv_elements
                                   ).view(NUM_FULL_BLOCKS, BLOCK_SIZE, KV_DIM)

        # ── 分配 full_k_rope 在 hugepage 上 ──
        rope_elements = NUM_FULL_BLOCKS * BLOCK_SIZE * K_ROPE_DIM
        rope_bytes = rope_elements * 2
        rope_aligned = ((rope_bytes + HUGEPAGE_SIZE - 1) // HUGEPAGE_SIZE) * HUGEPAGE_SIZE

        fd_rope = os.open(hugepage_path_rope, os.O_CREAT | os.O_RDWR, 0o600)
        fds.append(fd_rope)
        os.ftruncate(fd_rope, rope_aligned)
        mmap_rope = mmap.mmap(fd_rope, rope_aligned, flags=mmap.MAP_SHARED,
                              prot=mmap.PROT_READ | mmap.PROT_WRITE)
        mmaps.append(mmap_rope)

        host_rope = torch.frombuffer(mmap_rope, dtype=DTYPE, count=rope_elements
                                     ).view(NUM_FULL_BLOCKS, BLOCK_SIZE, K_ROPE_DIM)

        # ── 写入标记值 ──
        for b in range(NUM_FULL_BLOCKS):
            host_kv[b, 0, 0] = float(b + 400)
            host_rope[b, 0, 0] = float(b + 500)

        print(f"  Host KV:   shape={host_kv.shape}, ptr=0x{host_kv.data_ptr():x}")
        print(f"  Host Rope: shape={host_rope.shape}, ptr=0x{host_rope.data_ptr():x}")

        # ── NPU MMU 注册 ──
        _, npu_kv = zero_copy_npu.register_hugepage_as_npu_tensor(host_kv, 0)
        _, npu_rope = zero_copy_npu.register_hugepage_as_npu_tensor(host_rope, 0)

        print(f"  NPU KV:   device={npu_kv.device}, ptr=0x{npu_kv.data_ptr():x}")
        print(f"  NPU Rope: device={npu_rope.device}, ptr=0x{npu_rope.data_ptr():x}")

        # ── 创建 Selection Cache (Device) ──
        sel_k_rope = torch.zeros(
            TOTAL_SEL_BLOCKS, BLOCK_SIZE, K_ROPE_DIM,
            dtype=DTYPE, device=DEVICE)
        sel_kv_cache = torch.zeros(
            TOTAL_SEL_BLOCKS, BLOCK_SIZE, KV_DIM,
            dtype=DTYPE, device=DEVICE)
        sel_block_table = torch.arange(
            TOTAL_SEL_BLOCKS, dtype=torch.int32, device=DEVICE
        ).view(BATCH, NUM_SEL_BLOCKS_PER_BATCH)
        sel_block_status = -torch.ones(
            BATCH, 1, 1, TOPK + 1, dtype=torch.int32, device=DEVICE)

        topk_indices = torch.zeros(
            BATCH, 1, 1, TOPK, dtype=torch.int32, device=DEVICE)
        for b in range(BATCH):
            for k in range(TOPK):
                topk_indices[b, 0, 0, k] = b * TOPK + k

        full_block_table = torch.arange(
            NUM_FULL_BLOCKS, dtype=torch.int32, device=DEVICE
        ).unsqueeze(0).expand(BATCH, -1).contiguous()
        full_kv_actual_seq = torch.full(
            (BATCH,), SEQ_LEN, dtype=torch.int32, device=DEVICE)
        full_q_actual_seq = torch.ones(
            BATCH, dtype=torch.int32, device=DEVICE)

        # ── 调用算子: Host(NPU 视图) → Device ──
        print("  Calling GatherSelectionKvCache (Host→Device)...")
        result = gather_wrapper.npu_gather_selection_kv_cache(
            sel_k_rope, sel_kv_cache, sel_block_table, sel_block_status,
            topk_indices,
            npu_rope,   # Host hugepage 的 NPU 视图
            npu_kv,     # Host hugepage 的 NPU 视图
            full_block_table,
            full_kv_actual_seq, full_q_actual_seq, TOPK_BLOCK_SIZE)
        torch.npu.synchronize()

        # ── 验证 ──
        print(f"  block_status = {sel_block_status.cpu().numpy()}")
        print(f"  sel_kv_actual_seq = {result.cpu().tolist()}")

        passed = True
        for b in range(BATCH):
            status = sel_block_status[b, 0, 0, :TOPK].cpu()
            for slot_idx in range(TOPK):
                group_idx = status[slot_idx].item()
                if group_idx < 0:
                    continue
                full_block_idx = group_idx
                sel_block_idx = b * NUM_SEL_BLOCKS_PER_BATCH + slot_idx
                actual = sel_kv_cache[sel_block_idx, 0, 0].cpu().item()
                expected = host_kv[full_block_idx, 0, 0].item()
                match = abs(expected - actual) < 1e-2
                if not match:
                    print(f"  MISMATCH batch={b} slot={slot_idx} "
                          f"group={group_idx}: expected={expected:.1f} "
                          f"actual={actual:.1f}")
                    passed = False
                else:
                    print(f"  OK batch={b} slot={slot_idx} group={group_idx}: "
                          f"expected={expected:.1f} actual={actual:.1f}")

        print(f"  Test 4: {'PASS' if passed else 'FAIL'}")
        return passed

    except Exception as e:
        print(f"  Test 4: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        for m in mmaps:
            try: m.close()
            except: pass
        for f in fds:
            try: os.close(f)
            except: pass
        for p in [hugepage_path_kv, hugepage_path_rope]:
            try: os.unlink(p)
            except: pass


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("GatherSelectionKvCache Operator Test v2")
    print(f"torch_npu version: {torch_npu.__version__}")
    print(f"Device: {torch.npu.get_device_name(0)}")

    results = {}
    results["Test 1: Basic Gather"] = test_basic_gather()
    results["Test 2: Cache Reuse"] = test_cache_reuse()
    results["Test 3: Host Memory Registration"] = test_host_memory_registration()
    results["Test 4: Host→Device Gather"] = test_host_to_device_gather()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, result in results.items():
        if result is None:
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {name}: {status}")

    all_run = [r for r in results.values() if r is not None]
    if all_run and all(all_run):
        print("\nAll tests passed!")
    elif any(r is False for r in results.values()):
        print("\nSome tests FAILED!")
        sys.exit(1)
    else:
        print("\nSome tests skipped (hugepage not available?)")
