# TidalCache

> Sparse KV Cache Offload for DeepSeek Sparse Attention (DSA) on Ascend NPU

Like tides that surge shoreward only where the current pulls, TidalCache moves KV cache data from Host memory to Device HBM **only for the tokens the model actually attends to** — driven by DSA's Lightning Indexer top-k selection.

## What It Does

DeepSeek-V3/R1 uses **MLA (Multi-head Latent Attention)** with 512d compressed KV + 64d RoPE per token. Long sequences (64K–128K) exhaust Device HBM quickly. But DSA's sparse attention only touches a small fraction of the full KV cache each step.

TidalCache exploits this sparsity:

```
Full KV Cache (Host hugepage)
        │
        │  only top-k selected blocks
        │  (sparse Host→Device DMA)
        ▼
Selection Cache (Device HBM)  ──→  Sparse Attention Kernel
        ▲
        │
Indexer K Cache (Device, 64d only) ──→ Lightning Indexer ──→ top-k indices
```

1. **Full KV stays on Host** — registered via `aclrtHostRegisterV2` + `aclrtHostGetDevicePointer` on hugepages
2. **Indexer K Cache stays on Device** — only 64d per token, lightweight
3. **Lightning Indexer** scores all tokens and selects top-k important groups
4. **GatherSelectionKvCache** (CANN custom op) sparse-gathers selected blocks from Host to Device with cache reuse
5. **Sparse Attention** runs on the compact Selection Cache

## Key Numbers

| Metric | Value |
|--------|-------|
| KV per token (MLA) | 576 bytes (BF16) |
| HBM savings (8K seq) | ~25% |
| Per-step transfer (topk=4, 50% miss) | ~4.3 MB / request |
| TPOT overhead (small batch) | < 1% |

## Project Structure

```
TidalCache/
├── test_gather/              # Verified test suite (all 4 PASS on Ascend910_9392)
│   ├── test_gather_op.py     # 4 tests: basic gather, cache reuse, host reg, H2D gather
│   ├── gather_wrapper.cpp    # C++ extension: aclnn GatherSelectionKvCache wrapper
│   ├── zero_copy_npu.cpp     # C++ extension: hugepage → NPU tensor (zero-copy)
│   ├── tensor_register.cpp   # C interface for ACL host memory registration
│   ├── setup.py              # Build config (CANN include order critical!)
│   └── build.sh              # Build script (sequential to avoid ninja race)
├── ops_ascendc/              # CANN custom operator build framework
│   └── src/gather_selection_kv_cache/  # GatherSelectionKvCache operator source
├── docs/
│   └── gather_selection_kv_cache_explained.md  # Operator internals & architecture
├── kv_offload_integration_plan.md              # Integration design document
└── README.md
```

## Status

- [x] **Step 1**: GatherSelectionKvCache operator build & install (CANN 9.0.1)
- [x] **Step 2**: Host memory registration + H2D gather verification (4/4 tests PASS)
- [ ] **Step 3**: Integration into vllm-ascend (in progress)
- [ ] **Step 4**: Multi-batch performance benchmarking
- [ ] **Step 5**: Mooncake Store RDMA direct-write integration (P/D separation)

## Environment

| Component | Version |
|-----------|---------|
| CANN | 9.0.1 |
| torch | 2.10.0 |
| torch_npu | 2.10.0.post2 |
| Chip | Ascend910_9392 (Atlas A3) |
| Python | 3.12.13 |

## Quick Start (NPU Machine)

### Prerequisites

```bash
# Hugepage setup (requires root)
echo 512 > /proc/sys/vm/nr_hugepages
mkdir -p /dev/hugepages && mount -t hugetlbfs nodev /dev/hugepages

# Set LD_LIBRARY_PATH
export CANN_HOME=/usr/local/Ascend/cann-9.0.1
export LD_LIBRARY_PATH=$CANN_HOME/opp/vendors/customize/op_api/lib/:$CANN_HOME/lib64/:$LD_LIBRARY_PATH
```

### Build & Install Operator

```bash
cd ops_ascendc
bash build.sh -n "gather_selection_kv_cache" -c "ascend910_93" -p $CANN_HOME
sudo bash output/*.run
```

### Build & Run Tests

```bash
cd test_gather
bash build.sh
python3 test_gather_op.py
```

## Key Technical Findings

1. **NPU has an independent device address space** — `host_ptr` and `dev_ptr` differ; using `host_ptr` as device address causes SUSPECT REMOTE ERROR (507057)
2. **CANN include order matters** — `CANN_HOME/include` must precede `torch_npu/include` in the build, or old bundled ACL headers shadow `aclrtHostGetDevicePointer`
3. **GatherSelectionKvCache supports cache reuse** — `block_status` tracks which blocks are already in the Selection Cache, typically 50-80% hit rate across consecutive decode steps

## Future: Mooncake Store Synergy

TidalCache's Host hugepage pool is a natural landing zone for **Mooncake RDMA direct-write** in P/D (Prefill/Decode) separation:

- Prefill node computes KV → RDMA writes directly to Decode node's Host hugepage → zero-copy
- KV transfer latency hidden behind decode computation (pipeline)
- No KVConnector interface dependency — compatible with Mooncake Store

## License

Apache-2.0
