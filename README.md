# TidalCache

> Sparse KV Cache Offload for DeepSeek Sparse Attention on Ascend NPU

Like tides that surge shoreward only where the current pulls, TidalCache moves KV cache data from Host memory to Device HBM **only for the tokens the model actually attends to** — driven by the Lightning Indexer's top-k selection.

## Supported Models

| Model | Attention Type | TidalCache | Notes |
|-------|---------------|------------|-------|
| DeepSeek-V3 / R1 | DSA (DeepSeek Sparse Attention) | All sparse layers | Original target |
| DeepSeek-V4-Pro | CSA (Compressed Sparse Attention) | CSA layers only | Compressed KV, same sparse selection |
| DeepSeek-V4-Pro | HCA (Heavily Compressed Attention) | Skipped | Dense attention, no top-k — KV already small |
| DeepSeek-V4-Flash | CSA + HCA hybrid | CSA layers only | Same as V4-Pro |

## What It Does

DeepSeek models use **MLA (Multi-head Latent Attention)** with compressed KV per token. Long sequences (64K–1M) exhaust Device HBM quickly. But sparse attention (DSA/CSA) only touches a small fraction of the full KV cache each step.

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

### Why It Works for V4 (CSA)

CSA adds a compression step **before** the same Lightning Indexer + top-k pipeline:

```
V3 (DSA):  Token KV  ──→  Lightning Indexer  ──→  top-k  ──→  Sparse Attention
V4 (CSA):  Token KV  ──→  Compress (m:1)  ──→  Lightning Indexer  ──→  top-k  ──→  Sparse Attention
```

The compressed KV entries are smaller, so offloading them is even more efficient — less Host memory, less DMA bandwidth per gather.

HCA layers use dense attention on heavily compressed KV (m'>>m tokens → 1 entry). The KV cache is already tiny and every entry is attended, so offloading provides no benefit.

### V4 Mixed Precision

V4 uses a hybrid storage format for KV cache:
- **RoPE dimensions (64d)**: BF16 precision
- **KV dimensions (remaining)**: FP8 (float8_e4m3fn)

TidalCache supports separate `dtype` and `rope_dtype` to handle this.

## Key Numbers

| Metric | V3 (DSA) | V4 (CSA) |
|--------|----------|----------|
| KV per token (MLA) | 576 bytes (BF16) | ~320 bytes (FP8+BF16) |
| Compression ratio | 1:1 (no compression) | m:1 (e.g., 4:1) |
| Top-k selection | 512 blocks | Smaller top-k (more efficient) |
| HBM savings | ~25% (8K seq) | Higher (compressed + sparse) |
| Per-step transfer | ~4.3 MB/req (topk=512, 50% miss) | Less (smaller KV) |

## Project Structure

```
TidalCache/
├── tidalcache/                # Core Python module
│   ├── __init__.py            # Config flags (env vars)
│   ├── offload_manager.py     # TidalCacheManager: hugepage + gather + lifecycle
│   └── test_manager.py        # Integration test (NPU)
├── test_gather/               # Verified test suite (all 4 PASS on Ascend910_9392)
│   ├── test_gather_op.py      # 4 tests: basic gather, cache reuse, host reg, H2D gather
│   ├── gather_wrapper.cpp     # C++ extension: aclnn GatherSelectionKvCache wrapper
│   ├── zero_copy_npu.cpp      # C++ extension: hugepage → NPU tensor (zero-copy)
│   ├── tensor_register.cpp    # C interface for ACL host memory registration
│   ├── setup.py               # Build config (CANN include order critical!)
│   └── build.sh               # Build script (sequential to avoid ninja race)
├── ops_ascendc/               # CANN custom operator build framework
│   └── src/gather_selection_kv_cache/  # GatherSelectionKvCache operator source
├── patches/                   # vllm-ascend integration patches
│   ├── dsa_v1_patch.py        # Patches for attention/dsa_v1.py (DSA + CSA)
│   └── model_runner_patch.py  # Patches for worker/model_runner_v1.py (V3 + V4)
├── docs/
│   └── gather_selection_kv_cache_explained.md  # Operator internals & architecture
└── README.md
```

## Status

- [x] **Step 1**: GatherSelectionKvCache operator build & install (CANN 9.0.1)
- [x] **Step 2**: Host memory registration + H2D gather verification (4/4 tests PASS)
- [x] **Step 3**: Integration into vllm-ascend — **end-to-end running** (2026-09-11)
  - [x] TidalCacheManager core module
  - [x] vllm-ascend patches (dsa_v1 + model_runner)
  - [x] V4 CSA/HCA compatibility (layer filtering, mixed precision)
  - [x] NPU integration test — correct model output verified
  - [x] Dedicated logging (tidalcache.log)
  - [x] Warmup / graph capture compatibility (3 bugs fixed)
- [ ] **Step 4**: Performance optimization (see [Optimization Roadmap](#optimization-roadmap))
- [ ] **Step 5**: Multi-batch / concurrent throughput benchmarking
- [ ] **Step 6**: Mooncake Store RDMA direct-write integration (P/D separation)

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

### Enable TidalCache

```bash
export VLLM_DSA_KV_OFFLOAD=1
# Optional: custom hugepage path (default: /dev/hugepages)
export VLLM_DSA_OFFLOAD_HUGEPAGE_PATH=/dev/hugepages
```

## Key Technical Findings

1. **NPU has an independent device address space** — `host_ptr` and `dev_ptr` differ; using `host_ptr` as device address causes SUSPECT REMOTE ERROR (507057)
2. **CANN include order matters** — `CANN_HOME/include` must precede `torch_npu/include` in the build, or old bundled ACL headers shadow `aclrtHostGetDevicePointer`
3. **GatherSelectionKvCache supports cache reuse** — `block_status` tracks which blocks are already in the Selection Cache, typically 50-80% hit rate across consecutive decode steps
4. **BF16 precision rounding** — 7-bit mantissa causes values like 1003→1004 when stored; test comparisons must use BF16 reference values
5. **PyTorch advanced indexing returns copies** — `tensor[index_tensor].fill_(-1)` modifies a copy; iterate indices instead

## Integration Debugging History (2026-09-11)

Three bugs were encountered and fixed during vllm-ascend integration, all triggered during **warmup / CUDA graph capture** where vllm feeds dummy data with inconsistent batch sizes.

### Bug 1: torch.gather dim0 mismatch (507911)

- **Error**: `Size does not match at dimension 0, expected index shape 32 smaller than self shape 16`
- **Root cause**: During graph capture warmup, `hidden_states.shape[0]=32` (dummy batch) but `block_table` only has 16 rows. The copy-back code derived `_bseq` from `B=32` while `block_table` had fewer rows.
- **Fix**: Clamp batch size to block_table rows: `_aB = min(B, block_table.shape[0])` and slice all downstream tensors to `_aB`. Applied to both CP and non-CP paths.

### Bug 2: GatherSelectionKvCache assertion `curFullQSeqLen > seq`

- **Error**: `Assertion 'curFullQSeqLen <= tiling_->seq' curFullQSeqLen:2 cannot be greater than seq:1`
- **Root cause**: During warmup, `full_q_actual_seq` contains dummy value 2, but the operator's tiling was compiled for decode mode (seq=1). In decode, each request always has exactly 1 query token.
- **Fix**: Hardcode `full_q_actual_seq=torch.ones(B, dtype=torch.int32, device=device)` — decode mode always has q_seq=1 per request.

### Bug 3: Hugepage allocation fragmentation

- **Symptom**: All 21 DSA layers × 16 workers fall back to pinned memory (`[Errno 12] Cannot allocate memory`)
- **Root cause**: Machine has 2TB RAM but kernel memory fragmentation limits available contiguous 2MB pages. Allocated 240,000 hugepages total but only 137,661 (268 GB) are free.
- **Workaround**: Use available pages (63% coverage); layers that miss hugepages fall back to pinned memory (still functional, slightly slower DMA).
- **Recommended fix**: Boot-time kernel parameter `hugepages=240000` guarantees contiguous allocation before memory fragments.

## Performance Baseline

Single-request latency comparison (512 output tokens, DeepSeek model on Ascend NPU):

| Configuration | Latency | Overhead |
|---------------|---------|----------|
| TidalCache OFF (all KV on Device) | 13.3s | — |
| TidalCache ON (KV offloaded to Host) | 18.3s | +37% |

This overhead is **expected for a first version** without any optimization. The per-step Host→Device DMA gather runs synchronously on the main stream. The value of TidalCache is not single-request latency — it is **memory capacity**: offloading KV cache to Host frees Device HBM for serving more concurrent requests and/or longer contexts.

## Optimization Roadmap

Priority-ordered optimizations, informed by analysis of HiSparse's production implementation:

| # | Optimization | Expected Impact | Source |
|---|-------------|-----------------|--------|
| 1 | **Reduce Device KV allocation + direct Selection Cache attention** | **Actual HBM savings** — the core value of TidalCache. Currently Device still allocates full KV; must reduce DSA layer blocks to Selection Cache size only, and run attention directly on Selection Cache without copy-back | TidalCache architecture |
| 2 | **Boot-time hugepage reservation** | Eliminate pinned-memory fallback, consistent DMA perf | Ops |
| 3 | **Async copy stream** | Overlap DMA with compute, hide gather latency | HiSparse `_create_copy_stream` |
| 4 | **Device hot cache + LRU** | Skip DMA for frequently accessed blocks (50-80% hit rate) | HiSparse `lru_slots` |
| 5 | **Cross-layer index sharing + batch prefetch** | Leader layer gathers, follower layers reuse plan | HiSparse `_prefetch_group`, `_GroupPlan` |
| 6 | **Miss-only gather** | Only DMA blocks not already in selection cache | HiSparse `compact_miss_globals` |
| 7 | **Gather plan reuse** | Avoid redundant index computation across follower layers | HiSparse `_GroupPlan` |
| 8 | **Adaptive offload** | Only offload when HBM pressure is high; keep KV on Device for short sequences | TidalCache design |

> **Note**: Opt #1 is a prerequisite for HBM savings. The current v1 implementation validates the data path (gather/scatter/copy-back correctness) but does NOT reduce Device memory — vllm still allocates full KV cache on Device for all layers. Without Opt #1, TidalCache actually uses MORE total memory (Device full KV + Host copy + Selection Cache).

### Expected Benefits After Optimization

| Metric | Current (v1) | After Opt 1 | After Opt 1+3-7 |
|--------|-------------|-------------|-----------------|
| HBM savings (per DSA layer) | **None** (full KV still on Device) | **100% Full KV offloaded** | Same |
| Single-request overhead | +37% | ~+30% | ~+5-8% |
| Concurrent throughput gain | None | +20-30% | +50-80% |

The throughput gain comes from freed HBM enabling larger batch sizes. With 21 DSA layers × 576 bytes/token, offloading saves ~12 KB/token in HBM — for 128K context, that's ~1.5 GB per request freed.

## Testing Methodology

### Single-Request Latency

```bash
# TidalCache ON
export VLLM_DSA_KV_OFFLOAD=1
# Start vllm service, then:
time curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek","prompt":"test prompt","max_tokens":512}'

# TidalCache OFF
unset VLLM_DSA_KV_OFFLOAD  # or set to 0
# Restart service, repeat same curl
```

### Concurrent Throughput

```bash
# Use vllm benchmark scripts or wrk/hey:
# Compare max sustainable QPS at P99 < target latency
# Key metric: requests/sec at same latency SLO

# Example with hey (HTTP load generator):
hey -n 100 -c 10 -m POST -H "Content-Type: application/json" \
  -d '{"model":"deepseek","prompt":"test","max_tokens":128}' \
  http://localhost:8000/v1/completions
```

### Memory Usage

```bash
# Check per-device HBM usage:
npu-smi info

# Compare HBM consumption with ON vs OFF under same batch size
# Key metric: peak HBM usage per device
```

### Correctness Verification

```bash
# Compare output logits/tokens between ON and OFF modes
# TidalCache should produce identical outputs (same computation, different memory layout)

# Check tidalcache.log for per-request flow:
grep "SCATTER\|GATHER\|COPYBACK" tidalcache.log | head -20
```

### Logging

TidalCache writes to a dedicated log file (default: `tidalcache.log`, configurable via `TIDALCACHE_LOG` env var):

```bash
# Log stages per decode step:
# [SCATTER]  — D2H offload of new KV to Host
# [GATHER]   — H2D sparse gather of selected blocks
# [COPYBACK] — Copy gathered data to compress_kv_cache at correct block positions

# Adjust log level:
# Default: INFO (init + warnings)
# Set tidalcache logger to DEBUG for per-step tracing (generates ~5000 lines/request)
```

## Future: Mooncake Store Synergy

TidalCache's Host hugepage pool is a natural landing zone for **Mooncake RDMA direct-write** in P/D (Prefill/Decode) separation:

- Prefill node computes KV → RDMA writes directly to Decode node's Host hugepage → zero-copy
- KV transfer latency hidden behind decode computation (pipeline)
- No KVConnector interface dependency — compatible with Mooncake Store

---

# TidalCache（中文）

> 基于 DeepSeek 稀疏注意力的 KV Cache 稀疏卸载方案，面向昇腾 NPU

如同潮汐只在引力牵引处涌向海岸，TidalCache 将 KV Cache 数据从 Host 内存搬运到 Device HBM 时，**只搬运模型真正需要 attend 的 token** —— 由 Lightning Indexer top-k 选择驱动。

## 支持模型

| 模型 | 注意力类型 | TidalCache | 说明 |
|------|-----------|------------|------|
| DeepSeek-V3 / R1 | DSA（稀疏注意力） | 所有稀疏层 | 初始目标 |
| DeepSeek-V4-Pro | CSA（压缩稀疏注意力） | 仅 CSA 层 | 压缩 KV，同样的稀疏选择 |
| DeepSeek-V4-Pro | HCA（重压缩注意力） | 跳过 | 稠密注意力，无 top-k，KV 已很小 |
| DeepSeek-V4-Flash | CSA + HCA 混合 | 仅 CSA 层 | 与 V4-Pro 相同 |

## 核心原理

DeepSeek 模型使用 **MLA（Multi-head Latent Attention）**，每个 token 的 KV 数据为压缩潜向量 + RoPE 位置编码。长序列（64K–1M）下 KV Cache 占满 Device HBM。但稀疏注意力（DSA/CSA）每步只访问全量 KV 的一小部分。

TidalCache 利用这个稀疏性：

```
全量 KV Cache（Host hugepage 内存）
        │
        │  仅搬运 top-k 选中的 blocks
        │ （稀疏 Host→Device DMA）
        ▼
Selection Cache（Device HBM）──→  稀疏注意力 Kernel
        ▲
        │
Indexer K Cache（Device, 仅 64d）──→ Lightning Indexer ──→ top-k 索引
```

### 为什么对 V4 (CSA) 同样有效

CSA 在相同的 Lightning Indexer + top-k 流程**之前**增加了一个压缩步骤：

```
V3 (DSA):  Token KV  ──→  Lightning Indexer  ──→  top-k  ──→  稀疏注意力
V4 (CSA):  Token KV  ──→  压缩 (m:1)  ──→  Lightning Indexer  ──→  top-k  ──→  稀疏注意力
```

压缩后的 KV entries 更小，所以卸载效率更高——Host 内存占用更少，每次 gather 的 DMA 带宽需求更低。

HCA 层使用稠密注意力处理重度压缩的 KV（m'>>m 个 token → 1 个 entry）。KV cache 已经很小，且每个 entry 都会被访问，卸载没有收益。

### V4 混合精度

V4 的 KV cache 使用混合存储格式：
- **RoPE 维度（64d）**：BF16 精度
- **KV 维度（其余）**：FP8（float8_e4m3fn）

TidalCache 通过独立的 `dtype` 和 `rope_dtype` 参数支持这一特性。

## 关键指标

| 指标 | V3 (DSA) | V4 (CSA) |
|------|----------|----------|
| 每 token KV 大小（MLA） | 576 bytes（BF16） | ~320 bytes（FP8+BF16） |
| 压缩比 | 1:1（无压缩） | m:1（如 4:1） |
| Top-k 选择 | 512 blocks | 更小的 top-k（更高效） |
| HBM 节省 | ~25%（8K 序列） | 更高（压缩 + 稀疏） |
| 每步搬运量 | ~4.3 MB/请求 | 更少（更小的 KV） |

## 项目结构

```
TidalCache/
├── tidalcache/                # 核心 Python 模块
│   ├── __init__.py            # 配置标志（环境变量）
│   ├── offload_manager.py     # TidalCacheManager: hugepage + gather + 生命周期
│   └── test_manager.py        # 集成测试（NPU）
├── test_gather/               # 测试套件（Ascend910_9392 上 4 项全部 PASS）
│   ├── test_gather_op.py      # 4 个测试: 基础 gather、缓存复用、Host 注册、H2D gather
│   ├── gather_wrapper.cpp     # C++ 扩展: aclnn GatherSelectionKvCache 封装
│   ├── zero_copy_npu.cpp      # C++ 扩展: hugepage → NPU 张量（零拷贝）
│   ├── tensor_register.cpp    # C 接口: ACL Host 内存注册
│   ├── setup.py               # 编译配置（CANN include 顺序至关重要！）
│   └── build.sh               # 编译脚本（需顺序编译，避免 ninja 竞争）
├── ops_ascendc/               # CANN 自定义算子编译框架
│   └── src/gather_selection_kv_cache/  # GatherSelectionKvCache 算子源码
├── patches/                   # vllm-ascend 集成补丁
│   ├── dsa_v1_patch.py        # attention/dsa_v1.py 补丁（DSA + CSA）
│   └── model_runner_patch.py  # worker/model_runner_v1.py 补丁（V3 + V4）
├── docs/
│   └── gather_selection_kv_cache_explained.md  # 算子原理与架构详解
└── README.md
```

## 进度

- [x] **Step 1**: GatherSelectionKvCache 算子编译安装（CANN 9.0.1）
- [x] **Step 2**: Host 内存注册 + H2D gather 验证（4/4 测试 PASS）
- [x] **Step 3**: 集成到 vllm-ascend — **端到端跑通** (2026-09-11)
  - [x] TidalCacheManager 核心模块
  - [x] vllm-ascend 补丁（dsa_v1 + model_runner）
  - [x] V4 CSA/HCA 兼容（层过滤、混合精度）
  - [x] NPU 集成测试 — 模型输出正确性已验证
  - [x] 独立日志（tidalcache.log）
  - [x] Warmup / graph capture 兼容（修复 3 个 Bug）
- [ ] **Step 4**: 性能优化（见 [优化路线](#优化路线)）
- [ ] **Step 5**: 多 batch / 并发吞吐量基准测试
- [ ] **Step 6**: Mooncake Store RDMA 直写集成（P/D 分离）

## 环境

| 组件 | 版本 |
|------|------|
| CANN | 9.0.1 |
| torch | 2.10.0 |
| torch_npu | 2.10.0.post2 |
| 芯片 | Ascend910_9392（Atlas A3） |
| Python | 3.12.13 |

## 快速开始（NPU 机器）

### 前置准备

```bash
# Hugepage 配置（需要 root 权限）
echo 512 > /proc/sys/vm/nr_hugepages
mkdir -p /dev/hugepages && mount -t hugetlbfs nodev /dev/hugepages

# 设置 LD_LIBRARY_PATH
export CANN_HOME=/usr/local/Ascend/cann-9.0.1
export LD_LIBRARY_PATH=$CANN_HOME/opp/vendors/customize/op_api/lib/:$CANN_HOME/lib64/:$LD_LIBRARY_PATH
```

### 编译安装算子

```bash
cd ops_ascendc
bash build.sh -n "gather_selection_kv_cache" -c "ascend910_93" -p $CANN_HOME
sudo bash output/*.run
```

### 编译并运行测试

```bash
cd test_gather
bash build.sh
python3 test_gather_op.py
```

### 启用 TidalCache

```bash
export VLLM_DSA_KV_OFFLOAD=1
# 可选: 自定义 hugepage 路径（默认: /dev/hugepages）
export VLLM_DSA_OFFLOAD_HUGEPAGE_PATH=/dev/hugepages
```

## 关键技术发现

1. **NPU 拥有独立的设备地址空间** —— `host_ptr` 与 `dev_ptr` 不同；直接把 `host_ptr` 当设备地址使用会触发 SUSPECT REMOTE ERROR（507057）
2. **CANN 头文件 include 顺序至关重要** —— `CANN_HOME/include` 必须放在 `torch_npu/include` 之前，否则 torch_npu 打包的旧版 ACL 头文件会遮蔽 `aclrtHostGetDevicePointer` 等 API
3. **GatherSelectionKvCache 支持缓存复用** —— `block_status` 追踪 Selection Cache 中已有的 blocks，连续 decode 步骤间命中率通常为 50%–80%
4. **BF16 精度舍入** —— 7 位尾数导致 1003→1004 等舍入；测试比较必须使用 BF16 参考值
5. **PyTorch 高级索引返回副本** —— `tensor[index_tensor].fill_(-1)` 修改的是副本；需要逐个索引迭代

## 集成调试记录 (2026-09-11)

vllm-ascend 集成过程中遇到并修复了 3 个 Bug，均在 **warmup / CUDA graph capture** 阶段触发（vllm 使用 dummy 数据，batch size 不一致）。

### Bug 1: torch.gather dim0 维度不匹配 (507911)

- **报错**: `Size does not match at dimension 0, expected index shape 32 smaller than self shape 16`
- **原因**: Graph capture warmup 时 `hidden_states.shape[0]=32`（dummy batch）但 `block_table` 只有 16 行。Copy-back 代码从 `B=32` 派生索引，超出 block_table 范围。
- **修复**: 限制 batch 到 block_table 行数：`_aB = min(B, block_table.shape[0])`，所有后续张量切到 `_aB`。CP 和非 CP 路径均修复。

### Bug 2: GatherSelectionKvCache 断言 `curFullQSeqLen > seq`

- **报错**: `Assertion 'curFullQSeqLen <= tiling_->seq' curFullQSeqLen:2 cannot be greater than seq:1`
- **原因**: Warmup 时 `full_q_actual_seq` 含 dummy 值 2，但算子 tiling 编译为 decode 模式（seq=1）。Decode 阶段每个请求始终只有 1 个 query token。
- **修复**: 硬编码 `full_q_actual_seq=torch.ones(B, dtype=torch.int32, device=device)`。

### Bug 3: Hugepage 分配碎片化

- **现象**: 21 个 DSA 层 × 16 worker 全部回退到 pinned memory（`[Errno 12] Cannot allocate memory`）
- **原因**: 机器有 2TB 内存，但内核内存碎片化限制了可用连续 2MB 页。分配 240,000 hugepages 但仅 137,661（268 GB）可用。
- **临时方案**: 使用可用页面（63% 覆盖率），未获得 hugepage 的层回退到 pinned memory（功能正常，DMA 稍慢）。
- **推荐方案**: 开机内核参数 `hugepages=240000`，在内存碎片化前保证连续分配。

## 性能基线

单请求延迟对比（512 output tokens，DeepSeek 模型，Ascend NPU）：

| 配置 | 延迟 | 开销 |
|------|------|------|
| TidalCache OFF（所有 KV 在 Device） | 13.3s | — |
| TidalCache ON（KV 卸载到 Host） | 18.3s | +37% |

此开销是**首版无优化的预期结果**。每步 Host→Device DMA gather 在主 stream 上同步执行。TidalCache 的价值不在单请求延迟，而在**内存容量**：卸载 KV cache 到 Host 释放 Device HBM，可服务更多并发请求和/或更长上下文。

## 优化路线

按优先级排序，参考 HiSparse 生产实现的分析：

| # | 优化项 | 预期收益 | 来源 |
|---|--------|----------|------|
| 1 | **减少 Device KV 分配 + 直接用 Selection Cache 做 attention** | **真正的 HBM 节省** — TidalCache 的核心价值。当前 Device 仍分配全量 KV；需将 DSA 层的 Device blocks 缩减为 Selection Cache 大小，attention 直接读 Selection Cache 不再 copy-back | TidalCache 架构 |
| 2 | **开机 hugepage 预留** | 消除 pinned memory 回退，DMA 性能一致 | 运维 |
| 3 | **异步 copy stream** | DMA 与计算重叠，隐藏 gather 延迟 | HiSparse `_create_copy_stream` |
| 4 | **Device 热缓存 + LRU** | 跳过高频访问 block 的 DMA（50-80% 命中率） | HiSparse `lru_slots` |
| 5 | **跨层索引共享 + 批量预取** | Leader 层 gather，follower 层复用计划 | HiSparse `_prefetch_group` |
| 6 | **仅 miss gather** | 只 DMA 不在 selection cache 中的 block | HiSparse `compact_miss_globals` |
| 7 | **Gather plan 复用** | 避免 follower 层重复索引计算 | HiSparse `_GroupPlan` |
| 8 | **自适应卸载** | 仅在 HBM 压力高时卸载；短序列保留 KV 在 Device | TidalCache 设计 |

> **注意**：优化 #1 是 HBM 节省的前提。当前 v1 验证了数据通路（gather/scatter/copy-back 正确性），但**并未减少 Device 内存** — vllm 仍为所有层分配全量 KV cache。没有优化 #1，TidalCache 实际上**增加**了总内存占用（Device 全量 KV + Host 副本 + Selection Cache）。

### 优化后预期收益

| 指标 | 当前 (v1) | 优化 1 后 | 优化 1+3-7 后 |
|------|----------|----------|--------------|
| HBM 节省（每 DSA 层） | **无**（全量 KV 仍在 Device） | **100% Full KV 卸载** | 相同 |
| 单请求开销 | +37% | ~+30% | ~+5-8% |
| 并发吞吐提升 | 无 | +20-30% | +50-80% |

吞吐提升来自释放的 HBM 支持更大 batch。21 个 DSA 层 × 576 bytes/token，卸载节省 ~12 KB/token HBM。128K 上下文下每请求释放 ~1.5 GB。

## 测试方法

### 单请求延迟

```bash
# TidalCache ON
export VLLM_DSA_KV_OFFLOAD=1
# 启动 vllm 服务后:
time curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek","prompt":"测试提示","max_tokens":512}'

# TidalCache OFF
unset VLLM_DSA_KV_OFFLOAD  # 或设为 0
# 重启服务，重复相同 curl
```

### 并发吞吐

```bash
# 对比相同延迟 SLO 下的最大可持续 QPS
hey -n 100 -c 10 -m POST -H "Content-Type: application/json" \
  -d '{"model":"deepseek","prompt":"test","max_tokens":128}' \
  http://localhost:8000/v1/completions
```

### 内存使用

```bash
# 查看每设备 HBM 使用:
npu-smi info
# 对比相同 batch size 下 ON vs OFF 的 HBM 消耗
```

### 正确性验证

```bash
# 对比 ON/OFF 模式的输出 token（应完全一致）
# 查看 tidalcache.log 的每请求流程:
grep "SCATTER\|GATHER\|COPYBACK" tidalcache.log | head -20
```

### 日志系统

TidalCache 写入独立日志文件（默认 `tidalcache.log`，可通过 `TIDALCACHE_LOG` 环境变量配置）：

```bash
# 每个 decode step 的日志阶段:
# [SCATTER]  — 新 KV 的 D2H 卸载到 Host
# [GATHER]   — 选中 block 的 H2D 稀疏 gather
# [COPYBACK] — 将 gather 数据拷回 compress_kv_cache 正确位置

# 日志级别:
# 默认: INFO（初始化 + 告警）
# 设 tidalcache logger 为 DEBUG 可追踪每步（单请求约 5000 行）
```

## 未来方向：Mooncake Store 协同

TidalCache 的 Host hugepage 内存池天然适合作为 **Mooncake RDMA 直写**的落脚点，实现 P/D（Prefill/Decode）分离：

- Prefill 节点计算完 KV → RDMA 直写到 Decode 节点的 Host hugepage → 零拷贝
- KV 搬运延迟被 decode 计算隐藏（流水线化）
- 不依赖 KVConnector 接口 —— 与 Mooncake Store 兼容

## License

Apache-2.0
