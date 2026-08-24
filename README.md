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
- [ ] **Step 3**: Integration into vllm-ascend (in progress)
  - [x] TidalCacheManager core module
  - [x] vllm-ascend patches (dsa_v1 + model_runner)
  - [x] V4 CSA/HCA compatibility (layer filtering, mixed precision)
  - [ ] NPU integration test
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
- [ ] **Step 3**: 集成到 vllm-ascend（进行中）
  - [x] TidalCacheManager 核心模块
  - [x] vllm-ascend 补丁（dsa_v1 + model_runner）
  - [x] V4 CSA/HCA 兼容（层过滤、混合精度）
  - [ ] NPU 集成测试
- [ ] **Step 4**: 多 batch 性能基准测试
- [ ] **Step 5**: Mooncake Store RDMA 直写集成（P/D 分离）

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

## 未来方向：Mooncake Store 协同

TidalCache 的 Host hugepage 内存池天然适合作为 **Mooncake RDMA 直写**的落脚点，实现 P/D（Prefill/Decode）分离：

- Prefill 节点计算完 KV → RDMA 直写到 Decode 节点的 Host hugepage → 零拷贝
- KV 搬运延迟被 decode 计算隐藏（流水线化）
- 不依赖 KVConnector 接口 —— 与 Mooncake Store 兼容

## License

Apache-2.0
