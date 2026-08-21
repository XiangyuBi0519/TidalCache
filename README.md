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

---

# TidalCache（中文）

> 基于 DeepSeek 稀疏注意力（DSA）的 KV Cache 稀疏卸载方案，面向昇腾 NPU

如同潮汐只在引力牵引处涌向海岸，TidalCache 将 KV Cache 数据从 Host 内存搬运到 Device HBM 时，**只搬运模型真正需要 attend 的 token** —— 由 DSA 的 Lightning Indexer top-k 选择驱动。

## 核心原理

DeepSeek-V3/R1 使用 **MLA（Multi-head Latent Attention）**，每个 token 的 KV 数据为 512d 压缩潜向量 + 64d RoPE 位置编码 = 576 bytes（BF16）。长序列（64K–128K）下 KV Cache 占满 Device HBM。但 DSA 的稀疏注意力每步只访问全量 KV 的一小部分。

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

1. **全量 KV 保留在 Host** —— 通过 hugepage 上的 `aclrtHostRegisterV2` + `aclrtHostGetDevicePointer` 注册到 NPU MMU
2. **Indexer K Cache 保留在 Device** —— 每 token 仅 64d，占用很小
3. **Lightning Indexer** 对所有 token 打分，选出 top-k 个重要的 token 组
4. **GatherSelectionKvCache**（CANN 自定义算子）从 Host 稀疏采集选中的 blocks 到 Device，支持缓存复用
5. **稀疏注意力** 在紧凑的 Selection Cache 上执行

## 关键指标

| 指标 | 数值 |
|------|------|
| 每 token KV 大小（MLA） | 576 bytes（BF16） |
| HBM 节省（8K 序列） | ~25% |
| 每步搬运量（topk=4, 50% miss） | ~4.3 MB / 请求 |
| TPOT 开销（小 batch） | < 1% |

## 项目结构

```
TidalCache/
├── test_gather/              # 测试套件（Ascend910_9392 上 4 项全部 PASS）
│   ├── test_gather_op.py     # 4 个测试: 基础 gather、缓存复用、Host 注册、H2D gather
│   ├── gather_wrapper.cpp    # C++ 扩展: aclnn GatherSelectionKvCache 封装
│   ├── zero_copy_npu.cpp     # C++ 扩展: hugepage → NPU 张量（零拷贝）
│   ├── tensor_register.cpp   # C 接口: ACL Host 内存注册
│   ├── setup.py              # 编译配置（CANN include 顺序至关重要！）
│   └── build.sh              # 编译脚本（需顺序编译，避免 ninja 竞争）
├── ops_ascendc/              # CANN 自定义算子编译框架
│   └── src/gather_selection_kv_cache/  # GatherSelectionKvCache 算子源码
├── docs/
│   └── gather_selection_kv_cache_explained.md  # 算子原理与架构详解
├── kv_offload_integration_plan.md              # 集成设计文档
└── README.md
```

## 进度

- [x] **Step 1**: GatherSelectionKvCache 算子编译安装（CANN 9.0.1）
- [x] **Step 2**: Host 内存注册 + H2D gather 验证（4/4 测试 PASS）
- [ ] **Step 3**: 集成到 vllm-ascend（进行中）
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

## 关键技术发现

1. **NPU 拥有独立的设备地址空间** —— `host_ptr` 与 `dev_ptr` 不同；直接把 `host_ptr` 当设备地址使用会触发 SUSPECT REMOTE ERROR（507057）
2. **CANN 头文件 include 顺序至关重要** —— `CANN_HOME/include` 必须放在 `torch_npu/include` 之前，否则 torch_npu 打包的旧版 ACL 头文件会遮蔽 `aclrtHostGetDevicePointer` 等 API
3. **GatherSelectionKvCache 支持缓存复用** —— `block_status` 追踪 Selection Cache 中已有的 blocks，连续 decode 步骤间命中率通常为 50%–80%

## 未来方向：Mooncake Store 协同

TidalCache 的 Host hugepage 内存池天然适合作为 **Mooncake RDMA 直写**的落脚点，实现 P/D（Prefill/Decode）分离：

- Prefill 节点计算完 KV → RDMA 直写到 Decode 节点的 Host hugepage → 零拷贝
- KV 搬运延迟被 decode 计算隐藏（流水线化）
- 不依赖 KVConnector 接口 —— 与 Mooncake Store 兼容

## License

Apache-2.0
