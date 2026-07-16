# DSA KV Cache Offload VLLM集成方案

> 在 vllm-ascend 上实现 DeepSeek Sparse Attention 的 KV Cache Offload 方案
> 
> 日期: 2026-07-11
> 状态: Step 1 已完成（算子调用测试通过）

---

## 1. 背景与目标

### 1.1 问题

DeepSeek-V3/R1 使用 MLA (Multi-head Latent Attention)，KV Cache 包含：
- **c^KV**（MLA 压缩潜向量）: 512 维
- **k^R**（RoPE 位置编码）: 64 维
- 总计每 token: **576 bytes**（FP16）

128K 上下文长度时，单序列 KV Cache 占 ~70MB，批量推理时 Device HBM 成为瓶颈。

### 1.2 方案：DSA + KV Cache Offload

DeepSeek Sparse Attention (DSA) 的核心洞察：**decode 阶段不需要全部 KV Cache，只需要 Top-k 相关的稀疏子集**。

```
┌─────────────────────────────────────────────────────┐
│                  Decode 一步的流程                     │
│                                                      │
│  1. Lightning Indexer (FP8 MQA)                      │
│     - 用 Indexer K Cache (Device) 做全量打分           │
│     - 输出 Top-k indices (每个 index = 64 tokens)     │
│                                                      │
│  2. GatherSelectionKvCache (本算子)                    │
│     - 根据 Top-k indices 从 Host → Device 稀疏采集     │
│     - 支持缓存复用（block_status 跟踪）                 │
│                                                      │
│  3. Sparse Flash Attention                           │
│     - 只用 Selection Cache (Device) 做注意力计算        │
│     - 2048 * 64 = 131K tokens → 实际只读 ~131K tokens  │
└─────────────────────────────────────────────────────┘
```

### 1.3 三类缓存与 Offload 策略

| 缓存类型 | 大小/token | 位置 | 能否 Offload | 原因 |
|----------|-----------|------|-------------|------|
| MLA Full KV Cache (c^KV + k^R) | 576B (FP16) | **Host** | ✅ 可以 | 稀疏读取，通过 Top-k 选择 |
| Indexer K Cache (FP8 hash) | ~128B | **Device** | ❌ 不行 | 每步全量扫描，必须在 Device |
| Top-k Indices | ~8KB | Device | - | 太小，无需 Offload |

---

## 2. 架构设计

### 2.1 三层架构

```
┌──────────────────────────────────────────────────────────┐
│ Layer 3: DSA Attention Path                              │
│ (vllm_ascend/attention/dsa_v1.py)                        │
│                                                          │
│  修改 decode 路径:                                        │
│  prefill → offload_to_host()                             │
│  decode  → indexer → gather_selection → sparse_attn      │
├──────────────────────────────────────────────────────────┤
│ Layer 2: KV Offload Manager                              │
│ (新文件: vllm_ascend/attention/dsa_kv_offload.py)         │
│                                                          │
│  - Host 内存管理 (aclrtMallocHost)                        │
│  - Selection Cache 分配与管理                              │
│  - Block Table / Block Status 维护                        │
│  - Prefill 后 D2H 流水线                                  │
├──────────────────────────────────────────────────────────┤
│ Layer 1: Op Wrapper                                      │
│ (新文件: vllm_ascend/ops/gather_selection_kv_cache.py)     │
│                                                          │
│  - C++ extension 调用 aclnn 两阶段 API                     │
│  - Python 封装: gather_selection_kv_cache(...)             │
└──────────────────────────────────────────────────────────┘
```

### 2.2 数据流（Decode 阶段）

```
                    ┌─────────────┐
                    │   Query q   │
                    └──────┬──────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Lightning Indexer    │
              │   (FP8 MQA 全量打分)    │
              │                        │
              │ Indexer K Cache ◄──────│──── Device (不 offload)
              └────────────┬───────────┘
                           │
                    topk_indices
                    [B, S, H, topk]
                           │
                           ▼
            ┌──────────────────────────────┐
            │  GatherSelectionKvCache      │
            │                              │
            │  Full KV Cache ◄─── Host (pinned, NPU 可寻址)
            │       │                      │
            │       │ 稀疏 H2D DMA         │
            │       ▼                      │
            │  Selection Cache ──► Device  │
            │                              │
            │  block_status: 缓存复用跟踪    │
            └──────────────┬───────────────┘
                           │
                   Selection Cache
                   (只有 topk 组的 KV)
                           │
                           ▼
              ┌────────────────────────┐
              │  Sparse Flash Attn     │
              │  (只计算选中的 token)    │
              └────────────────────────┘
```

### 2.3 内存布局

```
Host (CPU pinned, NPU 可寻址)
├── full_k_rope:     [num_blocks, block_size, 64]      FP16
├── full_kv_cache:   [num_blocks, block_size, 512]     FP16
└── (通过 aclrtMallocHost 分配，有统一 GM 地址)

Device (NPU HBM)
├── Indexer K Cache: [num_blocks, block_size, 128]     FP8   ← 不 offload
├── Selection K Rope:    [sel_blocks, 64, 64]          FP16
├── Selection KV Cache:  [sel_blocks, 64, 512]         FP16
├── Selection Block Table: [B*S*H, max_sel_blocks]     INT32
├── Selection Block Status: [B, S, H, topk+1]         INT32
├── Full Block Table:    [B, max_full_blocks]           INT32
├── topk_indices:        [B, S, H, topk]               INT32
├── full_kv_actual_seq:  [B]                            INT32
└── full_q_actual_seq:   [B]                            INT32
```

---

## 3. Host 内存问题

### 3.1 问题描述

测试中发现：使用 PyTorch `pin_memory=True` 分配的 CPU 内存，其 `data_ptr()` 返回的是 **CPU 虚拟地址**（如 `0x7f8a...`），NPU 内核无法通过 DMA 访问。

```
CPU 地址空间:  0x7f8a_0000_0000  ← tensor.data_ptr() 返回这个
NPU 地址空间:  0x12c0_4000_0000  ← NPU 内核只能访问这个范围

aclCreateTensor(data_ptr=0x7f8a...)
    → NPU 内核 DMA 读 0x7f8a...
    → "DDR address of MTE instruction is out of range" 崩溃
```

### 3.2 解决方向

算子设计意图是 full KV cache 在 Host 上，NPU 通过 DMA 直接读取。需要的是 **NPU 可寻址的 Host 内存**：

| 方案 | 方式 | 说明 |
|------|------|------|
| A | `aclrtMallocHost` | CANN 提供的 Host 内存分配，自动映射到 NPU GM 地址空间 |
| B | DVPP / 大页内存 | 特殊的共享内存区域 |
| C | 设备测试先用 Device | 当前测试采用的方式，绕过问题验证算子逻辑 |

**Step 4 需要解决此问题**，关键是找到 `aclrtMallocHost` 分配的内存如何包装成 PyTorch tensor 传给算子。

---

## 4. 实现步骤

### Step 1: 算子调用测试 ✅ 已完成

**目标**: 验证 GatherSelectionKvCache 算子在 NPU 上能正常工作

**完成情况**:
- 编写 C++ extension wrapper（直接调用 aclnn 两阶段 API）
- 编译成功，链接 `libnnopbase.so` + `libcust_opapi.so` + `libascendcl.so` + `libtorch_npu.so`
- 测试通过：
  - Basic Gather: 从 full cache 采集到 selection cache，数据正确 ✓
  - Cache Reuse: 连续调用，缓存复用机制正常工作 ✓
- 文件位置（NPU 机器）: `/workspace/test_gather/`

**关键发现**:
- `EXEC_NPU_CMD` 宏不可用（编译进 libtorch_npu.so，无公开头文件）
- 必须手动调用 `aclCreateTensor` + 两阶段 API
- `aclCreateTensor` 在 `libnnopbase.so` 中
- Host pinned memory 不能直接传给 `aclCreateTensor`（地址空间不同）
- `block_status` 中 slot 顺序不等于 topk 顺序（复用时保留原位）

### Step 2: 数据结构分配

**目标**: 在 vllm-ascend 中正确分配 Offload 相关的数据结构

**工作内容**:
- 在模型初始化时分配 Host 端 Full KV Cache（需解决 NPU 可寻址问题）
- 分配 Device 端 Selection Cache、Block Table、Block Status
- 与 vllm 的 PagedAttention block manager 对接
- 确定 Selection Cache 的块数量（topk * batch_size * num_heads）

### Step 3: Decode 路径集成

**目标**: 在 DSA decode attention 路径中插入 GatherSelectionKvCache 调用

**工作内容**:
- 修改 `dsa_v1.py` 的 decode 路径
- 在 Lightning Indexer 之后、Sparse Flash Attention 之前调用 gather
- 管理 block_status 的生命周期（跨 decode step 保持）
- 处理多 batch、多 head 的维度映射

### Step 4: Prefill D2H 流水线

**目标**: Prefill 完成后将 Full KV Cache 从 Device offload 到 Host

**工作内容**:
- Prefill 阶段正常在 Device 上计算和存储 KV Cache
- Prefill 完成后，异步将 KV Cache 复制到 Host（D2H DMA）
- 解决 Host 内存分配问题（`aclrtMallocHost` 或等价方案）
- 实现双缓冲 / 流水线化，避免阻塞 decode

### Step 5: 端到端测试

**目标**: 完整推理流程验证

**工作内容**:
- 使用 DeepSeek-V3 模型进行端到端测试
- 验证长上下文场景下的正确性（输出一致性）
- 性能对比：Offload vs 不 Offload 的吞吐量和延迟
- 内存使用对比：HBM 节省量

---

## 5. 已有代码参考

### vllm-ascend 现有相关代码

| 文件 | 内容 | 与 Offload 的关系 |
|------|------|-------------------|
| `vllm_ascend/attention/dsa_v1.py` (2820行) | DSA attention 完整实现 | **主要修改点**，需在 decode 路径插入 gather |
| `vllm_ascend/kv_offload/cpu_npu.py` (261行) | 通用 block 级 NPU↔CPU 传输 | 参考其异步传输设计 |
| `vllm_ascend/simple_kv_offload/worker.py` (224行) | 通用 KV offload worker | 参考其 worker 模式 |
| `csrc/torch_binding.cpp` | C++ op 注册 | 参考 `EXEC_NPU_CMD` 用法模式 |

### 算子源码（已分析）

| 文件 | 内容 |
|------|------|
| `op_kernel/gather_selection_kv_cache_split_bs_reuse.h` | 标量路径 (topk≤32) |
| `op_kernel/gather_selection_kv_cache_split_bs_reuse_vec.h` | 向量化路径 (topk>32) |
| `op_host/gather_selection_kv_cache_tiling.h/cpp` | Tiling 数据结构与逻辑 |
| `op_host/gather_selection_kv_cache_def.cpp` | 算子注册（910_93, 910b）|

---

## 6. 风险与待确认项

| 风险项 | 影响 | 缓解方案 |
|--------|------|----------|
| Host 内存 NPU 寻址 | Step 4 关键阻塞 | 调研 `aclrtMallocHost`，参考 MindIE 实现 |
| Selection Cache 块管理与 vllm block manager 冲突 | Step 2-3 | 可能需要独立的 block allocator |
| 多卡 TP 场景下 block_status 同步 | Step 3 扩展 | 先单卡验证，后续处理 |
| 算子 topk 上限 2048 | 极长上下文场景 | 当前 DSA 设计就是 2048，无需突破 |
| Prefill D2H 延迟影响首 token | Step 4 | 异步 DMA + 双缓冲 |
