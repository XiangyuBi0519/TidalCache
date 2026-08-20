# GatherSelectionKvCache 算子原理与 DSA KV Offload 架构

> 日期: 2026-08-20
> 状态: Step 1/2 验证通过，4 个测试全部 PASS

---

## 1. DSA 的核心问题

DeepSeek-V3/R1 使用 MLA (Multi-head Latent Attention)，每个 token 的 KV 数据：
- **c^KV**（MLA 压缩潜向量）: 512 维
- **k^R**（RoPE 位置编码）: 64 维
- 总计: **576 bytes/token**（BF16）

长序列下 KV Cache 巨大（128K 序列 × 60 层 ≈ 4.2 GB / 请求），Device HBM 放不下。

但 **DSA（DeepSeek Sparse Attention）只对全量 KV 的一小部分做 attention**：
1. Lightning Indexer 先用 `k^R`（64d，轻量）做全量打分
2. 选出 topk 个"重要的 token 组"
3. 只对这些组做精确 attention

**核心思路：全量 KV 放 Host 内存，只把 topk 选中的部分搬到 Device。**

---

## 2. 整体架构

```
                    ┌─────────────────────────────────────────┐
                    │           HOST (Hugepage 内存)           │
                    │                                         │
                    │  Full KV Cache  [num_blocks, 64, 512]   │
                    │  Full K Rope    [num_blocks, 64, 64]    │
                    │  (NPU 视图: aclrtHostRegisterV2          │
                    │   + aclrtHostGetDevicePointer)           │
                    └───────────────┬─────────────────────────┘
                                    │
                        只搬 topk 选中的 blocks
                        (稀疏 Host→Device DMA)
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────┐
│                    DEVICE (NPU HBM)                           │
│                                                               │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐  │
│  │ Indexer K     │     │ Lightning    │     │ GatherSelect │  │
│  │ Cache (k^R)  │────▶│ Indexer      │────▶│ ionKvCache   │  │
│  │ 64d, Device  │     │ Q×K^R→Top-k  │     │ 算子          │  │
│  └──────────────┘     └──────────────┘     └──────┬───────┘  │
│                                                    │          │
│        ┌───────────┐                    ┌──────────▼───────┐  │
│        │ Q (decode) │                   │ Selection Cache  │  │
│        │ 当前 token  │                  │ sel_kv + sel_rope│  │
│        └─────┬─────┘                    │ topk 大小, Device│  │
│              │                          └──────────┬───────┘  │
│              │         ┌──────────────┐            │          │
│              └────────▶│ Sparse Attn  │◀───────────┘          │
│                        │ DSA Kernel   │◀── SWA Cache          │
│                        └──────────────┘    (滑动窗口, Device) │
└───────────────────────────────────────────────────────────────┘
```

### Decode 一步的流程

```
1. Q 生成（当前 token 的 query）
2. K^R scatter 到 Indexer K Cache（Device 上，始终保留，只有 64d）
3. Lightning Indexer: Q × K^R → topk_indices（选出重要的 token 组）
4.【GatherSelectionKvCache】: Host Full KV → Device Selection Cache
5. Sparse Attention: Q × Selection Cache + SWA Cache → output
```

第 4 步就是我们要集成的——目前 vllm-ascend 的 DSA 实现里 KV 全在 Device 上，没有 offload。

---

## 3. GatherSelectionKvCache 算子详解

### 3.1 功能

根据 Lightning Indexer 输出的 topk_indices，从 Full KV Cache（Host）中**稀疏采集**数据到 Selection Cache（Device），同时支持**缓存复用**。

### 3.2 一次调用的具体过程

**场景假设**：seq_len=8192 tokens，block_size=64，topk=4

- Full KV Cache（Host）：128 blocks × 64 tokens × 512d = 全量
- Selection Cache（Device）：4 blocks × 64 tokens × 512d = topk 大小

**算子执行步骤**：

```
输入: topk_indices = [3, 7, 51, 120]  （Indexer 选出的 4 个 group）
输入: block_status = [3, 7, 22, 88]   （上一轮 Selection Cache 缓存的 group）

比对:
  group 3  → 上轮 slot 0 已缓存 → 命中，不搬（zero copy）
  group 7  → 上轮 slot 1 已缓存 → 命中，不搬
  group 51 → 未缓存 → miss，从 Host block[51] DMA 到 Device slot 2
  group 120→ 未缓存 → miss，从 Host block[120] DMA 到 Device slot 3

输出: block_status = [3, 7, 51, 120]  （更新后的缓存状态）
输出: sel_kv_actual_seq = [256]         （selection cache 的有效 token 数）
```

**缓存复用的意义**：连续 decode 时，topk 选择有连续性（相邻 step 倾向于选相似的 token 组），命中率通常 50%~80%，大幅减少实际搬运量。

### 3.3 topk_block_size

每个 topk index 代表**一组连续的 64 个 token**（和 block_size 对齐）。

`topk_indices[b,0,0,k] = 3` → 选中第 3 组 → token 192~255。

### 3.4 block_status 语义

```
block_status[b, s, h, i]    = 第 i 个 slot 当前缓存的 group index
block_status[b, s, h, topk] = actual_seq_len（最后一个位置存序列长度）
初始值 -1 = 空 slot
```

算子内部用 block_status 判断命中/miss，然后原地更新：
- 命中的 slot：保持不变
- miss 的 group：填入被淘汰的 slot，更新 block_status

### 3.5 为什么不用多次 memcpy

topk 选中的 blocks 在 Full KV Cache 里是**散布的**（block 3, 7, 51, 120），需要从多个不连续位置 gather 到 Selection Cache 的连续 slots。GatherSelectionKvCache 是**单个 kernel 完成所有 scatter/gather + 缓存复用判断**，比多次 aclrtMemcpy 高效得多。

### 3.6 内核分支

- topk ≤ 32 → tilingKey=1，标量路径（SplitBsReuse）
- topk > 32 → tilingKey=2，向量化路径（SplitBsReuseVec，使用 Sort 指令）

### 3.7 张量接口

| 张量 | Shape | Dtype | 位置 | 说明 |
|------|-------|-------|------|------|
| selection_k_rope | [s_blk, 64, 64] | BF16 | Device | Selection RoPE（inplace 修改）|
| selection_kv_cache | [s_blk, 64, 512] | BF16 | Device | Selection KV（inplace 修改）|
| selection_kv_block_table | [B×H, s_max_blk] | INT32 | Device | Selection 物理块映射 |
| selection_kv_block_status | [B, S, H, topk+1] | INT32 | Device | 缓存状态（inplace 修改）|
| selection_topk_indices | [B, S, H, topk] | INT32 | Device | Indexer 输出的 Top-k 组索引 |
| full_k_rope | [f_blk, 64, 64] | BF16 | Host(NPU视图) | 完整 RoPE cache |
| full_kv_cache | [f_blk, 64, 512] | BF16 | Host(NPU视图) | 完整 KV cache |
| full_kv_block_table | [B, f_max_blk] | INT32 | Device | Full 物理块映射 |
| full_kv_actual_seq | [B] | INT32 | Device | 实际序列长度 |
| full_q_actual_seq | [B] | INT32 | Device | Query 长度（decode=1）|
| **attr**: selection_topk_block_size | int64 | - | - | 默认 64 |

**输出**：selection_kv_actual_seq [B×S×H] INT32

---

## 4. Host 内存注册链路

Full KV Cache 在 Host 上，但 NPU kernel 需要通过设备地址访问。注册链路：

```
hugepage mmap (2MB 对齐)
    → mlock() 锁定物理页
    → aclrtHostRegisterV2(ptr, size, ACL_HOST_REG_PINNED | ACL_HOST_REG_MAPPED)
    → aclrtHostGetDevicePointer(host_ptr, &dev_ptr, 0)
    → c10::DataPtr(dev_ptr) → Storage → npu_tensor.set_()
```

### 关键发现（2026-08-20 验证）

1. **NPU 有独立设备地址空间**：host_ptr(`0xfffee9c00000`) → dev_ptr(`0x3ffe9c00000`)，不是统一虚拟地址
2. 直接用 host_ptr 作为 dev_ptr 会导致 NPU kernel SUSPECT REMOTE ERROR (507057)
3. **CANN 头文件 include 顺序**：必须把 `CANN_HOME/include` 放在 `torch_npu/include` 之前，否则 torch_npu 打包的旧版 ACL 头文件会遮蔽 `aclrtHostGetDevicePointer` 等 API

---

## 5. 性能分析

### 5.1 HBM 节省

以 DeepSeek-V3、8K 序列为例：

| 配置 | Device HBM / 请求 | 说明 |
|------|-------------------|------|
| 无 offload | ~270 MB | 全量 KV 在 Device |
| 有 offload | ~203 MB | Indexer K(60MB) + Selection(8.4MB) + SWA(135MB) |
| **节省** | **~25%** | 更多并发 or 更长序列 |

超长序列（64K/128K）下收益更大：从 OOM 变为可用。

### 5.2 TPOT 影响

每步每层每请求的搬运量（topk=4，假设 50% miss 率）：

```
4 blocks × 64 tokens × 576 bytes × 50% miss = ~74 KB / 层
60 层 → ~4.3 MB / 请求 / step
```

PCIe/HCCS 带宽 30~60 GB/s → 搬运延迟 **0.07~0.14 ms**。
对比 decode step 总时延 10~30 ms → 开销 **< 1%**。

**风险场景**：大 batch（256 并发）下搬运总量 ~1.1 GB，可能占 18~36 ms，需要 pipeline 优化。

---

## 6. 测试验证（2026-08-20 全部 PASS）

| 测试 | 内容 | 状态 |
|------|------|------|
| Test 1 | Basic Gather (Device→Device) | PASS |
| Test 2 | Cache Reuse (重叠 topk 验证复用) | PASS |
| Test 3 | Host Memory Registration (hugepage + NPU MMU) | PASS |
| Test 4 | Host→Device Gather (完整链路) | PASS |

测试代码: `test_gather/test_gather_op.py`
编译脚本: `test_gather/build.sh`
