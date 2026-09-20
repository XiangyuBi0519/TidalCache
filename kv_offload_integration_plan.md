# DSA KV Cache Offload — vllm-ascend 集成方案 v2

> 在 vllm-ascend 上原生实现 DeepSeek Sparse Attention 的 KV Cache Offload
>
> 日期: 2026-07-11 (初版) → 2026-08-17 (v2 更新) → 2026-09-11 (v3 端到端跑通)
> 状态: **Step 3 已完成** · 端到端集成跑通 · 模型输出正确 · 首版性能基线已建立

---

## 1. 背景与目标

### 1.1 问题

DeepSeek-V3/R1 使用 MLA (Multi-head Latent Attention)，KV Cache 包含：
- **c^KV**（MLA 压缩潜向量）: 512 维
- **k^R**（RoPE 位置编码）: 64 维
- 总计每 token: **576 bytes**（BF16）

128K 上下文长度时，单序列 KV Cache 占 ~70MB，批量推理时 Device HBM 成为瓶颈。

### 1.2 方案：DSA + KV Cache Offload

DeepSeek Sparse Attention (DSA) 的核心洞察：**decode 阶段不需要全部 KV Cache，只需要 Top-k 相关的稀疏子集**。

```
┌──────────────────────────────────────────────────────┐
│                   Decode 一步的流程                    │
│                                                       │
│  1. Lightning Indexer (FP8 MQA)                       │
│     - 用 Indexer K Cache (Device) 做全量打分            │
│     - 输出 Top-k indices (每个 index = 64 tokens)      │
│                                                       │
│  2. GatherSelectionKvCache (CANN 自定义算子)            │
│     - 根据 Top-k indices 从 Host 稀疏采集到 Device      │
│     - NPU DMA 直接读 Host 内存（零拷贝 MMU 映射）       │
│     - 支持缓存复用（block_status 跟踪命中/未命中）       │
│                                                       │
│  3. Sparse Flash Attention                            │
│     - 只用 Selection Cache (Device) 做注意力计算        │
│     - topk=2048, 每组 64 tokens → 最多 131K tokens     │
└──────────────────────────────────────────────────────┘
```

### 1.3 三类缓存与 Offload 策略

| 缓存类型 | 大小/token | 位置 | 能否 Offload | 原因 |
|----------|-----------|------|-------------|------|
| MLA Full KV Cache (c^KV + k^R) | 576B | **Host** | ✅ 是 | 稀疏读取，通过 Top-k 选择 |
| Indexer K Cache (FP8) | ~128B | **Device** | ❌ 否 | 每步全量扫描，必须在 Device |
| Selection Cache (TopK 子集) | 按需 | **Device** | - | 算子输出，临时缓冲 |

### 1.4 定位：与 OmniCache 的区别

| 维度 | OmniCache | 本方案 |
|------|-----------|--------|
| 形态 | vLLM 外部 plugin，替换整个 KV cache 管理 | **vllm-ascend 内部原生能力** |
| KV Connector | 自带 OmniCacheConnector (OX)，占用接口 | **不占用 KV Connector**，留给 Mooncake |
| 适用场景 | 完整的 P/D 分离 + KV offload | 仅 DSA 层的 KV offload |
| 与 Mooncake Store | 冲突（都实现 KVConnectorBase_V1） | **兼容**（不同层面的能力） |

**核心价值**：集成到 vllm-ascend 后，可以与 Mooncake Store 搭配使用——Mooncake 负责跨节点 P/D KV 传输，本方案负责节点内 DSA 层的 HBM 卸载。这是 OmniCache 做不到的。

---

## 2. 架构设计

### 2.1 三层架构

```
┌──────────────────────────────────────────────────────────┐
│ Layer 3: DSA Attention Path                              │
│ (vllm_ascend/attention/dsa_v1.py)                        │
│                                                          │
│  修改 decode 路径:                                        │
│  prefill: 正常 Device 计算 → D2H offload 到 Host          │
│  decode:  indexer → gather_selection → sparse_attn       │
├──────────────────────────────────────────────────────────┤
│ Layer 2: KV Offload Manager                              │
│ (新文件: vllm_ascend/attention/dsa_kv_offload.py)         │
│                                                          │
│  - Host 内存池: hugepage mmap + NPU MMU 注册              │
│  - Selection Cache 预分配 (Device)                        │
│  - Block Table / Block Status 生命周期管理                 │
│  - Prefill 后 D2H 异步传输                                │
├──────────────────────────────────────────────────────────┤
│ Layer 1: Op Wrapper + Host Memory                        │
│ (新文件: vllm_ascend/ops/gather_selection_kv_cache.py)    │
│ (新文件: vllm_ascend/ops/csrc/zero_copy_npu.cpp)         │
│ (新文件: vllm_ascend/ops/csrc/tensor_register.cpp)       │
│                                                          │
│  - C++ extension: aclnn 两阶段 API 调用                   │
│  - C++ extension: Host 内存 NPU MMU 注册                  │
│  - Python 封装                                            │
└──────────────────────────────────────────────────────────┘
```

### 2.2 数据流

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
              │ Indexer K Cache ◄──────│──── Device HBM (不 offload)
              └────────────┬───────────┘
                           │
                    topk_indices
                    [B, S, H, topk]
                           │
                           ▼
            ┌──────────────────────────────┐
            │  GatherSelectionKvCache      │
            │                              │
            │  Full KV (c^KV + k^R) ◄─────│──── Host DDR
            │       │                      │     (hugepage + NPU MMU 映射)
            │       │ NPU DMA 零拷贝读取    │
            │       ▼                      │
            │  Selection Cache ──► Device  │     Device HBM
            │                              │     (预分配，跨 step 复用)
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
Host DDR (hugepage mmap, NPU MMU 注册)
├── full_kv_pool: [num_dsa_layers, num_blocks, block_size, 576]  BF16
│   ├── 前 512 维 = c^KV (MLA 压缩潜向量)
│   └── 后 64 维  = k^R  (RoPE 位置编码)
├── 通过 hugetlbfs 分配 (/dev/hugepages/vllm_ascend_dsa)
├── aclrtHostRegisterV2(MAPPED) 注册到 NPU MMU
└── aclrtHostGetDevicePointer() 获取 NPU GM 地址

Device HBM
├── Indexer K Cache:        [num_blocks, block_size, 128]       FP8   ← 不 offload
├── Selection K Rope:       [num_layers, total_sel_blocks, block_size, 64]   BF16
├── Selection KV Cache:     [num_layers, total_sel_blocks, block_size, 512]  BF16
├── Selection Block Table:  [batch_size, max_sel_blocks]        INT32
├── Selection Block Status: [num_layers, batch_size, topk+1]    INT32
├── Full Block Table:       [batch_size, max_full_blocks]       INT32
├── topk_indices:           [B, S, H, topk]                     INT32
├── full_kv_actual_seq:     [B]                                 INT32
└── full_q_actual_seq:      [B]                                 INT32

其中: total_sel_blocks = max_sel_blocks * batch_size
      max_sel_blocks = ceil(topk * topk_block_size / block_size)
```

---

## 3. Host 内存方案（已明确）

### 3.1 问题回顾

PyTorch `pin_memory=True` 分配的 CPU 内存，`data_ptr()` 返回 CPU 虚拟地址（`0x7f8a...`），NPU 内核无法 DMA 访问，报 "DDR address of MTE instruction is out of range"。

### 3.2 解决方案：Hugepage + NPU MMU 注册

参考 OmniCache 的 `tensor_register_lib/` 实现，完整链路如下：

```
Step 1: 分配 Hugepage 内存
    fd = os.open("/dev/hugepages/vllm_ascend_dsa", O_RDWR)
    os.ftruncate(fd, aligned_size)  # 2MB 对齐
    mmap_obj = mmap(fd, aligned_size, MAP_SHARED)
    host_tensor = torch.frombuffer(mmap_obj, dtype=torch.bfloat16)
    host_tensor = host_tensor.view(num_layers, num_blocks, block_size, 576)

Step 2: 注册到 NPU MMU (C++ extension: tensor_register.cpp)
    aclrtHostRegisterV2(host_ptr, size, ACL_HOST_REG_PINNED | ACL_HOST_REG_MAPPED)
    aclrtHostGetDevicePointer(host_ptr, &dev_ptr)
    // host_ptr: CPU 虚拟地址 0x7f8a...
    // dev_ptr:  NPU GM 地址   0x12c0... ← NPU 内核可以 DMA 读

Step 3: 包装为 PyTorch NPU Tensor (C++ extension: zero_copy_npu.cpp)
    // 创建一个 "假" NPU tensor，device=PrivateUse1(NPU)
    // 但 data_ptr 指向 dev_ptr（NPU MMU 映射后的 Host 地址）
    c10::DataPtr data_ptr(dev_ptr, dev_ptr, [](void*){}, npu_device);
    storage.set_data_ptr(std::move(data_ptr));
    npu_tensor.set_(storage, 0, host_tensor.sizes());

结果:
    host_tensor  → CPU 视图，可以用标准 PyTorch 操作读写
    npu_tensor   → NPU 视图，可以传给 aclCreateTensor / NPU 算子
    两者指向同一块物理内存（零拷贝）
```

### 3.3 关键约束

| 约束 | 说明 |
|------|------|
| 2MB 对齐 | hugepage 天然满足，`ftruncate` 时需向上对齐 |
| 4K 对齐校验 | `aclrtHostRegisterV2` 要求 4K 对齐（hugepage 自动满足） |
| mlock | 防止 hugepage 被 swap，OmniCache 使用 `mlock()` |
| 生命周期 | 必须 `aclrtHostUnregister` 后再释放 mmap |

### 3.4 C++ Extension 编译依赖

```python
# tensor_register.so (纯 C, ctypes 调用)
g++ -fPIC -shared -I${CANN}/include -L${CANN}/lib64 -lascendcl \
    tensor_register.cpp -o tensor_register.so

# zero_copy_npu (PyTorch extension, pybind11)
libraries = ["ascendcl"]
include_dirs = [CANN + "/include"]
```

---

## 4. Mooncake Store 兼容设计

### 4.1 为什么能兼容

```
┌─────────────────────────────────────────────────────────────┐
│                    两者作用于不同层面                          │
│                                                              │
│  Mooncake Store (KVConnectorBase_V1)                        │
│  ├── 作用: 跨节点 Prefill→Decode KV 传输                     │
│  ├── 粒度: block 级 GPU-to-GPU RDMA                          │
│  ├── 时机: prefill 完成后，decode 开始前                      │
│  └── 接口: vLLM KV Connector                                │
│                                                              │
│  本方案 (vllm-ascend 内部)                                   │
│  ├── 作用: 节点内 DSA 层 HBM → Host 卸载                     │
│  ├── 粒度: 按 DSA 层 + block 级                              │
│  ├── 时机: decode 阶段每步 gather                            │
│  └── 接口: 不占用 KV Connector                               │
│                                                              │
│  搭配使用时的流程:                                            │
│  Prefill Node:                                               │
│    prefill 计算 → Mooncake RDMA 传 KV 到 Decode Node         │
│  Decode Node:                                                │
│    收到 KV → Indexer K 留 Device, Full KV D2H offload 到 Host│
│    decode 每步: indexer → gather(Host→Device) → sparse attn  │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 关键设计约束

1. **不实现 KVConnectorBase_V1** — 完全不碰 KV Connector 接口
2. **不替换 `initialize_kv_cache`** — 只扩展 DSA attention 的 decode 路径
3. **Mooncake 传来的 KV 到 Device 后，由本方案 D2H offload** — 两者串行，无冲突
4. **Block Manager 不修改** — Selection Cache 用独立预分配缓冲，不走 vLLM 的 block allocator

### 4.3 OmniCache 为什么冲突

OmniCache 实现了 `OmniCacheConnector(KVConnectorBase_V1)`，自带 P/D KV 传输。vLLM 同一时刻只支持一个 KV Connector，所以 OmniCache 和 Mooncake Store 互斥。我们的方案绕开这个问题。

---

## 5. 实现步骤

### Step 1: 算子调用测试 ✅ 已完成

- 编写 C++ extension wrapper（直接调用 aclnn 两阶段 API）
- 编译成功，链接 libnnopbase.so + libcust_opapi.so + libascendcl.so + libtorch_npu.so
- 测试通过：Basic Gather ✓ · Cache Reuse ✓
- 文件位置（NPU 机器）: `/workspace/test_gather/`

关键发现：
- `EXEC_NPU_CMD` 宏不可用，必须手动 `aclCreateTensor` + 两阶段 API
- `aclCreateTensor` 在 `libnnopbase.so` 中
- `block_status` slot 顺序不等于 topk 顺序（复用时保留原位）
- Host pinned memory 不能直接传给算子（地址空间不同）

### Step 2: Host 内存注册验证 ✅ 已完成

**目标**: 在 NPU 上验证 hugepage + MMU 注册的完整链路

**结果**: 4/4 测试通过 — Basic Gather ✓ · Cache Reuse ✓ · Host Registration ✓ · H2D Gather ✓

### Step 3: Selection Cache 缓冲区管理 ✅ 已完成

**目标**: 实现 Selection Cache 的分配与生命周期管理

**工作内容**:
- 参考 OmniCache `buffers.py`，预分配 Selection 相关张量：
  - `selection_k_rope`: `[num_layers, total_sel_blocks, block_size, 64]` Device
  - `selection_kv_cache`: `[num_layers, total_sel_blocks, block_size, 512]` Device
  - `selection_kv_block_table`: `[batch_size, max_sel_blocks]` Device，连续递增
  - `selection_kv_block_status`: `[num_layers, batch_size, topk+1]` Device，初始 -1
- Selection Cache 不走 vLLM block manager，独立管理
- batch 变化时（请求增删）需要更新 block_status（参考 OmniCache `GatherSelectionUpdater`）

### Step 4: Decode 路径集成 ✅ 已完成

**目标**: 在 DSA decode attention 路径中插入 GatherSelectionKvCache 调用

**结果**: 通过 `apply_patches.py` 动态注入 vllm-ascend，包含：
- PATCH2 (gather): Indexer 之后调用 GatherSelectionKvCache，gather 结果 copy-back 到 compress_kv_cache
- PATCH3 (scatter): Attention 之后将新 KV D2H offload 到 Host hugepage
- CP 路径: 独立的 CP2/CP3 补丁处理 Context Parallel 模式（8 rank, local_topk=64）
- Warmup 兼容: 3 个 Bug 修复（见 README 调试记录）

### Step 5: 端到端验证 ✅ 已完成

**目标**: 完整推理流程验证

**结果**:
- 模型输出正确性：已验证（curl 请求返回合理 DeepSeek 文本）
- 性能基线：TidalCache ON = 18.3s, OFF = 13.3s（512 tokens, +37% 开销）
- 独立日志：tidalcache.log，含 SCATTER/GATHER/COPYBACK 每阶段追踪
- Hugepage：部分覆盖（63%），需开机参数完整分配

### Step 5.5: 数据通路验证 ✅ 已完成 (2026-09-12)

**目的**: 在改架构前先验证当前 gather 通路真的被 attention 使用

**方法**: `TIDALCACHE_POISON=1` 环境变量在 gather 后、copy-back 前把 sel_kv 填成 -1000，看输出是否变异

**结果**:
- 基线（POISON=0）: `"你好！我是DeepSeek..."` （正常）
- Poison（POISON=1）: `"וניב［ Septy学一做ahimut..."` （完全乱码）
- **结论**: gather 的数据确实进入了 attention，数据通路正常工作 ✓

### Step 6: 减少 Device KV 分配 + 直接 Selection Cache Attention ← 下一步（核心）

**目标**: 实现真正的 HBM 节省

**当前问题**: v1 验证了数据通路正确性，但 vllm 仍为 DSA 层分配全量 Device KV cache，TidalCache 实际增加了内存占用（Device 全量 KV + Host 副本 + Selection Cache）。

#### 6.1 三种 KV Cache 的定位

DeepSeek V4 DSA 的 attention 是混合的：dense（滑动窗口）+ sparse（压缩+topk）。存在三个 KV cache：

| 名字 | 类型 | 覆盖范围 | 大小 | Offload 策略 |
|------|------|---------|------|-------------|
| **swa_kv_cache** | Dense | 最近 W 个 token（如 4K） | 小 | 留 Device，不 offload |
| **compress_kv_cache** | Sparse（压缩） | 全部历史（m:1 压缩） | **大** | **主要 offload 目标** |
| **indexer_k_cache** | Dense (FP8, 64d) | 全部历史 | 中 | 每步全扫，必须 Device |

**只有 compress_kv_cache 值得 offload**。省的就是它。

#### 6.2 Prefill 和 Decode 的 KV 流程

**Prefill（一次处理整个 prompt）**:
```
Prompt → 计算 KV → scatter 到 compress/swa/indexer 三份 cache
       → Attention（每个 prompt token）:
          Indexer 打分 → topk
          SparseAttn = swa_dense(recent) + sparse(topk from compress)
```
**关键**: prefill 也做 topk 稀疏 attention（不只是 decode），也会**读** compress_kv_cache。

**Decode（每步一个新 token）**:
```
新 token → 计算 KV → scatter 到三份 cache
        → Attention（当前 token）:
           Indexer 在全量 indexer_cache 上打分 → topk
           SparseAttn = swa_dense + sparse(topk from compress)
```

#### 6.3 设计方案（三种候选，实现方案 A + B 带开关切换）

**方案 A**: Prefill 在 Device，Prefill 后 D2H sweep
- Prefill: scatter/attn 走原样（Device compress_kv_cache）
- Prefill 结束: 触发一次性 D2H 拷贝 → Host hugepage
- 释放 Device compress blocks
- Decode: gather from Host → sel_kv → attn 直接读 sel_kv

**方案 B**: Prefill 双写（Device + Host）
- Prefill: scatter 同时写 Device 和 Host（dual write）
- Prefill: attn 读 Device（原样）
- Prefill 结束: 直接释放 Device blocks（Host 已有数据）
- Decode: 同 A

**方案 C**（未来）: Prefill 也走 Host
- Scatter 只写 Host，prefill attention 也走 gather 通路
- 完全省掉 Device 全量 compress（激进方案）
- 需要 gather 算子支持 prefill 的多 query token 场景

#### 6.4 A/B 切换开关

```bash
TIDALCACHE_PREFILL_MODE=A     # 方案 A（默认）：post-prefill D2H sweep
TIDALCACHE_PREFILL_MODE=B     # 方案 B：dual write during prefill
TIDALCACHE_PREFILL_MODE=OFF   # 对照组：不做 offload，保留 Device 全量
```

**预期性能差异**（需要实测确认）:
- 短 prompt：B 的 dual write 累积开销 < A 的 sweep 开销 → B 可能更优
- 长 prompt：A 的 sweep 可异步（跟 decode 计算重叠） → A 可能更优

#### 6.5 共同基础设施（A、B 都要做）

1. **Prefill/Decode 边界检测** — 找到 vllm 里 prefill 完成的 hook 点（进行中）
2. **Post-prefill Device blocks 释放** — resize_(0) 或 pool 缩容
3. **Decode 路径改造**:
   - Gather from Host → sel_kv（已有）
   - attn_op 参数切换：`cmp_kv=sel_kv_cache`, `cmp_block_table=sel_block_table`, `cmp_sparse_indices=[0..topk-1]`（新增 PATCH4）
   - 去掉 copy-back（decode 不再写 compress_kv_cache）
4. **Device compress pool 缩容** — 初始化时只分配 max_concurrent × topk × few multiplier

#### 6.6 待确认的技术点

- [x] vllm 里 Prefill → Decode 切换的 hook 点 ✅ **已定位**（见 6.7）
- [ ] compress_kv_cache 是跨请求共享 pool，如何按请求粒度释放
- [ ] Device pool 应缩到多大（Selection Cache + hot cache buffer）
- [ ] SparseAttnSharedkv 算子对 block_size 的约束（sel_kv 用 compress_block_size=64，vllm block_size=128，能否直接兼容）

#### 6.7 Prefill→Decode 边界检测（Hook 点）

**关键信号**（`model_runner_v1.py:3266`）:
```python
is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
```
Per-request 布尔数组，`True` 表示该请求还在 prefill 阶段。

**"本步刚完成 prefill"的判定**:
```python
was_prefilling = num_computed[R] < num_prompt[R]                            # 步前
will_be_done  = num_computed[R] + num_scheduled[R] >= num_prompt[R]         # 步后
just_finished_prefill[R] = was_prefilling and will_be_done
```

**推荐 hook 位置：dsa_v1.py `forward` 方法**（选项 2）
- 每层 `_forward_prefill` 完成后，检测本步有没有 request 完成 prefill
- 有的话，本层立刻 D2H copy 该 request 占用的 blocks → Host
- 天然 per-layer 触发，21 层各自执行一次
- 从 `attn_metadata[0].prefill.block_table` 就能拿到 request-to-block 映射
- 改动集中在 dsa_v1.py（跟现有 PATCH2/PATCH3 一个文件）

**方案 B 的 hook 更简单**:
- Dual write 不需要边界检测（scatter 时就写两处）
- 完成 prefill 只触发**释放 Device 存储**，可以延迟到下一步 decode 懒释放

**验收标准**: `npu-smi info` 显示 ON 模式比 OFF 模式 HBM 占用明显更低（预期减少 ~1.5 GB/请求 × 并发数）。

#### 6.8 Phase B3 v2 — Host-backed Compress Group（2026-09-14 定案，**已被 6.9 修正**）

> ⚠️ **本节结论被 6.9 推翻**：`kv_cache_groups` 的 group 划分和 `kv_cache_tensors.shared_by` 是**两个正交的概念**。
> group 独立 ≠ raw_tensor 独立。真实情况是**一个 raw_tensor 被跨 group 的多个 spec 共享**——这也是 B3 v2 首个版本 HBM 完全不降的根因。
> 保留本节作为历史记录。

**背景**：B3.1 尝试"物理替换 Device tensor"失败——vllm 的 kv_cache 视图共享 raw_tensor storage，`resize_(0)` 会连带干碎 state_cache。同时"分配新 Device tensor" HBM 双倍无法承受。

**KV Cache Group 侦查结果**（rank 0）:

| Group | Layers | 后缀 | 说明 |
|-------|--------|------|------|
| 0 | 42 | `self_attn.indexer.k_cache` | Indexer |
| 1 | 20 | `self_attn.attn` | compress_kv_cache（B3 目标）|
| 2 | 22 | `self_attn.swa_cache` | SWA |
| 3 | 22 | `self_attn.swa_cache` | SWA-另一波 |
| 4 | 42 | `self_attn.compressor.state_cache` | 压缩器状态 |
| 5 | 20 | `self_attn.compressor.state_cache` | 压缩器状态-另一波 |

**当时结论（后被修正）**：compress (group 1) 独立成 group → 以为改这一 group 就完事。

**方案：Host-backed 替换**（首个版本）
- 在 `_allocate_kv_cache_tensors` 返回前，识别属于 compress group 的 raw_tensors（layer 名 `.self_attn.attn`）
- 分配同大小的 Host hugepage + `aclrtHostRegisterV2` NPU MMU 映射
- 用 Host-backed NPU tensor 替换 vllm 原 Device 分配
- `torch.npu.empty_cache()` 强制释放原 Device tensor

**开关**：`TIDALCACHE_HOST_COMPRESS=1`  
**实现位置**：`apply_patches.py` MR_PATCH3

**首个版本运行结果（2026-09-20）**：
- ✅ 服务能起来，推理正常
- ✅ `replaced=41 layers, total=23.48 GB → Host` — dict 替换生效
- ❌ **`memory_allocated: 51.64→51.64→51.64 GB`（drop=0）**
- ❌ `npu-smi HBM-Usage: 61 GB`（跟 baseline 完全一样，没省任何 HBM）

⇒ 问题定位见 6.9。

#### 6.9 shared_by 共享 raw_tensor 的真相（2026-09-20 定位）

**用 `gc.get_referrers` 追引用链定位到根因**：

- Python 层加了完整链条的 HBM diag：`torch.npu.memory_allocated / reserved`（自 self.device 而非 current_device）、`gc.collect()` 触发释放、`sys.getrefcount` 数引用、`torch.zeros` self-probe 验证 API 有效
- api-probe 结果：`allocated 51.64 → 51.89 → 51.64 GB (delta up=256 MB down=256 MB)` ✅ API 精确追踪
- probe old tensor 只剩 **1 个 referrer**，是 `kv_cache_raw_tensors` 自己
- 展开该 dict 找"还指向老 tensor 的 keys" — 得到 **stubborn refs**：

  ```
  ['model.layers.0.self_attn.swa_cache',
   'model.layers.1.self_attn.swa_cache',
   'model.layers.2.self_attn.compressor.state_cache',
   'model.layers.3.self_attn.compressor.state_cache']
  ```

**真实结构**（推翻 6.8 的独立 group 假设）：

```
一个 kv_cache_tensor（一次 torch.zeros ~1.1 GB）
  └── shared_by 列表包含跨 group 的多个 spec：
      ├── layer_0.swa_cache        ← 在 group 2/3
      ├── layer_1.swa_cache        ← 在 group 2/3
      ├── layer_2.self_attn.attn   ← 在 group 1 (compress)
      ├── layer_3.self_attn.attn   ← 在 group 1 (compress)
      ├── layer_2.compressor.state_cache  ← 在 group 4/5
      └── layer_3.compressor.state_cache  ← 在 group 4/5
```

每个 spec 从这个 raw_tensor 的**不同 offset 切 view**，共同拼成完整分配。

**推翻的结论**：
- ❌ "group 独立 ⇒ raw_tensor 独立"
- ✅ 实际是：`kv_cache_groups` 是逻辑分组（管理 block 分配），`kv_cache_tensors[i].shared_by` 是物理分配的共享列表，两者**跨切**

**为什么首版 MR_PATCH3 失败**：
- 只替换了 `.self_attn.attn` 后缀（2 个 dict entry）
- swa/state 的 4 个 dict entry 依然指向老 Device tensor
- 老 tensor refcount > 0 → Python GC 不释放 → `torch.npu.empty_cache()` 无块可回收 → HBM 不降
- 而且额外分了 23 GB Host —— **净效果：多用 23 GB Host，HBM 一点没省**

#### 6.10 三条修复路径（2026-09-20 讨论）

##### 路径 A：把 swa + state 也 offload 到 Host（已试，**HBM 省成但正确性崩**）

**做法演化**：
- v1 [93206a3]：MR_PATCH3 匹配后缀扩宽 `['.self_attn.attn', '.swa_cache', '.compressor.state_cache']`。→ 崩，Host OOM（多分配了纯 SWA / 纯 state raw_tensor，突破 Host 内存上限）
- v2 [bbfe14f]：改成**按 tensor identity 两遍扫描**。Pass 1 只从 `.self_attn.attn` 收集 compress raw_tensor 的 `id()`；Pass 2 走全 dict，只替换 id 命中集合的条目。这样只碰"和 compress 共享 raw_tensor 的" swa/state，不动纯 swa / 纯 state 的独立分配。

**v2 实测结果（2026-09-20）**：

| 指标 | 结果 |
|------|------|
| `replaced=124 layers, total=23.48 GB` | ✅ 124 个 dict entry 全被正确替换到同一批 21 个 Host tensor |
| `HBM diag: 51.64→28.16→28.16 GB` | ✅ **Device HBM 真降 23.48 GB**——首次达成"物理 HBM 节省"目标 |
| `npu-smi: HBM-Usage 61 → 37 GB / chip` | ✅ 每张卡节省 24 GB |
| 短请求（13 tokens）| ✅ 正常，0.89 s |
| 中请求（128 tokens）| ✅ 正常，3.7 s（比 baseline 甚至更快，疑似 cold graph 加暖导致的对比失真）|
| **长 prompt（1604 tokens）→ 32 decode** | ❌ **首 token 起就乱码**：`#EA software aspectsus AIDS (nullius...` |
| **长输出（1024 tokens）** | ❌ 前 200 tokens 正常，之后 `MB-MB-MB-MB...` → 随机 garbage → 无限重复 `unnecessarily unnecessarily...` |

**根因**：**path A 把 swa_cache / state_cache 也搬到 Host 破坏了它们的 kernel 正确性**。

- swa_cache 的 sliding-window attention kernel 假设 Device HBM 语义（顺序、一致性）
- state_cache 是 indexer 的 stateful compressor 缓存，每步 read-then-write，对内存 coherency 敏感
- Host-mapped NPU 内存不提供 NPU L1/L2 cache 参与的 coherency 保证
- 结果：写后读拿到 stale 数据 → attention 计算错 → hidden states 逐步偏离 → decode 到一定长度后完全崩坏

**为什么短请求看着行**：污染在 KV cache 里累积，token 少的时候数据量小、topk 命中随机侥幸没炸；一旦 decode 步数上去，错误数据主导 attention → 崩

**结论**：**路径 A 不可用**。HBM 省了但推理坏了，等价于没用。撤退。

##### 路径 B：改 vllm 让 compress 独立成 kv_cache_tensor（**next step**）

**为什么必须走这条**：path A v2 证明了 Host offload 机制本身没问题（HBM 真降、短请求正常），但**只要 swa/state 被拽到 Host 就崩**。要既省 HBM 又保正确性，唯一办法是**让 compress 有自己独立的 raw_tensor**，替换它不影响 swa/state。

**改动位置**（待挖）：vllm 生成 `kv_cache_config` 的地方，负责决定 `kv_cache_tensors[i].shared_by` 列表的函数。搜索关键点：

- `vllm/v1/core/sched/kv_cache_manager.py` 或 `vllm/v1/kv_cache_interface.py`：`get_kv_cache_config` 逻辑
- `vllm/v1/core/kv_cache_utils.py`：可能有 `_generate_kv_cache_config` / `_group_kv_cache_tensors` 之类
- vllm-ascend 的 `worker/model_runner_v1.py`：可能 override 或调用 vllm 主库的 config 生成

**改动策略**（初步）：
1. 找到 spec → shared_by 分组的逻辑
2. 加一个 hook / patch：对 `.self_attn.attn` 后缀的 spec，强制它独占一个 `kv_cache_tensors[i]`，不与 `.swa_cache` / `.compressor.state_cache` 打包
3. 副作用：分配次数变多、总内存可能略增（对齐 padding 增加），但从 15+% 到几个 %，可接受

**优点**：只 offload 真正稀疏访问的 compress，swa/state 保持 Device 高速访问且正确性不变。真正兑现 DSA 稀疏红利。

**缺点**：改动深入 vllm 内部内存规划器，需要理解 KVCacheManager 的 group→tensor 映射逻辑。风险中等。

##### 路径 C：raw_tensor 底层 storage 替换（zero-copy 极限）

保持 raw_tensor 对象不变（所有 dict entry 继续引用），**换掉它底层 storage 的物理页面**。所有基于该 tensor 的 view 自动跟随到 Host。

**优点**：不需要理解 shared_by 结构，天然覆盖所有 spec。

**缺点**：需要 hack PyTorch storage 层，Ascend NPU 的 storage 语义不完全清楚；即便做成，swa/state 依然是 Host-backed（性能问题同路径 A）。

**结论（2026-09-20 更新）**：路径 A 已试并证伪——机制通、HBM 省，但正确性崩。**转路径 B**：改 vllm 让 compress 独占 raw_tensor。这是唯一能既省 HBM 又保对齐推理的道路。

### Step 7: 性能优化

**目标**: 将单请求开销从 ~+30% 降至 +5-8%

**优化路线**（参考 HiSparse 生产实现）：
1. 开机 hugepage 预留 — 消除 pinned memory 回退
2. 异步 copy stream — DMA 与计算重叠
3. Device 热缓存 + LRU — 跳过高频 block 的 DMA
4. 跨层索引共享 + 批量预取 — Leader/follower 模式
5. 仅 miss gather — 只搬运不在 selection cache 中的 block
6. Gather plan 复用 — follower 层复用 leader 的索引计划
7. 自适应卸载 — 短序列不卸载，HBM 压力高时启用

### Step 8: 多 batch 并发吞吐基准测试

**目标**: 验证 HBM 节省带来的吞吐提升

### Step 9: Mooncake Store RDMA 直写集成（P/D 分离）

---

## 6. 关键数据结构与接口

### 6.1 GatherSelectionKvCache 算子接口

```
输入 (10 个张量 + 1 个属性):
  selection_k_rope:          [s_blk, s_blk_size, 64]     BF16  Device (inplace)
  selection_kv_cache:        [s_blk, s_blk_size, kv_dim] BF16  Device (inplace)
  selection_kv_block_table:  [B*S*H, s_max_blk]          INT32 Device
  selection_kv_block_status: [B, S, H, topk+1]           INT32 Device (inplace)
  selection_topk_indices:    [B, S, H, topk]              INT32 Device
  full_k_rope:               [f_blk, f_blk_size, 64]     BF16  Host (NPU MMU 映射)
  full_kv_cache:             [f_blk, f_blk_size, kv_dim]  BF16  Host (NPU MMU 映射)
  full_kv_block_table:       [B, f_max_blk]               INT32 Device
  full_kv_actual_seq:        [B]                           INT32 Device
  full_q_actual_seq:         [B]                           INT32 Device
  selection_topk_block_size: int64 = 64

输出 (1 个):
  selection_kv_actual_seq:   [B*S*H]                      INT32

内核分支:
  topk ≤ 32  → tilingKey=1, 标量路径
  topk > 32  → tilingKey=2, 向量化路径 (Sort 指令)
```

### 6.2 block_status 语义

```
block_status[b, s, h, i] = 第 i 个 slot 当前缓存的 group index
block_status[b, s, h, topk] = actual_seq_len
初始值 -1 = 空 slot

缓存复用判断 (算子内部执行):
  slot 值 == 新 topk 中的值  → 命中 (zero copy, 不搬运)
  slot 值不在新 topk 中      → miss (从 Full Cache DMA 拷贝)
  新请求加入时              → 整行重置为 -1
```

### 6.3 Host 内存注册 API

```cpp
// tensor_register.cpp (ctypes 调用)
extern "C" int register_tensor(
    void* cpu_ptr,      // hugepage mmap 地址
    size_t size,        // 内存大小
    void** dev_ptr,     // 输出: NPU GM 地址
    int device_id       // NPU 设备号
);

// zero_copy_npu.cpp (pybind11)
std::tuple<Tensor, Tensor> register_hugepage_as_npu_tensor(
    Tensor host_tensor,  // CPU tensor (hugepage)
    int device_id        // NPU 设备号
);
// 返回: (host_tensor, npu_tensor) — 同一物理内存的两个视图
```

---

## 7. 已完成的验证

| 验证项 | 状态 | 说明 |
|--------|------|------|
| 算子编译与链接 | ✅ | aclnn 两阶段 API, 4 个库 |
| Basic Gather 正确性 | ✅ | Full → Selection 数据一致 |
| Cache Reuse 正确性 | ✅ | block_status 复用机制正常 |
| block_status 语义理解 | ✅ | slot 保留原位，非 topk 顺序 |
| Host 内存方案调研 | ✅ | hugepage + aclrtHostRegisterV2 |
| OmniCache 源码分析 | ✅ | 零拷贝注册、内存池、gather 调用 |
| Mooncake 兼容性分析 | ✅ | 不占 KV Connector，无冲突 |
| Host 内存注册 + H2D gather | ✅ | 4/4 测试通过（NPU 机器） |
| topk>32 分片 gather | ✅ | 2×32 split 绕过 561002 向量路径限制 |
| vllm-ascend 集成 | ✅ | apply_patches.py 动态注入，CP + 非 CP |
| Warmup 兼容 | ✅ | 修复 3 个 graph capture 阶段 Bug |
| 端到端推理 | ✅ | 模型输出正确，curl 验证通过 |
| 独立日志 | ✅ | tidalcache.log，SCATTER/GATHER/COPYBACK |
| 性能基线 | ✅ | ON=18.3s, OFF=13.3s, 512 tokens (+37%) |
| HiSparse 优化分析 | ✅ | 7 项优化方向已识别（async stream, LRU, prefetch 等） |

---

## 8. 参考代码

### OmniCache 关键实现 (已分析)

| 文件 | 作用 | 我们的参考价值 |
|------|------|---------------|
| `tensor_register_lib/tensor_register.cpp` | ACL Host 注册 C 接口 | **直接复用方案** |
| `tensor_register_lib/zero_copy_npu.cpp` | PyTorch NPU tensor 包装 | **直接复用方案** |
| `cache/memory/hugepage_ops.py` | hugepage 分配 + MMU 注册 | 参考流程 |
| `cache/memory/memory_pool.py` | KV cache 内存池管理 | 参考 reshape/视图拆分 |
| `cache/memory/dsa_host_pool.py` | DSA 二级 Host 池 | 参考 head_size=576 设计 |
| `gather_selection/core/buffers.py` | Selection Cache 分配 | **直接参考** |
| `gather_selection/core/gather_selection.py` | 算子调用封装 | **直接参考** |
| `attention/backends/dsa_ext.py` | DSA backend 扩展 | 参考集成点 |
| `connector/connector.py` | KV Connector (OX) | **需要避开的设计** |
| `plugin.py` | 插件注册 | 我们不用插件模式 |

### vllm-ascend 现有相关代码

| 文件 | 内容 | 与 Offload 的关系 |
|------|------|-------------------|
| `vllm_ascend/attention/dsa_v1.py` | DSA attention 完整实现 | **主要修改点** |
| `vllm_ascend/kv_offload/cpu_npu.py` | 通用 block 级 NPU↔CPU 传输 | 参考异步传输 |
| `csrc/torch_binding.cpp` | C++ op 注册 | 参考编译模式 |

### 测试文件 (NPU 机器)

```
/workspace/test_gather/
├── gather_wrapper.cpp    — C++ extension (aclnn 两阶段 API)
├── setup.py              — 编译脚本
└── test_gather_op.py     — 测试用例 (Basic + Reuse)
```

---

## 9. 风险与缓解

| 风险项 | 影响 | 状态 | 缓解方案 |
|--------|------|------|----------|
| Host 内存 NPU 寻址 | 核心阻塞 | ✅ 已解决 | hugepage + aclrtHostRegisterV2 |
| hugetlbfs 系统配置 | 部署前置 | ✅ 已验证 | /dev/hugepages 可用，需开机参数 `hugepages=240000` 保证完整分配 |
| Selection Cache 独立管理 | 内存碎片 | ✅ 已验证 | 预分配 + 连续递增，运行正常 |
| TP 多卡 block_status 同步 | 扩展 | ✅ 已验证 | 16 worker TP 运行正常，各卡独立 gather |
| topk>32 CANN 向量路径 | 核心阻塞 | ✅ 已解决 | 拆分为 2×32 chunk，绕过 561002 限制 |
| Warmup dummy 数据不一致 | 集成阻塞 | ✅ 已解决 | 3 个 Bug 修复（batch clamp, q_seq=1, 等） |
| Hugepage 碎片化 | 性能 | ⚠️ 部分 | 63% 覆盖率，未获得 hugepage 的层回退到 pinned memory |
| 单请求延迟开销 | 性能 | ⚠️ +37% | 优化路线已定（async stream, LRU, prefetch） |
| D2H offload 首 token 延迟 | 性能 | 待优化 | 异步 DMA + 独立 stream（优化路线 #2） |
| 算子 topk 上限 2048 | 设计限制 | 无风险 | DSA 规范就是 2048 |
| Mooncake + 本方案搭配时序 | 集成 | 待设计 | Mooncake RDMA 完成 → D2H offload |
