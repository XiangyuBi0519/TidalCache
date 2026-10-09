# TidalCache 方案汇总（2026-10-09 跑通版）

## 一句话定位

把 DeepSeek-V4-Flash 的 **DSA 压缩 KV cache** 从 910 NPU 的 HBM 搬到 **Host DRAM**，
用 **NPU MMU 映射**让 NPU 地址能访问 Host 内存；decode 时按 Lightning Indexer 选出的
topk 块，用 **DMA gather** 把需要的少量 KV 拉回 Device 的小缓冲，attention 只读这个小缓冲。
**省 HBM，输出正确。**

---

## 一、整体数据通路

```
         ┌─────────────────────── Host DRAM (hugepage) ───────────────────────┐
         │  compress_kv_cache 全量压缩 KV (21 个 CSA 层, ~13GB)                  │
         │     ▲ 写:模型原生 scatter 直接写 Host(因为这块本就是 Host 视图)      │
         └─────┼──────────────────────────────────────────────────────────────┘
               │ NPU MMU 映射 (aclrtHostRegisterV2 + aclrtHostGetDevicePointer)
               │         Host 地址 ←→ Device 可寻址地址 (UVA)
   ┌───────────┼────────────────────── 910 NPU HBM ──────────────────────────┐
   │           │  decode 每步:                                                 │
   │   Lightning Indexer ──topk(512)──▶ gather (DMA 引擎, 分 32 块)             │
   │                                      │ 只拉 topk 命中的块                  │
   │                                      ▼                                     │
   │                               sel_kv (Device 小缓冲)                       │
   │                                      │                                     │
   │                          attn_op 读 sel_kv ──▶ attention 输出              │
   └────────────────────────────────────────────────────────────────────────┘
```

**核心洞察**：vector core 不能可靠读 Host 映射内存（507035 崩），但 **DMA 引擎能**。
所以不是让 attention 直接读 Host，而是先 DMA gather 把 topk 的那几块拉到 Device，
attention 只读 Device 小缓冲。

---

## 二、关键机制分层（L1–L5）

| 层 | 机制 | 实现位置 |
|---|---|---|
| **L1 Host 分配** | hugepage mmap（失败回落 pinned），`mlock` 锁页 | `offload_manager._alloc_hugepage_tensor` |
| **L2 MMU 映射** | `aclrtHostRegisterV2(PINNED\|MAPPED)` + `aclrtHostGetDevicePointer`，包装成 PrivateUse1 torch tensor | `zero_copy_npu.cpp` + `_register_npu` |
| **L3 compress 落 Host** | vllm 分配 kv_cache 后，把 CSA 层的 compress raw_tensor **替换**成 Host-backed 视图（identity 两遍匹配，只动 ratio=4） | `MR_PATCH3` |
| **L4 写** | 模型原生 scatter 直接写 compress_kv_cache(=Host)。**mode-B dual-write 已禁用** | `PATCH3 / CP_PATCH3` |
| **L5 读(gather+rebind)** | decode 时 Indexer 算 topk → `mgr.gather()` DMA 拉到 sel_kv → 把 attn_op 的 cmp_block_table 重指向 mini 表，让它读 sel_kv | `PATCH2/CP_PATCH2` + `PATCH4/CP_PATCH4` |

---

## 三、补丁清单（源码级，`apply_patches.py`）

针对 vllm-ascend 两个文件：`mla_v1.py`（MR_*）和 `dsa_cp.py`（CP_*，带 context-parallel）。

- **PATCH1_init** — `__init__` 里设 `kv_offload_enabled` flag（锚 `self.index_topk = self.indexer.index_topk`，只有带 Lightning Indexer 的 CSA 层有）
- **MR_PATCH2_INIT** — 在 vllm 分配前就创建 manager（保证 MR_PATCH3 能用）
- **MR_PATCH3** — 把 CSA compress raw_tensor 换成 Host 视图（省 HBM 的核心）
- **PATCH2 / CP_PATCH2** — decode 入口插入 gather，拉 sel_kv
- **PATCH3 / CP_PATCH3** — scatter 写（加 `HOST_COMPRESS` 守卫禁 dual-write）
- **PATCH4 / CP_PATCH4** — attn_op 的 cmp_block_table 改读 `_tc_cmp_block_table`（B2 时指向 sel_kv 的 mini 表）
- **KV_UTIL_PATCH** — 源码级隔离 compress group，不让它和 swa/state 共享 raw_tensor（Path B）

---

## 四、运行时开关

| 环境变量 | 作用 | 生产值 |
|---|---|---|
| `VLLM_DSA_KV_OFFLOAD` | 总开关 | 1 |
| `TIDALCACHE_HOST_COMPRESS` | compress 落 Host | 1 |
| `TIDALCACHE_ATTN_ON_SEL` | attention 读 sel_kv（B2） | 1 |
| `TIDALCACHE_CSA_ONLY` | 只 offload ratio=4 的 CSA 层 | 1（默认） |
| `TIDALCACHE_PREFILL_MODE` | A=单写 / B=dual-write（已被 HOST_COMPRESS 覆盖禁用）/ OFF | A（默认） |
| `TIDALCACHE_POISON` | 诊断：sel_kv 填 -1000 | 仅测试 |

---

## 五、已实现 ✅ vs 未做 ⬜

**已实现（可复现、有量化）**：
- ✅ Host hugepage + NPU MMU 映射全链路（官方 ACL API，非 hack）
- ✅ CSA compress KV 落 Host：**HBM 每片省 ~12-13GB**（61→48）
- ✅ Path B 源码级隔离 compress（不污染 swa/state）
- ✅ 层结构查清（43 层 = 2 SWA + 交替 21 CSA / 20 HCA），只 offload CSA
- ✅ gather（DMA，topk>32 分块）+ attention 读 sel_kv
- ✅ **长上下文正确性**：1024 token 输出连贯（POISON 证实走本链路）
- ✅ hugepage 泄漏清理（atexit + signal）

**未做**：
- ⬜ 性能基准（TTFT/TPOT vs baseline）← 下一步
- ⬜ prefill 阶段 offload（目前重点在 decode）
- ⬜ HCA（ratio=128 稠密层）的 offload（稠密读 Host 不划算，暂留 Device）
- ⬜ Mooncake RDMA 直写 / P-D 分离集成
- ⬜ post-prefill D2H sweep（mode-A 的 TBD 项）

---

## 六、硬件/语义边界（踩过的坑，别再犯）

1. **vector core 读 Host = 507035 崩** → 必须 gather 到 Device，不能让 attention 直读 Host
2. **compress_ratio 不同层语义不同** → CSA(4) 稀疏可 offload，HCA(128) 稠密不划算
3. **kv_cache raw_tensor 跨 group 共享** → 必须源码隔离，否则动 compress 连带动 swa/state → 崩
4. **dual-write 是 compress-on-Device 的历史包袱** → HOST_COMPRESS 下冗余且越界，已修

---

## 七、根因备忘：长上下文崩坏（2026-10-09 锁定）

**真凶**：mode-B 的 dual-write scatter（往 `layers[name].npu_kv_cache` 再写一份）。
`HOST_COMPRESS=1` 下 compress_kv_cache 本身就是 Host 视图，模型原生 scatter 已经写了 Host，
dual-write 既冗余又**越界写入，踩坏相邻的 Lightning Indexer 输出缓冲** → topk_idxs 从 decode
第 2 步起变垃圾 → 选错块 → 累积失真 → NaN。

**证据**：`TIDALCACHE_LOG_TOPK_IDXS` + hidden_states 探针显示「输入正常、Indexer 输出被踩坏」；
关掉 dual-write 后 topk 全程有效、1024 token 连贯；POISON 复测 1+1 立刻乱码证明 attention 真读 sel_kv。

详见 `kv_offload_integration_plan.md` §6.15。
