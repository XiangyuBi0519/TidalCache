[root@mep-mirror-280t-ga-az5-turbo-86 test_gather]# python3 test_gather_op.py 
GatherSelectionKvCache Operator Test
torch_npu version: 2.9.0.post1+gitee7ba04
Device: Ascend910B3

============================================================
TEST 1: Basic Gather
============================================================
Test configuration:
  batch_size=1, num_heads=1, topk=4
  sel_topk_block_size=64
  full_seq_len=1024, full_block_size=128
  kv_cache_dim=512, k_rope_dim=64
  num_full_blocks=8, total_sel_blocks=12
  topk_indices=[1, 3, 4, 5]
  full_k_rope: shape=torch.Size([12, 128, 64]), device=npu:0
  full_kv_cache: shape=torch.Size([12, 128, 512]), device=npu:0
  selection_k_rope: shape=torch.Size([12, 64, 64]), device=npu:0
  selection_kv_cache: shape=torch.Size([12, 64, 512]), device=npu:0
  selection_kv_block_table: shape=torch.Size([1, 4])
  selection_kv_block_status: shape=torch.Size([1, 1, 1, 5])
  selection_topk_indices: shape=torch.Size([1, 1, 1, 4])

Calling GatherSelectionKvCache...

selection_kv_actual_seq: tensor([256], dtype=torch.int32)
selection_kv_block_status:
tensor([[[[  1,   3,   4,   5, 256]]]], dtype=torch.int32)
  OK: group 1, sel_blk 0, value=65.0 (expected 65.0)
  OK: group 3, sel_blk 1, value=193.0 (expected 193.0)
  OK: group 4, sel_blk 2, value=257.0 (expected 257.0)
  OK: group 5, sel_blk 3, value=321.0 (expected 321.0)

TEST 1 PASSED

============================================================
TEST 2: Cache Reuse
============================================================
Test configuration:
  batch_size=1, num_heads=1, topk=4
  sel_topk_block_size=64
  full_seq_len=1024, full_block_size=128
  kv_cache_dim=512, k_rope_dim=64
  num_full_blocks=8, total_sel_blocks=12
  topk_indices=[3, 4, 11, 12]
  full_k_rope: shape=torch.Size([12, 128, 64]), device=npu:0
  full_kv_cache: shape=torch.Size([12, 128, 512]), device=npu:0
  selection_k_rope: shape=torch.Size([12, 64, 64]), device=npu:0
  selection_kv_cache: shape=torch.Size([12, 64, 512]), device=npu:0
  selection_kv_block_table: shape=torch.Size([1, 4])
  selection_kv_block_status: shape=torch.Size([1, 1, 1, 5])
  selection_topk_indices: shape=torch.Size([1, 1, 1, 4])

First gather call...
After first call - block_status:
tensor([[[[  3,   4,  11,  12, 256]]]], dtype=torch.int32)

Second call: old_indices=[3, 4, 11, 12], new_indices=[0, 1, 3, 4]
  Overlapping: [3, 4] (should be reused)
  New: [0, 1] (should be fetched from host)

Second gather call...
After second call - block_status:
tensor([[[[  3,   1,   0,   4, 256]]]], dtype=torch.int32)
  FAIL [NEW]: group 0, expected 1.0, got 193.0
  OK [NEW]: group 1, value=65.0
  FAIL [REUSED]: group 3, expected 193.0, got 1.0
  OK [REUSED]: group 4, value=257.0

TEST 2 FAILED

============================================================
SUMMARY
============================================================
  Basic Gather: PASSED
  Cache Reuse: FAILED
