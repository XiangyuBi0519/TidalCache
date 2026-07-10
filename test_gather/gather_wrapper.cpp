#include <torch/extension.h>

#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>

// Include the custom op's aclnn header (installed via the .run package)
#include "aclnn_gather_selection_kv_cache.h"

// EXEC_NPU_CMD is provided by torch_npu's op_api headers.
// If not available, we include it explicitly.
#ifndef EXEC_NPU_CMD
#include "aclnn_torch_adapter/op_api_common.h"
#endif

at::Tensor npu_gather_selection_kv_cache(
    at::Tensor& selection_k_rope,
    at::Tensor& selection_kv_cache,
    at::Tensor& selection_kv_block_table,
    at::Tensor& selection_kv_block_status,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    int64_t selection_topk_block_size)
{
    at::Tensor selection_kv_actual_seq = at::empty(
        {selection_kv_block_table.size(0)},
        selection_topk_indices.options());

    EXEC_NPU_CMD(aclnnGatherSelectionKvCache,
        selection_k_rope,
        selection_kv_cache,
        selection_kv_block_table,
        selection_kv_block_status,
        selection_topk_indices,
        full_k_rope,
        full_kv_cache,
        full_kv_block_table,
        full_kv_actual_seq,
        full_q_actual_seq,
        selection_topk_block_size,
        selection_kv_actual_seq);

    return selection_kv_actual_seq;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_gather_selection_kv_cache",
          &npu_gather_selection_kv_cache,
          "GatherSelectionKvCache operator wrapper");
}
