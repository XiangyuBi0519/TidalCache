#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "aclnn/acl_meta.h"
#include "aclnn_gather_selection_kv_cache.h"

namespace {

aclDataType torchDtypeToAcl(at::ScalarType dtype) {
    switch (dtype) {
        case at::kHalf:    return ACL_FLOAT16;
        case at::kBFloat16: return ACL_BF16;
        case at::kFloat:   return ACL_FLOAT;
        case at::kInt:     return ACL_INT32;
        case at::kLong:    return ACL_INT64;
        case at::kChar:    return ACL_INT8;
        case at::kByte:    return ACL_UINT8;
        default:
            TORCH_CHECK(false, "Unsupported dtype: ", dtype);
    }
}

aclTensor* createAclTensor(const at::Tensor& tensor) {
    auto contiguous = tensor.contiguous();
    auto sizes = contiguous.sizes();
    auto strides = contiguous.strides();

    std::vector<int64_t> dims(sizes.begin(), sizes.end());
    std::vector<int64_t> str(strides.begin(), strides.end());
    std::vector<int64_t> storageDims = dims;

    return aclCreateTensor(
        dims.data(), dims.size(),
        torchDtypeToAcl(contiguous.scalar_type()),
        str.data(),
        contiguous.storage_offset(),
        ACL_FORMAT_ND,
        storageDims.data(), storageDims.size(),
        contiguous.data_ptr());
}

} // namespace

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

    auto* t0  = createAclTensor(selection_k_rope);
    auto* t1  = createAclTensor(selection_kv_cache);
    auto* t2  = createAclTensor(selection_kv_block_table);
    auto* t3  = createAclTensor(selection_kv_block_status);
    auto* t4  = createAclTensor(selection_topk_indices);
    auto* t5  = createAclTensor(full_k_rope);
    auto* t6  = createAclTensor(full_kv_cache);
    auto* t7  = createAclTensor(full_kv_block_table);
    auto* t8  = createAclTensor(full_kv_actual_seq);
    auto* t9  = createAclTensor(full_q_actual_seq);
    auto* t10 = createAclTensor(selection_kv_actual_seq);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnGatherSelectionKvCacheGetWorkspaceSize(
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9,
        selection_topk_block_size,
        t10,
        &workspaceSize, &executor);
    TORCH_CHECK(ret == 0,
        "aclnnGatherSelectionKvCacheGetWorkspaceSize failed, ret=", ret);

    void* workspace = nullptr;
    at::Tensor ws_tensor;
    if (workspaceSize > 0) {
        ws_tensor = at::empty({static_cast<int64_t>(workspaceSize)},
            at::TensorOptions().dtype(at::kByte).device(selection_k_rope.device()));
        workspace = ws_tensor.data_ptr();
    }

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    ret = aclnnGatherSelectionKvCache(workspace, workspaceSize, executor, stream);
    TORCH_CHECK(ret == 0,
        "aclnnGatherSelectionKvCache execute failed, ret=", ret);

    aclDestroyTensor(t0);
    aclDestroyTensor(t1);
    aclDestroyTensor(t2);
    aclDestroyTensor(t3);
    aclDestroyTensor(t4);
    aclDestroyTensor(t5);
    aclDestroyTensor(t6);
    aclDestroyTensor(t7);
    aclDestroyTensor(t8);
    aclDestroyTensor(t9);
    aclDestroyTensor(t10);

    return selection_kv_actual_seq;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_gather_selection_kv_cache",
          &npu_gather_selection_kv_cache,
          "GatherSelectionKvCache operator wrapper");
}
