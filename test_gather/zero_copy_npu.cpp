#include <torch/extension.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <stdexcept>
#include <sys/mman.h>
#include <iostream>

std::tuple<torch::Tensor, torch::Tensor> register_hugepage_as_npu_tensor(
    torch::Tensor host_tensor, int device_id)
{
    TORCH_CHECK(host_tensor.device().is_cpu(), "host_tensor must be on CPU");

    void* host_ptr = reinterpret_cast<void*>(host_tensor.data_ptr());
    size_t size = static_cast<size_t>(host_tensor.nbytes());

    if (mlock(host_ptr, size) != 0) {
        std::cerr << "[ZeroCopy] mlock warning: " << strerror(errno) << std::endl;
    }

    aclrtHostUnregister(host_ptr);
    aclrtSetDevice(device_id);

    void* dev_ptr = nullptr;
    aclError ret = aclrtHostRegisterV2(
        host_ptr, size, ACL_HOST_REG_PINNED | ACL_HOST_REG_MAPPED);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(
            "aclrtHostRegisterV2 failed: " + std::to_string(ret));
    }

    aclError ret_ptr = aclrtHostGetDevicePointer(host_ptr, &dev_ptr, 0);
    if (ret_ptr != ACL_SUCCESS) {
        aclrtHostUnregister(host_ptr);
        throw std::runtime_error(
            "aclrtHostGetDevicePointer failed: " + std::to_string(ret_ptr));
    }

    std::cout << "[ZeroCopy] registered: host=" << host_ptr
              << " dev=" << dev_ptr << " size=" << size << std::endl;

    c10::DeviceType device_type = c10::DeviceType::PrivateUse1;

    auto options = torch::TensorOptions()
        .dtype(host_tensor.dtype())
        .device(torch::Device(device_type, device_id));
    auto npu_tensor = torch::empty(host_tensor.sizes(), options);

    size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(
        host_tensor.sizes(), host_tensor.dtype().itemsize());

    c10::DataPtr data_ptr(
        dev_ptr, dev_ptr,
        [](void*) {},
        c10::Device(c10::DeviceType::PrivateUse1, device_id));

    auto fptr = c10::GetStorageImplCreate(device_type);
    auto allocator = c10::GetAllocator(device_type);
    at::Storage storage = fptr(
        c10::StorageImpl::use_byte_size_t(), 0,
        allocator->allocate(0), allocator, true);
    storage.unsafeGetStorageImpl()->set_nbytes(tensor_nbytes);
    storage.set_data_ptr(std::move(data_ptr));

    npu_tensor.set_(storage, 0, host_tensor.sizes());

    return std::make_tuple(host_tensor, npu_tensor);
}

void unregister_host(torch::Tensor host_tensor) {
    void* ptr = host_tensor.data_ptr();
    aclrtHostUnregister(ptr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("register_hugepage_as_npu_tensor",
          &register_hugepage_as_npu_tensor,
          "Register hugepage host tensor as NPU-addressable tensor");
    m.def("unregister_host", &unregister_host,
          "Unregister host tensor from NPU MMU");
}
