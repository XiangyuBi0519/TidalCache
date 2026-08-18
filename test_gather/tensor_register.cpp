#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <mutex>

struct RegisteredTensor {
    void* cpu_ptr;
    void* dev_ptr;
    size_t size;
    int device_id;
    RegisteredTensor() : cpu_ptr(nullptr), dev_ptr(nullptr), size(0), device_id(-1) {}
};

static std::unordered_map<void*, RegisteredTensor> g_registry;
static std::mutex g_registry_mutex;

extern "C" int register_tensor(
    void* cpu_ptr,
    size_t size,
    void** dev_ptr,
    int device_id)
{
    if (!cpu_ptr || size == 0 || dev_ptr == nullptr) return -1;

    if (aclrtSetDevice(device_id) != ACL_ERROR_NONE) {
        std::cerr << "aclrtSetDevice(" << device_id << ") failed\n";
        return -1;
    }

    if (reinterpret_cast<uintptr_t>(cpu_ptr) % 4096 != 0) {
        std::cerr << "Warning: CPU pointer not 4K aligned: " << cpu_ptr << std::endl;
    }

    void* out_dev_ptr = nullptr;
    aclError ret = aclrtHostRegisterV2(
        cpu_ptr, size, ACL_HOST_REG_PINNED | ACL_HOST_REG_MAPPED);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtHostRegisterV2 failed: " << ret << std::endl;
        return static_cast<int>(ret);
    }

    aclError ret_ptr = aclrtHostGetDevicePointer(cpu_ptr, &out_dev_ptr, 0);
    if (ret_ptr != ACL_SUCCESS) {
        std::cerr << "aclrtHostGetDevicePointer failed: " << ret_ptr << std::endl;
        aclrtHostUnregister(cpu_ptr);
        return static_cast<int>(ret_ptr);
    }

    std::cout << "Registered: cpu_ptr=" << cpu_ptr
              << " dev_ptr=" << out_dev_ptr
              << " size=" << size << std::endl;

    {
        std::lock_guard<std::mutex> lk(g_registry_mutex);
        RegisteredTensor info;
        info.cpu_ptr = cpu_ptr;
        info.dev_ptr = out_dev_ptr;
        info.size = size;
        info.device_id = device_id;
        g_registry[cpu_ptr] = info;
    }

    *dev_ptr = out_dev_ptr;
    return 0;
}

extern "C" int unregister_tensor(void* cpu_ptr) {
    if (cpu_ptr == nullptr) return -1;

    RegisteredTensor info;
    {
        std::lock_guard<std::mutex> lk(g_registry_mutex);
        auto it = g_registry.find(cpu_ptr);
        if (it == g_registry.end()) return -1;
        info = it->second;
        g_registry.erase(it);
    }

    if (info.device_id >= 0) aclrtSetDevice(info.device_id);
    aclError ret = aclrtHostUnregister(info.cpu_ptr);
    return static_cast<int>(ret);
}

extern "C" void* get_dev_ptr_from_cpu(void* cpu_ptr) {
    std::lock_guard<std::mutex> lk(g_registry_mutex);
    auto it = g_registry.find(cpu_ptr);
    if (it != g_registry.end()) return it->second.dev_ptr;
    return nullptr;
}
