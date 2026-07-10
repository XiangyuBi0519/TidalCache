"""
Build script for GatherSelectionKvCache test wrapper.

Usage on NPU machine:
    export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
    python setup.py build_ext --inplace
"""
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

CANN_HOME = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-8.5.1")
CUSTOM_OPP = os.path.join(CANN_HOME, "opp", "vendors", "customize")

# torch_npu include/lib paths
try:
    import torch_npu
    torch_npu_dir = os.path.dirname(torch_npu.__file__)
    torch_npu_include = os.path.join(torch_npu_dir, "include")
    torch_npu_lib = os.path.join(torch_npu_dir, "lib")
except ImportError:
    raise RuntimeError("torch_npu is not installed")

# CANN include/lib paths
cann_include = os.path.join(CANN_HOME, "include")
cann_lib = os.path.join(CANN_HOME, "lib64")

# Custom op include/lib paths
custom_op_include = os.path.join(CUSTOM_OPP, "op_api", "include")
custom_op_lib = os.path.join(CUSTOM_OPP, "op_api", "lib")

# aclnn_torch_adapter include path (for EXEC_NPU_CMD macro)
# This is typically in torch_npu or in the CANN opp directory
aclnn_adapter_include = os.path.join(CANN_HOME, "opp", "built-in", "op_api", "include")

include_dirs = [
    torch_npu_include,
    cann_include,
    custom_op_include,
    aclnn_adapter_include,
]

library_dirs = [
    torch_npu_lib,
    cann_lib,
    custom_op_lib,
]

# Filter to only include existing directories
include_dirs = [d for d in include_dirs if os.path.isdir(d)]
library_dirs = [d for d in library_dirs if os.path.isdir(d)]

print("Include dirs:", include_dirs)
print("Library dirs:", library_dirs)

setup(
    name="gather_kv_test",
    ext_modules=[
        CppExtension(
            name="gather_kv_wrapper",
            sources=["gather_wrapper.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=["cust_opapi", "ascendcl", "torch_npu"],
            extra_compile_args=["-std=c++17", "-D__EXPORT_API__"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
