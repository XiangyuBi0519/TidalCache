import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

CANN_HOME = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-8.5.1")
CUSTOM_OPP = os.path.join(CANN_HOME, "opp", "vendors", "customize")

try:
    import torch_npu
    torch_npu_dir = os.path.dirname(torch_npu.__file__)
except ImportError:
    raise RuntimeError("torch_npu is not installed")

include_dirs = [
    os.path.join(torch_npu_dir, "include"),
    os.path.join(torch_npu_dir, "include", "third_party", "acl", "inc"),
    os.path.join(CANN_HOME, "include"),
    os.path.join(CUSTOM_OPP, "op_api", "include"),
]

library_dirs = [
    os.path.join(torch_npu_dir, "lib"),
    os.path.join(CANN_HOME, "lib64"),
    os.path.join(CUSTOM_OPP, "op_api", "lib"),
]

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
            libraries=["torch_npu", "cust_opapi", "nnopbase", "ascendcl"],
            extra_compile_args=["-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
