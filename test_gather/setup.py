import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

torch_npu_dir = os.path.dirname(os.path.abspath(
    __import__('torch_npu').__file__))

CANN_HOME = os.environ.get(
    "ASCEND_TOOLKIT_HOME",
    os.environ.get("ASCEND_HOME_PATH",
                   "/usr/local/Ascend/cann-8.5.1"))

CUSTOM_OPP = os.path.join(CANN_HOME, "opp/vendors/customize")

common_include = [
    CANN_HOME + "/include",
    torch_npu_dir + "/include",
    torch_npu_dir + "/include/third_party/acl/inc",
]

common_lib_dirs = [
    torch_npu_dir + "/lib",
    CANN_HOME + "/lib64",
]

extensions = [
    CppExtension(
        name="gather_wrapper",
        sources=["gather_wrapper.cpp"],
        include_dirs=common_include + [
            CUSTOM_OPP + "/op_api/include",
        ],
        library_dirs=common_lib_dirs + [
            CUSTOM_OPP + "/op_api/lib",
        ],
        libraries=["torch_npu", "cust_opapi", "nnopbase", "ascendcl"],
        extra_compile_args=["-std=c++17", "-O2"],
    ),
    CppExtension(
        name="zero_copy_npu",
        sources=["zero_copy_npu.cpp"],
        include_dirs=common_include,
        library_dirs=common_lib_dirs,
        libraries=["ascendcl"],
        extra_compile_args=["-std=c++17", "-O2"],
    ),
]

setup(
    name="dsa_offload_test",
    version="0.2",
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension},
)
