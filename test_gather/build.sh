#!/bin/bash
# Build all components for DSA KV Cache Offload testing
# Run on NPU machine inside the vllm container
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CANN_HOME="${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.1}}"
CUSTOM_OPP="$CANN_HOME/opp/vendors/customize"

echo "=== Environment ==="
echo "CANN_HOME: $CANN_HOME"
python3 -c "import torch; print(f'torch: {torch.__version__}')"
python3 -c "import torch_npu; print(f'torch_npu: {torch_npu.__version__}')"
echo ""

# Check prerequisites
echo "=== Checking prerequisites ==="
CUSTOM_OP_HEADER="${CUSTOM_OPP}/op_api/include/aclnn_gather_selection_kv_cache.h"
if [ -f "$CUSTOM_OP_HEADER" ]; then
    echo "Custom op header: FOUND"
else
    echo "ERROR: Custom op header not found at: $CUSTOM_OP_HEADER"
    echo "Install the GatherSelectionKvCache operator first."
    exit 1
fi
echo ""

# Build tensor_register.so (pure C, loaded via ctypes)
echo "=== Building tensor_register.so (ctypes) ==="
g++ -fPIC -shared -std=c++11 \
    -I${CANN_HOME}/include \
    -L${CANN_HOME}/lib64 -lascendcl \
    tensor_register.cpp -o tensor_register.so
echo "OK: tensor_register.so"
echo ""

# Build PyTorch extensions sequentially (parallel builds have race conditions)
echo "=== Building gather_wrapper ==="
python3 -c "
from setup import extensions
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
setup(name='gw', version='0.1', ext_modules=[extensions[0]], cmdclass={'build_ext': BuildExtension})
" build_ext --inplace 2>&1 | tail -3
echo ""

echo "=== Building zero_copy_npu ==="
python3 -c "
from setup import extensions
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
setup(name='zc', version='0.1', ext_modules=[extensions[1]], cmdclass={'build_ext': BuildExtension})
" build_ext --inplace 2>&1 | tail -3
echo ""

echo "=== Build complete ==="
ls -la *.so 2>/dev/null || true
echo ""

# Print LD_LIBRARY_PATH setup command
TORCH_LIB=$(python3 -c 'import torch; print(torch.__path__[0])')/lib
TORCH_NPU_LIB=$(python3 -c 'import torch_npu; print(torch_npu.__path__[0])')/lib

echo "=== Before running tests, set: ==="
echo "export LD_LIBRARY_PATH=${CUSTOM_OPP}/op_api/lib/:${CANN_HOME}/lib64/:${TORCH_LIB}:${TORCH_NPU_LIB}:\${LD_LIBRARY_PATH}"
echo ""
echo "Then run: python3 test_gather_op.py"
