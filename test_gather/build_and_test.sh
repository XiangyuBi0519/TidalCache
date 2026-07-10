#!/bin/bash
# Build and test GatherSelectionKvCache operator wrapper
# Run this on the NPU machine

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.1}
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib/:${ASCEND_HOME_PATH}/lib64/:${LD_LIBRARY_PATH}

echo "=== Environment ==="
echo "ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
echo "Python: $(python3 --version)"
echo ""

# Check prerequisites
echo "=== Checking prerequisites ==="
python3 -c "import torch; print(f'torch: {torch.__version__}')"
python3 -c "import torch_npu; print(f'torch_npu: {torch_npu.__version__}')"

CUSTOM_OP_HEADER="${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/include/aclnn_gather_selection_kv_cache.h"
if [ -f "$CUSTOM_OP_HEADER" ]; then
    echo "Custom op header: FOUND"
else
    echo "ERROR: Custom op header not found at: $CUSTOM_OP_HEADER"
    echo "Did you install the operator .run package?"
    exit 1
fi

CUSTOM_OP_LIB="${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib/libcust_opapi.so"
if [ -f "$CUSTOM_OP_LIB" ]; then
    echo "Custom op library: FOUND"
else
    echo "ERROR: Custom op library not found at: $CUSTOM_OP_LIB"
    exit 1
fi

echo ""

# Build
echo "=== Building wrapper ==="
python3 setup.py build_ext --inplace 2>&1 | tail -20

echo ""

# Test
echo "=== Running tests ==="
python3 test_gather_op.py
