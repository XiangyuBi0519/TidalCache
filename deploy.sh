#!/bin/bash
# TidalCache Deploy Script
# Backs up original vllm-ascend files and replaces with patched versions.
#
# Usage:
#   bash deploy.sh /path/to/vllm-ascend       # deploy (backup + replace)
#   bash deploy.sh /path/to/vllm-ascend rollback  # restore from backup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_ASCEND_DIR="${1:?Usage: bash deploy.sh /path/to/vllm-ascend [rollback]}"
ACTION="${2:-deploy}"

DSA_SRC="$VLLM_ASCEND_DIR/vllm_ascend/attention/dsa_v1.py"
MR_SRC="$VLLM_ASCEND_DIR/vllm_ascend/worker/model_runner_v1.py"

DSA_PATCHED="$SCRIPT_DIR/patches/dsa_v1_patched.py"
MR_PATCHED="$SCRIPT_DIR/patches/model_runner_v1_patched.py"
TIDALCACHE_DIR="$SCRIPT_DIR/tidalcache"

if [ "$ACTION" = "rollback" ]; then
    echo "=== TidalCache Rollback ==="
    for f in "$DSA_SRC" "$MR_SRC"; do
        if [ -f "${f}.bak" ]; then
            cp "${f}.bak" "$f"
            echo "  Restored: $f"
        else
            echo "  No backup found: ${f}.bak"
        fi
    done
    echo "Done. Restart vllm to take effect."
    exit 0
fi

echo "=== TidalCache Deploy ==="
echo "  vllm-ascend: $VLLM_ASCEND_DIR"

# Verify targets exist
for f in "$DSA_SRC" "$MR_SRC"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found"
        exit 1
    fi
done

# Verify patches exist
for f in "$DSA_PATCHED" "$MR_PATCHED"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Run from TidalCache repo root."
        exit 1
    fi
done

# Backup
echo ""
echo "--- Backup ---"
for f in "$DSA_SRC" "$MR_SRC"; do
    if [ ! -f "${f}.bak" ]; then
        cp "$f" "${f}.bak"
        echo "  Backed up: ${f}.bak"
    else
        echo "  Backup exists: ${f}.bak (skipped)"
    fi
done

# Replace
echo ""
echo "--- Patch ---"
cp "$DSA_PATCHED" "$DSA_SRC"
echo "  Patched: $DSA_SRC"
cp "$MR_PATCHED" "$MR_SRC"
echo "  Patched: $MR_SRC"

# Symlink tidalcache module
echo ""
echo "--- Module ---"
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if [ -n "$SITE_PACKAGES" ] && [ -d "$SITE_PACKAGES" ]; then
    if [ ! -e "$SITE_PACKAGES/tidalcache" ]; then
        ln -s "$TIDALCACHE_DIR" "$SITE_PACKAGES/tidalcache"
        echo "  Linked: $TIDALCACHE_DIR -> $SITE_PACKAGES/tidalcache"
    else
        echo "  Already linked: $SITE_PACKAGES/tidalcache"
    fi
else
    echo "  site-packages not found. Add to PYTHONPATH manually:"
    echo "    export PYTHONPATH=$SCRIPT_DIR:\$PYTHONPATH"
fi

echo ""
echo "=== Deploy Complete ==="
echo ""
echo "To enable TidalCache:"
echo "  export VLLM_DSA_KV_OFFLOAD=1"
echo ""
echo "To rollback:"
echo "  bash $0 $VLLM_ASCEND_DIR rollback"
