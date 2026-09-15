#!/usr/bin/env bash
# Install span_attention CUDA operator into site-packages.
#
# Usage:
#   bash build_span_attn.sh
#   PYTHON=/path/to/python bash build_span_attn.sh
#
# NOTE: --no-build-isolation is required because CUDAExtension needs torch at
#       build time, which is not available in pip's default isolated sandbox.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(which python)}"
export PATH="$(dirname "$(readlink -f "$PYTHON")"):$PATH"
export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"

echo "=============================="
echo " span_attention CUDA operator"
echo "=============================="
echo "Script dir : $SCRIPT_DIR"
echo "Python     : $PYTHON"
echo "PyTorch    : $($PYTHON -c 'import torch; print(torch.__version__)')"
echo "CUDA       : $($PYTHON -c 'import torch; print(torch.version.cuda)')"
echo "C++        : $CXX"
echo "=============================="

"$PYTHON" -m pip install . --no-build-isolation

SO=$("$PYTHON" -c "import torch; import span_attention; print(span_attention.__file__)" 2>/dev/null || true)
if [ -z "$SO" ]; then
    echo "[ERROR] Install succeeded but import failed."
    exit 1
fi

echo ""
echo "[OK] Installed: $SO"
