#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-spanv2_official}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_EXE:-/home/qhm/miniconda3/bin/conda}"

if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[skip] Conda environment already exists: ${ENV_NAME}"
else
    if ! "${CONDA_BIN}" env create -n "${ENV_NAME}" -f "${ROOT_DIR}/environment.yml"; then
        echo "[fallback] Repository access failed; cloning local conda_py38 environment"
        "${CONDA_BIN}" create -n "${ENV_NAME}" --clone conda_py38 -y
    fi
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -e "${ROOT_DIR}" --no-deps
"${CONDA_BIN}" run -n "${ENV_NAME}" python -c \
    "import basicsr; print('BasicSR', basicsr.__version__, '->', basicsr.__file__)"

echo "Environment ready. Activate with: conda activate ${ENV_NAME}"
echo "Build the optional inference kernel with:"
echo "PYTHON=\$(conda run -n ${ENV_NAME} which python) bash span_attention_op/build_span_attn.sh"
