#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${SPANV2_ENV_NAME:-spanv2_official}"

if [[ -n "${SPANV2_PYTHON:-}" ]]; then
    PYTHON="${SPANV2_PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" \
        && "$(basename "${CONDA_PREFIX}")" == "${ENV_NAME}" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
else
    CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
    if [[ -z "${CONDA_BIN}" ]]; then
        echo "Conda was not found. Activate ${ENV_NAME} or set SPANV2_PYTHON." >&2
        exit 1
    fi
    ENV_PREFIX="$(${CONDA_BIN} env list | awk -v name="${ENV_NAME}" '$1 == name {print $NF; exit}')"
    if [[ -z "${ENV_PREFIX}" || ! -x "${ENV_PREFIX}/bin/python" ]]; then
        echo "Conda environment not found: ${ENV_NAME}" >&2
        echo "Run: bash scripts/setup_environment.sh ${ENV_NAME}" >&2
        exit 1
    fi
    PYTHON="${ENV_PREFIX}/bin/python"
fi

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python executable is unavailable: ${PYTHON}" >&2
    exit 1
fi

printf '%s\n' "${PYTHON}"
