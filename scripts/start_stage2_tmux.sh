#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="spanv2-stage2"
RESUME_ITER=""
WANDB_RESUME_MODE=""
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/qhm/miniconda3/envs/spanv2_official/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume-iter)
            RESUME_ITER="${2:?--resume-iter requires an iteration}"
            shift 2
            ;;
        --wandb-mode)
            WANDB_RESUME_MODE="${2:?--wandb-mode requires resume, rewind, or fork}"
            shift 2
            ;;
        --session)
            SESSION_NAME="${2:?--session requires a name}"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--resume-iter N] [--wandb-mode resume|rewind|fork] [--session NAME]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

case "${WANDB_RESUME_MODE:-resume}" in
    resume|rewind|fork) ;;
    *)
        echo "Invalid --wandb-mode: ${WANDB_RESUME_MODE}" >&2
        exit 2
        ;;
esac

EXISTING_PIDS="$(pgrep -f '[b]asicsr/train.py.*configs/stage2_report.yaml' || true)"
if [[ -n "${EXISTING_PIDS}" ]]; then
    echo "Stage 2 training is already running (PID: ${EXISTING_PIDS//$'\n'/, })." >&2
    echo "Stop the existing process before starting another Stage 2 session." >&2
    exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}"
    echo "Attach with: tmux attach -t ${SESSION_NAME}"
    exit 1
fi

TRAIN_COMMAND=(
    env "CUDA_VISIBLE_DEVICES=${GPU}"
    "${PYTHON}" basicsr/train.py
    -opt configs/stage2_report.yaml
    --auto_resume
)
if [[ -n "${RESUME_ITER}" ]]; then
    TRAIN_COMMAND+=(--resume_iter "${RESUME_ITER}")
fi
if [[ -n "${WANDB_RESUME_MODE}" ]]; then
    TRAIN_COMMAND+=(--wandb_resume_mode "${WANDB_RESUME_MODE}")
fi
printf -v TRAIN_COMMAND_STRING '%q ' "${TRAIN_COMMAND[@]}"
printf -v PROJECT_ROOT_QUOTED '%q' "${PROJECT_ROOT}"

tmux new-session -d -s "${SESSION_NAME}" \
    "cd ${PROJECT_ROOT_QUOTED} && exec ${TRAIN_COMMAND_STRING}"

echo "Stage 2 training started in tmux session: ${SESSION_NAME}"
echo "Checkpoint: ${RESUME_ITER:-latest complete pair, or Stage 1 initialization}"
echo "W&B mode: ${WANDB_RESUME_MODE:-resume (from configs/stage2_report.yaml)}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Detach: Ctrl+B, then D"
