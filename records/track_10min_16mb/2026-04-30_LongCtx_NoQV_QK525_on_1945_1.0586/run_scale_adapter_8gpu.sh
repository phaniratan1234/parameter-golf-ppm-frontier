#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CASEOPS_RECORD="records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611"
DEFAULT_DATA_PATH="$REPO_ROOT/$CASEOPS_RECORD/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
DEFAULT_TOKENIZER_PATH="$REPO_ROOT/$CASEOPS_RECORD/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"

DATA_PATH="${DATA_PATH:-$DEFAULT_DATA_PATH}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$DEFAULT_TOKENIZER_PATH}"
SEED="${SEED:-42}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TTT_SCALE_ADAPTER_LIMIT="${TTT_SCALE_ADAPTER_LIMIT:-0.02}"
LOG_FILE="${LOG_FILE:-run_1953_scale_adapter_seed${SEED}_limit${TTT_SCALE_ADAPTER_LIMIT}.log}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "Missing DATA_PATH: $DATA_PATH" >&2
  echo "Build/copy the CaseOps dataset first, or export DATA_PATH=/path/to/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved" >&2
  exit 1
fi

if [[ ! -f "$TOKENIZER_PATH" ]]; then
  echo "Missing TOKENIZER_PATH: $TOKENIZER_PATH" >&2
  echo "Export TOKENIZER_PATH=/path/to/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model" >&2
  exit 1
fi

cd "$SCRIPT_DIR"
rm -f final_model.pt final_model.int6.ptz

echo "Running #1953 frontier + TTT activation scale adapter"
echo "  seed: $SEED"
echo "  gpus: $NPROC_PER_NODE"
echo "  scale limit: $TTT_SCALE_ADAPTER_LIMIT"
echo "  data: $DATA_PATH"
echo "  tokenizer: $TOKENIZER_PATH"
echo "  log: $LOG_FILE"
echo "Expected critical flags in hyperparameter print:"
echo "  ttt_eval_seq_len=2560, ttt_mask=no_qv, ttt_lora_rank=80, ttt_weight_decay=0.5"
echo "  awq_lite_enabled=True, qk_gain_init=5.25, compressor=pergroup"

CASEOPS_ENABLED=1 \
DATA_PATH="$DATA_PATH" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
SEED="$SEED" \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
PHASED_TTT_ENABLED=1 PHASED_TTT_NUM_PHASES=3 PHASED_TTT_PREFIX_DOCS=2500 \
EVAL_SEQ_LEN=2560 TTT_EVAL_SEQ_LEN=2560 \
TTT_MASK=no_qv TTT_Q_LORA=0 TTT_V_LORA=0 TTT_LOCAL_LR_MULT=0.75 \
TTT_LORA_RANK=80 TTT_WEIGHT_DECAY=0.5 TTT_BETA2=0.99 \
TTT_SCALE_ADAPTER_ENABLED=1 TTT_SCALE_ADAPTER_LIMIT="$TTT_SCALE_ADAPTER_LIMIT" \
QK_GAIN_INIT=5.25 \
EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
WARMDOWN_FRAC=0.85 BETA2=0.99 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GPTQ_RESERVE_SECONDS=4.0 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
AWQ_LITE_ENABLED=1 ASYM_LOGIT_RESCALE=1 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py 2>&1 | tee "$LOG_FILE"
