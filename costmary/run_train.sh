#!/bin/bash
set -e

COSTMARY_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$COSTMARY_DIR"
MAIN_PY="$COSTMARY_DIR/main.py"

PYTHON=""
REF_FILE=""
FEATURE_PATH=""
SAVE_DIR=""
COHORT="TCGA"
SEED=29
VIS_DEPTH=6
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  bash run_train.sh --ref_file <path> --feature_path <path> --save_dir <path> [options]

Options:
  --ref_file <path>      Path to ref_file.csv. Default: empty.
  --feature_path <path>  Path to feature root. Default: empty.
  --save_dir <path>      Path to output root. Default: empty.
  --cohort <name>        Cohort name. Default: TCGA.
  --seed <int>           Random seed. Default: 29.
  --vis_depth <int>      ViS depth for the main ViS run. Default: 6.
  --python <path>        Python executable. Default: auto-detect.
  --dry_run              Print commands without running them.
  -h, --help             Show this help message.
EOF
}

resolve_costmary_path() {
    local value="$1"
    if [ -z "$value" ]; then
        printf '%s' ""
        return
    fi
    if [[ "$value" =~ ^([A-Za-z]:[\\/]|/) ]]; then
        printf '%s' "$value"
    else
        printf '%s/%s' "$COSTMARY_DIR" "$value"
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --ref_file)
            REF_FILE="$2"
            shift 2
            ;;
        --feature_path)
            FEATURE_PATH="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --cohort)
            COHORT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --vis_depth)
            VIS_DEPTH="$2"
            shift 2
            ;;
        --python)
            PYTHON="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ -z "$PYTHON" ]; then
    if [ -f "$COSTMARY_DIR/.venv/bin/python" ]; then
        PYTHON="$COSTMARY_DIR/.venv/bin/python"
    elif [ -f "$COSTMARY_DIR/.venv/Scripts/python.exe" ]; then
        PYTHON="$COSTMARY_DIR/.venv/Scripts/python.exe"
    elif [ -f "$COSTMARY_DIR/../.venv/bin/python" ]; then
        PYTHON="$COSTMARY_DIR/../.venv/bin/python"
    elif [ -f "$COSTMARY_DIR/../.venv/Scripts/python.exe" ]; then
        PYTHON="$COSTMARY_DIR/../.venv/Scripts/python.exe"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    else
        PYTHON=python
    fi
fi

REF_FILE="$(resolve_costmary_path "$REF_FILE")"
FEATURE_PATH="$(resolve_costmary_path "$FEATURE_PATH")"
SAVE_DIR="$(resolve_costmary_path "$SAVE_DIR")"

if [ -z "$REF_FILE" ] || [ -z "$FEATURE_PATH" ] || [ -z "$SAVE_DIR" ]; then
    echo "Error: --ref_file, --feature_path, and --save_dir are required." >&2
    usage >&2
    exit 1
fi

if [ ! -f "$REF_FILE" ]; then
    echo "Error: ref_file not found: $REF_FILE" >&2
    exit 1
fi

if [ ! -d "$FEATURE_PATH" ]; then
    echo "Error: feature_path not found: $FEATURE_PATH" >&2
    exit 1
fi

# Backbone set to run.
backbone_list=(vis vis_d2 spd he2rna mean projmean)

# Shared SPD parameters for spd / mean / projmean.
SPD_BACKBONE_SPD_R=96

# When head=spex, run one experiment per K; experiment name includes k.
spex_k_list=(8 16 32 64)

total_exps=0
for backbone in "${backbone_list[@]}"; do
    total_exps=$(( total_exps + 1 ))
    if [ "$backbone" != "projmean" ] && [ "$backbone" != "vis_d2" ]; then
        total_exps=$(( total_exps + ${#spex_k_list[@]} ))
    fi
done
echo "TOTAL experiments: ${total_exps}"

COMMON_ARGS=(
    --ref_file "$REF_FILE"
    --feature_path "$FEATURE_PATH"
    --save_dir "$SAVE_DIR"
    --cohort "$COHORT"
    --seed "$SEED"
    --batch_size 16
    --depth "$VIS_DEPTH"
    --train
)

for BACKBONE in "${backbone_list[@]}"; do
    MODEL_TYPE="$BACKBONE"
    EXTRA_ARGS=()
    if [ "$BACKBONE" = "spd" ] || [ "$BACKBONE" = "mean" ] || [ "$BACKBONE" = "projmean" ]; then
        SPD_ARGS=(
            --spd_r "$SPD_BACKBONE_SPD_R"
        )
    else
        SPD_ARGS=()
    fi

    if [ "$BACKBONE" = "vis" ]; then
        EXTRA_ARGS=(--depth "$VIS_DEPTH" --num-heads 16 --num_epochs 200)
    elif [ "$BACKBONE" = "vis_d2" ]; then
        MODEL_TYPE="vis"
        EXTRA_ARGS=(--depth 2 --num-heads 16 --num_epochs 200)
    else
        EXTRA_ARGS=(--num_epochs 200)
    fi

    HEAD_ARGS=(--head_type linear)
    EXP_NAME="${BACKBONE}_linear"
    CMD=(
        "$PYTHON" "$MAIN_PY"
        --model_type "$MODEL_TYPE"
        "${COMMON_ARGS[@]}"
        "${EXTRA_ARGS[@]}"
        "${SPD_ARGS[@]}"
        "${HEAD_ARGS[@]}"
        --exp_name "$EXP_NAME"
    )
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%q ' "${CMD[@]}"
        echo ""
    else
        echo ""
        echo "============================================"
        echo "  backbone=${BACKBONE} head=linear"
        echo "  exp_name: ${EXP_NAME}"
        echo "============================================"
        "${CMD[@]}"
        echo "Finished: ${EXP_NAME}"
    fi

    if [ "$BACKBONE" = "projmean" ] || [ "$BACKBONE" = "vis_d2" ]; then
        continue
    fi

    for SPEX_K in "${spex_k_list[@]}"; do
        HEAD_ARGS=(
            --head_type spex
            --spex_k "$SPEX_K"
        )
        EXP_NAME="${BACKBONE}_spex_k${SPEX_K}"
        CMD=(
            "$PYTHON" "$MAIN_PY"
            --model_type "$MODEL_TYPE"
            "${COMMON_ARGS[@]}"
            "${EXTRA_ARGS[@]}"
            "${SPD_ARGS[@]}"
            "${HEAD_ARGS[@]}"
            --exp_name "$EXP_NAME"
        )
        if [ "$DRY_RUN" -eq 1 ]; then
            printf '%q ' "${CMD[@]}"
            echo ""
        else
            echo ""
            echo "============================================"
            echo "  backbone=${BACKBONE} head=spex spex_k=${SPEX_K}"
            echo "  exp_name: ${EXP_NAME}"
            echo "============================================"
            "${CMD[@]}"
            echo "Finished: ${EXP_NAME}"
        fi
    done
done

echo ""
echo "All experiments completed."
