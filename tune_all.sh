#!/bin/bash

DATASET_DIR="$1"
MLFLOW_EXPERIMENT="$2"
if [ -z "$DATASET_DIR" ] || [ -z "$MLFLOW_EXPERIMENT" ]; then
    echo "Usage: $0 <dataset_dir> <mlflow_experiment>"
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset directory $DATASET_DIR does not exist"
    exit 1
fi


NTRIALS=50
PATIENCE=10
TOLERANCE=0.01

# git tags for different models
BASELINE_BASE="baseline_base"
BASELINE_DERIVED_1="baseline-derived_1"
BASELINE_DERIVED_2="baseline-derived_2"

REPR1_BASE="repr1-base"
REPR1_DERIVED_1="repr1-derived_1"
REPR1_DERIVED_2="repr1-derived_2"

REPR2_BASE="repr2-base"
REPR2_DERIVED_1="repr2-derived_1"
REPR2_DERIVED_2="repr2-derived_2"

echo "Running tuning for all models"
echo "Dataset directory: $DATASET_DIR"
echo "MLflow experiment: $MLFLOW_EXPERIMENT"
echo "Parameters:"
echo "  NTRIALS: $NTRIALS"
echo "  PATIENCE: $PATIENCE"
echo "  TOLERANCE: $TOLERANCE"

tune() {
    echo "Running tuning for model $1"
    echo "  BATCH_SIZE: $BATCH_SIZE"
    echo "  EPOCHS: $3"
    # $1: git tag
    # $2: model name
    # $3: epochs
    git checkout "$1"
    python tune.py \
        "$DATASET_DIR" \
        "$2" \
        --epochs "$3" \
        --batch_size "$BATCH_SIZE" \
        --patience "$PATIENCE" \
        --trials "$NTRIALS" \
        --tolerance "$TOLERANCE" \
        --mlflow-experiment "$MLFLOW_EXPERIMENT" \
        --mlflow-run "$1"
}

echo "Removing processed dataset"
BATCH_SIZE=64
EPOCHS=200
rm "$DATASET_DIR"/processed/*
tune "$BASELINE_BASE" "baseline" "$EPOCHS"
# tune "$BASELINE_DERIVED_1" "baseline" "$EPOCHS"
# tune "$BASELINE_DERIVED_2" "baseline" "$EPOCHS"

BATCH_SIZE=64
EPOCHS=200
echo "Removing processed dataset"
rm "$DATASET_DIR"/processed/*
tune "$REPR1_BASE" "repr1" "$EPOCHS"
# tune "$REPR1_DERIVED_1" "repr1" "$EPOCHS"
# tune "$REPR1_DERIVED_2" "repr1" "$EPOCHS"

BATCH_SIZE=128
EPOCHS=200
echo "Removing processed dataset"
rm "$DATASET_DIR"/processed/*
tune "$REPR2_BASE" "repr2" "$EPOCHS"
# tune "$REPR2_DERIVED_1" "repr2" "$EPOCHS"
# tune "$REPR2_DERIVED_2" "repr2" "$EPOCHS"

echo "Tuning finished"
