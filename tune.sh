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

echo "Running tuning for all models"
echo "Dataset directory: $DATASET_DIR"
echo "MLflow experiment: $MLFLOW_EXPERIMENT"
echo "Parameters:"
echo "  NTRIALS: $NTRIALS"

tune() {
    echo "Running tuning for model $1"
    echo "  EPOCHS: $2"
    echo "  BATCH: $3"
    echo "  PAT: $4"
    echo "  TOL: $5"
    python3 tune.py \
	"$DATASET_DIR/$(echo "$1" | cut -d '-' -f 1)" \
	"$1" \
        --epochs "$2" \
        --batch_size "$3" \
        --patience "$4" \
        --trials "$NTRIALS" \
        --tolerance "$5" \
        --mlflow-experiment "$MLFLOW_EXPERIMENT" \
        --mlflow-run "$1" \
    	--device cuda
}

	tune "repr2-graphconv" 30 256 5 0.01
	tune "repr2-sage" 30 256 5 0.01
	tune "repr2-gat" 30 256 5 0.01

	tune "repr1-graphconv" 40 256 5 0.01
	tune "repr1-gcn" 40 256 5 0.01
	tune "repr1-gat" 40 256 5 0.01

	tune "baseline-sum-max" 40 256 5 0.01
	tune "baseline-max-max" 40 256 5 0.01
	tune "baseline-mean-max" 40 256 5 0.01
	tune "baseline-mean-both" 40 256 5 0.01


echo "Tuning finished"
