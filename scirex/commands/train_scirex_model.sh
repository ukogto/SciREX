if [ $# -eq  0 ]
  then
    echo "No argument supplied for experiment name"
    exit 1
fi
rm -r outputs/pwc_outputs/experiment_scirex_full/main

export LONGFORMER_BASE_FOLDER=./longformer_weights
export LONGFORMER_VOCAB=$LONGFORMER_BASE_FOLDER/vocab.json
export LONGFORMER_WEIGHTS=$LONGFORMER_BASE_FOLDER/

export CONFIG_FILE=scirex/training_config/scirex_full.jsonnet

export CUDA_DEVICE=0
export IS_LOWERCASE=true

export DATA_BASE_PATH=scirex_dataset/release_data

export TRAIN_PATH=$DATA_BASE_PATH/train.jsonl
export DEV_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_PATH=$DATA_BASE_PATH/test.jsonl

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs/pwc_outputs/experiment_scirex_full/$1}

export longformer_fine_tune=10,11,pooler

nw=1 lw=1 rw=1 em=false \
relation_cardinality=4 \
allennlp train -s $OUTPUT_BASE_PATH --include-package scirex $RECOVER $CONFIG_FILE
