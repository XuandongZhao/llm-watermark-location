#!/bin/bash

JSON_PATH="./detect_file.jsonl"
TEXT_KEY="text"
TOKENIZER_DIR="baffo32/decapoda-research-llama-7B-hf"
METHOD="marylandz"
SEEDING="hash"
NGRAM=2
N_POSITIVE=500
GAMMA=0.25
HASH_KEY=35317
SCORING_METHOD="none"
PAYLOAD=0
PAYLOAD_MAX=4
DELTA=2.0
BATCH_SIZE=16
SEED=0
OUTPUT_JSON_PATH="output.jsonl"

# Execute the Python script with the defined arguments
python ../Geometry_Cover_detection.py --json_path $JSON_PATH \
                 --output_json_path $OUTPUT_JSON_PATH \
                 --text_key $TEXT_KEY \
                 --tokenizer_dir $TOKENIZER_DIR \
                 --method $METHOD \
                 --seeding $SEEDING \
                 --ngram $NGRAM \
                 --n_positive $N_POSITIVE \
                 --gamma $GAMMA \
                 --hash_key $HASH_KEY \
                 --scoring_method $SCORING_METHOD \
                 --payload $PAYLOAD \
                 --payload_max $PAYLOAD_MAX \
                 --delta $DELTA \
                 --batch_size $BATCH_SIZE \
                 --seed $SEED 