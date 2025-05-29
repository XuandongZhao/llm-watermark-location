#!/bin/bash

JSON_PATH="detect.jsonl"
TEXT_KEY="text"
TOKENIZER_DIR="mistralai/Mistral-7B-Instruct-v0.2"
METHOD="marylandz"
SEEDING="hash"
NGRAM=2
N_POSITIVE=500
HASH_KEY=35317
SCORING_METHOD="none"
PAYLOAD=0
PAYLOAD_MAX=4
SEED=0
GAMMA=0.25
DELTA=2

python ../AOL.py --json_path $JSON_PATH \
                 --text_key $TEXT_KEY \
                 --tokenizer_dir $TOKENIZER_DIR \
                 --method $METHOD \
                 --seeding $SEEDING \
                 --ngram $NGRAM \
                 --n_positive $N_POSITIVE \
                 --hash_key $HASH_KEY \
                 --scoring_method $SCORING_METHOD \
                 --payload $PAYLOAD \
                 --payload_max $PAYLOAD_MAX \
                 --seed $SEED \
                 --gamma $GAMMA \
                 --delta $DELTA 
