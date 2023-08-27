#!/bin/bash

TARGET_COLUMN="$1"
EARLY_STOPPING_ROUNDS="$2"
NUM_BOOST_ROUND="$3"

pip install -r requirements.txt

python3 "src/TaskOne.py" --target_column ${TARGET_COLUMN} --early_stopping_rounds ${EARLY_STOPPING_ROUNDS} --num_boost_round ${NUM_BOOST_ROUND}