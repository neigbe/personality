#!/bin/bash
source /nlp/scr/neigbe/miniconda3/etc/profile.d/conda.sh
conda activate personality
cd /nlp/scr/neigbe/pers_proj/src/fsdp_qlora/
# CHECK TEST SIZE
# CHECK THAT MODEL_NAME AND OUTPUT_DIR MATCH
python train.py --world_size 5 --context_length 8192 --num_epochs 3 --dataset personality --model_name "meta-llama/Meta-Llama-3-70B-Instruct" --save_model True --verbose True --output_dir "/scr/neigbe/`date '+%F|%T'`" --test_size 0.975 --reentrant_checkpointing True