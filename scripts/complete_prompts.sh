#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=complete
#SBATCH --array=1-25

MAIN="/home/gridsan/akyurek/git/bias-lm"
# declare -a Cats=("asis" "combined" "qas" "cds" "is" "re" "om" "an" "or" "svp" "asm" "pr" "dmc" "asf" "ddp" "ps" "pa" "rci" "ddf")

# MODEL_NAME="gpt2"
# MODEL_PATH=$MAIN/$MODEL_NAME
# PROMPT_SET="bold"
# DOMAIN="gender"
# cnt=0
# for LEN in 15 25 50 75 100; do
#     for NUMGEN in 1 5 10 25 50; do
#         (( cnt++ ))
#         if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
#             SAVE_PATH=$MAIN/outputs/generations/$MODEL_NAME/${PROMPT_SET}/${DOMAIN}/num_gen_${NUMGEN}/length_${LEN}
#             mkdir -p $SAVE_PATH
#             echo "Generations from ${MODEL_PATH} for $PROMPT_SET $DOMAIN" > $SAVE_PATH/info.txt
#             python complete_prompts.py \
#                     --model_name $MODEL_NAME \
#                     --model_path $MODEL_PATH \
#                     --prompt_domain $DOMAIN \
#                     --prompt_set $PROMPT_SET \
#                     --save_path $SAVE_PATH/gens.csv \
#                     --num_gens $NUMGEN \
#                     --max_length $LEN
#         fi
#     done
# done

# Debug
MODEL_PATH=$MAIN/"gpt2"
PROMPT_SET="bold"
DOMAIN="gender"
SAVE_PATH=$MAIN/outputs/generations/${PROMPT_SET}/${DOMAIN}
mkdir -p $SAVE_PATH
echo "Generations from ${MODEL_PATH} for $PROMPT_SET $DOMAIN" > $SAVE_PATH/info.txt
python complete_prompts.py \
                --model_name $MODEL_NAME \
                --model_path $MODEL_PATH \
                --prompt_domain $DOMAIN \
                --prompt_set $PROMPT_SET \
                --save_path $SAVE_PATH/gens.csv \
                --num_gens 25
