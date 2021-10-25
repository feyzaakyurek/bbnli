#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=complete
#SBATCH --array=1-19

MAIN="/home/gridsan/akyurek/git/bias-lm"
declare -a Cats=("asis" "combined" "qas" "cds" "is" "re" "om" "an" "or" "svp" "asm" "pr" "dmc" "asf" "ddp" "ps" "pa" "rci" "ddf")

cnt=0
for CAT in ${Cats[@]}; do
    (( cnt++ ))
    if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
        MODEL_PATH=$MAIN/"gpt-2-fine-tuned"/${CAT}_50000
        PROMPT_SET="bold"
        DOMAIN="gender"
        SAVE_PATH=$MAIN/outputs/generations/${PROMPT_SET}_${DOMAIN}_${CAT}_nosampling_50000_50
        mkdir -p $SAVE_PATH
        echo "Generations from ${MODEL_PATH} for $PROMPT_SET $DOMAIN" > $SAVE_PATH/info.txt
        python complete_prompts.py \
                --model_path $MODEL_PATH \
                --prompt_domain $DOMAIN \
                --prompt_set $PROMPT_SET \
                --save_path $SAVE_PATH/gens.csv \
                --do_not_sample \
                --num_gens 1 \
                --max_length 50
    fi
done

# Debug
# CAT="asis"
# MODEL_PATH=$MAIN/"gpt-2-fine-tuned"/$CAT
# PROMPT_SET="bold"
# DOMAIN="gender"
# SAVE_PATH=$MAIN/outputs/${PROMPT_SET}_${DOMAIN}_${CAT}
# mkdir -p $SAVE_PATH
# echo "Generations from ${MODEL_PATH} for $PROMPT_SET $DOMAIN" > $SAVE_PATH/info.txt
# python complete_prompts.py \
#                 --model_path $MODEL_PATH \
#                 --prompt_domain $DOMAIN \
#                 --prompt_set $PROMPT_SET \
#                 --save_path $SAVE_PATH/gens.csv \
#                 --num_gens 25
