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
# declare -a Cats=("asis" "combined" "qas" "cds" "is" "re" "om" "an" "or" "svp" "asm" "pr" "dmc" "asf" "ddp" "ps" "pa" "rci" "ddf")

# cnt=0
# for CAT in ${Cats[@]}; do
#     (( cnt++ ))
#     if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
#         PROMPT_SET="bold"
#         DOMAIN="gender"
#         python compute_tox_sent.py \
#                 --prompt_domain $DOMAIN \
#                 --prompt_set $PROMPT_SET \
#                 --save_path 
#                 --inpute_file $MAIN/gpt-2-output-dataset/data_asis/web
#                 --category $CAT
#     fi    
# done


# Debug
CAT="webtext.train"
PROMPT_SET="bold"
DOMAIN="gender"
SAVE_PATH="$MAIN/outputs/webtext"
mkdir -p $SAVE_PATH
python compute_tox_sent.py \
        --category $CAT \
        --input_file $MAIN/gpt-2-output-dataset/data_asis/$CAT.txt \
        --save_path $SAVE_PATH