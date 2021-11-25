#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=complete
#SBATCH --array=1-5
MAIN="/home/gridsan/akyurek/git/bias-lm"


MODEL_NAME="gpt2"
PROMPT_SET="bold"
DOMAIN="gender"
        
cnt=0
for LEN in 100; do
    for NUMGEN in 1 5 10 25 50; do
        (( cnt++ ))
        if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
            INPUT_PATH=$MAIN/outputs/generations/$MODEL_NAME/$PROMPT_SET/$DOMAIN/num_gen_${NUMGEN}/length_${LEN}

            python compute_tox_sent.py \
                    --prompt_domain $DOMAIN \
                    --prompt_set $PROMPT_SET \
                    --save_path $INPUT_PATH \
                    --input_file $INPUT_PATH/gens.csv
        fi  
    done
done


# Debug
# CAT="webtext.train"
# PROMPT_SET="bold"
# DOMAIN="gender"
# SAVE_PATH="$MAIN/outputs/webtext"
# mkdir -p $SAVE_PATH
# python compute_tox_sent.py \
#         --category $CAT \
#         --input_file $MAIN/gpt-2-output-dataset/data_asis/$CAT.txt \
#         --save_path $SAVE_PATH