#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-7
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=genderbias


MAIN=$PWD
# cnt=0
# for i in $MAIN/data/nli/gender/*.csv; do
#     (( cnt++ ))
#     if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then 
#         LOG_OUT=$MAIN/dumped/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
#         LOG_ERR=$MAIN/dumped/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err
#         # python $MAIN/json2csv.py $i > $LOG_OUT 2> $LOG_ERR
#         v=${i::-5}
#         python $MAIN/hf_inference_api.py bart $v.csv > $LOG_OUT 2> $LOG_ERR
#     fi
# done

# Debug
i=$MAIN/data/nli/gender/man_is_to_surgeon
# python $MAIN/json2csv.py ${i}.json
python $MAIN/hf_inference_api.py t0pp ${i}.csv

