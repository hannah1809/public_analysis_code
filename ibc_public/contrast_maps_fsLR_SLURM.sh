#!/bin/bash

#SBATCH --job-name=contrast_fsLR
#SBATCH --output=/ptmp/hmueller2/GLM_logs/output/%A_%x.out
#SBATCH --error=/ptmp/hmueller2/GLM_logs/errors/%A_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
# #SBATCH --array=0-7 # (subjects-1)

export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"
container=/home/rglz/containers/gfae.sif

# SUBJECTS_FILE=/ptmp/hmueller2/Downloads/subjects_resting.txt
# subject=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $SUBJECTS_FILE)

subject="06"

echo "---- Starting contrast map calculation for subject $subject ----"

srun apptainer exec ${container} \
    bash -c "export PYTHONPATH=/home/hmueller2/ibc_code/ibc_latent/public_analysis:\$PYTHONPATH && python /home/hmueller2/ibc_code/ibc_latent/public_analysis/ibc_public/contrast_maps_fsLR.py $subject"

exit 0


# run: sbatch /home/hmueller2/ibc_code/ibc_latent/public_analysis/ibc_public/contrast_maps_fsLR_SLURM.sh