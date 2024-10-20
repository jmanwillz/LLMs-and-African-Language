#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --job-name=nlp-fine-tune
#SBATCH --output=/home-mscluster/kkaruppen/jason/result.txt

echo ------------------------------------------------------
echo -n 'Job is running on node ' $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: sbatch is running on $SLURM_SUBMIT_HOST
echo SLURM: job ID is $SLURM_JOB_ID
echo SLURM: submit directory is $SLURM_SUBMIT_DIR
echo SLURM: number of nodes allocated is $SLURM_JOB_NUM_NODES
echo SLURM: number of cores is $SLURM_NTASKS
echo SLURM: job name is $SLURM_JOB_NAME
echo ------------------------------------------------------

/bin/hostname
cd /home-mscluster/kkaruppen/jason
python3 fine_tune.py