#!/bin/bash
#PBS -l walltime=06:00:00
#PBS -q gpuq
#PBS -e sweep.err
#PBS -o sweep.log
#PBS -l select=1:ncpus=16:ngpus=2

conda init bash
module load anaconda3_2020
source activate pytorch_transunet

cd /lfs/usrhome/btech/ed17b047/transunet/TransUNet

jobid=`echo $PBS_JOBID | cut -f 1 -d .`
logfilename=$HOME/transunet/job_logs/sweep_$jobid

$HOME/.conda/envs/pytorch_transunet/bin/wandb agent $SWEEP_ID >> $logfilename.log 2>> $logfilename.err