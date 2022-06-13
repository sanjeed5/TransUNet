#!/bin/bash
#PBS -l walltime=23:59:59
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

CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/pytorch_transunet/bin/python -u '/lfs/usrhome/btech/ed17b047/transunet/TransUNet/sweep_bayes.py' >> $logfilename.log 2>> $logfilename.err
