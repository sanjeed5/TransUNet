#!/bin/bash
#PBS -l walltime=05:00:00
#PBS -q gpuq
#PBS -e test.err
#PBS -o test.log
#PBS -l select=1:ncpus=8:ngpus=2
#PBS -m eb
#PBS -M ed17b047@smail.iitm.ac.in

conda init bash
module load anaconda3_2020
source activate pytorch_transunet
cd /lfs/usrhome/btech/ed17b047/transunet/TransUNet

jobid=`echo $PBS_JOBID | cut -f 1 -d .`
logfilename=$HOME/transunet/job_logs/test_$jobid.log

CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/pytorch_transunet/bin/python -u '/lfs/usrhome/btech/ed17b047/transunet/TransUNet/test.py' --dataset CBIS-DDSM --vit_name R50-ViT-B_16 &>> $logfilename