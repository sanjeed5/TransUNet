#!/bin/bash
#PBS -l walltime=23:59:59
#PBS -q gpuq
#PBS -e train.err
#PBS -o train.log
#PBS -l select=1:ncpus=16:ngpus=2

conda init bash
module load anaconda3_2020
source activate pytorch_transunet
cat $PBS_O_WORKDIR
cd /lfs/usrhome/btech/ed17b047/transunet/TransUNet

jobid=`echo $PBS_JOBID | cut -f 1 -d .`

logfilename=$HOME/transunet/job_logs/train_$jobid.log

CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/pytorch_transunet/bin/python -u '/lfs/usrhome/btech/ed17b047/transunet/TransUNet/train.py' --dataset CBIS-DDSM --root_path ../data/CBIS-DDSM/Train_npz --vit_name R50-ViT-B_16 --img_size 512 --batch_size 8 &>> $logfilename

testlogfilename=$HOME/transunet/job_logs/test_$jobid.log

CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 ~/.conda/envs/pytorch_transunet/bin/python -u '/lfs/usrhome/btech/ed17b047/transunet/TransUNet/test.py' --dataset CBIS-DDSM --vit_name R50-ViT-B_16 --img_size 512 --batch_size 8 &>> $testlogfilename