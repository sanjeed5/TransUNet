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

tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .

CUDA_VISIBLE_DEVICES=0,1 ~/.conda/envs/pytorch_transunet/bin/python '/lfs/usrhome/btech/ed17b047/transunet/TransUNet/train.py' --dataset CBIS-DDSM --root_path ../data/CBIS-DDSM/Train_npz
source deactivate
