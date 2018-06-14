#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30
#SBATCH --mem=4G
#SBATCH -c 16

export CUDA_MPS_LOG_DIRECTORY=$HOME/tmp
nvidia-cuda-mps-control -d

time ~/CUDA_day2/P100/HQ.sh 4
