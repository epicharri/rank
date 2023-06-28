#!/bin/bash

#SBATCH --job-name=rank_search
#SBATCH --account=project_2005883
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=results/results-%x.%j.out
#SBATCH --error=results/log-%x.%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load gcc
module load cuda/11.5.0


shf="--with-shuffles"
empty="--not-any-parameter"

params=(256 512 1024 1024 2048 2048 4096)
shuffles=($empty $empty $empty $shf $empty $shf $empty)


srun build/search --do-not-store-results --rank-structure=cum-poppy --one-zero-and-then-all-ones-bit-vector --bits-in-bit-vector=100000000000 --query-positions-count=1000000000 --bits-in-superblock=1024 --no-shuffles --threads-per-block=1024 --device-set-limit-presearch=64 --device-set-limit-search=64 

