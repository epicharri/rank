#!/bin/bash

#SBATCH --job-name=search
#SBATCH --account=project_2005883
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=results/results-%x.%j.out
#SBATCH --error=results/log-%x.%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load gcc
module load cuda/11.5.0


params=(256 512 1024 2048 4096)

for len in 10000000000 100000000000
do
        for m in poppy cum-poppy
        do
          for i in {0..4}
          do
            for a in --sequential-positions --random-positions
            do
              for content in --random-bit-vector
              do
                srun build/search --bits-in-bit-vector=$len $a $content --query-positions-count=1000000000 --start-position=4000000000 --do-not-store-results --rank-structure=$m --bits-in-superblock=${params[i]} --threads-per-block=1024 --device-set-limit-search=64
              done
            done
          done
        done
done
