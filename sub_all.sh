#!/bin/bash

#SBATCH --job-name=search
#SBATCH --account=project_2005883
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
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

for m in {10..13}
do
  for i in {0..6}
    do
      for a in ./querydata100/queries.fna ./query_data_random/random_queries.fna ./query_data_random/queries_2464000_lines_and_random_2365442_lines.fna
        do
          srun build/search --do-not-store-results --rank-structure=cum-poppy --one-zero-and-then-all-ones-bit-vector --bits-in-bit-vector=10000000000--query-positions-count=1000000000 --bits-in-superblock=${params[i]} ${shuffles[i]} --threads-per-block=1024 --device-set-limit-presearch=64 --device-set-limit-search=64 
        done
    done
done

  
