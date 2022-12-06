#!/usr/bin/env bash
#SBATCH -A chpc
#SBATCH -p notchpeak-shared
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --time=00:15:00

host=$(hostname)
echo "Running on $host"
date +'Starting at %R'

module load python/3.10.3
module load gcc/8.5.0
module load mpich/4.0.2

time mpiexec -n 10 ./laplace_mpi.py --dim 100 --method arrays --verbose

date +'Finished at %R'
