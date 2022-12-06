#!/usr/bin/env bash
#SBATCH -A chpc
#SBATCH -p notchpeak-shared
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --time=4:00:00

date +'Starting at %R'

module load python/3.10.3
module load gcc/8.5.0
module load mpich/4.0.2

for method in arrays loops
do
	for dim in 100 200 500
	do
		echo "Method $method dimension $dim"
		time mpiexec -n 10 ./laplace_mpi.py --dim $dim --method $method > /dev/null
	done
done

date +'Finished at %R'
