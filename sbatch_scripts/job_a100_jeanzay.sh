#!/bin/bash
#SBATCH -A pri@a100
#SBATCH --job-name=ginkgobench        # name of job
#SBATCH -C a100                     # uncomment for gpu_p5 partition (80GB A100 GPU)

# Here, reservation of 10 CPUs (for 1 task) and 1 GPU on a single node:
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p5)

#SBATCH --exclusive                  # Reserve the entire node exclusively for this job

# The number of CPUs per task must be adapted according to the partition used. Knowing that here
# only one GPU is reserved (i.e. 1/4 or 1/8 of the GPUs of the node depending on the partition),
# the ideal is to reserve 1/4 or 1/8 of the CPUs of the node for the single task:
#SBATCH --cpus-per-task=8           # number of cores per task for gpu_p5 (1/8 of 8-GPUs A100 node)

# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=00:10:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=ginkgobench%j.out    # name of output file
#SBATCH --error=ginkgobench%j.out     # name of error file (here, in common with the output file)

# Cleans out the modules loaded in interactive and inherited by default 
module purge

module load arch/a100

# Loading of modules
module load cuda/12.4.1 openmpi/4.1.5 cmake/3.18.0

# Echo of launched commands
set -x

./main cuda

