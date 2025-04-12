#!/bin/zsh -l
#SBATCH --job-name=sg   # create a name for your job
#SBATCH --output=logs/gpu/%x.%j.log # %x.%j.%N expands to JobName.JobID.NodeName
#SBATCH --error=logs/gpu/%x.%j.log
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hz54@njit.edu

# Create log directory if it doesn't exist
# mkdir -p logs; mkdir -p logs/gpu

# Remove all .err and .out files in the logs/gpu folder
# rm -f logs/gpu/*.err logs/gpu/*.out

# Remove all .err and .out files in the logs folder
# rm -f logs/*.log


export OMPI_MCA_mpi_warn_on_fork=0
export RDMAV_FORK_SAFE=1
export UCX_TLS=rc,sm,self
export UCX_NET_DEVICES=mlx5_0:1
export CUDA_LAUNCH_BLOCKING=1

eval "$(micromamba shell hook --shell zsh)"
micromamba activate /project/md748/hpcadmin/hz54/env/sae-sd
export PYTHONPATH=$PYTHONPATH:/mmfs1/project/md748/hpcadmin/hz54/env/sae-sd/lib/python3.11/site-packages/
export PROJ=/project/md748/hz54/kav/sae-kav

# srun python squad_baseline_after.py
srun python baseline/baseline.py

# Capture the exit status
exit_status=$?

# Cancel the job regardless of its status
scancel $SLURM_JOB_ID

# Exit with the original exit status
exit $exit_status