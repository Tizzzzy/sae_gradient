#!/bin/bash
#SBATCH --job-name=pti18   # create a name for your job
#SBATCH --partition=general
#SBATCH --output=logs/cpu/%x.%j.out
#SBATCH --error=logs/cpu/%x.%j.err
#SBATCH --qos=standard
#SBATCH --nodes=1  # node count
#SBATCH --ntasks-per-node=1               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=150G         # memory per cpu-core
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)

# Create log directory if it doesn't exist
# mkdir -p logs; mkdir -p logs/cpu/ 

# Remove all .out and .err files in the log directory
# rm -f logs/cpu/*.err logs/cpu/*.out logs/main*.log

# set up fork safety for ucx
export OMPI_MCA_mpi_warn_on_fork=0
export OMPI_MCA_btl_openib_want_fork_support=1

export PROJ=/mmfs1/project/md748/hz54/sae-sd
eval "$(micromamba shell hook --shell bash)"
micromamba activate /project/md748/hpcadmin/hz54/env/sae-sd
export PYTHONPATH=$PYTHONPATH:/mmfs1/project/md748/hpcadmin/hz54/env/sae-sd/lib/python3.11/site-packages/

# Run the Python script
# srun python crop_chart.py --year 2015

# Capture the exit status
exit_status=$?

# Cancel the job regardless of its status
scancel $SLURM_JOB_ID
# Exit with the original exit status
exit $exit_status
