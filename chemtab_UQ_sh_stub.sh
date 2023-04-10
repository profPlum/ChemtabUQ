#!/bin/bash

# NOTE: this is a dynamic SH stub which can be reused for debug or preduction versions of chemtab UQ experiments!

srun nvidia-smi

# --nodes cannot be set dynamically!! (in sheebang)
# but it can be detected dynammically using srun magic!
num_nodes=$(srun hostname | sort | uniq -c | wc -l)
# logic is: prints all hostnames, then sorts, then groups & counts group sizes, then counts number of groups

# this version is deprecated...
#num_nodes=$((num_nodes/2)) # WARNING: this will fail if you use more than 2 tasks per node!
echo num nodes: $num_nodes

# let the user know if their EXTRA_PL_ARGS went through or not
[ -z "$EXTRA_PL_ARGS" ] && echo EXTRA_PL_ARGS is empty!! >&2

# clear previous variables (NOT an input env variable)
diagnostic_CLI_args=
#diagnostic_CLI_args='--fast_dev_run 1000'
#diagnostic_CLI_args="--fast_dev_run 1000 --overfit_batches 0.01 --logger True --profiler simple --device-stats-monitor --detect_anomaly" # use these only for "debug mode"
diagnostic_CLI_args="--logger True --profiler simple --device-stats-monitor --detect_anomaly --track_grad_norm 2" # use these only for "debug mode"
EXTRA_PL_ARGS1="$EXTRA_PL_ARGS $PL_MAX_EPOCHS $diagnostic_CLI_args" # NOTE: this should be an environment variable given by the user when submitting sbatch!!
lightning_CLI_args="$EXTRA_PL_ARGS1 --num_nodes=$num_nodes --devices=2 --accelerator=gpu --strategy=ddp --gradient_clip_algorithm=value --gradient_clip_val 0.25"
# NOTE: gradient_clip_value=0.5 recommended by PL docs
# NOTE: --max_epochs -1 --> means no max epochs (i.e. fit for entire allocation time)

echo diagnostic_CLI_args: $diagnostic_CLI_args
echo EXTRA_PL_ARGS: $EXTRA_PL_ARGS
echo lightning_CLI_args: $lightning_CLI_args

#Let's start some work
source /user/dwyerdei/.bash_profile
conda activate pytorch_distributed_cuda3
# IMPORTANT: pytorch_distributed_cuda3 is the "stable version" with pytorch-lightning==1.9.0 and regular CLI stuff working
# pytorch_distributed_cuda is a possible "newer version" which is capable of using the A100 GPUs but also has a new API 
# for the CLI interface which would require new code & I'm too lazy to do that right now

# NOTE: this command is for slurmi sessions!
#srun --ntasks-per-node=2 python ChemtabUQ.py --num_nodes=$num_nodes --devices=2 --accelerator=gpu --strategy=ddp --gradient_clip_algorithm=value --gradient_clip_val 0.5
srun --ntasks-per-node=2 python ChemtabUQ.py $lightning_CLI_args #--num_nodes=$num_nodes --devices=2 --accelerator=gpu --strategy=ddp --gradient_clip_algorithm=value --gradient_clip_val 0.5
#Let's finish some work
