#!/bin/bash

# NOTE: this is a dynamic SH stub which can be reused for debug or preduction versions of chemtab UQ experiments!
# Example USAGE: EXTRA_PL_ARGS='--data.data_fn=../data/chrest_contiguous_group_sample100k.csv --data.batch_size 4500 --trainer.fast_dev_run True' ./chemtab_UQ_slurm-debug.job

srun nvidia-smi

# --nodes cannot be set dynamically!! (in sheebang)
# but it can be detected dynammically using srun magic!
num_nodes=$(srun hostname | sort | uniq -c | wc -l)
# logic is: prints all hostnames, then sorts, then groups & counts group sizes, then counts number of groups

echo num nodes: $num_nodes

# let the user know if their EXTRA_PL_ARGS went through or not
[ -z "$EXTRA_PL_ARGS" ] && echo EXTRA_PL_ARGS is empty!! >&2

# clear previous variables (NOT an input env variable)
diagnostic_CLI_args="--trainer.logger True --trainer.profiler simple --model.device_stats_monitor True --trainer.track_grad_norm 2" # use these only for "debug mode"
EXTRA_PL_ARGS1="$EXTRA_PL_ARGS $PL_MAX_EPOCHS $diagnostic_CLI_args" # NOTE: EXTRA_PL_ARGS should be an environment variable given by the user when submitting sbatch!!
lightning_CLI_args="$EXTRA_PL_ARGS1 --trainer.num_nodes=$num_nodes --trainer.devices=2 --trainer.accelerator=gpu --trainer.strategy=ddp --trainer.gradient_clip_algorithm=value --trainer.gradient_clip_val 0.25"
# NOTE: gradient_clip_value=0.5 recommended by PL docs
# NOTE: --max_epochs -1 --> means no max epochs (i.e. fit for entire allocation time)

echo diagnostic_CLI_args: $diagnostic_CLI_args
echo EXTRA_PL_ARGS: $EXTRA_PL_ARGS
echo lightning_CLI_args: $lightning_CLI_args

#Let's start some work
source /user/dwyerdei/.bash_profile
conda activate pytorch_distributed_cuda3
# IMPORTANT: pytorch_distributed_cuda3 is the "stable version" with pytorch-lightning==1.9.0 and regular CLI stuff working
# pytorch_distributed_cuda is a possible "newer version" which is capable of using the A100 GPUs

srun --ntasks-per-node=2 python ChemtabUQ.py fit --data.class_path=MeanRegressorDataModule $lightning_CLI_args 
mkdir mean_regressor
mv model.* mean_regressor
srun --ntasks-per-node=2 python ChemtabUQ.py fit --data.class_path=UQRegressorDataModule --data.mean_regressor_fn=mean_regressor/model.ckpt $lightning_CLI_args 
mkdir UQ_regressor
mv model.* UQ_regressor

# Example Commands for training of mean_regressor and UQ model
#srun --ntasks-per-node=2 python ChemtabUQ.py fit --data.class_path=MeanRegressorDataModule --data.data_fn=../data/chrest_contiguous_group_sample100k.csv --trainer.accelerator=gpu --trainer.devices=2 --trainer.num_nodes=2
#srun --ntasks-per-node=2 python ChemtabUQ.py fit --data.class_path=UQRegressorDataModule --data.data_fn=../data/chrest_contiguous_group_sample100k.csv --data.mean_regressor_fn=mean_regressor/model.ckpt --trainer.accelerator=gpu --trainer.devices=2 --trainer.num_nodes=2
