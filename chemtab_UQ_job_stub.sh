#!/bin/bash

# NOTE: this is a dynamic SH stub which can be reused for debug or preduction versions of chemtab UQ experiments!
# Example USAGE: EXTRA_PL_ARGS='--data.data_fn=../data/chrest_contiguous_group_sample100k.csv --data.batch_size 4500 --trainer.fast_dev_run True' ./chemtab_UQ-debug.slurm
# HINT, try this: python ChemtabUQ.py fit --data.help MeanRegressorDataModule !! Gives you great overview of possible CLI args to the data module class for training more general Chemtab mean models

srun nvidia-smi

# --nodes cannot be set dynamically!! (in sheebang)
# but it can be detected dynammically using srun magic!
num_nodes=$(srun hostname | sort | uniq -c | wc -l)
# logic is: prints all hostnames, then sorts, then groups & counts group sizes, then counts number of groups

echo num nodes: $num_nodes

# let the user know if their EXTRA_PL_ARGS went through or not
[ -z "$EXTRA_PL_ARGS" ] && echo EXTRA_PL_ARGS is empty!! >&2
# NOTE: EXTRA_PL_ARGS should be an environment variable given by the user when submitting sbatch!!

diagnostic_CLI_args="--trainer.logger=pytorch_lightning.loggers.TensorBoardLogger --trainer.logger.save_dir=. --trainer.logger.name=$SLURM_JOB_NAME --trainer.logger.version=$SLURM_JOB_ID" #--trainer.logger.save_dir=ChemtabUQ_TBlogs"
diagnostic_CLI_args="$diagnostic_CLI_args --trainer.profiler simple --trainer.callbacks pytorch_lightning.callbacks.DeviceStatsMonitor --trainer.track_grad_norm 2"
lightning_CLI_args="$EXTRA_PL_ARGS $diagnostic_CLI_args --trainer.num_nodes=$num_nodes --trainer.devices=2 --trainer.accelerator=gpu --trainer.strategy=ddp"

# IMPORTANT: gradient_clip defaults set to --trainer.gradient_clip_algorithm=value --trainer.gradient_clip_val=0.5 (inside ChemtabUQ.py)
# NOTE: gradient_clip_value=0.5 recommended by PL docs
# NOTE: --max_epochs -1 --> means no max epochs (i.e. fit for entire allocation time)

find_last_ckpt() {
    last_checkpoint=$(/bin/ls -t $(/bin/find "$1" -name "*.ckpt") | head -n 1)
    #last_checkpoint=$(/bin/ls -t $1/*/checkpoints/*.ckpt | head -n 1)
    echo $last_checkpoint
}

echo RESUME: $RESUME
if [ $RESUME == T ]; then
    # NOTE: this works for BOTH Sigma & Mean models b/c we use --trainer.default_root_dir which these checkpoint paths will implicitly be relative to!
    cd CT_logs_Mu
    last_checkpoint=$(find_last_ckpt ./$SLURM_JOB_NAME/)
    cd -
    echo loading last_checkpoint: $last_checkpoint
    lightning_CLI_args="$lightning_CLI_args --ckpt_path=$last_checkpoint"
else
    echo We arent resuming last ckpt... >&2
fi

echo diagnostic_CLI_args: $diagnostic_CLI_args
echo EXTRA_PL_ARGS: $EXTRA_PL_ARGS
echo lightning_CLI_args: $lightning_CLI_args

#Let's start some work
source /user/dwyerdei/.bash_profile
conda activate pytorch_distributed_cuda3
# IMPORTANT: pytorch_distributed_cuda3 is the "stable version" with pytorch-lightning==1.9.0 and regular CLI stuff working
# pytorch_distributed_cuda is a possible "newer version" which is capable of using the A100 GPUs

cd CT_logs_Mu
srun --ntasks-per-node=2 python ../ChemtabUQ.py fit --data.class_path=MeanRegressorDataModule $lightning_CLI_args #--trainer.default_root_dir=CT_logs_Mu
mkdir mean_regressors
mv model.ckpt mean_regressors/model-${SLURM_JOB_ID}.ckpt
cd -

if [ $MEAN_ONLY != T ]; then
    cd CT_logs_Sigma
	# TODO: fix bad mean regressor loading for cases with multiple jobs running at once
    srun --ntasks-per-node=2 python ../ChemtabUQ.py fit --data.class_path=UQRegressorDataModule --data.mean_regressor_fn=~/ChemtabUQ/CT_logs_Mu/mean_regressors/model-${SLURM_JOB_ID}.ckpt $lightning_CLI_args #--trainer.default_root_dir=CT_logs_Sigma 
	mkdir UQ_regressors
    mv model.ckpt UQ_regressors/model-${SLURM_JOB_ID}.ckpt
	cd -
else
	rm -r UQ_regressor
fi

# Example Commands for training of mean_regressor and UQ model
#srun --ntasks-per-node=2 python ChemtabUQ.py fit --data.class_path=MeanRegressorDataModule --data.data_fn=../data/chrest_contiguous_group_sample100k.csv --trainer.accelerator=gpu --trainer.devices=2 --trainer.num_nodes=2
#srun --ntasks-per-node=2 python ChemtabUQ.py fit --data.class_path=UQRegressorDataModule --data.data_fn=../data/chrest_contiguous_group_sample100k.csv --data.mean_regressor_fn=mean_regressor/model.ckpt --trainer.accelerator=gpu --trainer.devices=2 --trainer.num_nodes=2

## Example LR & BatchSize tune command: output is lr=0.0004365158322401656, batch_size=40
# python ChemtabUQ.py tune --data.class_path=MeanRegressorDataModule --data.data_fn=../data/Chemtab_data_MassR2.csv.gz --data.inputs_like=mass_CPV --data.outputs_like=source_CPV --data.scale_output True --trainer.accelerator=gpu --trainer.auto_lr_find True --trainer.auto_scale_batch_size power
## Example LR tune command: output, lr=7.585775750291837e-08
# srun --ntasks-per-node=2 python ChemtabUQ.py tune --data.class_path=MeanRegressorDataModule --data.data_fn=../data/chrest_contiguous_group_sample100k.csv --trainer.accelerator=gpu --trainer.devices=2 --trainer.num_nodes=2 --trainer.auto_lr_find True
## Example BatchSize tuner command: output, batch_size=27310
# python ChemtabUQ.py tune --data.class_path=MeanRegressorDataModule --data.data_fn=../data/chrest_contiguous_group_sample.csv --trainer.accelerator=gpu --trainer.auto_scale_batch_size power
