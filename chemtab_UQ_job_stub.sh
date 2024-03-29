#!/bin/bash

# NOTE: this is a dynamic SH stub which can be reused for debug or preduction versions of chemtab UQ experiments!
# Example USAGE: EXTRA_PL_ARGS='--data.data_fn=../data/chrest_contiguous_group_sample100k.csv --data.batch_size 4500 --trainer.fast_dev_run True' ./chemtab_UQ-debug.slurm
# HINT, try this: python ChemtabUQ.py fit --data.help MeanRegressorDataModule !! Gives you great overview of possible CLI args to the data module class for training more general Chemtab mean models

# actual GPU detection done below!
srun nvidia-smi

# --nodes cannot be set dynamically!! (in sheebang)
# but it can be detected dynammically using srun magic!
# logic is: prints all hostnames, then sorts, then groups & counts group sizes, then counts number of groups
num_nodes=$(srun hostname | sort | uniq -c | wc -l)
echo num nodes: $num_nodes

# let the user know if their EXTRA_PL_ARGS went through or not
[ -z "$EXTRA_PL_ARGS" ] && echo EXTRA_PL_ARGS is empty!! >&2
# NOTE: EXTRA_PL_ARGS should be an environment variable given by the user when submitting sbatch!!

# IMPORTANT: gradient_clip defaults set to --trainer.gradient_clip_algorithm=value --trainer.gradient_clip_val=0.5 (inside ChemtabUQ.py)
# NOTE: gradient_clip_value=0.5 recommended by PL docs
# NOTE: --max_epochs -1 --> means no max epochs (i.e. fit for entire allocation time)
# NOTE: --trainer.track_grad_norm 2 is now done automatically inside the model in a way that is compatible with PL v2.*

# add diagnostic cli args which should always be set & clear any existing lightning_CLI_args value
experiment_version="version_$SLURM_JOB_ID" # used later to reference the mean_regressor save path
diagnostic_CLI_args="--trainer.logger=pytorch_lightning.loggers.TensorBoardLogger --trainer.logger.save_dir=. --trainer.logger.name=$SLURM_JOB_NAME --trainer.logger.version=$experiment_version" #--trainer.logger.save_dir=ChemtabUQ_TBlogs"
diagnostic_CLI_args="$diagnostic_CLI_args --trainer.profiler simple --trainer.callbacks+=pytorch_lightning.callbacks.LearningRateMonitor --trainer.callbacks.logging_interval=epoch --trainer.callbacks+=pytorch_lightning.callbacks.DeviceStatsMonitor --trainer.callbacks+=pytorch_lightning.callbacks.ModelCheckpoint --trainer.callbacks.monitor=loss --trainer.callbacks.filename={epoch}-{loss:.4f}-{val_loss:.4f}-{val_MAPE:.4f}-{val_R2_avg:.4f}-{val_R2_var_weighted:.4f}" #--trainer.track_grad_norm 2" 
lightning_CLI_args="--trainer.num_nodes=$num_nodes --trainer.devices=2 --trainer.accelerator=gpu --trainer.strategy=ddp" # NOTE: everything will be combined later, NOT HERE

echo diagnostic_CLI_args: $diagnostic_CLI_args
echo EXTRA_PL_ARGS: $EXTRA_PL_ARGS
echo lightning_CLI_args: $lightning_CLI_args

echo RESUME: $RESUME

find_last_ckpt() {
    last_checkpoint=$(/bin/ls -t $(/bin/find "$1" -name "*.ckpt") | head -n 1)
    echo $last_checkpoint
}
if ((RESUME)); then
    # do a bunch of sanity/error checking
    if [[ $SLURM_JOB_NAME == InteractiveJob ]]; then
        echo Error! SLURM_JOB_NAME == InteractiveJob \& RESUME=T!! >&2
        echo It is not possible or wise to resume InteractiveJob, instead set SLURM_JOB_NAME manually! >&2
        return 1 || exit 1
    fi
    # NOTE: keep this even if we remove ban on any existing $EXTRA_PL_ARGS during RESUME
    if [[ "$EXTRA_PL_ARGS" =~ --data\.split_seed= ]]; then
        echo Error! You can\'t set the split seed when resuming this will break the validation split!! >&2
        return 3 || exit 3
    fi

    # TODO: remove this if/else if you think that RESUME should allow override params
    # NOTE: for resume we should exclude $diagnostic_CLI_args & $EXTRA_PL_ARGS!!
    if [[ "$EXTRA_PL_ARGS" ]]; then
        echo Error! EXTRA_PL_ARGS must be empty in order to resume properly!! >&2
        echo EXTRA_PL_ARGS=$EXTRA_PL_ARGS
        return 2 || exit 2
    else
        diagnostic_CLI_args= # we don't want to pass unnecesary arguments
    fi
 
    cd CT_logs_Mu # NOTE: this works for BOTH CT_logs_Mu & CT_logs_Sigma models b/c they have the same relative path structure
    echo SLURM_JOB_NAME: $SLURM_JOB_NAME
    last_checkpoint=$(find_last_ckpt ./$SLURM_JOB_NAME/)
    last_cfg=$(dirname $last_checkpoint)/../config.yaml # config loads hyper-parameter settings
    cd -
    echo loading last_checkpoint: $last_checkpoint \& last config: $last_cfg
    lightning_CLI_args="--ckpt_path=$last_checkpoint -c $last_cfg $lightning_CLI_args"
else
    echo We arent resuming last ckpt... >&2
fi

# combine everything, AFTER RESUME if/else (b/c it can modify diagnostic_CLI_args)
lightning_CLI_args="$EXTRA_PL_ARGS $lightning_CLI_args $diagnostic_CLI_args"

# setup env based on GPUs allocated
source /user/dwyerdei/.bash_profile
conda deactivate
# IMPORTANT: pytorch_distributed_cuda3 is the "stable version" with pytorch-lightning==1.9.0 for V100s,
# pytorch_distributed_cuda is a possible "newer version" which is capable of using the A100+ GPUs
GPU_info=$(srun nvidia-smi)
if [[ $GPU_info =~ .*(V100|P100).* ]]; then 
    echo legacy GPUs detected! activating: pytorch_distributed_cuda3
    conda activate pytorch_distributed_cuda3
else # ^ verified to work on P100 debug node, 9/24/23
    echo modern GPUs detected! activating: pytorch_distributed_cuda
    conda activate pytorch_distributed_cuda
fi

if ((TRAIN_MU)); then
    ! [[ -e CT_logs_Mu ]] && mkdir CT_logs_Mu
    cd CT_logs_Mu
    srun --ntasks-per-node=2 python ../ChemtabUQ.py fit --data.class_path=MeanRegressorDataModule $lightning_CLI_args #--trainer.default_root_dir=CT_logs_Mu
    mkdir mean_regressors 2> /dev/null
    mv model.ckpt mean_regressors/model-${SLURM_JOB_ID}.ckpt
    cd -
fi

if ((TRAIN_SIGMA)); then
    ! [[ -e CT_logs_Sigma ]] && mkdir CT_logs_Sigma
    cd CT_logs_Sigma
    #default_mean_regressor_path=~/ChemtabUQ/CT_logs_Mu/mean_regressors/model-${SLURM_JOB_ID}.ckpt
    default_mean_regressor_path="$(find_last_ckpt ~/ChemtabUQ/CT_logs_Mu/$SLURM_JOB_NAME/$experiment_version)" # can be overridden in either EXTRA_PL_ARGS or EXTRA_UQ_ARGS
    #![[ -e default_mean_regressor_path ]] default_mean_regressor_path=~/ChemtabUQ/CT_logs_Mu/mean_regressors/model-${SLURM_JOB_ID}.ckpt
    echo default_mean_regressor_path="$default_mean_regressor_path"
    srun --ntasks-per-node=2 python ../ChemtabUQ.py fit --data.class_path=UQRegressorDataModule --data.mean_regressor_fn="$default_mean_regressor_path" $lightning_CLI_args $EXTRA_UQ_ARGS #--trainer.default_root_dir=CT_logs_Sigma 
	mkdir UQ_regressors 2> /dev/null
    mv model.ckpt UQ_regressors/model-${SLURM_JOB_ID}.ckpt
    cd -
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
