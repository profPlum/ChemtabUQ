#!/bin/bash
#
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --mail-user=dwyerdei@buffalo.edu
#SBATCH --mail-type=end
#SBATCH --cluster=ub-hpc
#SBATCH --threads-per-core=1  # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=5     # number of CPUs per process (we want extra for data loaders)
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...), Also: it seems that GPU nodes have only 2 GPUs per node anyways...
#SBATCH --gpus-per-node=2     # We need to request it "per-node" because pytorch needs visibility of all node's GPUs for some reason...
#SBATCH --mem=50G
#SBATCH --nodes=1 # This needs to match Trainer(nodes=...)
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --time=04:00:00 # TODO: add more time?
#SBATCH --job-name="BIG-SousCPV" # sort by information entropy, e.g. CT tells us nothing, MAPE or Big tells us a lot

# #SBATCH --partition=debug
# #SBATCH --qos=debug
# #SBATCH --nodes=1 # This needs to match Trainer(nodes=...)
# #SBATCH --time=00:05:00 # TODO: add more time?
# #SBATCH --job-name="Debug-PCA-CT" # sort by information entropy, e.g. CT tells us nothing, MAPE or Big tells us a lot

# #SBATCH --gres=gpu:V100:2    # A100's require version of PL > v1.9.0 (e.g. pytorch_distributed_cuda3) 
# #SBATCH --signal=SIGUSR1@90

## example command to tune lr & batch size 
#python ChemtabUQ.py tune --data.class_path=MeanRegressorDataModule --data.data_fn=../data/TChem+CPV_MassR2.csv.gz --data.inputs_like=mass_CPV --data.outputs_like=source_CPV --data.scale_output True --trainer.accelerator=gpu --trainer.auto_lr_find True --trainer.auto_scale_batch_size power

#####################################################
# These are the settings that when changed make other experiments
# 'incomparable' & thus requires new experiment name
#####################################################
INPUTS=mass_CPV
#OUTPUTS=Yi
OUTPUTS=source_CPV
#OUTPUTS=souener
LOSS= #'--model.MAPE_loss True'
RESUME=F # Tell job stub to load last ckpts from this experiment
#####################################################

# IMPORTANT: use --data.group_key=null to ensure uncoursened grid, this avoids trivial case of 50 data!
# IMPORTANT: currently relative --data.data_fn paths aren't supported b/c job stub cd's before launching script!
EXTRA_PL_ARGS="$EXTRA_PL_ARGS --data.data_fn=~/data/TChem+CPV_MassR2.csv.gz --data.inputs_like=$INPUTS --data.outputs_like=$OUTPUTS --data.scale_output True --data.group_key=null --trainer.benchmark=True"

# NOTE: LR finder found: --model.learning_rate 0.0001445439770745928 (big batch=200k)
# NOTE: LR finder found: --model.learning_rate 7.585775750291837e-08 (big batch=100k) <-- anomoly!
# NOTE: LR finder found: --model.learning_rate 4.786300923226383e-07 (batch=20000)
TRAIN_CFG="--trainer.max_epochs -1 $LOSS --model.SELU True --model.hidden_size=2000 --data.batch_size=100000 --lr_scheduler pytorch_lightning.cli.ReduceLROnPlateau --lr_scheduler.monitor=epoch --model.learning_rate 4.786300923226383e-07" # --trainer.gradient_clip_val 0.25"
MEAN_ONLY=T # Turn off UQ for now

EXTRA_PL_ARGS="$EXTRA_PL_ARGS $TRAIN_CFG" #--ckpt_path=last" # last isn't supported, it appears it doesn't work 

