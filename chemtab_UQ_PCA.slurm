#!/bin/bash
#
#SBATCH --output=slurm_logs/R-%x.%j.out
#SBATCH --error=slurm_logs/R-%x.%j.err
#SBATCH --mail-user=dwyerdei@buffalo.edu
#SBATCH --mail-type=end
#SBATCH --cluster=ub-hpc
#SBATCH --threads-per-core=1  # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=5     # number of CPUs per process (we want extra for data loaders)
#SBATCH --gpus-per-node=2     # We need to request it "per-node" because pytorch needs visibility of all node's GPUs for some reason...
#SBATCH --constraint="V100|P100"   # V100s are best supported and A100s would be overkill for 1d
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...), Also: it seems that GPU nodes have only 2 GPUs per node anyways...
#SBATCH --mem=25G
#SBATCH --nodes=1 # This needs to match Trainer(nodes=...)
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --time=6:00:00 # TODO: add more time?
#SBATCH --job-name="Best-SousCPV" # sort by information entropy, e.g. CT tells us nothing, MAPE or Souener tells us a lot

# #SBATCH --constraint="V100|P100"
# #SBATCH --constraint="A100|H100"
# #SBATCH --gpus-per-node=2     # We need to request it "per-node" because pytorch needs visibility of all node's GPUs for some reason...
# #SBATCH --signal=SIGUSR1@90

## example command to tune lr & batch size 
#python ChemtabUQ.py tune --data.class_path=MeanRegressorDataModule --data.data_fn=~/data/TChem+CPVs+Zmix_MassR2.csv.gz --data.inputs_like=mass_CPV --data.outputs_like=source_CPV_PC --data.scale_output True --trainer.accelerator=gpu --trainer.auto_lr_find True --trainer.auto_scale_batch_size power

#####################################################
# These are the settings that when changed make other experiments
# 'incomparable' & thus requires new experiment name
#####################################################
INPUTS=mass_CPV # we OMIT the _PC postfix b/c we want to include zmix in inputs
#OUTPUTS=Yi
OUTPUTS=source_CPV_PC # we INCLUDE the _PC postflix b/c zmix_source=0 by constraints
#OUTPUTS=souener
#LOSS='--model.MAPE_loss True'
RESUME=F # Tell job stub to load last ckpts from this experiment
#####################################################

# IMPORTANT: use --data.group_key=null to ensure uncoursened grid, this avoids trivial case of 50 data!
# IMPORTANT: currently relative --data.data_fn paths aren't supported b/c job stub cd's before launching script!
EXTRA_PL_ARGS="$EXTRA_PL_ARGS --data.data_fn=~/data/TChem+CPVs+Zmix_MassR2.csv.gz --data.inputs_like=$INPUTS --data.outputs_like=$OUTPUTS --data.scale_output True --data.group_key=null --trainer.benchmark=True --trainer.max_epochs -1 $LOSS"
# this set of args are relatively constant across experiments or they take args specified elsewhere

# NOTE: LR finder found: --model.learning_rate 0.0001445439770745928 (big batch=200k)
# NOTE: LR finder found: --model.learning_rate 5.608e-06 (big batch=100k, median over 20+ trials)
# NOTE: LR finder found: --model.learning_rate 0.0002596525 (big batch=100k, mean over 20+ trials)
# NOTE: LR finder found: --model.learning_rate 4.786300923226383e-07 (batch=20000)
TRAIN_CFG="--model.hidden_size=500 --data.batch_size=10000 --model.SELU True --model.hidden_size=500 --model.reduce_lr_on_plateu_shedule=True --model.learning_rate 0.00014454 --trainer.gradient_clip_val 0.25"
MEAN_ONLY=T # Turn off UQ for now

#IMPORTANT: "best config"
#TRAIN_CFG="--trainer.max_epochs -1 $LOSS --model.SELU True --model.hidden_size=500 --data.batch_size=10000 --lr_scheduler pytorch_lightning.cli.ReduceLROnPlateau --lr_scheduler.monitor=epoch --model.learning_rate 0.00014454 --trainer.gradient_clip_val 0.25"

EXTRA_PL_ARGS="$EXTRA_PL_ARGS $TRAIN_CFG" #--ckpt_path=last" # last isn't supported, it appears it doesn't work 
. chemtab_UQ_job_stub.sh
