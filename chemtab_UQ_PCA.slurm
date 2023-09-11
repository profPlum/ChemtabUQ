#!/bin/bash
#
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --mail-user=dwyerdei@buffalo.edu
#SBATCH --mail-type=end
#SBATCH --cluster=ub-hpc
#SBATCH --threads-per-core=1    # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=5       # number of CPUs per process (we want extra for data loaders)
#SBATCH --ntasks-per-node=2     # This needs to match Trainer(devices=...), Also: it seems that GPU nodes have only 2 GPUs per node anyways...
#SBATCH --gpus-per-node=2       # We need to request it "per-node" because pytorch needs visibility of all node's GPUs for some reason...
#SBATCH --mem=50G
#SBATCH --partition=general-compute #debug
#SBATCH --qos=general-compute #debug
#SBATCH --nodes=2 # This needs to match Trainer(nodes=...)
#SBATCH --time=01:30:00 # TODO: add more time?
#SBATCH --job-name="Scaled-PCA-CT" # sort by information entropy, e.g. CT tells us nothing, MAPE or Big tells us a lot

# #SBATCH --partition=debug
# #SBATCH --qos=debug
# #SBATCH --nodes=1 # This needs to match Trainer(nodes=...)
# #SBATCH --time=00:05:00 # TODO: add more time?

# #SBATCH --gres=gpu:V100:2    # A100's require version of PL > v1.9.0 (e.g. pytorch_distributed_cuda3) 
# #SBATCH --signal=SIGUSR1@90

## example command to tune lr & batch size 
#python ChemtabUQ.py tune --data.class_path=MeanRegressorDataModule --data.data_fn=../data/Chemtab_data_MassR2.csv.gz --data.inputs_like=mass_CPV --data.outputs_like=source_CPV --data.scale_output True --trainer.accelerator=gpu --trainer.auto_lr_find True --trainer.auto_scale_batch_size power

#####################################################
# These are the settings that when changed make other experiments
# 'incomparable' & thus requires new experiment name
#####################################################
#OUTPUTS=souener
OUTPUTS=source_CPV
LOSS= #'--model.MAPE_loss True'
RESUME=F # Tell job stub to load last ckpts from this experiment
#####################################################

# IMPORTANT: currently relative --data.data_fn paths aren't supported b/c job stub cd's before launching script!
EXTRA_PL_ARGS="$EXTRA_PL_ARGS --data.data_fn=~/data/TChem+CPV_MassR2.csv.gz --data.inputs_like=mass_CPV --data.outputs_like=$OUTPUTS --data.scale_output True"
TRAIN_CFG="--trainer.max_epochs -1 --model.learning_rate 0.0004365158322401656 $LOSS" # --model.hidden_size=500"
MEAN_ONLY=T # Turn off UQ for now

EXTRA_PL_ARGS="$EXTRA_PL_ARGS $TRAIN_CFG" #--ckpt_path=last" # last isn't supported, it appears it doesn't work 
. chemtab_UQ_job_stub.sh
