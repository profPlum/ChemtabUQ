#!/bin/bash
#
#SBATCH --output=slurm_logs/R-%x.%j.out
#SBATCH --error=slurm_logs/R-%x.%j.err
#SBATCH --cluster=ub-hpc
#SBATCH --threads-per-core=1  # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=5     # number of CPUs per process (we want extra for data loaders)
#SBATCH --gpus-per-node=2     # We need to request it "per-node" because pytorch needs visibility of all node's GPUs for some reason...
#SBATCH --constraint="V100|P100"   # V100s are best supported and A100s would be overkill for 1d
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...), Also: it seems that GPU nodes have only 2 GPUs per node anyways...
#SBATCH --mem=25G
#SBATCH --nodes=1 # This needs to match Trainer(nodes=...)
#SBATCH --partition=scavenger #general-compute
#SBATCH --qos=scavenger #general-compute
#SBATCH --time=24:00:00
#SBATCH --job-name="Methane-SS-Inv-10_CPVs" 
## sort by information entropy

# #SBATCH --constraint="V100|P100"
# #SBATCH --constraint="A100|H100"
# #SBATCH --signal=SIGUSR1@90

## example command to tune lr & batch size 
#python ChemtabUQ.py tune --data.class_path=MeanRegressorDataModule --data.data_fn=~/data/TChem+CPVs+Zmix_MassR2.csv.gz --data.inputs_like=mass_CPV --data.outputs_like=source_CPV_PC --data.scale_output True --trainer.accelerator=gpu --trainer.auto_lr_find True --trainer.auto_scale_batch_size power

source ~/.bash_aliases
regex_number_range() { x=$(seq $1 $2 | tr '\n' '|'); echo "(?<![0-9eE.-])(?:${x%%|$2|})(?![0-9eE.])"; }
export NUM_PERTURB_SPREAD=0.2 # we've already got pretty good parameters

#####################################################
# These are the settings that when changed make other experiments
# 'incomparable' & thus requires new experiment name
#####################################################

# QR datasets:
#DATA=~/data/TChem+Perturbedx2/25_CPV_data/TChem+CPVs+Zmix_QR_MassR2.csv.gz

# NO QR datasets:
#DATA=~/data/TChem+Perturbedx2/10_CPV_data/TChem+CPVs+Zmix_MassR2.csv.gz
#DATA=~/data/Identity_CPV_data/TChem+CPVs+Zmix.csv.gz
#DATA=~/data/25_CPV_data/TChem+CPVs+Zmix_MassR2.csv.gz

DATA=~/data/ablate_methane_steadystate_collated/TChem+CPVs+Zmix_MassR2.csv.gz

N_CPVS=10 # we also want to include zmix in inputs
VALID_CPV_RANGE="$(regex_number_range 0 $N_CPVS)"
INPUTS="mass_CPV_(zmix|PC_$VALID_CPV_RANGE)"
#OUTPUTS="source_CPV_PC_$VALID_CPV_RANGE" # we exclude zmix b/c zmix_source=0 by constraints
#OUTPUTS=souener
OUTPUTS=Yi
#LOSS='--model.MAPE_loss True'
#LOSS='--model.sMAPE_loss True'
#LOSS='--model.MSE_loss True'
AUTO_PERTURB_ARGS=0 # turn on for auto-hparam search!
RESUME=1 # Tell job stub to load last ckpts from this experiment
USE_TIMESORTED_SPLIT=1 # Tell job to exclude only the latest part of the simulation
#USE_BEST_SEED=1 # Tell job to use the best known split seed

#touch CT_logs_Mu/Methane-SS-Inv-10_CPVs/version_14345154 # batch_size=5000 (1st best)
#touch CT_logs_Mu/Methane-SS-Inv-10_CPVs/version_14361329 # batch_size=1752 (2nd best)
#touch CT_logs_Mu/Methane-SS-Inv-10_CPVs/version_14361326 # batch_size=876 (3rd best)

####################################################

echo INPUTS=$INPUTS # Wow it works!
echo OUTPUTS=$OUTPUTS

# IMPORTANT: use --data.group_key=null to ensure uncoursened grid, this avoids trivial case of 50 data!
# IMPORTANT: currently relative --data.data_fn paths aren't supported b/c job stub cd's before launching script
JOB_CFG="$JOB_CFG --data.data_fn=$DATA --data.inputs_like=$INPUTS --data.outputs_like=$OUTPUTS --data.sort_key=time"
JOB_CFG="$JOB_CFG --data.scale_output True --data.group_key=null --trainer.benchmark=True --trainer.max_epochs -1 $LOSS"
# this set of args are relatively constant across experiments or they take args specified elsewhere

# NOTE: LR finder found: --model.learning_rate 0.0001445439770745928 (big batch=200k)
# NOTE: LR finder found: --model.learning_rate 5.608e-06 (big batch=100k, median over 20+ trials)
# NOTE: LR finder found: --model.learning_rate 0.0002596525 (big batch=100k, mean over 20+ trials)
# ---------------------------------------------------------------------------------------------
# NOTE: LR finder found: --model.learning_rate 1.318e-07 (batch=20000, median over 200 trials)
# NOTE: LR finder found: --model.learning_rate 1.663e-05 (batch=20000, mean over 200 trials)
# NOTE: set gradient_clip_val=null to disable it! (default is 0.25)
TRAIN_CFG="--model.hidden_size=500 --data.batch_size=10000 --model.SELU True --trainer.gradient_clip_val=0.25 --model.learning_rate 0.0001445439770745928"
TRAIN_CFG="$TRAIN_CFG --model.reduce_lr_on_plateu_shedule=True --model.RLoP_patience=100 --model.RLoP_cooldown=20 --model.RLoP_factor=0.95"
#TRAIN_CFG="$TRAIN_CFG --model.cosine_annealing_lr_schedule=True --model.cos_T_0=60 --model.cos_T_mult=null" # cosine annealing + warm restarts with increasing period
#TRAIN_CFG="$TRAIN_CFG --model.cosine_annealing_lr_schedule=True --model.cos_T_0=1 --model.cos_T_mult=2" # consine annealing vanilla

#IMPORTANT: "best config"
#TRAIN_CFG="--trainer.max_epochs -1 $LOSS --model.SELU True --model.hidden_size=500 --data.batch_size=10000 -model.learning_rate 0.00014454 --trainer.gradient_clip_val 0.25"
#TRAIN_CFG="$TRAIN_CFG --model.reduce_lr_on_plateu_shedule=True --model.RLoP_patience=100 --model.RLoP_cooldown=20 --model.RLoP_factor=0.95"

((AUTO_PERTURB_ARGS)) && TRAIN_CFG=$(auto_cli_perturb "$TRAIN_CFG")
echo TRAIN_CFG: $TRAIN_CFG

# GOTCHA: Currently this is rather misleading b/c actually both 10_CPV_data & 5_CPV_data have different preshuffling 
# so the seeds are incomparable. Hopefully they generalize across souener & souscpv.
BEST_SEED_5_CPVS=5605 # epoch=22759-val_loss=0.0024-val_MAPE=0.1256-val_R2_avg=0.9735 (Souener, hidden_size=500, CPVs=5)
BEST_SEED_10_CPVS=11436 # epoch=21993-val_loss=0.0013-val_MAPE=0.0553-val_R2_avg=0.9841 (SousCPV, hidden_size=1000, CPVs=10)
BEST_SEED_25_CPVS=26415 # epoch=9688-val_loss=0.0027-val_MAPE=0.0165-val_R2_avg=0.9543 (Souener, hidden_size=1500, CPVs=25)

if [[ $DATA =~ /5_CPV_data ]]; then
    # tested once on Souener (see above)
    BEST_SEED=$BEST_SEED_5_CPVS
    echo 5_CPV_data
elif [[ $DATA =~ /10_CPV_data ]]; then
    # tested once on SousCPV (see above)
    BEST_SEED=$BEST_SEED_10_CPVS
    echo 10_CPV_data
elif [[ $DATA =~ /25_CPV_data ]]; then
    # tested once on Souener (see above)
    BEST_SEED=$BEST_SEED_25_CPVS
    echo 25_CPV_data
fi

if ((USE_BEST_SEED)); then
    # we turn off sort key for compatibility with older datasets which came preshuffled
    JOB_CFG="$JOB_CFG --data.split_seed=$BEST_SEED --data.sort_key=null"
else
    split_seed=$RANDOM
    ((USE_TIMESORTED_SPLIT)) && split_seed=null # null means last X% (sorted by time by default)
    echo adding random split seed: $split_seed
    JOB_CFG="$JOB_CFG --data.split_seed=$split_seed"
fi

#MEAN_ONLY=0 # Turn on UQ
MEAN_ONLY=1 # Turn off UQ for now
JOB_CFG="$TRAIN_CFG $JOB_CFG" #--ckpt_path=last" # last isn't supported, it appears it doesn't work 
EXTRA_PL_ARGS="$JOB_CFG $EXTRA_PL_ARGS" # inject manually specified user arguments (with priority over job cfg)
((RESUME)) && EXTRA_PL_ARGS= # if RESUME==1 we don't want to mess with existing settings

. chemtab_UQ_job_stub.sh
