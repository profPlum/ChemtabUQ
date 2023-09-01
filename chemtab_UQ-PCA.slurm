#!/bin/bash
#
#SBATCH --job-name="ChemTab-PCA"
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --mail-user=dwyerdei@buffalo.edu
#SBATCH --mail-type=end
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute #debug
#SBATCH --qos=general-compute #debug
#SBATCH --threads-per-core=1    # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=5       # number of CPUs per process (we want extra for data loaders)
#SBATCH --ntasks-per-node=2     # This needs to match Trainer(devices=...), Also: it seems that GPU nodes have only 2 GPUs per node anyways...
## #SBATCH --gres=gpu:V100:2    # A100's require version of PL > v1.9.0 (e.g. pytorch_distributed_cuda3) 
#SBATCH --gpus-per-node=2       # We need to request it "per-node" because pytorch needs visibility of all node's GPUs for some reason...
#SBATCH --mem=50G
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --nodes=1 # This needs to match Trainer(nodes=...)
#SBATCH --time=00:05:00 # TODO: add more time?

#   #SBATCH --partition=general-compute #debug
#   #SBATCH --qos=general-compute #debug
#   #SBATCH --nodes=2 # This needs to match Trainer(nodes=...)
#   #SBATCH --time=01:30:00 # TODO: add more time?

## #SBATCH --signal=SIGUSR1@90

#ver=$(/bin/ls -t lightning_logs | head -n 1)
#last_checkpoint= $(/bin/ls -t ./lightning_logs/$ver/checkpoints/*.ckpt | head -n 1) 
#echo last_checkpoint: $last_checkpoint

#OUTPUTS=souener
OUTPUTS=source_CPV
EXTRA_PL_ARGS="$EXTRA_PL_ARGS --data.data_fn=../data/TChem+CPV_MassR2.csv.gz --data.inputs_like=mass_CPV --data.outputs_like=$OUTPUTS --data.scale_output True"

# # example command to tune lr & batch size 
#python ChemtabUQ.py tune --data.class_path=MeanRegressorDataModule --data.data_fn=../data/Chemtab_data_MassR2.csv.gz --data.inputs_like=mass_CPV --data.outputs_like=source_CPV --data.scale_output True --trainer.accelerator=gpu --trainer.auto_lr_find True --trainer.auto_scale_batch_size power
TRAIN_CFG='--trainer.max_epochs -1 --model.learning_rate 0.0004365158322401656' # --model.MAPE_loss True' #--model.hidden_size=500' # --data.batch_size 40'

EXTRA_PL_ARGS="$EXTRA_PL_ARGS $TRAIN_CFG" # --ckpt_path last"
MEAN_ONLY=T # Turn off UQ for now

. chemtab_UQ_job_stub.sh
