#!/bin/bash
#
#SBATCH --output=slurm_logs/R-%x.%j.out
#SBATCH --error=slurm_logs/R-%x.%j.err
#SBATCH --mail-user=dwyerdei@buffalo.edu
#SBATCH --mail-type=end
#SBATCH --cluster=ub-hpc
#SBATCH --threads-per-core=1  # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=5     # number of CPUs per process (we want extra for data loaders)
#SBATCH --gpus-per-node=1     # We need to request it "per-node" because pytorch needs visibility of all node's GPUs for some reason...
#SBATCH --constraint="V100"   # V100s are best supported and A100s would be overkill for 1d
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...), Also: it seems that GPU nodes have only 2 GPUs per node anyways...
#SBATCH --mem=25G
#SBATCH --nodes=1 # This needs to match Trainer(nodes=...)
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --time=01:00:00 # TODO: add more time?
#SBATCH --job-name="LR_finder_off" # sort by information entropy, e.g. CT tells us nothing, MAPE or Big tells us a lot

# #SBATCH --constraint="V100"
# #SBATCH --constraint="A100"
# #SBATCH --gpus-per-node=2     # We need to request it "per-node" because pytorch needs visibility of all node's GPUs for some reason...
# #SBATCH --signal=SIGUSR1@90

## example command to tune lr & batch size 
#python ChemtabUQ.py tune --data.class_path=MeanRegressorDataModule --data.data_fn=~/data/TChem+CPVs+Zmix_MassR2.csv.gz --data.inputs_like=mass_CPV --data.outputs_like=source_CPV --data.scale_output True --trainer.accelerator=gpu --trainer.auto_lr_find True --trainer.auto_scale_batch_size power

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

conda activate pytorch_distributed_cuda3 
for i in {1..100}; do
    srun -n 1 python ChemtabUQ.py tune --data.class_path=MeanRegressorDataModule --data.data_fn=~/data/TChem+CPVs+Zmix_MassR2.csv.gz --data.inputs_like=$INPUTS --data.outputs_like=$OUTPUTS --data.scale_output True --trainer.accelerator=gpu --trainer.auto_lr_find True --data.batch_size=100000 --lr_scheduler pytorch_lightning.cli.ReduceLROnPlateau --lr_scheduler.monitor=epoch --data.group_key=null
done
