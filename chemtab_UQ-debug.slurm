#!/bin/bash
#
#SBATCH --job-name="ChemTab UQ Distributed-debug"
#SBATCH --output=ChemTab_UQ_Distributed.out
#SBATCH --mail-user=dwyerdei@buffalo.edu
#SBATCH --mail-type=end
#SBATCH --cluster=ub-hpc
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --threads-per-core=1    # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=5       # number of CPUs per process (we want extra for data loaders)
#SBATCH --ntasks-per-node=2     # This needs to match Trainer(devices=...), Also: it seems that GPU nodes have only 2 GPUs per node anyways...
## #SBATCH --gres=gpu:V100:2    # We cannot use A100's with the current version of PL (v1.9.0)
#SBATCH --gpus-per-node=2       # We need to request it "per-node" because pytorch needs visibility of all node's GPUs for some reason...
#SBATCH --mem=50G
#SBATCH --nodes=2 # This needs to match Trainer(nodes=...)
## #SBATCH --time=00:30:00 # TODO: add more time?

## #SBATCH --signal=SIGUSR1@90

PL_MAX_EPOCHS='--trainer.max_epochs 500'

. chemtab_UQ_job_stub.sh
