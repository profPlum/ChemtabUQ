
import os
from typing import Any, Optional
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchmetrics.functional as F_metrics
from torchmetrics.regression import MeanAbsolutePercentageError

import pytorch_lightning as pl
import pytorch_lightning.cli
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pytorch_lightning.callbacks import DeviceStatsMonitor, EarlyStopping
from pytorch_lightning.utilities import grad_norm # for grad-norm tracking

# TODO: put inside prepare method for LigthningDataModule! (should only happen once then result saved to disk)
class UQMomentsDataset(Dataset):
    def __init__(self, csv_fn: str, inputs_like='Yi', outputs_like='souspec', group_key='group',
                 sort_key='time', scale_output=False, **kwd_args):
        """
        Generic UQ Dataset which uses split-apply-combine on group_key to Produce UQ moments (mean & variance)
        :param csv_fn: name of csv file containing chrest data
        :param inputs_like: matches (glob) *inputs_like* columns from csv for input (e.g. 'Yi')
        :param outputs_like: matches (glob) *outputs_like* columns from csv for output (e.g. 'souspec')
        :param group_key: the key to group on to get distributions' moments
        :param sort_key: the key to sort the dataset on, can be set to None to disable sorting. 
        (default is time so if data_split_seed=None then we train on all transcience)
        :param scale_output: whether to scale the data for the model (i.e. using standard scaler)
        """
        df = pd.read_csv(csv_fn)
        print('original df len: ', len(df))

        if sort_key:
            df=df.sort_values(by=sort_key)
        if not group_key:
            group_key='group'
            df['group']=df.index
            print('no grouping!')

        def filter_and_scale(like, scale: bool):
            scaler = None
            subset_df = df.filter(like=like)
            if scale:
                scaler = StandardScaler()
                print(f'old columns: {subset_df.columns}')
                subset_df = pd.DataFrame(scaler.fit_transform(subset_df), index=subset_df.index,
                                         columns=subset_df.columns)
                print(f'new columns: {subset_df.columns}')
            subset_df.index = df[group_key]
            return subset_df, scaler

        inputs_df, self.input_scaler = filter_and_scale(inputs_like, scale=False)
        outs_df, self.output_scaler = filter_and_scale(outputs_like, scale=scale_output)
        assert scale_output ^ (self.output_scaler is None) # sanity check 

        self.df_mu = th.Tensor(inputs_df.groupby(group_key).mean().values).detach()
        self.df_sigma = th.Tensor(inputs_df.groupby(group_key).std(ddof=0).values).detach()
        self.outs_df = th.Tensor(outs_df.groupby(group_key).mean().values).detach()

        self.input_col_names = inputs_df.columns
        self.output_col_names = outs_df.columns

        print('reduced df len: ', self.df_sigma.shape[0])
        assert self.df_sigma.shape[0]>1000, 'did you use too few groups?'

    def to(self, device):
        self.df_mu=self.df_mu.to(device)
        self.df_sigma=self.df_sigma.to(device)
        self.outs_df=self.outs_df.to(device)
        return self

    def __len__(self):
        return self.df_mu.shape[0]

    def __getitem__(self, idx):
        #inputs = th.randn((self.df_mu.shape[1]),)*self.df_sigma[idx,:] + self.df_mu[idx,:]
        outputs = self.outs_df[idx,:]
        return (self.df_mu[idx,:], self.df_sigma[idx,:]), outputs

class UQSamplesDataset(Dataset):
    """ Wrapper for UQMomentsDataset which produces samples from the corresponding moments """
    def __init__(self, moments_dataset, constant=False):
        self.moments_dataset = moments_dataset
        self.rand_coef = (int)(not constant)

    def sample(self, mu, sigma):
        return mu + th.randn(*mu.shape).to(mu.device)*sigma*self.rand_coef

    def __len__(self):
        return len(self.moments_dataset)

    def __getitem__(self, idx):
        (mu, sigma), outputs = self.moments_dataset[idx]
        inputs = self.sample(mu, sigma)
        return inputs, outputs

# NOTE: this takes MULTIPLE samples per distribution then get the SE for the entire distribution & save that as training target!
# you can do this partially by using split-apply-combine with pandas
class UQErrorPredictionDataset(Dataset):
    def __init__(self, target_model: nn.Module, moments_dataset: UQMomentsDataset, samples_per_distribution=50):
        self.target_model = target_model
        self.moments_dataset = moments_dataset
        self.sampling_dataset = UQSamplesDataset(moments_dataset)

        # NOTE: idea is for each UQ distribution we sample n=samples_per_distribution times
        # then we derive SE from accumulated SSE. This is better than MAE because we can assume
        # that errors are i.i.d which lets us compute total uncertainty during demo 
        # P.S. this slick addition method lets us avoid using groupby! groups are implicitly positions!

        self.SE_model = 0 # dummy value to be replaced by matrix
        for i in range(samples_per_distribution):
            input_samples, outputs = self.sampling_dataset[:]
            preds = self.target_model(input_samples)
            self.SE_model = self.SE_model + ((preds-outputs)**2).detach()
            # accumulate SSE, NOTE: VAR(X+Y)=VAR(X)+VAR(Y) | X indep Y
    
        self.SE_model /= samples_per_distribution # derive MSE
        self.SE_model = self.SE_model**(1/2) # MSE --> Standard Error

    def __len__(self):
        return len(self.moments_dataset)
    
    def __getitem__(self, index):
        (mu, sigma), outputs = self.moments_dataset[index]
        return  th.cat((mu, sigma), axis=-1), self.SE_model[index]

class FFRegressor(pl.LightningModule):
    def __init__(self, input_size: int, output_size: int=None, hidden_size: int=100,
                 n_layers: int=8, learning_rate: float=0.0001445439770745928, lr_coef: float=1.0,
                 MAPE_loss: bool=False, sMAPE_loss: bool=False, MSE_loss: bool=False, SELU: bool = True,
                 reduce_lr_on_plateu_shedule: bool=False, RLoP_patience=100, RLoP_cooldown=20, RLoP_factor=0.9, 
                 cosine_annealing_lr_schedule: bool=False, cos_T_0: int=60, cos_T_mult: int=1):
        """
        Just a simple FF Network that scales

        :param input_size: NN input size
        :param output_size: NN output size
        :param hidden_size: NN hidden layer size
        :param n_layers: number of NN layers
        :param learning_rate: NN learning rate (default provided by auto_lr_finder)
        :param lr_coef: the learning_rate scaling coefficient (i.e. from larger batch size across gpus)
        :param MAPE_loss: (experimental) use MAPE as loss function
        :param sMAPE_loss: (experimental) use sMAPE as a loss function
        :param MSE_loss: use MSE as loss function
        :param SELU: uses SELU activation & initialization for normalized acivations (better than batch norm)
        """
        super().__init__()

        self.save_hyperparameters() # save hyper-params to TB logs for better analysis later! 

        learning_rate *= lr_coef; del lr_coef
        if not output_size: output_size = input_size

        self.loss = F.l1_loss # MAE is default loss
        if MSE_loss: self.loss = F.mse_loss 
        elif MAPE_loss: self.loss=F_metrics.mean_absolute_percentage_error #MeanAbsolutePercentageError()
        elif sMAPE_loss: self.loss=F_metrics.symmetric_mean_absolute_percentage_error
        assert MAE_loss + MAPE_loss + sMAPE_loss <= 1 # all loss flags are mutually exclusive 

        vars(self).update(locals()); del self.self
        self.example_input_array=th.randn(16, self.input_size)

        #hidden_size = input_size*4
        activation=nn.SELU if SELU else nn.ReLU 
        bulk_layers = []
        for i in range(self.n_layers-1): # this should be safer then potentially copying layers by reference...
            bulk_layers.extend([activation(), nn.Linear(hidden_size,hidden_size)])
        # IMPORTANT: don't include any batchnorm layers! They break ONNX & are redundant with selu anyways...
        self.regressor = nn.Sequential(nn.Linear(input_size,hidden_size),*bulk_layers, nn.Linear(hidden_size, output_size))
        # last layer is just to change size, doesn't count as a "layer" since it's linear
   
        if SELU: 
            ## NOTE: docs specifically instruct to use nonlinearity='linear' for original SNN implementation
            # SNN_gain=torch.nn.init.calculate_gain(nonlinearity='linear', param=None)
            # it just so happens this gives gain=1, which is default hence no function call
            def init_weights_glorot_normal(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    m.bias.data.fill_(0.00) # bias=0 is simplist & common
            self.regressor.apply(init_weights_glorot_normal)

    def forward(self, inputs):
        return self.regressor(inputs)

    def configure_optimizers(self):
        min_lr = 1e-8
        assert self.learning_rate>=min_lr, 'learning rate < 1e-8 is crazy!!'
        opt = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        assert not (self.reduce_lr_on_plateu_shedule and self.cosine_annealing_lr_schedule), 'lr scheduler options are mutually exclusive!'
        if self.reduce_lr_on_plateu_shedule:
            lr_scheduler = pl.cli.ReduceLROnPlateau(opt, monitor='loss', cooldown=self.RLoP_cooldown, factor=self.RLoP_factor, patience=self.RLoP_patience, min_lr=min_lr)
            return {'optimizer': opt, 'lr_scheduler': lr_scheduler, 'monitor': 'loss'}
        elif self.cosine_annealing_lr_schedule:
            if self.cos_T_mult: lr_scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=self.cos_T_0, T_mult=self.cos_T_mult, eta_min=min_lr)
            else: lr_scheduler = th.optim.lr_scheduler.CosineAnnealing(opt, T_max=self.cos_T_0, eta_min=min_lr)
            return {'optimizer': opt, 'lr_scheduler': lr_scheduler} 
        else: return opt

    # Track grad norm, instructions from here: 
    # https://github.com/Lightning-AI/lightning/pull/16745
    def optimizer_step(self, *args, **kwd_args):
        super().optimizer_step(*args, **kwd_args)
        self.log_dict(grad_norm(self, norm_type='inf')) 
        # inspect (unscaled) gradients here

    # sync dist makes metrics more accurate (by syncing across devices), but slows down training
    def log_metrics(self, Y_pred, Y, val_metrics=False, sync_dist=True):
        """ computes/logs metrics & loss for one step """
        prefix = 'val_' if val_metrics else ''
        self.log(prefix+'MSE',  F.mse_loss(Y_pred, Y), sync_dist=sync_dist)
        self.log(prefix+'MAE', F_metrics.mean_absolute_error(Y_pred, Y), sync_dist=sync_dist)
        self.log(prefix+'R2_var_weighted', F_metrics.r2_score(Y_pred, Y, multioutput='variance_weighted'),sync_dist=sync_dist)
        self.log(prefix+'R2_avg', F_metrics.r2_score(Y_pred, Y, multioutput='uniform_average'),sync_dist=sync_dist)
        self.log(prefix+'MAPE', F_metrics.mean_absolute_percentage_error(Y_pred, Y),sync_dist=sync_dist)
        self.log(prefix+'sMAPE', F_metrics.symmetric_mean_absolute_percentage_error(Y_pred, Y),sync_dist=sync_dist)      
 
        if val_metrics: # We are now using val_R2 again b/c we found out that it is more important than MAPE
           self.log('hp_metric', F_metrics.r2_score(Y_pred, Y),sync_dist=sync_dist)
 
        loss = self.loss(Y_pred, Y)
        self.log(prefix+'loss', loss, sync_dist=sync_dist)
        return loss

    def training_step(self, training_batch, batch_id, val_metrics=False):
        X, Y = training_batch
        Y_pred = self.forward(X)
        assert not (th.isnan(X).any() or th.isnan(Y).any())
        loss = self.log_metrics(Y_pred, Y, val_metrics) 
        return loss

    # reuse training_step(), but log validation loss
    def validation_step(self, val_batch, batch_id):
        self.training_step(val_batch, batch_id, val_metrics=True)
        # with val_metrics=True it will log hp_metric too! 

# TODO: should do moments preprocessing in the prep function stage then save the file to pickle with hash based on arguments & reload later
# doubly important since we are seperating the mean regressor fitting from the UQ model fitting!
# Lol I just crammed the previous functions into this class... I think it should work fine though?
class UQ_DataModule(pl.LightningDataModule):
    def __init__(self, dataset: Dataset, batch_size: int=10000, train_portion: float=0.8, 
                 data_workers: int=4, split_seed: int=None, **kwd_args):
        """
        UQ data module (or mean regressor data module)
        :param dataset: this is the dataset you want to fit your "UQ" model to (can also be mean regressor)
        :param batch_size: the batch size for training & validation (default set by auto_batch_size_finder)
        :param train_portion: portion of dataset to be trained on
        :param data_workers: number of paralell workers to load the dataset per GPU
        :param split_seed: the seed to be used for dataset splitting, encouraged to pass a random number for diversity.
            But the default is None, which means that we just take the last X% of the dataset (sorted by time).
        """
        super().__init__()
        self.save_hyperparameters(ignore=['dataset'])

        vars(self).update(locals()); del self.self # gotcha to make trick work
        self.prepare_data_per_node=False
    
    def setup(self, stage=None): # simple version 
        assert float('.'.join(th.__version__.split('.')[:2]))>=1.13, 'torch.__version__ must be >= 1.13.0 in order to use random_split portions feature!'        
        if self.split_seed:
            fixed_split_seed = th.Generator().manual_seed(self.split_seed) # IMPORTANT: must be fixed so that this works across processes
            train, val = random_split(self.dataset, [self.train_portion, 1-self.train_portion], generator=fixed_split_seed) # requires torch version >= 1.13.0!
        else: # if split_seed is None then we act like keras and just get last X% of the data for validation (useful for time sorted split)
            train_len = int(len(self.dataset)*self.train_portion)
            train, val = self.dataset[:train_len], self.dataset[train_len:]
        assert len(train) % self.batch_size < len(self.dataset)//10, f'Batch size is too inefficient! It will require dropping {len(train) % self.batch_size} samples.'

        # IMPORTANT: drop_last is needed b/c it prevents unstable training at large batch sizes (e.g. batch_size=100k w/ trunc batch size 40)
        # NOTE: drop last here actually isn't so bad b/c the dataset is shuffled meaning that although an epoch doesn't cover everything, 2 epochs should!!
        self.train_loader = DataLoader(train, batch_size=self.batch_size, num_workers=self.data_workers, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val, batch_size=min(self.batch_size, len(val)), num_workers=self.data_workers)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_loader

################### Specialized Data Modules for Mean & UQ Regressors: ###################

class MeanRegressorDataModule(UQ_DataModule):
    def __init__(self, data_fn: str, constant=True, **kwargs):
        """
        This is the dataset used for fitting a mean regressor model (i.e. dummy chemtab model)
        :param data_fn: grouped chrest csv for getting moments
        :param constant: whether to keep the (uncertain) inputs constant (i.e. use only mean), I believe constant is a good idea
        """
        moments_dataset = UQMomentsDataset(data_fn, **kwargs)
        regressor_dataset = UQSamplesDataset(moments_dataset, constant=constant)
        super().__init__(regressor_dataset, **kwargs)

def load_mean_regressor_factory(model_fn, cols):
    """ load model factory (originally intended for mean regressor) """
    if model_fn.endswith('.ckpt'):
        model = FFRegressor.load_from_checkpoint(model_fn, input_size=len(cols))#.to('cuda')
    else:
        import TF2PL_chemtab_wrapper
        model = TF2PL_chemtab_wrapper.wrap_mean_regressor(model_fn)
        TF2PL_chemtab_wrapper.check_Yi_consistency(cols)
    return model

class UQRegressorDataModule(UQ_DataModule):
    def __init__(self, data_fn: str, mean_regressor_fn: str, **kwargs):
        """
        This is the dataset used for fitting a UQ model (i.e. 2nd moment aka SE regressor)
        :param data_fn: grouped chrest csv for getting moments
        :param constant: whether to keep the (uncertain) inputs constant (i.e. use only mean), I believe constant is a good idea
        """
        moments_dataset = UQMomentsDataset(data_fn, **kwargs)
        (mu, sigma), outs = random.choice(moments_dataset)

        mean_regressor = load_mean_regressor_factory(mean_regressor_fn, moments_dataset.input_col_names)
        regressor_dataset = UQErrorPredictionDataset(mean_regressor, moments_dataset)
        super().__init__(regressor_dataset, **kwargs)

#########################################################################

# NOTE: if you want to juggle between pytorch v2.0 and v<2.0 (e.g. for legacy or A100 GPUs) then you just need to juggle between pytorch_distributed_cuda3 (for legacy) and pytorch_distributed (for A100s)
class MyLightningCLI(pl.cli.LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.set_defaults({'trainer.num_nodes': 1, 'trainer.devices': 1, # we want lr_coef to work properly! 
        'trainer.gradient_clip_algorithm': 'value', 'trainer.gradient_clip_val': 0.5}) # turn on grad clipping comparable to Ada-Sum (4 parallel)
        parser.link_arguments(['trainer.devices', 'trainer.num_nodes'], 'model.lr_coef', apply_on='parse', compute_fn=lambda devices, num_nodes: int(num_nodes)*int(devices))
        parser.link_arguments(['data.dataset'], 'model.input_size', compute_fn=lambda dataset: next(iter(dataset))[0].shape[0], apply_on='instantiate') # holyshit this works!
        parser.link_arguments(['data.dataset'], 'model.output_size', compute_fn=lambda dataset: next(iter(dataset))[1].shape[0], apply_on='instantiate')
        #get_shape = lambda ds, output=False: next(iter(dataset))[int(output)].shape[0] # shape 0 size batching is not applied yet, it is a 1d vector...    
        #parser.link_arguments(['data.dataset'], 'model.input_size', apply_on='instantiate', compute_fn=lambda ds: get_shape(ds, output=False)) # holyshit this works!
        #parser.link_arguments(['data.dataset'], 'model.output_size', apply_on='instantiate', compute_fn=lambda ds: get_shape(ds, output=True))

# TODO: fix me, currently doesn't work: now new hyper-params are recorded this way!
# NOTE: logs all CLI args (e.g. gradient clipping) as hyper-params for tensorboard logger
class LoggerSaveConfigCallback(SaveConfigCallback):
    """ from PytorchLightning website as recommended for recording the 'config' as 'hyperparamters' for the logger """
    def save_config(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
            trainer.logger.log_hyperparams({"config": config})

# HINT, try this: python ChemtabUQ.py fit --data.help MeanRegressorDataModule !!
# ^ Gives you great overview of possible CLI args to the data module class for training more general Chemtab mean models
# Example Usage: srun --ntasks-per-node=2 python ChemtabUQ.py fit --data.class_path=MeanRegressorDataModule 
# --data.data_fn=../data/chrest_contiguous_group_sample100k.csv --trainer.accelerator=gpu --trainer.devices=2 --trainer.num_nodes=2
def cli_main():
    cli=MyLightningCLI(FFRegressor, UQ_DataModule, subclass_mode_data=True, #save_config_callback=LoggerSaveConfigCallback, 
        save_config_kwargs={"overwrite": True})
    cli.trainer.save_checkpoint("model.ckpt")

if __name__=='__main__':
    cli_main()
