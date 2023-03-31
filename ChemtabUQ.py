import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torchmetrics.functional as F_metrics


class UQMomentsDataset(Dataset):
	""" Generic UQ Dataset which uses split-apply-combine on group_key to Produce UQ moments (mean & variance)"""
	def __init__(self, csv_fn, inputs_like, outputs_like, group_key):
		df = pd.read_csv(csv_fn)
		print('original df len: ', len(df))

		valid_group_keys = df.groupby(group_key).std().dropna().index
		mask = np.isin(df[group_key], valid_group_keys)
		outs_df = inputs_df = df = df[mask] # mask df to remove all group keys with only 1 element (which gives nan std)
		df.index=df[group_key]
		print('masked df len: ', len(df))

		if inputs_like: inputs_df = df.filter(like=inputs_like)
		if outputs_like: outs_df = df.filter(like=outputs_like)
		self.df_mu = th.Tensor(inputs_df.groupby(group_key).mean().values)
		self.df_sigma = th.Tensor(inputs_df.groupby(group_key).std().values)
		self.outs_df = th.Tensor(outs_df.groupby(group_key).mean().values)

		print('reduced df len: ', self.df_sigma.shape[0])

	def __len__(self):
		return self.df_mu.shape[0]

	def __getitem__(self, idx):
		#inputs = th.randn((self.df_mu.shape[1]),)*self.df_sigma[idx,:] + self.df_mu[idx,:]
		outputs = self.outs_df[idx,:]
		return (self.df_mu[idx,:], self.df_sigma[idx,:]), outputs


class UQSamplesDataset(Dataset):
	""" Wrapper for UQMomentsDataset which produces samples from the corresponding moments """
	def __init__(self, moments_dataset):
		self.moments_dataset = moments_dataset

	def __len__(self):
		return len(self.moments_dataset)

	def __getitem__(self, idx):
		(mu, sigma), outputs = self.moments_dataset[idx]
		inputs = th.randn(*mu.shape)*sigma + mu
		return inputs, outputs

class UQErrorPredictionDataset(Dataset):
	def __init__(self, target_model, moments_dataset):
		self.target_model = target_model
		self.moments_dataset = moments_dataset
		self.sampling_dataset = UQSamplesDataset(moments_dataset)

		input_samples, outputs = self.sampling_dataset[:]
		preds = self.target_model(input_samples)
		self.abs_err_model = th.abs(preds-outputs).detach()

	def __len__(self):
		return len(self.moments_dataset)
	
	def __getitem__(self, index):
		(mu, sigma), outputs = self.moments_dataset[index]
		#input_samples, outputs = self.sampling_dataset[index]
		#preds = self.target_model(input_samples.view(1,-1))
		#abs_err = th.abs(preds-outputs)
		return	th.cat((mu, sigma), axis=-1), self.abs_err_model[index]

from pytorch_lightning.callbacks import DeviceStatsMonitor
#trainer = Trainer(callbacks=[DeviceStatsMonitor()])
# ^ example usage

class UQModel(pl.LightningModule):
	def __init__(self, input_size=53, output_size=None, n_layers=8, lr=1e-8, device_stats_monitor=False):
		super().__init__()
		if not output_size: output_size = input_size
	
		hidden_size = input_size*4	
		bulk_layers = []
		for i in range(n_layers-1): # this should be safer then potentially copying layers by reference...
			bulk_layers.extend([nn.SELU(), nn.Linear(hidden_size,hidden_size)])
		self.regressor = nn.Sequential(nn.BatchNorm1d(input_size),nn.Linear(input_size,hidden_size),*bulk_layers, nn.Linear(hidden_size, output_size)) 
		# last layer is just to change size, doesn't count as a "layer" since it's linear
		self.lr=lr
		self.device_stats_monitor=device_stats_monitor
		# whether to profile GPU usage
		
	def forward(self, inputs):
		return self.regressor(inputs)

	def configure_optimizers(self):
		return th.optim.Adam(self.parameters(), lr=self.lr)

	def configure_callbacks(self):
		"""We want to log accelerator usage statistics for profiling,
		& only way to do this with CLI is to use this hook"""
		return [DeviceStatsMonitor()] if self.device_stats_monitor else []

	def log_metrics(self, Y_pred, Y, prefix=''):
		loss = F.mse_loss(Y_pred, Y)
		self.log(prefix+'mse_loss', loss, sync_dist=True)
		self.log(prefix+'R2_var_weighted', F_metrics.r2_score(Y_pred, Y, multioutput='variance_weighted'), sync_dist=True)
		self.log(prefix+'R2_avg', F_metrics.r2_score(Y_pred, Y, multioutput='uniform_average'),sync_dist=True)
		self.log(prefix+'MAPE', F_metrics.mean_absolute_percentage_error(Y_pred, Y), sync_dist=True)	

	def training_step(self, training_batch, batch_id, log_prefix=''):
		X, Y = training_batch
		Y_pred = self.forward(X)
		assert not (th.isnan(X).any() or th.isnan(Y).any())
		loss = F.mse_loss(Y_pred, Y)
		self.log_metrics(Y_pred, Y, log_prefix)	

	# reuse training_step(), but log validation loss
	def validation_step(self, val_batch, batch_id):
		self.training_step(val_batch, batch_id, log_prefix='val_')
		

import os

def _get_split_sizes(full_dataset: Dataset, train_portion) -> tuple:
	len_full = len(full_dataset)
	len_train = int(train_portion*len_full)
	len_val = len_full - len_train
	return len_train, len_val

def make_data_loaders(dataset, batch_size=64, train_portion=0.8, workers=8, **kwd_args):
	train, val = random_split(dataset, _get_split_sizes(dataset, train_portion))#[train_portion, 1-train_portion])
	get_batch_size = lambda df: len(df) if batch_size is None else batch_size
	train_loader = DataLoader(train, batch_size=get_batch_size(train), num_workers=workers, shuffle=True)
	val_loader = DataLoader(val, batch_size=get_batch_size(val), num_workers=workers)
	return train_loader, val_loader

def fit_UQ_model(dataset, name, trainer, **kwd_args):
	train_loader, val_loader = make_data_loaders(dataset, **kwd_args)
	x,y=next(iter(val_loader)) # hack to find dataset input size!
	mean_regressor = UQModel(input_size=x.shape[1], output_size=y.shape[1], device_stats_monitor=kwd_args['device_stats_monitor'])
	trainer.fit(mean_regressor, train_loader, val_loader)
	with open(f'{name}.pt', 'wb') as f:
		th.save(mean_regressor, f)
	print(f'done fitting {name}!')
	return mean_regressor

from argparse import ArgumentParser

if __name__=='__main__':
	parser = ArgumentParser()
	parser = pl.Trainer.add_argparse_args(parser)
	parser.add_argument('--device-stats-monitor', action='store_true', 
		help='Manually added CLI arg to support configuration of DeviceStatsMonitor() profiling (i.e. measuring utilization of GPUs). Turn on for more thorough profiling/troubleshooting.')
	parser.add_argument('--dataset', default='chrest_rand10000.csv', help='dataset file name (should be inside the ~/data folder)')
	parser.add_argument('--batch_size', default=500, help='total batch_size for the entire training process (i.e. across nodes), default (None) is entire dataset!')  
	args = parser.parse_args()
	
	##################### Fit Mean Regressor: #####################
	df_fn = f'{os.environ["HOME"]}/data/{args.dataset}'
	moments_dataset = UQMomentsDataset(df_fn, inputs_like='Yi', outputs_like='souspec', group_key='group')
	samples_dataset = UQSamplesDataset(moments_dataset)

	# TODO: use pl.LightningDataModule, see this url: https://lightning.ai/docs/pytorch/stable/data/datamodule.html 
	# TODO: make batchs size bigger or dynamic...
	train_settings = {'batch_size': args.batch_size, 'workers': 4, 'device_stats_monitor': args.device_stats_monitor}#, 'max_epochs': 100000, 'accelerator': 'horovod', 'gpus': 2} 
	trainer = pl.Trainer.from_argparse_args(args)

	mean_regressor = fit_UQ_model(samples_dataset, 'mean_regressor', trainer=trainer, **train_settings)
	#########################################################################
	
	##################### Fit Standard Deviation Regressor: #####################
	STD_dataset = UQErrorPredictionDataset(mean_regressor, moments_dataset)
	mean_regressor = fit_UQ_model(STD_dataset, 'std_regressor', trainer=trainer, **train_settings)
	#########################################################################
