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

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pytorch_lightning.callbacks import DeviceStatsMonitor

# TODO: put inside prepare method for LigthningDataModule! (should only happen once then result saved to disk)
class UQMomentsDataset(Dataset):
	def __init__(self, csv_fn: str, inputs_like: str, outputs_like: str, group_key: str, scale=False):
		"""
		Generic UQ Dataset which uses split-apply-combine on group_key to Produce UQ moments (mean & variance)
		:param csv_fn: name of csv file containing chrest data
		:param inputs_like: pattern to match input columns in csv
		:param outputs_like: pattern to match output columns in csv
		:param scale: whether to scale the data for the model (i.e. using standard scaler)
		"""
		df = pd.read_csv(csv_fn)
		print('original df len: ', len(df))

		valid_group_keys = df.groupby(group_key).std().dropna().index
		mask = np.isin(df[group_key], valid_group_keys)
		df = df[mask] # mask df to remove all group keys with only 1 element (which gives nan std)
		print('masked df len: ', len(df))

		def filter_and_scale(like):
			scaler = None
			subset_df = df.filter(like=like)
			if scale:
				scaler = StandardScaler()
				subset_df[:] = scaler.fit_transform(subset_df)
			subset_df.index = df[group_key]
			return subset_df, scaler			

		inputs_df, self.input_scaler = filter_and_scale(inputs_like)
		outs_df, self.output_scaler = filter_and_scale(outputs_like)

		self.df_mu = th.Tensor(inputs_df.groupby(group_key).mean().values).detach()
		self.df_sigma = th.Tensor(inputs_df.groupby(group_key).std().values).detach()
		self.outs_df = th.Tensor(outs_df.groupby(group_key).mean().values).detach()

		self.input_col_names = inputs_df.columns
		self.output_col_names = outs_df.columns

		print('reduced df len: ', self.df_sigma.shape[0])

	def to(self, device):
		self.df_mu=self.df_mu.to(device)
		self.df_sigma=self.df_mu.to(device)
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
	def __init__(self, target_model: nn.Module, moments_dataset: UQMomentsDataset, samples_per_distribution=30):
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
		self.SE_model = self.SE_model**(1/2)

	def __len__(self):
		return len(self.moments_dataset)
	
	def __getitem__(self, index):
		(mu, sigma), outputs = self.moments_dataset[index]
		return	th.cat((mu, sigma), axis=-1), self.SE_model[index]

class FFRegressor(pl.LightningModule):
	def __init__(self, input_size: int, output_size: int=None, n_layers: int=8, 
	      		learning_rate: float=7.585775750291837e-08, lr_coef: float=1.0,
				device_stats_monitor=False):
		"""
		Just a simple FF Network that scales

		:param input_size: NN input size
		:param output_size: NN output size
		:param n_layers: number of NN layers
		:param learning_rate: NN learning rate (default provided by auto_lr_finder)
		:param lr_coef: the learning_rate scaling coefficient (i.e. from larger batch size across gpus)
		:param device_stats_monitor: whether to use a device monitor (i.e. log device usage)
		"""
		super().__init__()
		learning_rate *= lr_coef; del lr_coef
		if not output_size: output_size = input_size
		vars(self).update(locals()); del self.self

		hidden_size = input_size*4
		bulk_layers = []
		for i in range(self.n_layers-1): # this should be safer then potentially copying layers by reference...
			bulk_layers.extend([nn.SELU(), nn.Linear(hidden_size,hidden_size)])
		self.regressor = nn.Sequential(nn.BatchNorm1d(input_size),nn.Linear(input_size,hidden_size),*bulk_layers, nn.Linear(hidden_size, output_size))
		# last layer is just to change size, doesn't count as a "layer" since it's linear
	
	def forward(self, inputs):
		return self.regressor(inputs)

	def configure_optimizers(self):
		return th.optim.Adam(self.parameters(), lr=self.learning_rate)

	def configure_callbacks(self):
		"""We want to log accelerator usage statistics for profiling,
		& only way to do this with CLI is to use this hook"""
		return [DeviceStatsMonitor()] if self.device_stats_monitor else []

	# sync dist makes metrics more accurate (by syncing across devices), but slows down training
	def log_metrics(self, Y_pred, Y, prefix='', sync_dist=True):
		loss = F.mse_loss(Y_pred, Y)
		self.log(prefix+'mse_loss', loss, sync_dist=sync_dist)
		self.log(prefix+'R2_var_weighted', F_metrics.r2_score(Y_pred, Y, multioutput='variance_weighted'),sync_dist=sync_dist)
		self.log(prefix+'R2_avg', F_metrics.r2_score(Y_pred, Y, multioutput='uniform_average'),sync_dist=sync_dist)
		self.log(prefix+'MAPE', F_metrics.mean_absolute_percentage_error(Y_pred, Y),sync_dist=sync_dist)	

	def training_step(self, training_batch, batch_id, log_prefix=''):
		X, Y = training_batch
		Y_pred = self.forward(X)
		assert not (th.isnan(X).any() or th.isnan(Y).any())
		loss = F.mse_loss(Y_pred, Y)
		self.log_metrics(Y_pred, Y, log_prefix)	
		return loss

	# reuse training_step(), but log validation loss
	def validation_step(self, val_batch, batch_id):
		self.training_step(val_batch, batch_id, log_prefix='val_')

# TODO: should do moments preprocessing in the prep function stage then save the file to pickle with hash based on arguments & reload later
# doubly important since we are seperating the mean regressor fitting from the UQ model fitting!
# Lol I just crammed the previous functions into this class... I think it should work fine though?
class UQ_DataModule(pl.LightningDataModule):
	def __init__(self, dataset: Dataset, batch_size=27310, train_portion=0.8, data_workers=4):
		"""
		UQ data module (or mean regressor data module)
		:param dataset: this is the dataset you want to fit your "UQ" model to (can also be mean regressor)
		:param batch_size: the batch size for training & validation (default set by auto_batch_size_finder)
		"""
		super().__init__()
		vars(self).update(locals())
		del self.self # gotcha to make trick work
		self.prepare_data_per_node=False
	
	def setup(self, stage=None):
		def make_data_loaders(dataset, batch_size, train_portion=0.8, data_workers=4, **kwd_args):
			# TODO: revert to simpler method of providing portions (should work I think)
			def _get_split_sizes(full_dataset: Dataset, train_portion) -> tuple:
				len_full = len(full_dataset)
				len_train = int(train_portion*len_full)
				len_val = len_full - len_train
				return len_train, len_val
			train, val = random_split(dataset, _get_split_sizes(dataset, train_portion))#[train_portion, 1-train_portion])
			get_batch_size = lambda df: len(df) if batch_size is None else min(batch_size, len(df)) # min ensures we don't choose invalid values!
			train_loader = DataLoader(train, batch_size=get_batch_size(train), num_workers=data_workers, shuffle=True)
			val_loader = DataLoader(val, batch_size=get_batch_size(val), num_workers=data_workers)
			return train_loader, val_loader
		self.train_loader, self.val_loader = make_data_loaders(**vars(self))
	
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
		moments_dataset = UQMomentsDataset(data_fn, inputs_like='Yi', outputs_like='souspec', group_key='group')
		regressor_dataset = UQSamplesDataset(moments_dataset, constant=constant)
		super().__init__(regressor_dataset, **kwargs)

def load_mean_regressor_factory(model_fn, cols):
	""" load model factory (originally intended for mean regressor) """
	if model_fn.endswith('.ckpt'):
        model = FFRegressor.load_from_checkpoint(model_fn, input_size=len(cols))#.to('cuda')
    else:
        model = TF2PL_chemtab_wrapper.wrap_mean_regressor(model_fn)
        TF2PL_chemtab_wrapper.check_Yi_consistency(cols)
	return mean_regressor

class UQRegressorDataModule(UQ_DataModule):
	def __init__(self, data_fn: str, mean_regressor_fn: str, **kwargs):
		"""
		This is the dataset used for fitting a UQ model (i.e. 2nd moment aka SE regressor)
		:param data_fn: grouped chrest csv for getting moments
		:param constant: whether to keep the (uncertain) inputs constant (i.e. use only mean), I believe constant is a good idea
		"""
		moments_dataset = UQMomentsDataset(data_fn, inputs_like='Yi', outputs_like='souspec', group_key='group')#.to('cuda')
		(mu, sigma), outs = random.choice(moments_dataset)

		mean_regressor = load_mean_regressor_factory(mean_regressor_fn, moments_dataset.input_col_names)
		#if mean_regressor_fn.endswith('.ckpt'):
		#	# TODO: support loading tensorflow models too!! How do we do this? Different file/module?
		#	mean_regressor = FFRegressor.load_from_checkpoint(mean_regressor_fn, input_size=mu.shape[0])#.to('cuda')
		#else:
		#	mean_regressor = TF2PL_chemtab_wrapper.wrap_mean_regressor(mean_regressor_fn) #'./PCDNNV2_decomp_ablate-filtered-97%R2')
		#	TF2PL_chemtab_wrapper.check_Yi_consistency(moments_dataset.input_col_names)
		regressor_dataset = UQErrorPredictionDataset(mean_regressor, moments_dataset)
		super().__init__(regressor_dataset, **kwargs)

#########################################################################

# NOTE: if you want to juggle between pytorch v2.0 and v<2.0 (e.g. for legacy or A100 GPUs) then you just need to juggle between pytorch_distributed_cuda3 (for legacy) and pytorch_distributed (for A100s)
class MyLightningCLI(pl.cli.LightningCLI):
	def add_arguments_to_parser(self, parser):
		parser.set_defaults({'trainer.num_nodes': 1, 'trainer.devices': 1}) # we want lr_coef to work properly!
		parser.link_arguments(['trainer.devices', 'trainer.num_nodes'], 'model.lr_coef', apply_on='parse', compute_fn=lambda devices, num_nodes: int(num_nodes)*int(devices))
		parser.link_arguments(['data.dataset'], 'model.input_size', compute_fn=lambda dataset: next(iter(dataset))[0].shape[0], apply_on='instantiate') # holyshit this works!
		parser.link_arguments(['data.dataset'], 'model.output_size', compute_fn=lambda dataset: next(iter(dataset))[1].shape[0], apply_on='instantiate')
		#parser.add_argument('--model_name', type=str, default=None, help='name of the model for saving', required=False)	
		
		#get_shape = lambda ds, output=False: next(iter(dataset))[int(output)].shape[0] # shape 0 size batching is not applied yet, it is a 1d vector...	
		#parser.link_arguments(['data.dataset'], 'model.input_size', apply_on='instantiate', compute_fn=lambda ds: get_shape(ds, output=False)) # holyshit this works!
		#parser.link_arguments(['data.dataset'], 'model.output_size', apply_on='instantiate', compute_fn=lambda ds: get_shape(ds, output=True))

# Example Usage: srun --ntasks-per-node=2 python ChemtabUQ.py fit --data.class_path=MeanRegressorDataModule --data.data_fn=../data/chrest_contiguous_group_sample100k.csv --trainer.accelerator=gpu --trainer.devices=2 --trainer.num_nodes=2
def cli_main():
	cli=MyLightningCLI(FFRegressor, UQ_DataModule, subclass_mode_data=True, save_config_kwargs={"overwrite": True})
	cli.trainer.save_checkpoint("model.ckpt")
	#th.save(cli.model, "model.pt") # it seems this type of save is unnecessary and even problematic	

if __name__=='__main__':
    cli_main()

#def DataSet_factory(data_fn: str, UQ_data: bool=False, mean_regressor_fn: str=None):
#	"""
#	Factory pattern for building the datasets, composition > inheritance
#	:param data_fn: data_csv for building moments dataset, whether fitting a mean regressor or a UQ-SE model we need this
#	:param UQ_data: whether we are building the dataset for a UQ model or a mean regressor
#	:param mean_regressor_fn: if we are building the dataset for a UQ model it is required that we load a mean regressor to acquire error data (this is the model's file name)
#	"""
#	df_fn = f'{os.environ["HOME"]}/data/{data_fn}'
#	moments_dataset = UQMomentsDataset(df_fn, inputs_like='Yi', outputs_like='souspec', group_key='group')
#	if UQ_data:
#		regressor_dataset = UQErrorPredictionDataset(mean_regressor, moments_dataset)
#	else:
#		regressor_dataset = UQSamplesDataset(moments_dataset, constant=True)
#	return regressor_dataset


## TODO: make into main() function?
#def fit_UQ_model(name, trainer, regressor, datamodule):
#	# TODO: make this a linked argument!
#	#x,y=next(iter(datamodule.val_dataloader())) # hack to find dataset input size!
#	#regressor = FFRegressor(input_size=x.shape[1], output_size=y.shape[1], # TODO: all this should be automated by LightningCLI
#	#						lr_coef=kwd_args['lr_coef'], 
#	#			    		device_stats_monitor=kwd_args['device_stats_monitor'])
#	trainer.fit(regressor, datamodule=datamodule)
#	trainer.save_checkpoint(f'{name}.ckpt')
#	print(f'done fitting {name}!')
#	return regressor

#if __name__=='__main__':
#	# NOTE: it seems that regular LightningCLI is too generic for your use case since only trains 1 model on one dataset at a time...
#	parser = LightningArgumentParser() # this should be strictly better than regular argument parser & more general too!
#	#parser.add_argument('--device-stats-monitor', action='store_true', 
#	#	help='Manually added CLI arg to support configuration of DeviceStatsMonitor() profiling (i.e. measuring utilization of GPUs). Turn on for more thorough profiling/troubleshooting.')
#	#parser.add_argument('--dataset', default='chrest_contiguous_group_sample100k.csv', help='dataset file name (should be inside the ~/data folder)')
#	#parser.add_argument('--batch_size', default=1000, type=int, help='batch_size per GPU') # TODO: remove since this should be redundant??
#	#parser.add_argument('--constant-training-data', action='store_true',  # TODO: consider removing this since it seems like it doesn't even make sense to have anymore?
#	#					help='Experimental mode where constant training data is used for primary regressor then samples are taken afterward for UQ portion.')
#	parser.add_lightning_class_args(pl.Trainer, 'Trainer')
#	parser.add_lightning_class_args(UQ_DataModule, 'DataModule') # does this even make sense?? How does it have the non-trivial Dataset Argument?? Fuck it lets try it!
#	parser.add_lightning_class_args(FFRegressor, 'FFRegressor')
#	dataset_fact_args = parser.add_function_arguments(DataSet_factory, 'DataSet_factory')
#	parser.link_arguments(dataset_fact_args, 'UQ_DataModule.dataset', compute_fn=DataSet_factory, apply_on='parse')
#	parser.link_arguments(['UQ_DataModule.dataset'], 'FFRegressor.input_size', compute_fn=lambda dataset: next(iter(dataset))[0].shape[1], apply_on='parse')
#	parser.link_arguments(['UQ_DataModule.dataset'], 'FFRegressor.output_size', compute_fn=lambda dataset: next(iter(dataset))[1].shape[1], apply_on='parse')
#	parser.link_arguments(['Trainer.devices', 'Trainer.num_nodes'], 'FFRegressor.lr_coef', compute_fn=lambda devices, num_nodes: int(num_nodes)*int(devices), apply_on='parse')
#	args = parser.parse_args()
#
#	##################### Fit Mean Regressor: #####################
#	#df_fn = f'{os.environ["HOME"]}/data/{args.dataset}'
#	#moments_dataset = UQMomentsDataset(df_fn, inputs_like='Yi', outputs_like='souspec', group_key='group')
#	#samples_dataset = UQSamplesDataset(moments_dataset, constant=args.constant_training_data)
#	#samples_data_module = UQ_DataModule(samples_dataset, **vars(args.DataModule))
#	samples_data_module = UQ_DataModule(**vars(args.DataModule))
#
#	# TODO: use pl.LightningDataModule, see this url: https://lightning.ai/docs/pytorch/stable/data/datamodule.html
#	#train_settings = {'batch_size': args.batch_size, 'workers': 4, # these should be from LightningCLI args directly
#	#				  'lr_coef': int(args.num_nodes)*int(args.devices), # should be a link
#	#				  'device_stats_monitor': args.device_stats_monitor} # again should be from LightningCLI args directly
#	trainer = pl.Trainer(**vars(args.Trainer)) # TODO: check that this works? It does!! But you should just use LightningCLI after making data module factory...
#	mean_regressor = FFRegressor(**vars(args.FFRegressor))
#	mean_regressor = fit_UQ_model('mean_regressor', trainer, mean_regressor, samples_data_module)
#	#########################################################################
#	
#	###################### Fit Standard Deviation Regressor: #####################
#	#
#	#STD_dataset = UQErrorPredictionDataset(mean_regressor, moments_dataset)
#	##del trainer # I'm sorry why do this??
#	##trainer = pl.Trainer.from_argparse_args(args) 
#	#std_regressor = FFRegressor(**vars(args.FFRegressor))
#	#std_regressor = fit_UQ_model('std_regressor', trainer, std_regressor, STD_dataset)
#	##########################################################################
