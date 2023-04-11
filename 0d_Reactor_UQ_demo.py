from ChemtabUQ import *

sample_from_uncertainty=False

df_fn = f'./data/chrest_contiguous_group_sample100k.csv'
moments_dataset = UQMomentsDataset(df_fn, inputs_like='Yi', outputs_like='souspec', group_key='group')

# get ICs
(mu, sigma), outs = moments_dataset[0]
Yi_state=mu + th.randn(*sigma.shape)*sigma*sample_from_uncertainty
Yi_sigma=sigma

n_Yi = mu.shape[0]
assert n_Yi == 53

mean_regressor = UQModel.load_from_checkpoint('mean_regressor.ckpt', input_size=n_Yi).cpu()
std_regressor = UQModel.load_from_checkpoint('std_regressor.ckpt', input_size=n_Yi*2, output_size=n_Yi).cpu()

mean_regressor.eval()
std_regressor.eval()

n_time_steps = 100
step_multiplier=100 # gives higher accuracy without slowing animation
dt = 1e-2 # TODO: make much smaller

#class TransWrapModel(pl.LightningModule):
#	def __init__(self, pl_module, input_trans, output_trans):
#		super().__init__()
#		self.input_trans = input_trans
#		self.output_trans = output_trans
#		self._module = pl_module
#	def forward(self, inputs):
#		inputs = th.from_numpy(self.input_trans.transform(inputs.detach().numpy()))
#		outputs = th.from_numpy(self.output_trans.inverse_transform(self._module(inputs).numpy()))
#		return outputs

# TODO: have this automatically store uncertainty information & accumulate it via VAR(X+Y)=VAR(X)+VAR(Y)
#class UQ_tensor

import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd

#mean_regressor = TransWrapModel(mean_regressor, moments_dataset.input_scaler, moments_dataset.output_scaler)

df = pd.DataFrame(columns=moments_dataset.input_col_names)

def constrain_state(Yi_state):
	Yi_state = np.maximum(Yi_state, 0)
	Yi_state = np.minimum(Yi_state, 1)
	Yi_state /= Yi_state.sum()
	return Yi_state

import warnings
warnings.simplefilter("ignore")
for i in range(n_time_steps*step_multiplier):
	Yi_dot = mean_regressor(Yi_state.reshape(1,-1)).squeeze().detach()
	Yi_state += Yi_dot*dt

	if i%step_multiplier==0: # log interval is determined by original steps
		Yi_state_pd = Yi_state.numpy()
		Yi_state_pd = moments_dataset.input_scaler.inverse_transform(Yi_state_pd.reshape(1,-1))
		Yi_state_pd = constrain_state(Yi_state_pd)
		Yi_state = th.from_numpy(moments_dataset.input_scaler.transform(Yi_state_pd).squeeze())
		# apply constraint back to state!
		Yi_state_pd = pd.Series(Yi_state_pd.squeeze(), index=moments_dataset.input_col_names)
		df.loc[i//step_multiplier]=Yi_state_pd


	#Yi_state_pd.plot(kind='bar')
	#plt.show()

	#Yi_dot_std = std_regressor()

df['id'] = df.index
print(df)
# Example: pd.wide_to_long(df, stubnames='ht', i=['famid', 'birth'], j='age')
df_long = pd.wide_to_long(df, stubnames=['Yi'], i='id', j='Yi_name', suffix='.+')
df_long = df_long.reset_index() # merges multi-index into the columns

print(df_long)
fig = px.bar(df_long, x="Yi_name", y="Yi", animation_frame="id")#, animation_group="country")
fig.show()
