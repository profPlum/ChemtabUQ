from ChemtabUQ import *

sample_from_uncertainty=True

df_fn = './data/chrest_contiguous_group_sample100k.csv'
moments_dataset = UQMomentsDataset(df_fn, inputs_like='Yi', outputs_like='souspec', group_key='group')

import random

# get ICs
(mu, sigma), outs = random.choice(moments_dataset)
Yi_state=mu.reshape(1,-1) + th.randn(*sigma.shape)*sample_from_uncertainty*0.1
Yi_var=(sigma**2).reshape(1,-1)

n_Yi = mu.shape[0]
assert n_Yi == 53

#mean_regressor = UQModel.load_from_checkpoint('mean_regressor.ckpt', input_size=n_Yi).cpu()
mean_regressor = TF2PL_chemtab_wrapper.wrap_mean_regressor('./PCDNNV2_decomp_ablate-filtered-97%R2')
#TF2PL_chemtab_wrapper.check_Yi_consistency(moments_dataset.input_col_names)
std_regressor = UQModel.load_from_checkpoint('std_regressor.ckpt', input_size=n_Yi*2, output_size=n_Yi).cpu()

#mean_regressor.model.eval()
std_regressor.eval()

n_time_steps=100
step_multiplier=100 # gives higher accuracy without slowing animation
dt = 1e-2 # TODO: make much smaller

import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd

df_state = pd.DataFrame(columns=moments_dataset.input_col_names)
df_SE = pd.DataFrame(columns=moments_dataset.input_col_names)

def constrain_state(Yi_state):
	Yi_state = np.maximum(Yi_state, 0)
	Yi_state = np.minimum(Yi_state, 1)
	Yi_state /= Yi_state.sum()
	return Yi_state

import time

import warnings
warnings.simplefilter("ignore")
for i in range(n_time_steps*step_multiplier):
	Yi_dot = mean_regressor(Yi_state).detach()
	print('time: ', i*dt)
	print('||Yi_dot||: ', np.linalg.norm(Yi_dot, 2))
	print('Yi_dot: ', Yi_dot)
	Yi_state += Yi_dot*dt
	print('Yi_state: ', Yi_state)

	#time.sleep(1)

	# scale the SE by the dt coef then convert to variance for VAR(X+Y)=VAR(X)+VAR(Y)
	Yi_var += (std_regressor(th.cat([Yi_state, Yi_var],axis=1)).detach()*dt)**2

	if i%step_multiplier==0: # log interval is determined by original steps
		Yi_state_pd = Yi_state.numpy()
		Yi_state_pd = Yi_state_pd.reshape(1,-1)
		Yi_state_pd = constrain_state(Yi_state_pd)
		Yi_state = th.from_numpy(Yi_state_pd)
		# apply constraint back to state!

		# convert to SE then scale with scaler
		Yi_SE_pd = pd.Series((Yi_var**(1/2)).squeeze(), index=moments_dataset.input_col_names)
		df_SE.loc[i//step_multiplier]=Yi_SE_pd

		Yi_state_pd = pd.Series(Yi_state_pd.squeeze(), index=moments_dataset.input_col_names)
		df_state.loc[i//step_multiplier]=Yi_state_pd

df_state['id'] = df_state.index
df_SE['id'] = df_SE.index
print(df_state)
# Example: pd.wide_to_long(df, stubnames='ht', i=['famid', 'birth'], j='age')
df_long = pd.wide_to_long(df_state, stubnames=['Yi'], i='id', j='Yi_name', suffix='.+')
df_long_SE = pd.wide_to_long(df_SE, stubnames=['Yi'], i='id', j='Yi_name', suffix='.+')
df_long_SE['Yi_SE']=df_long_SE['Yi']
del df_long_SE['Yi']

df_long = pd.concat([df_long,df_long_SE],axis=1)

#df_long_SE = df_long_SE.reset_index() # merges multi-index into the columns
df_long = df_long.reset_index() # merges multi-index into the columns

print(df_long)
fig = px.bar(df_long, x="Yi_name", y="Yi", animation_frame="id", error_y='Yi_SE', title='0d Reactor Chemtab UQ Demo (NOTICE: UQ Error Bars)')#, animation_group="country")
fig.show()
