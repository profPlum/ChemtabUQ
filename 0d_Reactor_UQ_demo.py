from ChemtabUQ import *

sample_from_uncertainty=False

df_fn = f'{os.environ["HOME"]}/data/chrest_contiguous_group_sample100k.csv'
moments_dataset = UQMomentsDataset(df_fn, inputs_like='Yi', outputs_like='souspec', group_key='group')

# get ICs
(mu, sigma), outs = moments_dataset[0]
Yi_state=mu + th.randn(*sigma.shape)*sigma*sample_from_uncertainty
Yi_sigma=sigma

n_Yi = mu.shape[0]
assert n_Yi == 53

mean_regressor = UQModel.load_from_checkpoint('mean_regressor.ckpt', input_size=n_Yi)
std_regressor = UQModel.load_from_checkpoint('std_regressor.ckpt', input_size=n_Yi*2, output_size=n_Yi)

mean_regressor.eval()
std_regressor.eval()

n_time_steps = 100
dt = 1e-7

import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd

for i in range(n_time_steps):
	Yi_dot = mean_regressor(Yi_state.reshape(1,-1)).squeeze().detach()
	Yi_state += Yi_dot*dt

	Yi_state_pd = pd.Series(Yi_state.numpy(), index=moments_dataset.input_col_names)
	Yi_state_pd.plot(kind='bar')
	plt.show()

	#Yi_dot_std = std_regressor()


