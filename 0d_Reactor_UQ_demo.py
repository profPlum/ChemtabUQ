from ChemtabUQ import *

sample_from_uncertainty=False

df_fn = f'{os.environ["HOME"]}/data/chrest_contiguous_group_sample.csv'
moments_dataset = UQMomentsDataset(df_fn, inputs_like='Yi', outputs_like='souspec', group_key='group')

# get ICs
(mu, sigma), outs = moments_dataset[0]
Yi_state=mu + th.randn(*sigma.shape)*sigma*sample_from_uncertainty
Yi_sigma=sigma

mean_regressor = th.load('mean_regressor.pt')
std_regressor = th.load('std_regressor.pt') # technically this one was trained on MAE, close enough

n_time_steps = 100
dt = 1e-7

import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd

for i in range(n_time_steps):
	Yi_dot = mean_regressor(Yi_state)
	Yi_state += Yi_dot*dt

	Yi_state_pd = pd.Series(Yi_state.numpy(), index=moments_dataset.input_col_names)
	Yi_state_pd.plot(kind='bar')
	plt.show()

	#Yi_dot_std = std_regressor()


