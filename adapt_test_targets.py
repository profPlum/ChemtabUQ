from tensorflow import keras
import pandas as pd
import numpy as np
import yaml
import os

model_path = 'PCDNNV2_decomp'

W = pd.read_csv(f'{model_path}/weights.csv', index_col=0)
Winv = np.linalg.pinv(W.T) #Winv = pd.read_csv(f'{model_path}/weights_inv.csv', index_col=0)
print_array = lambda x: ','.join([str(i) for i in np.asarray(x).squeeze()]) 
print_str_array = lambda x: ','.join([f'"{i}"' for i in np.asarray(x).squeeze()]) 
print(W)

#input_mass = m = np.random.rand(len(W.index))
#input_mass = m = m/m.sum() # should sum to 1 as per definition 
#m = m[:,np.newaxis]

# Here we derive the masses from several known constraints (to make them more realistic)
#INTERESTING: we proved given orthonormality constraint (via scharts inequality) that all |CPVs| < 1 (including Zmix)
start_cpvs = np.random.rand(len(W.columns))*2-1 # NOTE: gives uniform distribution between -1 & 1 to satisfy above mentioned constraint
masses = np.dot(Winv, start_cpvs[:,np.newaxis])
masses = np.maximum(masses, 0) # must be positive!
masses = np.minimum(masses, 1) # must be < 1!
masses /= np.sum(masses) # should sum to 1 as per definition 
input_mass = m = masses

#INTERESTING: we proved given orthonormality constraint (via scharts inequality) that all CPVs < 1 (including Zmix)
output_cpv = cpv = np.dot(W.T,m).flatten()
print('CPVs <-- mass_fractions:')
print('CPVs:', print_array(cpv), '<-- mass_fractions:', print_array(m))

# use only 1 input CPV for both tests (for simplicity)
input_cpv = cpv# = np.random.rand(len(cpv))#np.linspace(0.1,1,len(cpv)) 
cpv = cpv[:,np.newaxis]

# pad with Zmix value
#input_cpv = np.concatenate([[0], input_cpv])
#output_cpv = np.concatenate([[0], output_cpv])

regressor = keras.models.load_model(f"{model_path}/regressor")
source_output = regressor(input_cpv[:,np.newaxis].T)
souener_output = source_output['static_source_prediction'].numpy()[0,0]
# Source energy is always first static source term!

print('CPVs --> Source Terms:')
print(f'CPVs: {print_array(cpv)} -->')
print('pred CPV source terms: ', print_array(source_output['dynamic_source_prediction']))
print('pred Source Energy: ', souener_output) 

# confirmed that [0,1:] is correct indexing 12/16/22
output_mass = m = source_output['static_source_prediction'].numpy()[0,1:]
#output_mass = m = np.dot(Winv.T,cpv).flatten()
print('CPVs --> mass_fractions:')
print('CPVs:', print_array(cpv), '--> mass_fractions:', print_array(m))

import os
test_targets = {'testName': f'{os.path.basename(model_path)}_parameter_set_1', 
				'cpv_names': list(W.columns),
				'species_names': list(W.index),
				'input_cpvs': input_cpv.tolist(),
				'output_mass_fractions': output_mass.tolist(),
				'input_mass_fractions': input_mass.squeeze().tolist(),
				'output_cpvs': output_cpv.tolist(),
				'output_source_terms': [0.0]+source_output['dynamic_source_prediction'].numpy().squeeze().tolist(),
				'output_source_energy': souener_output.item()}
test_targets = [test_targets] # Matt wants a 'sequence' for some reason?

print('\n'+'='*50)
print('test_targets ([dict]):')
print('='*50)
print(test_targets)

import yaml
with open(f'{model_path}/testTargets.yaml', 'w') as f:
	yaml.dump(test_targets, f, default_flow_style=None)
