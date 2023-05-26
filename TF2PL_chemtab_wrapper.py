import os
import tensorflow as tf
from tensorflow import keras
import torch as th
import pandas as pd
import numpy as np

def _load_mean_regressor(model_fn: str):
    model_dir = os.path.dirname(model_fn)
    weight_mat = pd.read_csv(model_dir+'/weights.csv',index_col=0)
    model = keras.models.load_model(model_fn)
    print(model.summary())
    print(weight_mat)
    return model, weight_mat

Yi_names=None
def wrap_mean_regressor(model_dir: str):
    model, weight_mat = _load_mean_regressor(model_dir+'/regressor')
    wrap_mean_regressor.model = model
    wrap_mean_regressor.weight_mat = weight_mat

    global Yi_names
    Yi_names=tuple(weight_mat.index)

    def mean_regressor(Yi_inputs: th.Tensor):
        # TODO: sanity check that the species are in the correct order!!
        Zmix_and_CPVs=Yi_inputs.cpu().numpy().dot(weight_mat) # Zmix should be at position 0
        outputs = model(Zmix_and_CPVs)['dynamic_source_prediction'].numpy()
        assert type(outputs) is np.ndarray
        return th.from_numpy(outputs).to(Yi_inputs.device)
    return mean_regressor

def check_Yi_consistency(other_Yi_names):
    global Yi_names
    assert tuple(other_Yi_names)==tuple(Yi_names), "Yi Species names are inconsistent!!"
