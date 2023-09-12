#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pytorch_lightning as pl
import torch as th
import ChemtabUQ


# In[ ]:


ckpt_path='../CT_logs_Mu/MAPE-PCA-CT/version_13420071/checkpoints/epoch=10891-step=10892.ckpt'
PL_module = ChemtabUQ.FFRegressor.load_from_checkpoint(ckpt_path, input_size=25)


# In[ ]:


import os
from glob import glob
print(os.getcwd())
print(glob('../CT_logs_Mu/MAPE-PCA-CT/*/checkpoints/*'))


# In[ ]:


onnx_model_path = os.path.dirname(ckpt_path)+'/model.onnx'
PL_module.to_onnx(onnx_model_path)


# In[ ]:


import onnx
from onnx_tf.backend import prepare
onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(os.path.dirname(ckpt_path)+'/model.pb')


# In[ ]:


"""%pip install onnx
%pip install onnx_tf
%pip install tensorflow_probability"""

get_ipython().run_line_magic('pip', 'uninstall -y tensorflow')
get_ipython().run_line_magic('pip', 'install tensorflow')
get_ipython().run_line_magic('pip', 'uninstall -y tensorflow_probability')
get_ipython().run_line_magic('pip', 'install tensorflow_probability==0.19.0')


# In[ ]:


get_ipython().run_line_magic('pip', 'show tensorflow_probability')


# In[ ]:




