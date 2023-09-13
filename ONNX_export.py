#!/usr/bin/env python
# coding: utf-8

import pytorch_lightning as pl
import torch as th
import ChemtabUQ
import os, sys

## from here: https://stackoverflow.com/questions/76839366/tf-rep-export-graphtf-model-path-keyerror-input-1
#def convert_names_workaround(onnx_model):
#    from onnx import helper
#    
#    # Define a mapping from old names to new names
#    name_map = {"input.1": "input_1"}
#    
#    # Initialize a list to hold the new inputs
#    new_inputs = []
#    
#    # Iterate over the inputs and change their names if needed
#    for inp in onnx_model.graph.input:
#        if inp.name in name_map:
#            # Create a new ValueInfoProto with the new name
#            new_inp = helper.make_tensor_value_info(name_map[inp.name],
#                                                    inp.type.tensor_type.elem_type,
#                                                    [dim.dim_value for dim in inp.type.tensor_type.shape.dim])
#            new_inputs.append(new_inp)
#        else:
#            new_inputs.append(inp)
#    
#    # Clear the old inputs and add the new ones
#    onnx_model.graph.ClearField("input")
#    onnx_model.graph.input.extend(new_inputs)
#    
#    # Go through all nodes in the model and replace the old input name with the new one
#    for node in onnx_model.graph.node:
#        for i, input_name in enumerate(node.input):
#            if input_name in name_map:
#                node.input[i] = name_map[input_name]
#    return onnx_model

def convert_FFRegressor_to_TF(ckpt_path):
    print(f'loading PL ckpt: {ckpt_path}')
    PL_module = ChemtabUQ.FFRegressor.load_from_checkpoint(ckpt_path, input_size=25)
    print(PL_module)

    # save new ONNX model
    onnx_model_path = os.path.dirname(ckpt_path)+'/model.onnx'
    print(f'ONNX version being saved at: {onnx_model_path}')
    PL_module.to_onnx(onnx_model_path, #IMPORTANT: that input/output does not contain periods! (e.g. 'input.1') 
                    input_names=['input'], output_names=['output'], 
                    dynamic_axes={'input' : {0 : 'batch_size'}, # variable length axes
                                'output' : {0 : 'batch_size'}})
    
    import onnx
    from onnx_tf.backend import prepare
    onnx_model = onnx.load(onnx_model_path) #convert_names_workaround(onnx.load(onnx_model_path))
    tf_rep = prepare(onnx_model)
    tf_path=os.path.dirname(ckpt_path)+'/model_TF'
    print(f'TF version being saved at: {tf_path}')
    tf_rep.export_graph(tf_path)

if __name__=='__main__':
    # TODO: use argparser
    ckpt_path='../CT_logs_Mu/MAPE-PCA-CT/version_13420071/checkpoints/epoch=10891-step=10892.ckpt'
    if len(sys.argv)>1: ckpt_path=sys.argv[1]
    convert_FFRegressor_to_TF(ckpt_path) 
