#!/usr/bin/env python
# coding: utf-8

import pytorch_lightning as pl
import torch as pt
import ChemtabUQ
import os, sys
import matplotlib

def test_outputs(yt, yp):
    import tensorflow as tf
    MAE = lambda yt, yp, axis=None: tf.reduce_mean(tf.math.abs(yp-yt), axis=axis).numpy().item()
    MSE = lambda yt, yp, axis=None: tf.reduce_mean((yp-yt)**2, axis=axis).numpy().item()
    R2 = lambda yt, yp: 1-tf.reduce_mean(tf.reduce_mean((yp-yt)**2, axis=0)/tf.math.reduce_variance(yt, axis=0)).numpy().item()
    print('sanity R^2: ', R2(yt, yp))
    print('sanity MAE: ', MAE(yt, yp))
    print('sanity MSE: ', MSE(yt, yp))
    
    import matplotlib.pyplot as plt
    plt.imshow(yp[:5,:])
    plt.title('Predicted Output Sample:')
    plt.savefig('Predicted_Output_Sample.png')
    #plt.show() # this is blocking
    plt.imshow(yt[:5,:])
    plt.title('Expected Output Sample:')
    plt.savefig('Expected_Output_Sample.png')
    #plt.show() # this is blocking

    # Machine error is approximately 1e-7, 
    # so we make sure it is within machine error tolerance.
    assert MAE(yt, yp) < 1e-6

def test_models(torch_model, TF_model):
    in_shape = (128,25)
    inputs = pt.randn(*in_shape).cpu()

    target_outputs = torch_model.cpu().forward(inputs).detach().cpu().numpy()
    test_inputs=inputs.cpu().detach().numpy()

    try: # first assume TF_model is keras model, if it fails try treating it like raw TF model
        actual_output=TF_model(test_inputs)
    except Exception as e:
        print('Failed to use TF model as keras model, with exception:\n', e, file=sys.stderr)
        print('Retrying as raw TF model', file=sys.stderr)
        actual_output=TF_model(input=test_inputs)['output']
    
    test_outputs(target_outputs, actual_output)

# verified to work 9/14/23 (sanity check builtin!)
# Also supported onnx-TF 'backup', in case onnx2keras fails
def convert_FFRegressor_to_TF(ckpt_path):
    print(f'loading PL ckpt: {ckpt_path}')
    PL_module = ChemtabUQ.FFRegressor.load_from_checkpoint(ckpt_path)
    print(PL_module)

    # save new ONNX model
    onnx_model_path = os.path.dirname(ckpt_path)+'/model.onnx'
    print(f'ONNX version being saved at: {onnx_model_path}')
    PL_module.to_onnx(onnx_model_path, #IMPORTANT: that input/output does not contain periods! (e.g. 'input.1') 
                    input_names=['input'], output_names=['output'], 
                    dynamic_axes={'input' : {0 : 'batch_size'}, # variable length axes
                                  'output' : {0 : 'batch_size'}})
    
    import onnx
    from onnx2keras import onnx_to_keras
    onnx_model = onnx.load(onnx_model_path)
    tf_path=os.path.dirname(ckpt_path)+'/model_TF'
    try: # To Keras
        tf_rep = onnx_to_keras(onnx_model, ['input'], name_policy="renumerate")
        print(f'*Keras* version being saved at: {tf_path}')
        tf_rep.save(tf_path)
    except Exception as e: # To TF
        print('Conversion Keras failed, with exception:\n', e, file=sys.stderr)
        print('Retrying conversion to raw TF', file=sys.stderr)
        from onnx_tf.backend import prepare
        tf_rep = prepare(onnx_model)
        print(f'*raw TF* version being saved at: {tf_path}')
        tf_rep.export_graph(tf_path)
   
    # sanity check conversion (so we know models are the same)
    test_models(PL_module, tf_rep)
    print('done converting')
    return tf_rep, tf_path 

from fit_rescaling_layer import fit_rescaling_layer, add_unit_L1_layer_constraint

# TODO: finish/debug me!!
def export_CT_model_for_ablate(ckpt_path, add_hard_l1_constraint=False):
    """ adds additional rescaling & L1 unit constraint post-processing via layers """
    keras_rep, model_path = convert_FFRegressor_to_TF(ckpt_path)
    try: keras_rep.summary()
    except: raise RuntimeError('keras representation is required for rescaling layer!')

    PLD_module = ChemtabUQ.MeanRegressorDataModule.load_from_checkpoint(ckpt_path)  
    moments_dataset = PLD_module.dataset.moments_dataset    
    
    output = input_ = keras.layers.Input(shape=keras_rep.input_shape[1:], name='input_1')

    if moments_dataset.input_scaler:
        print('fitting input rescaling layer!')
        # IMPORTANT: you can't use fit_rescaling_layer for the input scaler b/c it is designed for output scaler only!
        #input_scaling_layer, (m,b) = fit_rescaling_layer(moments_dataset.input_scaler, layer_name='input_rescaling')
        output = moments_dataset.input_scaler.transform(output) #input_scaling_layer(input_)
    output = keras_rep(output)
    if moments_dataset.output_scaler:
        print('fitting input rescaling layer!')
        #output_scaling_layer, (m,b) = fit_rescaling_layer(moments_dataset.output_scaler, layer_name='output_rescaling')
        output=moments_dataset.output_scaler.inverse_transform(output) # does this work??

    wrapper = keras.models.Model(inputs=input_, outputs=output) 
    return wrapper

# this is the test case
if __name__=='__main__':
    
    ckpt_path='../CT_logs_Mu/Selu-Scaled-PCA-CT/version_13479785/checkpoints/epoch=5082-step=5083.ckpt'
    if len(sys.argv)>1: ckpt_path=sys.argv[1]
    keras_rep = convert_FFRegressor_to_TF(ckpt_path)
    try: keras_rep.summary()
    except: raise RuntimeError('keras representation is required for rescaling layer!') 
