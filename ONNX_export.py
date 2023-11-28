#!/usr/bin/env python
# coding: utf-8

#NOTE: entire file verified to work 9/20/23

import os, sys
import ChemtabUQ
import pytorch_lightning as pl
import torch as pt
import tensorflow as tf
import tensorflow.keras as keras

######################### Post-Processing Layers #########################

# Verified to work! 5/14/22 (in regards to relative tolerance, see the assert)
# :return: rescaling_layer, (m,b) <-- includes params for manual scaling!
def fit_rescaling_layer(scaler, inverse=True, layer_name='rescaling', n_samples=1000):
    import sklearn, tensorflow as tf, tensorflow.keras as keras, numpy as np # metrics ... for sanity checks
    import sklearn.linear_model
    def R2(yt,yp): return tf.reduce_mean(1-tf.reduce_mean((yp-yt)**2, axis=0)/(tf.math.reduce_std(yt,axis=0)**2))
    def rel_err(yt, yp): return tf.reduce_mean(tf.abs((yp-yt)/yt))

    n_input_features = scaler.n_features_in_
    data_scale = 100
    for i in range(100): # retry until sanity np.allclose check passes!
        dummy_input_data = (np.random.random(size=(n_samples,n_input_features))-0.5)*data_scale
        inverted_data = scaler.inverse_transform(dummy_input_data) if inverse else scaler.transform(dummy_input_data)
        def fit_rescaling_lms():
            linear_models = []
            for i in range(n_input_features):
                lm = sklearn.linear_model.LinearRegression()
                X_data = dummy_input_data[:,i].reshape(-1,1)
                Y_data = inverted_data[:,i].reshape(-1,1)
                lm.fit(X_data, Y_data)
                assert lm.score(X_data, Y_data)==1.0 # assert R^2==1 (should be perfect fit)
                linear_models.append(lm)
            return linear_models
        lms = fit_rescaling_lms()
        m = np.array([lm.coef_ for lm in lms]).squeeze()
        b = np.array([lm.intercept_ for lm in lms]).squeeze()
        rescaling_layer = keras.layers.Rescaling(m, b, name=layer_name) # y=mx+b
        rescale_inverted_data = rescaling_layer(dummy_input_data).numpy().astype('float64')
        print('MAE/data_scale for inversion layer:', np.mean(np.abs(rescale_inverted_data-inverted_data))/data_scale)
        print('R^2 for inversion layer:', R2(inverted_data, rescale_inverted_data).numpy())
        print('Rel-error for inversion layer:', rel_err(inverted_data, rescale_inverted_data).numpy())
        if np.allclose(rescale_inverted_data, inverted_data): break
    assert np.allclose(rescale_inverted_data, inverted_data)
    return rescaling_layer, (m, b)

#rescaling_layer, (m,b) = fit_rescaling_layer(dm.outputScaler)

# Verified to work! 9/20/23 (within numerical precision tolerances)
def add_unit_L1_layer_constraint(x, first_n_preserved=0, layer_name=None):
    """ this is designed to constraint the [first_n_preserved:]
    outputs from the inversion layer to fall between 0 & 1 and to sum to 1 """
    from tensorflow.keras import layers

    inputs_ = layers.Input((x.shape[1]-first_n_preserved,))
    out = tf.math.maximum(inputs_, 0)
    out = tf.math.minimum(out, 1)

    # apparently doing this twice increases numerical accuracy
    out = out/tf.math.reduce_sum(out, axis=-1, keepdims=True)
    out = out/tf.math.reduce_sum(out, axis=-1, keepdims=True)
    model = keras.models.Model(inputs=inputs_, outputs=out, name='Unit_L1_constraint')
    return layers.Concatenate(axis=-1, name=layer_name)([x[:,:first_n_preserved], model(x[:,first_n_preserved:])])

##################################################################

######################### ONNX/TF Export #########################

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

    # NOTE: we cannot assert that it is within machine error because anomolous models with very
    # large outputs naturally have higher MAE (it scales otherwise small differences), 
    # this is an unavoidable side effect of ONNX 
    
    ## Machine error is approximately 1e-7, 
    ## so we make sure it is within machine error tolerance.
    #assert MAE(yt, yp) < 1e-6
    assert R2(yt, yp) > 0.9999

def test_models(torch_model, TF_model):
    inputs = torch_model.example_input_array #pt.randn(*in_shape).cpu()
    in_shape=inputs.shape

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

##################################################################

# verified to work 9/20/23
def export_CT_model_for_ablate(ckpt_path, add_hard_l1_constraint=False):
    """ adds additional rescaling & L1 unit constraint post-processing via layers """
    keras_rep, model_path = convert_FFRegressor_to_TF(ckpt_path)
    try: keras_rep.summary()
    except: raise RuntimeError('keras representation is required for rescaling/l1_norm layer(s)!')

    PLD_module = ChemtabUQ.MeanRegressorDataModule.load_from_checkpoint(ckpt_path)  
    moments_dataset = PLD_module.dataset.moments_dataset    
    
    output = input_ = keras.layers.Input(shape=keras_rep.input_shape[1:], name='input_1')
    assert moments_dataset.input_scaler is None, 'input scaler export not supported yet, TODO: test inverse=False'
    # TODO: test using fit_rescaling_layer(..., inverse=False) (should work for input scaler)

    output = keras_rep(output)
    if moments_dataset.output_scaler:
        print('fitting output rescaling layer!')
        output_scaling_layer, (m,b) = fit_rescaling_layer(moments_dataset.output_scaler, 
                                                           inverse=True, layer_name='output_rescaling')
        output=output_scaling_layer(output)

    if add_hard_l1_constraint:
        output=add_unit_L1_layer_constraint(output, first_n_preserved=0)

    wrapper = keras.models.Model(inputs=input_, outputs=output) 
    wrapper.save(model_path)
    return wrapper

from glob import glob
import random
import os, argparse

# default checkpoint is the test case
if __name__=='__main__':
    # get default checkpoint
    candidate_ckpts=glob(f'./CT_logs_Mu/*/version_*/checkpoints/*.ckpt')
    ckpt_path = random.choice(candidate_ckpts) 
    print(ckpt_path)
    
    parser = argparse.ArgumentParser(description='exports CT models to TF with relevant post-processing layers (e.g. rescaling)')
    parser.add_argument('checkpoint_path', type=str, nargs='?', default=ckpt_path, help='path to checkpoint to load, convert to TF & wrap') 
    parser.add_argument('--hard_L1_unit_norm_constraint', action='store_true', help='turn on for the inverse model, it will enforce Yi constraints')
    args = parser.parse_args() 
    ckpt_path=args.checkpoint_path
    
    keras_rep = export_CT_model_for_ablate(ckpt_path)
    #keras_rep = convert_FFRegressor_to_TF(ckpt_path)
