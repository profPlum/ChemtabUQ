## NOTE: improvement over fit_rescaling_layer was negligible so we ditched it
## NOTE: turns out that most rescaling layer error is due to 32bit precision rather than lm fitting
#def make_exact_rescaling_layer(scaler, inverse=True, layer_name='rescaling'):
#    """ 
#    Exact version of fit_rescaling_layer which only works for StandardScaler, 
#    trades generality for better numerical accuracy.
#    """
#    import sklearn, tensorflow as tf, tensorflow.keras as keras, numpy as np # metrics ... for sanity checks
#    def R2(yt,yp): return tf.reduce_mean(1-tf.reduce_mean((yp-yt)**2, axis=0)/(tf.math.reduce_std(yt,axis=0)**2))
#    def rel_err(yt, yp): return tf.reduce_mean(tf.abs((yp-yt)/yt))
#    
#    data_scale=100
#    n_input_features = scaler.n_features_in_
#    dummy_input_data = (np.random.random(size=(n_samples,n_input_features))-0.5)*data_scale
#
#    assert type(scaler) is sklearn.preprocessing.StandardScaler, 'other scalers not implemented exactly!' 
#    if inverse: # NOTE: scaler.mean_:= original data mean, scaler.scale_:= original data SD, 
#        m, b = scaler.scale_, scaler.mean_
#        scaled_data = scaler.inverse_transform(dummy_input_data)
#    else: # m & b values are derived quite easily
#        m, b = 1/scaler.scale_, -scaler.mean_/scaler.scale_
#        scaled_data = scaler.transform(dummy_input_data)
#
#    rescaling_layer = keras.layers.Rescaling(m, b, name=layer_name) # y=mx+b
#    layer_rescaled_data = rescaling_layer(dummy_input_data).numpy().astype('float64')
#    print('MAE/data_scale for Rescaling layer:', np.mean(np.abs(layer_rescaled_data-inverted_data))/data_scale)
#    print('R^2 for Rescaling layer:', R2(scaled_data, layer_rescaled_data).numpy())
#    print('Rel-error for Rescaling layer:', rel_err(scaled_data, layer_rescaled_data).numpy())
#
#    assert np.allclose(layer_rescaled_data, scaled_data), 'Rescaling layer has bad numerical accuracy!'
#    return rescaling_layer, (m, b)

