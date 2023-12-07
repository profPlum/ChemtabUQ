from glob import glob
import random
import os, argparse

import pytorch_lightning as pl
import tensorflow as tf # MUST import tensorflow SECOND or else you will encounter ambiguous error!
from tensorflow import keras
from tensorflow.keras import layers as L

# NOTE: currently unused, was used previously to verify make_aggregate_regressor()
def inspect_outputs(outputs, model_name: str):
    plt.imshow(outputs['dynamic_source_prediction'])
    plt.title(f'CPV_source-{model_name}_outputs')
    plt.show()
    plt.imshow(outputs['static_source_prediction'][:,1:])
    plt.title(f'Inv-{model_name}_outputs')
    plt.show()
    plt.hist(np.asarray(outputs['static_source_prediction'][:,:1]).squeeze())
    plt.title(f'Souener-{model_name}_outputs')
    plt.show()

# verified to work! 9/20/23
def make_aggregate_regressor(CPV_source_ckpt: str, souener_ckpt: str, inv_ckpt: str):
    """ makes aggregate regressor with the same interfaces as CT V1 from individual models given from CT V2"""
    print('CPV_source checkpoint: ', CPV_source_ckpt)
    print('Inv checkpoint: ', inv_ckpt)
    print('Energy_source checkpoints', souener_ckpt)
    
    import ONNX_export
    CPV_source_model = ONNX_export.export_CT_model_for_ablate(CPV_source_ckpt)
    Souener_model = ONNX_export.export_CT_model_for_ablate(souener_ckpt)
    Inv_model = ONNX_export.export_CT_model_for_ablate(inv_ckpt, add_hard_l1_constraint=True)
    # hard_l1_constraint is needed for Inv model only due to constraints on Yis
   
    input_shape = CPV_source_model.input_shape[1:]
    print('input_shape: ', input_shape)
 
    assert Souener_model.output_shape[1]==1, 'invalid souener model!'
    assert CPV_source_model.input_shape[1]==CPV_source_model.output_shape[1]+1, 'invalid CPV_Source model!'
    assert Inv_model.input_shape[1]<Inv_model.output_shape[1], 'invalid inverse model!'
    assert CPV_source_model.input_shape==Souener_model.input_shape==Inv_model.input_shape, 'incompatible models!'

    input_=L.Input(shape=CPV_source_model.input_shape[1:], name='input_1')
    
    dynamic_source_prediction = L.Rescaling(1.0, name='dynamic_source_prediction')(CPV_source_model(input_))
    static_source_prediction = L.Concatenate(name='static_source_prediction')([Souener_model(input_), Inv_model(input_)])
    
    outputs={'static_source_prediction': static_source_prediction, 'dynamic_source_prediction': dynamic_source_prediction}
    aggregate_regressor = keras.models.Model(inputs=input_, outputs=outputs, name='V2_aggregate_regressor')
    print('aggregate regressor summary: ')
    aggregate_regressor.summary()
    print('aggregate regressor input_shape: ', aggregate_regressor.input_shape)
    print('aggregate regressor output_shape: ', aggregate_regressor.output_shape)
    tf.keras.utils.plot_model(aggregate_regressor, to_file='aggregate_regressor.png', show_shapes=True, show_dtype=True)
    return aggregate_regressor

# Verified to work 12/5/23
def find_best_ckpt(search_dir):
    ckpts = os.popen(f'find {search_dir} -name "*.ckpt"').read().strip()

    if ckpts: ckpts=ckpts.split('\n') # sanity checks...
    else: raise FileNotFoundError('No checkpoints found in search path!')
    assert all(['-val_loss=' in ckpt_i for ckpt_i in ckpts]), 'Checkpoint naming format is incompatible! (must contain val_loss=*)'
    assert all(['-loss=' in ckpt_i for ckpt_i in ckpts]), 'Checkpoint naming format is incompatible! (must contain loss=*)'
    #contains_train_loss=all(['-loss=' in ckpt_i for ckpt_i in ckpts])

    import re
    get_loss = lambda fn, prefix='': float(re.sub(r"^.*-{}loss=([0-9.]+).*$".format(prefix), r"\1", fn))
    get_val_loss = lambda fn: get_loss(fn, prefix='val_')
    get_worse_loss = lambda fn: max(get_loss(fn), get_val_loss(fn))
    #if not contains_train_loss: get_worse_loss=get_val_loss
    ckpts = sorted(ckpts, key=get_worse_loss)
    print('Sorted ckpts (top 5): \n'+'\n'.join(ckpts[:5])+'\n')
    return ckpts[0]


if __name__=='__main__':
    # show candidate checkpoint search paths
    candidate_ckpts=glob(f'./CT_logs_Mu/*')
    description='\n'+'='*100+'\n'+'='*100+'\n'
    description+='ALL CANDIDATE CHECKPOINT SEARCH PATHS (aka experiment names): \n'
    description+=', '.join(map(lambda x: x.replace('./CT_logs_Mu/', ''), candidate_ckpts))+'\n'
    description+='='*100+'\n'+'='*100+'\n'
    print(description)
    import time; time.sleep(0.5)
 
    #print('\n'+'='*100+'\n'+'='*100)
    #print('All candidate checkpoint search paths (aka experiment names): ')
    #print(', '.join(map(lambda x: x.replace('./CT_logs_Mu/', ''), candidate_ckpts)))
    #print('='*100+'\n'+'='*100+'\n')

    parser = argparse.ArgumentParser(description='Packages/aggregates V2 CT models for ablate.')
    parser.add_argument('--CPV_source_path', type=str, required=True, help='path to checkpoint *search_directory* of V2 CPV_source model')
    parser.add_argument('--Souener_path', type=str, required=True, help='path to checkpoint *search_directory* of V2 Souener model')
    parser.add_argument('--Inverse_path', type=str, required=True, help='path to checkpoint *search_directory* of V2 Inverse model')
    parser.add_argument('--CPV_Weight_matrix_path', type=str, required=True, help='path to the W-matrix csv (the matrix used to compress Yis to CPVs)')
    args = parser.parse_args()
 
    # IMPORTANT: MUST MATCH the 'model_name' variable in adapt_test_targets.py!!
    out_dir = 'PCDNNV2_decomp' # (advised not to change from PCDNNV2_decomp) 
    os.system('rm -r PCDNNV2_decomp 2> /dev/null') 
    # done without reference to out_dir as a sanity check on adapt_test_targets

    # search for checkpoints inside the search paths
    ckpt_search_args=['CPV_source_path', 'Souener_path', 'Inverse_path']
    for arg_name in ckpt_search_args:
        print(f'searching for {arg_name} checkpoint:') # we test prepending 'CT_logs_Mu/' for brevity
        try: vars(args)[arg_name]=find_best_ckpt('CT_logs_Mu/'+vars(args)[arg_name])
        except FileNotFoundError: vars(args)[arg_name]=find_best_ckpt(vars(args)[arg_name])
    
    aggregate_regressor = make_aggregate_regressor(args.CPV_source_path, args.Souener_path, args.Inverse_path)
    os.system(f'mkdir -p {out_dir}/experiment_records') # for config.yaml files
    aggregate_regressor.save(f'{out_dir}/regressor')
 
    config_path = lambda ckpt_path: os.path.dirname(ckpt_path) + '/../config.yaml'
    os.system(f'cp {config_path(args.CPV_source_path)} {out_dir}/experiment_records/CPV_source_config.yaml')
    os.system(f'cp {config_path(args.Inverse_path)} {out_dir}/experiment_records/Inverse_config.yaml')
    os.system(f'cp {config_path(args.Souener_path)} {out_dir}/experiment_records/Souener_config.yaml')
 
    import pandas as pd
    weights = pd.read_csv(args.CPV_Weight_matrix_path, index_col=0)
    weights.to_csv(f'{out_dir}/weights.csv')
    import adapt_test_targets # this will automatically build/save test targets for use by ablate
    print('='*50)
    print('Built Aggregate Regressor using these models: ')
    for arg_name in ckpt_search_args:
        print(f'{arg_name}={vars(args)[arg_name]}')
    print('='*50)
