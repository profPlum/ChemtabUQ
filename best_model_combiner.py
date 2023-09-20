from glob import glob
import random
import os, argparse

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
    """ makes aggergate regressor with the same interfaces as CT V1 from individual models given from CT V2"""
    print('Inv checkpoint: ', Inv_ckpt)
    print('CPV_source checkpoint: ', CPV_source_ckpt)
    print('Energy_source checkpoints', Souener_ckpt)
    
    import ONNX_export
    CPV_source_model = ONNX_export.export_CT_model_for_ablate(CPV_source_ckpt)
    Souener_model = ONNX_export.export_CT_model_for_ablate(souener_ckpt)
    Inv_model = ONNX_export.export_CT_model_for_ablate(inv_ckpt, add_hard_l1_constraint=True)
    # hard_l1_constraint is needed for Inv model only due to constraints on Yis
    
    assert Souener_model.output_shape[1]==1, 'invalid souener model!'
    assert CPV_source_model.input_shape==CPV_source_model.output_shape, 'invalid CPV_Source model!'
    assert Inv_model.input_shape[1]<Inv_model.output_shape[1], 'invalid inverse model!'
    assert CPV_source_model.input_shape==Souener_model.input_shape==Inv_model.input_shape, 'incompatible models!'

    input_shape = CPV_source_model.input_shape[1:]
    print('input_shape: ', input_shape)
    input_=L.Input(shape=CPV_source_model.input_shape[1:], name='input_1')
    
    dynamic_source_prediction = L.Rescaling(1.0, name='dynamic_source_prediction')(CPV_source_model(input_))
    static_source_prediction = L.Concatenate(name='static_source_prediction')([Souener_model(input_), Inv_model(input_)])
    
    outputs={'static_source_prediction': static_source_prediction, 'dynamic_source_prediction': dynamic_source_prediction}
    aggergate_regressor = keras.models.Model(inputs=input_, outputs=outputs, name='V2_aggergate_regressor')
    print('aggregate regressor summary: ')
    aggergate_regressor.summary()
    print('aggregate regressor input_shape: ', aggergate_regressor.input_shape)
    print('aggregate regressor output_shape: ', aggergate_regressor.output_shape)
    tf.keras.utils.plot_model(aggergate_regressor, to_file='aggregate_regressor.png', show_shapes=True, show_dtype=True)
    return aggergate_regressor

# default checkpoint is the test case
if __name__=='__main__':
    # get default checkpoint
    candidate_ckpts=glob(f'./CT_logs_Mu/*/version_*/checkpoints/*.ckpt')
    print('all candidate checkpoints: ', candidate_ckpts)

    parser = argparse.ArgumentParser(description='packages/aggergates V2 CT models for ablate')
    parser.add_argument('--CPV_source_path', type=str, required=True, help='path to checkpoint of V2 CPV_source model')
    parser.add_argument('--Souener_path', type=str, required=True, help='path to checkpoint of V2 Souener model')
    parser.add_argument('--Inverse_path', type=str, required=True, help='path to checkpoint of V2 Inverse model')
    parser.add_argument('--CPV_Weight_matrix_path', type=str, required=True, help='path to the W-matrix csv (the matrix used to compress Yis to CPVs)')
    args = parser.parse_args()

    config_path = lambda ckpt_path: os.path.dirname(ckpt_path) + '/../config.yaml'

    aggergate_regressor = make_aggregate_regressor(args.CPV_source_path, args.Souener_path, args.Inverse_path)
    os.system('mkdir -p PCDNNV2_decomp/experiment_records') # for config.yaml files
    aggergate_regressor.save('PCDNNV2_decomp/regressor')
    os.system(f'cp {config_path(args.CPV_source_path)} PCDNNV2_decomp/experiment_records/CPV_source_config.yaml')
    os.system(f'cp {config_path(args.Inverse_path)} PCDNNV2_decomp/experiment_records/Inverse_config.yaml')
    os.system(f'cp {config_path(args.Souener_path)} PCDNNV2_decomp/experiment_records/Souener_config.yaml')
    os.system(f'cp {args.CPV_Weight_matrix_path} PCDNNV2_decomp/weights.csv')
    import adapt_test_targets # this will automatically build/save test targets for use by ablate
