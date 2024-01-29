import os, argparse
import yaml
import sys

# example metadata.yaml:
"""
---
# all paths are assumed to be relative
name: PCDNNV2_decomp-Wax-Orthog+Zmix
rpath: regressor
wpath: weights.csv
mechanism: GRIMech30.yaml
version: 1.0
"""

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'makes metadata.yaml for Matt')
    parser.add_argument('model_folder', type=str, help='model folder to create metadata.yaml file for')
    parser.add_argument('--IC_cfg_path', type=str, default='./Initial_Condition_Configs/sampleDiffusionFlame.yaml', help='relative path to Initial_Condtion yaml file (which implicity contains path to the corresponding mechanism file), prototypical example is: sampleDiffusionFlame.yaml')
    parser.add_argument('--version', type=float, default=2.0, help='version of model, assuming CTV2 by default')
    args = parser.parse_args()

    with open(args.IC_cfg_path, 'r') as f:
        IC_cfg = yaml.safe_load(f)
    assert 'initializers' in IC_cfg, 'invalid IC_cfg file! (initial conditions missing)'
    mech_fn = os.path.dirname(args.IC_cfg_path) + '/' + IC_cfg['mechanism']

    # NOTE: as of right now rpath & wpath are constants
    cfg = {'rpath': 'regressor', 'wpath': 'weights.csv', 'version': args.version}
    cfg['name']=os.path.basename(args.model_folder)
    cfg['mechanism']=os.path.basename(mech_fn)
    cfg['initializers']=IC_cfg['initializers']

    # add mechanism file & write the meta_data
    os.system(f'cp {mech_fn} {args.model_folder}/.')
    with open(f'{args.model_folder}/metadata.yaml', 'w') as f:
        yaml.dump(cfg, f)
