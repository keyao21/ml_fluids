import pickle 
import os 
import sys 
import shutil 
import argparse 
import pandas as pd 
import numpy as np 
from pprint import pprint 
import matplotlib.pyplot as plt 


curr_fullpath = os.getcwd()
ML_Fluid_fullpath = os.path.abspath("../ML_Fluid/src/")
QG_FTLE_fullpath = os.path.abspath("../QG_FTLE/src/")
ML_Fluid_RESULTS_fullpath = os.path.abspath("../ML_Fluid/results/")
QG_FTLE_INPUTS_fullpath = os.path.abspath("../QG_FTLE/inputs/")
ML_Fluid_raw_inputs_fullpath = os.path.abspath("../ML_Fluid/inputs/raw/")

# DIRTY HACK, list all overlapping modules (by name) in the two dirs
OVERLAPPING_MODULES = ['config', 'util']  

"""
Some notes regarding functions switch_to_qgftle_src_dir() and switch_to_mlfluids_src_dir(): 

It is necessary to change the directory to import the python modules in the relevant directory
(e.g. QG_FTLE or ML_Fluids); however, it is **ALSO** necessary to insert the src to the top of the
path -- this is because there may be modules which share names (e.g. configs.py) across both QG_FTLE
and ML_Fluids src directories, so the namespace must be explicit and the correct src dir must be 
at the top of the path with highest precedence. We must delete the modules that share names
"""

def _switch_to_dir(fullpath): 
    os.chdir(fullpath)
    sys.path.insert(0,fullpath)
    # delete overlapping modules with same name
    for module_name in OVERLAPPING_MODULES: 
        try:
            del sys.modules[module_name]
        except: 
            pass 

def switch_to_qgftle_src_dir(): 
    _switch_to_dir(QG_FTLE_fullpath)

def switch_to_home_dir(): 
    _switch_to_dir(curr_fullpath)

def switch_to_mlfluids_src_dir(): 
    _switch_to_dir(ML_Fluid_fullpath)

def generate_experiment_id(): 
    # from experiment parmas create unique string to be used as key for config dicts 
    pass


def generate_ml_fluid_params(   training_length, trained_model_params, testing_length, 
                                trained_model_filename, stream_function_prefix,
                                dt=0.1, elapsedTime=2500, xct=128, yct=64, amp=0.1, epsilon=0.2, unique_id=1 ): 
    # params to pass into config for ML_Fluid 
    params_dict = {
        'PREPROCESS' : { 
            'sf_filename' : f'dgsf_{dt}_{xct}_{yct}_{amp}_{epsilon}',
            'training_length'         : training_length,
            'training_data_filename'  : f'dgsf_{dt}_{xct}_{yct}_{amp}_{epsilon}_id{unique_id}.TRAIN',
            'testing_data_filename'  : f'dgsf_{dt}_{xct}_{yct}_{amp}_{epsilon}_id{unique_id}.TEST'  
        },
        'TRAIN' : { 
            'training_data_filename'  : f'dgsf_{dt}_{xct}_{yct}_{amp}_{epsilon}_id{unique_id}.TRAIN',
            'trained_model_params'    : trained_model_params,
            'trained_model_filename'  : trained_model_filename,
        },
        'TEST' : { 
            'testing_data_filename'   : f'dgsf_{dt}_{xct}_{yct}_{amp}_{epsilon}_id{unique_id}.TEST',
            'testing_length'          : testing_length,
            'trained_model_filename'  : trained_model_filename, 
            'output_filenames'        : { # Davis wuz here!!
                'stream_function_estimated' : f'{stream_function_prefix}_id{unique_id}.est',
                'stream_function_actual'    : f'{stream_function_prefix}_id{unique_id}.actual',
            }
        },
        'GENERATE_STREAM_FUNCTION_FIELDS' : {
            'dt' : dt, 
            'elapsedTime' : elapsedTime, 
            'xct': xct, 
            'yct': yct,
            'amp': amp,
            'epsilon': epsilon,
            'stream_function_filename' : f'dgsf_{dt}_{xct}_{yct}_{amp}_{epsilon}'
        }
    }
    return params_dict


def load_trajectories(experiment_prefixes, num_samples, elapsed_time, dt, dim, noise): 
    # e.g. 
    # experiment_prefix = "QGds0.01di0.05dm0.03p0.5rs1000sr3.0dens0.1lr0.5insc0.1reg0.1"
    import config, util 
    states = dict()
    for experiment_prefix in experiment_prefixes: 
        velocity_func_filenames = [f"{experiment_prefix}.uvinterp.actual"]
        vfuncs_list = []
        for velocity_func_filename in velocity_func_filenames:
            velocity_func_fullpath = os.path.join( config.INTERP_VELOCITY_PATH_DIR, velocity_func_filename)
            vfuncs_list.append( util.load_velocity_field(velocity_func_fullpath) )

        state = compare_trajectories.calculate_trajectory(vfuncs_list, num_samples, elapsed_time, dt, dim, noise, concentrated=False)
        states[experiment_prefix] = state
    return states


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectral_radius', const=3.5, type=float)
    parser.add_argument('--resSize', const=2000, type=int)
    parser.add_argument('--training_length', const=7000, type=int)
    parser.add_argument('--init_length', const=200, type=int)
    parser.add_argument('--ridge_reg', const=1.0, type=float)
    parser.add_argument('--density', const=0.5, type=float)
    parser.add_argument('--input_scaling', const=0.5, type=float)
    parser.add_argument('--leaking_rate', const=0.0, type=float)
    parser.add_argument('--id', type=int)
    args = parser.parse_args()
    # spectral_radius = args.spectral_radius
    # resSize = args.resSize
    # training_length = args.training_length    
    # init_length = args.init_length
    # ridge_reg = args.ridge_reg    
    # unique_id = args.id 


    switch_to_qgftle_src_dir() 
    import compare_trajectories
    # experiment_prefix = 'QGds0.01di0.05dm0.03p0.5rs5000sr1.4dens0.5lr0.0insc0.1reg1.0_id0'
    experiment_prefix = 'dgsf_0.01_200_100_0.1_0.2_1000_2.0'
    num_samples = 1
    elapsed_time = 99
    dt = 0.01
    dim = 0
    noise = 0.1
    trajectories = load_trajectories([experiment_prefix], num_samples, elapsed_time, dt, dim, noise)
    traj = trajectories[experiment_prefix]
    traj = np.transpose(traj, (2,3,1,0))
    traj = traj.squeeze(axis=3)
    
    train_traj = traj[:args.training_length , :, :]
    test_traj = traj[args.training_length :, :, -1]
    
    # import pdb;pdb.set_trace() 

    switch_to_mlfluids_src_dir()
    import util, config
    # util.save_data( train_traj, os.path.join(config.PREPROCESS_INPUT_PATH_DIR, experiment_prefix + '_trajs_{0}.TRAIN'.format(args.id ) ))
    # util.save_data( test_traj, os.path.join(config.PREPROCESS_INPUT_PATH_DIR, experiment_prefix + '_trajs_{0}.TEST'.format(args.id ) ))

    import MESN
    from MESN import MultiEchoStateNetwork
    trained_model_params = { 
        "initLen"           : args.init_length, 
        "resSize"           : args.resSize, 
        "partial_know"      : False, 
        "noise"             : 1e-2, 
        "density"           : args.density, 
        "spectral_radius"   : args.spectral_radius, 
        "leaking_rate"      : args.leaking_rate, 
        "input_scaling"     : args.input_scaling, 
        "ridgeReg"          : args.ridge_reg, 
        "mute"              : False 
    }
    mesn = MultiEchoStateNetwork(loaddata = train_traj, **trained_model_params)
    mesn.train()
    mesn.test(testing_data = test_traj)
    traj_est = mesn.v_.T
    traj_actual = mesn.v_tgt_
    
    # util.save_data(traj_est, os.path.join(config.RESULTS_PATH_DIR, experiment_prefix + '_{0}.est'.format(args.id )))
    # util.save_data(traj_actual, os.path.join(config.RESULTS_PATH_DIR, experiment_prefix + '_{0}.actual'.format(args.id )))
    
    # import pdb;pdb.set_trace()
    plt.figure()
    plt.scatter(traj_est[:100,0], traj_est[:100,1], color='b')
    plt.scatter(traj_actual[:100,0], traj_actual[:100,1], color='r') 
    plt.savefig(os.path.join(config.RESULTS_PATH_DIR, experiment_prefix + '_traj_{0}.compare.jpg'.format(args.id )))
    mesn.plot(length=5000, name=os.path.join(config.RESULTS_PATH_DIR, experiment_prefix + '_traj_ts_{0}.compare'.format(args.id )))
    

