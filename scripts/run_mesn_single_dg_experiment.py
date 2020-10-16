import pickle 
import os 
import sys 
import shutil 
import argparse 
import pandas as pd 
from pprint import pprint 

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
    
    switch_to_qgftle_src_dir() 
    import compare_trajectories
    experiment_prefix = 'QGds0.01di0.05dm0.03p0.5rs5000sr1.4dens0.5lr0.0insc0.1reg1.0_id0'
    num_samples = 10
    elapsed_time = 100 
    dt = 0.01
    dim = 1
    noise = 0.1
    trajectories = load_trajectories([experiment_prefix], num_samples, elapsed_time, dt, dim, noise)
    traj = trajectories[experiment_prefix]
    traj = traj.reshape( traj.shape[2], traj.shape[1], traj.shape[3] )
    
    train_traj = traj[:, :, :5]
    test_traj = traj[:, :, 5:]

    switch_to_mlfluids_src_dir()
    import train 
    train.train_ESN()
