import pickle 
import os 
import sys 
import shutil 
from pprint import pprint
import pandas as pd 
from multiprocessing import Pool
import argparse 

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

def generate_experiment_id(): 
    # from experiment parmas create unique string to be used as key for config dicts 
    pass

def generate_ml_fluid_params(  training_length, trained_model_params, testing_length, 
                               trained_model_filename, stream_function_prefix, 
                               xct, yct, dt, ds, di, dm, pertamp, unique_id ): 
    # params to pass into config for ML_Fluid 
    params_dict = {
        'PREPROCESS' : { 
            # 'MATLAB_filename' : 'QG_psi_ds0.02_di0.02_dm0.02_pertamp0.3.mat', 
            # 'MATLAB_filename' : 'QGpsi_100_200_dt0.01_ds0.03_di0.00_dm0.00_pertamp0.3.mat',
            'MATLAB_filename' :f'QGpsi_{xct}_{yct}_dt{dt:.2f}_ds{ds:.2f}_di{di:.2f}_dm{dm:.2f}_pertamp{pertamp:.1f}.mat',
            # 'MATLAB_filename' : 'QGpsi_128_256_dt0.01_ds0.03_di0.01_dm0.00_pertamp0.3.mat'
            # 'MATLAB_filename' : 'QGpsi_128_256_dt0.01_ds0.03_di0.00_dm0.00_pertamp0.3',
            'training_length'         : training_length,
            'training_data_filename'  : f'QGpsi_{xct}_{yct}_dt{dt:.2f}_ds{ds:.2f}_di{di:.2f}_dm{dm:.2f}_pertamp{pertamp:.1f}_id{unique_id}.TRAIN',
            'testing_data_filename'  : f'QGpsi_{xct}_{yct}_dt{dt:.2f}_ds{ds:.2f}_di{di:.2f}_dm{dm:.2f}_pertamp{pertamp:.1f}_id{unique_id}.TEST'  
        },
        'TRAIN' : { 
            'training_data_filename'  : f'QGpsi_{xct}_{yct}_dt{dt:.2f}_ds{ds:.2f}_di{di:.2f}_dm{dm:.2f}_pertamp{pertamp:.1f}_id{unique_id}.TRAIN',
            'trained_model_params'    : trained_model_params,
            'trained_model_filename'  : trained_model_filename,
        },
        'TEST' : { 
            'testing_data_filename'   : f'QGpsi_{xct}_{yct}_dt{dt:.2f}_ds{ds:.2f}_di{di:.2f}_dm{dm:.2f}_pertamp{pertamp:.1f}_id{unique_id}.TEST',
            'testing_length'          : testing_length,
            'trained_model_filename'  : trained_model_filename, 
            'output_filenames'        : { # Davis wuz here!!
                'stream_function_estimated' : f'{stream_function_prefix}_id{unique_id}.est',
                'stream_function_actual'    : f'{stream_function_prefix}_id{unique_id}.actual',
            }

        }
    }
    return params_dict

def generate_qgftle_params( stream_function_prefix, mapped_dt=20, dt=0.1, iters=10, xct=128, yct=64, unique_id=1 ):
    """
    params to pass into config for QG_FTLE
    stream_function_prefix - part of the stream function file name 
    **BEFORE** est and actual, e.g. QGds02di02dm02p3.1000.0p3

    return a dict with keys: <stream_function_prefix>.actual and <stream_function_prefix>.est
    """

    params_dict = dict()
    for actual_flag in ['actual', 'est']:   
        params_dict[ f"{stream_function_prefix}_id{unique_id}.{actual_flag}" ] = { 
            'GENERATE_VELOCITY_FIELDS' : {
                'stream_function_filename' : f"{stream_function_prefix}_id{unique_id}.{actual_flag}", 
                'velocity_filename' : f"{stream_function_prefix}_id{unique_id}.uv.{actual_flag}",
                'velocity_func_filename' : f"{stream_function_prefix}_id{unique_id}.uvinterp.{actual_flag}"
            }, 
            'GENERATE_FTLE_MAPPING' : {
                'iters' : iters, 
                'mapped_dt' : mapped_dt,
                'dt' : dt,
                'xct': xct, 
                'yct': yct,
                # Davis wuz here!!
                'velocity_func_filename': f"{stream_function_prefix}_id{unique_id}.uvinterp.{actual_flag}",
                'mapping_path_dir': f"{stream_function_prefix}_id{unique_id}.{actual_flag}"
            },
            'GENERATE_FTLE_FIELDS': {
                'iters' : iters,
                'xct': xct, 
                'yct': yct,
                'mapping_path_dir': f"{stream_function_prefix}_id{unique_id}.{actual_flag}",
                'ftle_path_dir' : f"{stream_function_prefix}_id{unique_id}.{actual_flag}"
            }, 
            'GENERATE_FTLE_ANIMATIONS': {
                'iters' : iters,
                'xct': xct, 
                'yct': yct,
                'ftle_path_dir' : f"{stream_function_prefix}_id{unique_id}.{actual_flag}",
                'ftle_animation_filename' : f"{stream_function_prefix}_id{unique_id}.{actual_flag}.gif",
            }
        }
    return params_dict



def run_experiment_without_ftle(resSize, spectral_radius, noise=1e-2, density=1e-1, 
                                leaking_rate=0.2, input_scaling=0.3, ridgeReg=0.01,
                                xct=64, yct=128, dt=0.01, ds=0.04, di=0.02, dm=0.0, pertamp=0.2, 
                                training_length=10000, init_length=0, unique_id=1): 
    ## Preparing parameters for entire experiment
    ## ml fluid params should link to qgftle params by  
    ## stream_function_estimated and stream_function_actual
    trained_model_params = { 
        "initLen"           : init_length, 
        "resSize"           : resSize, 
        "partial_know"      : False, 
        "noise"             : noise, 
        "density"           : density, 
        "spectral_radius"   : spectral_radius, 
        "leaking_rate"      : leaking_rate, 
        "input_scaling"     : input_scaling, 
        "ridgeReg"          : ridgeReg, 
        "mute"              : False 
    }
    training_length = training_length
    testing_length = 14000
        
    # these actually don't matter because they are only used for FTLE calculation
    mapped_dt = 5
    iters = 10

    stream_function_prefix = \
        f"QGds{ds:.2f}di{di:.2f}dm{dm:.2f}p{pertamp:.1f}rs{resSize}sr{spectral_radius:.1f}dens{density:.1f}lr{leaking_rate:.1f}insc{input_scaling:.1f}reg{ridgeReg:.1f}"
    
    ######################################
    print("""\n====================================================================================""")
    print("RUNNING NEW EXPERIMENT (WITHOUT FTLE): ")
    print("trained_model_params: ")
    pprint(trained_model_params)
    print("qg parameters:")
    print(f"""
    training_length = {training_length}
    testing_length = {training_length}
    dt ={dt}
    xct = {xct}
    yct = {yct}
    ds = {ds}
    di = {di}
    dm = {dm}
    pertamp = {pertamp}
    mapped_dt = {mapped_dt}
    iters = {iters}
    """)
    print("stream_function_prefix:")
    print(stream_function_prefix)
    ######################################



    trained_model_filename = f'{stream_function_prefix}_id{unique_id}.ESN'
    ml_fluid_params_dict = generate_ml_fluid_params(    training_length=training_length, 
                                        trained_model_params=trained_model_params,
                                        testing_length=testing_length,
                                        trained_model_filename=trained_model_filename,
                                        stream_function_prefix=stream_function_prefix,
                                        xct=xct, yct=yct, dt=dt, ds=ds, di=di, dm=dm, pertamp=pertamp, unique_id=unique_id
                                    )
    
    
    qgftle_params_dict = generate_qgftle_params( stream_function_prefix=stream_function_prefix,
                                                 mapped_dt=mapped_dt, dt=dt, iters=iters, xct=xct,yct=yct, unique_id=unique_id )

    """
    1. Running ML_Fluids procedure - train a model on training data and generate 
    and testing sample to compare. Specifically, we must
        a. Preprocess data 
        b. train data 
        c. test data 
    """
    # ensure correct directory
    switch_to_mlfluids_src_dir()
    print(os.getcwd())
    # a. preprocessing data
    import preprocess
    preprocess.preprocess_input_data( **ml_fluid_params_dict['PREPROCESS'] )
    # b. training data
    import train 
    train.train_ESN(**ml_fluid_params_dict['TRAIN'])
    # c. testing data 
    import test 
    test.test_ESN(**ml_fluid_params_dict['TEST'])
    switch_to_home_dir()

    ## COPY STREAM FUNCTION FILES OVER TO QG_FTLE INPUT DIRECTORY
    MLFluid_streamfunction_fullpaths = [os.path.join(ML_Fluid_RESULTS_fullpath,streamfunction_filename) 
                        for _,streamfunction_filename in ml_fluid_params_dict['TEST']['output_filenames'].items()]
    for streamfunction_fullpath in MLFluid_streamfunction_fullpaths:
        # qgftle_streamfunction_fullpath = os.path.join(QG_FTLE_INPUTS_fullpath, os.path.basename(streamfunction_fullpath))
        shutil.copy(streamfunction_fullpath, QG_FTLE_INPUTS_fullpath)



    # d. compare stream function files
    switch_to_qgftle_src_dir() 
    import compare_streamfunctions
    data = compare_streamfunctions.compare_stream_functions(max_iters=12000, 
                                                            sf_filenames=[streamfunction_filename for _,streamfunction_filename 
                                                                in ml_fluid_params_dict['TEST']['output_filenames'].items()])
   

    for params_key, params_dict in qgftle_params_dict.items(): 
        # generate velocity fields 
        import generate_velocity_fields 
        generate_velocity_fields.generate_velocity_fields( **params_dict['GENERATE_VELOCITY_FIELDS']) 

    
    # import pdb;pdb.set_trace()

    for params_key, params_dict in qgftle_params_dict.items():
        # ensure correct directory
        switch_to_qgftle_src_dir()
        # a. generate velcoity fields
        # import generate_velocity_fields
        # generate_velocity_fields.generate_velocity_fields( **params_dict['GENERATE_VELOCITY_FIELDS'] )
        # b. generate ftle mappings
        import generate_FTLE_mapping
        generate_FTLE_mapping.generate_mapping_files( **params_dict['GENERATE_FTLE_MAPPING'] )
        # c. gerenate ftle files
        import generate_FTLE_fields 
        generate_FTLE_fields.generate_FTLE_fields( **params_dict['GENERATE_FTLE_FIELDS'] )

    # d. compare ftle files     
    # ensure correct directory
    # switch_to_qgftle_src_dir()
    # import compare_FTLE_fields 
    # ssi = compare_FTLE_fields.compare_FTLE_animation(iters=10, ftle_path_dirs=[ params_dict['GENERATE_FTLE_FIELDS']['ftle_path_dir'] 
    #                                                                       for _ , params_dict in qgftle_params_dict.items() ],
    #                                                   ftle_animation_filename=f"{stream_function_prefix}.gif")
    switch_to_home_dir()
    return data 

    print("END OF EXPERIMENT.")
    print("""====================================================================================\n""")
    



if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--spectral_radius', type=float)
    parser.add_argument('--resSize', type=int)
    parser.add_argument('--training_length', type=int)
    parser.add_argument('--init_length', type=int)
    parser.add_argument('--ridge_reg', type=float)
    parser.add_argument('--density',type=float)
    parser.add_argument('--input_scaling',type=float)
    parser.add_argument('--leaking_rate',type=float)
    parser.add_argument('--id', type=int)
    args = parser.parse_args()
    spectral_radius = args.spectral_radius
    resSize = args.resSize
    training_length = args.training_length    
    init_length = args.init_length
    ridge_reg = args.ridge_reg
    density = args.density
    leaking_rate = args.leaking_rate
    input_scaling = args.input_scaling    
    unique_id = args.id 

    # density = 0.1
    # leaking_rate = 0.5
    # input_scaling = 0.1
    
    
    ds = 0.01
    di = 0.05
    dm = 0.03
    pertamp = 0.5
    
    xct=64
    yct=128
    dt=0.01

    
    supdata = {}
    data = run_experiment_without_ftle(resSize=resSize, spectral_radius=spectral_radius, 
                             noise=1e-2, density=density, leaking_rate=leaking_rate, 
                             input_scaling=input_scaling, ridgeReg=ridge_reg,
                             xct=xct, yct=yct, dt=dt, ds=ds, di=di, dm=dm, pertamp=pertamp, 
                             training_length=training_length, init_length=init_length, unique_id=unique_id)
           
    supdata[ (resSize, spectral_radius, training_length, init_length, ridge_reg, density, leaking_rate, input_scaling, unique_id) ] = data
    data_df = pd.DataFrame.from_dict(supdata)
    # import pdb;pdb.set_trace()
    data_df.to_pickle(os.path.join('./experiments/', 
        f"qg_id{unique_id}"))

            
    print('done.')





