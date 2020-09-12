import pickle 
import os 
import sys 
import shutil 

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
                                dt=0.1, elapsedTime=2500, xct=128, yct=64, amp=0.1, epsilon=0.2 ): 
    # params to pass into config for ML_Fluid 
    params_dict = {
        'PREPROCESS' : { 
            'sf_filename' : f'dgsf_{dt}_{xct}_{yct}_{amp}_{epsilon}',
            'training_length'         : training_length,
            'training_data_filename'  : f'dgsf_{dt}_{xct}_{yct}_{amp}_{epsilon}.TRAIN',
            'testing_data_filename'  : f'dgsf_{dt}_{xct}_{yct}_{amp}_{epsilon}.TEST'  
        },
        'TRAIN' : { 
            'training_data_filename'  : f'dgsf_{dt}_{xct}_{yct}_{amp}_{epsilon}.TRAIN',
            'trained_model_params'    : trained_model_params,
            'trained_model_filename'  : trained_model_filename,
        },
        'TEST' : { 
            'testing_data_filename'   : f'dgsf_{dt}_{xct}_{yct}_{amp}_{epsilon}.TEST',
            'testing_length'          : testing_length,
            'trained_model_filename'  : trained_model_filename, 
            'output_filenames'        : { # Davis wuz here!!
                'stream_function_estimated' : f'{stream_function_prefix}.est',
                'stream_function_actual'    : f'{stream_function_prefix}.actual',
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

def generate_qgftle_params( stream_function_prefix, mapped_dt=20, dt=0.1, iters=10, xct=128, yct=64 ):
    """
    params to pass into config for QG_FTLE
    stream_function_prefix - part of the stream function file name 
    **BEFORE** est and actual, e.g. QGds02di02dm02p3.1000.0p3

    return a dict with keys: <stream_function_prefix>.actual and <stream_function_prefix>.est
    """

    params_dict = dict()
    for actual_flag in ['actual', 'est']:   
        params_dict[ f"{stream_function_prefix}.{actual_flag}" ] = { 
            'GENERATE_VELOCITY_FIELDS' : {
                'stream_function_filename' : f"{stream_function_prefix}.{actual_flag}", 
                'velocity_filename' : f"{stream_function_prefix}.uv.{actual_flag}",
                'velocity_func_filename' : f"{stream_function_prefix}.uvinterp.{actual_flag}"
            }, 
            'GENERATE_FTLE_MAPPING' : {
                'iters' : iters, 
                'mapped_dt' : mapped_dt,
                'dt' : dt,
                'xct': xct, 
                'yct': yct,
                # Davis wuz here!!
                'velocity_func_filename': f"{stream_function_prefix}.uvinterp.{actual_flag}",
                'mapping_path_dir': f"{stream_function_prefix}.{actual_flag}"
            },
            'GENERATE_FTLE_FIELDS': {
                'iters' : iters,
                'xct': xct, 
                'yct': yct,
                'mapping_path_dir': f"{stream_function_prefix}.{actual_flag}",
                'ftle_path_dir' : f"{stream_function_prefix}.{actual_flag}"
            }, 
            'GENERATE_FTLE_ANIMATIONS': {
                'iters' : iters,
                'xct': xct, 
                'yct': yct,
                'ftle_path_dir' : f"{stream_function_prefix}.{actual_flag}",
                'ftle_animation_filename' : f"{stream_function_prefix}.{actual_flag}.gif",
            }
        }
    return params_dict


def run_experiment(resSize, spectral_radius): 
    ## Preparing parameters for entire experiment
    ## ml fluid params should link to qgftle params by  
    ## stream_function_estimated and stream_function_actual
    trained_model_params = { 
        "initLen"           : 0, 
        "resSize"           : resSize, 
        "partial_know"      : False, 
        "noise"             : 1e-2, 
        "density"           : 1e-1, 
        "spectral_radius"   : spectral_radius, 
        "leaking_rate"      : 0.2, 
        "input_scaling"     : 0.3, 
        "ridgeReg"          : 0.01, 
        "mute"              : False 
    }
    training_length = 10000
    testing_length = 10000
    dt = 0.05
    elapsedTime = 2100
    xct = 400
    yct = 200
    amp = 0.1
    epsilon = 0.2
    stream_function_prefix = f"dgsf_{dt}_{xct}_{yct}_{amp}_{epsilon}_{resSize}_{spectral_radius:.1f}"
    trained_model_filename = f'{stream_function_prefix}.ESN'
    ml_fluid_params_dict = generate_ml_fluid_params(    training_length=training_length, 
                                        trained_model_params=trained_model_params,
                                        testing_length=testing_length,
                                        trained_model_filename=trained_model_filename,
                                        stream_function_prefix=stream_function_prefix,
                                        dt=dt, elapsedTime=elapsedTime, xct=xct, yct=yct,
                                        amp=amp, epsilon=epsilon
                                    )
    mapped_dt = 20
    iters = 10
    qgftle_params_dict = generate_qgftle_params( stream_function_prefix=stream_function_prefix,
                                                 mapped_dt=mapped_dt, dt=dt, iters=iters, xct=xct,yct=yct )
    """
    0. Generate double gyre raw data (src code is located in QG_FTLE directory)
    Transfer raw data file to ML_Fluids directory for step 1. 
    """
    # ensure correct directory
    switch_to_qgftle_src_dir()
    import double_gyre
    double_gyre.generate_streamfunction_values(**ml_fluid_params_dict['GENERATE_STREAM_FUNCTION_FIELDS'] )
    switch_to_home_dir()
    ## COPY DOUBLE GYRE STREAM FUNCTION FILES OVER TO ML_FLUIDS INPUT DIRECTORY    
    dg_raw_streamfunction_filename = ml_fluid_params_dict['GENERATE_STREAM_FUNCTION_FIELDS']['stream_function_filename']
    dg_raw_streamfunction_fullpath = os.path.join(QG_FTLE_INPUTS_fullpath, dg_raw_streamfunction_filename)
    shutil.copy(dg_raw_streamfunction_fullpath, ML_Fluid_raw_inputs_fullpath)


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
    preprocess.preprocess_numpy_input_data( **ml_fluid_params_dict['PREPROCESS'] )
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


    """
    2. Running QG_FTLE procedure **FOR BOTH estimated and actual stream function files
    - interpolate discrete velocity fields to continuous, using them to generate ftle mappings and fields
        a. generate velcoity fields 
        b. generate ftle mappings 
        c. generate ftle fields 
    """
    # ensure correct directory
    for params_key, params_dict in qgftle_params_dict.items():
        # ensure correct directory
        switch_to_qgftle_src_dir()
        # a. generate velcoity fields
        import generate_velocity_fields
        generate_velocity_fields.generate_velocity_fields( **params_dict['GENERATE_VELOCITY_FIELDS'] )
        # b. generate ftle mappings
        import generate_FTLE_mapping
        generate_FTLE_mapping.generate_mapping_files( **params_dict['GENERATE_FTLE_MAPPING'] )
        # c. gerenate ftle files
        import generate_FTLE_fields 
        generate_FTLE_fields.generate_FTLE_fields( **params_dict['GENERATE_FTLE_FIELDS'] )
    # d. compare ftle files     
    # ensure correct directory
    switch_to_qgftle_src_dir()
    import compare_FTLE_fields 
    ssi = compare_FTLE_fields.compare_FTLE_animation(iters=10, ftle_path_dirs=[ params_dict['GENERATE_FTLE_FIELDS']['ftle_path_dir'] 
                                                                            for _ , params_dict in qgftle_params_dict.items() ],
                                                        ftle_animation_filename=f"{stream_function_prefix}.gif")
    switch_to_home_dir()
    return ssi 




def run_experiment_without_ftle(resSize, spectral_radius): 
    ## Preparing parameters for entire experiment
    ## ml fluid params should link to qgftle params by  
    ## stream_function_estimated and stream_function_actual
    trained_model_params = { 
        "initLen"           : 0, 
        "resSize"           : resSize, 
        "partial_know"      : False, 
        "noise"             : 1e-2, 
        "density"           : 1e-1, 
        "spectral_radius"   : spectral_radius, 
        "leaking_rate"      : 0.2, 
        "input_scaling"     : 0.3, 
        "ridgeReg"          : 0.01, 
        "mute"              : False 
    }
    training_length = 500
    testing_length = 500
    dt = 0.1
    elapsedTime = 700
    xct = 160
    yct = 80
    amp = 0.1
    epsilon = 0.2
    stream_function_prefix = f"dgsf_{dt}_{xct}_{yct}_{amp}_{epsilon}_{resSize}_{spectral_radius:.1f}"
    trained_model_filename = f'{stream_function_prefix}.ESN'
    ml_fluid_params_dict = generate_ml_fluid_params(    training_length=training_length, 
                                        trained_model_params=trained_model_params,
                                        testing_length=testing_length,
                                        trained_model_filename=trained_model_filename,
                                        stream_function_prefix=stream_function_prefix,
                                        dt=dt, elapsedTime=elapsedTime, xct=xct, yct=yct,
                                        amp=amp, epsilon=epsilon
                                    )
    mapped_dt = 20
    iters = 10
    qgftle_params_dict = generate_qgftle_params( stream_function_prefix=stream_function_prefix,
                                                 mapped_dt=mapped_dt, dt=dt, iters=iters, xct=xct,yct=yct )


    """
    0. Generate double gyre raw data (src code is located in QG_FTLE directory)
    Transfer raw data file to ML_Fluids directory for step 1. 
    """
    # ensure correct directory
    switch_to_qgftle_src_dir()
    import double_gyre
    double_gyre.generate_streamfunction_values(**ml_fluid_params_dict['GENERATE_STREAM_FUNCTION_FIELDS'] )
    switch_to_home_dir()
    ## COPY DOUBLE GYRE STREAM FUNCTION FILES OVER TO ML_FLUIDS INPUT DIRECTORY    
    dg_raw_streamfunction_filename = ml_fluid_params_dict['GENERATE_STREAM_FUNCTION_FIELDS']['stream_function_filename']
    dg_raw_streamfunction_fullpath = os.path.join(QG_FTLE_INPUTS_fullpath, dg_raw_streamfunction_filename)
    shutil.copy(dg_raw_streamfunction_fullpath, ML_Fluid_raw_inputs_fullpath)


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
    preprocess.preprocess_numpy_input_data( **ml_fluid_params_dict['PREPROCESS'] )
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


    """
    2. Running QG_FTLE procedure **FOR BOTH estimated and actual stream function files
    - interpolate discrete velocity fields to continuous, using them to generate ftle mappings and fields
        a. generate velcoity fields 
        b. generate ftle mappings 
        c. generate ftle fields 
    """
    # ensure correct directory
    for params_key, params_dict in qgftle_params_dict.items():
        # ensure correct directory
        switch_to_qgftle_src_dir()
        # a. generate velcoity fields
        import generate_velocity_fields
        generate_velocity_fields.generate_velocity_fields( **params_dict['GENERATE_VELOCITY_FIELDS'] )
    #     # b. generate ftle mappings
    #     import generate_FTLE_mapping
    #     generate_FTLE_mapping.generate_mapping_files( **params_dict['GENERATE_FTLE_MAPPING'] )
    #     # c. gerenate ftle files
    #     import generate_FTLE_fields 
    #     generate_FTLE_fields.generate_FTLE_fields( **params_dict['GENERATE_FTLE_FIELDS'] )
    # # d. compare ftle files     
    # # ensure correct directory
    # switch_to_qgftle_src_dir()
    # import compare_FTLE_fields 
    # ssi = compare_FTLE_fields.compare_FTLE_animation(iters=10, ftle_path_dirs=[ params_dict['GENERATE_FTLE_FIELDS']['ftle_path_dir'] 
    #                                                                         for _ , params_dict in qgftle_params_dict.items() ],
    #                                                     ftle_animation_filename=f"{stream_function_prefix}.gif")





    switch_to_home_dir()
    # return ssi 



def run_ftle_experiment(stream_function_prefix):
    """
    2. Running QG_FTLE procedure **FOR BOTH estimated and actual stream function files
    - interpolate discrete velocity fields to continuous, using them to generate ftle mappings and fields
        a. generate velcoity fields 
        b. generate ftle mappings 
        c. generate ftle fields 
    """
    dt = 0.05
    xct = 400
    yct = 200
    mapped_dt = 20
    iters = 10
    qgftle_params_dict = generate_qgftle_params( stream_function_prefix=stream_function_prefix,
                                                 mapped_dt=mapped_dt, dt=dt, iters=iters, xct=xct,yct=yct )
    # ensure correct directory
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
    switch_to_qgftle_src_dir()
    import compare_FTLE_fields 
    ssi = compare_FTLE_fields.compare_FTLE_animation(iters=10, ftle_path_dirs=[ params_dict['GENERATE_FTLE_FIELDS']['ftle_path_dir'] 
                                                                            for _ , params_dict in qgftle_params_dict.items() ],
                                                        ftle_animation_filename=f"{stream_function_prefix}.gif")
    switch_to_home_dir()
    return ssi 

if __name__ == '__main__':
    

    # resSize = 1000
    # spectral_radius = 2.0

    # RUNNING MULTIPLE WHOLE EXPERIMENTS
    # results = []
    # for spectral_radius in [1.0, 3.0]:
    #     for resSize in [100,1000,10000,100000]: 
    #         try: # wrap in try except block in case of memory issues...
    #             ssi = run_experiment(resSize=resSize, spectral_radius=spectral_radius)
    #             results.append([ssi, resSize, spectral_radius])
    #         except MemoryError:
    #             print('memory error...skipping')

    # import pandas as pd 
    # df = pd.DataFrame(results,columns=['SSI','resSize','spectral_density'])  
    # df = df.pivot_table(values='SSI',index='spectral_density',columns='resSize')
    # df.to_clipboard()
    # df.to_csv('experiment_results.csv')
    # import pdb;pdb.set_trace()

    ## RUNNING WHOLE  EXPERIMENT
    # resSize = 5000
    # spectral_radius = 2.0

    # for spectral_radius in [1.33, 1.66, 2.0, 2.33, 2.66, 3.0, 3.33]:
    for spectral_radius in [2.0]:
         for resSize in [13000]: 
            # try: 
            run_experiment_without_ftle(resSize=resSize, spectral_radius=spectral_radius)
            # except: 
            #     pass
    
    import pdb;pdb.set_trace()

    ## RUNNING ONLY PART 1 EXPERIMENT FOR GENERATING FTLE PLOTS
    # resSize = 100000
    # spectral_radius = 2.0
    # run_ftle_experiment(resSize=resSize, spectral_radius=spectral_radius)    
    # print( 'done.')
    # import pdb;pdb.set_trace()

    ## RUNNING ONLY PART 2 EXPERIMENT FOR GENERATING FTLE PLOTS
    # ssi = run_ftle_experiment(stream_function_prefix="dgsf_0p1_128_64_0p1_0p21000_2.0")    
    # import pdb;pdb.set_trace()
