import argparse 
import pickle 
import os 
import logging 
from scipy import io
import numpy as np 
import joblib
logging.basicConfig(level=logging.WARNING)

def load_data( fullpath ): 
    """
    load python object, 
    good for loading relatively smaller data
    """
    print('loading data from ', fullpath, '...')
    with open( fullpath, 'rb' ) as _file: 
        obj = pickle.load( _file ) 
    print('loaded data!')
    return obj

def save_data( _object, fullpath ): 
    """
    Save python object to fullpath, 
    good for saving relatively smaller data
    """
    print('Saving data to ', fullpath, '...')
    with open( fullpath , 'wb') as _file: 
        pickle.dump( _object, _file , protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved data fields')

def load_model( fullpath ): 
    """load saved ESNs"""
    model = joblib.load(fullpath)
    return model 

def save_model( _object, fullpath ): 
    """good for saving trained ESNs"""
    print('Saving model to ', fullpath, '...')
    joblib.dump(_object, fullpath)  
    print('Saved model')

def flatten_time_series( data ):
    """
    Convert N-D time series to 1-D time series 
    data : (N+1)-dimensional numpy array (LAST dimension corresponds to time) 
    rtype: 2-dimensional numpy array (LAST dimension corresponds to time)
    """
    if max(data.shape) != data.shape[-1]: 
        logging.warning("Last dimension is not the largest, check order of dimensions")
    flattened_data = data.reshape( -1, data.shape[-1] )
    return flattened_data

def set_up_dir(path_dir):
    """
    Create new directory if directory does not exist, else ignore
    """
    try: 
        os.makedirs(path_dir)
        print("Created new directory ", path_dir)
    except FileExistsError: 
        pass

def reset_dir(path_dir): 
    """
    Create new directory if directory does not exist, else wipe contents of directory
    """
    try: 
        os.makedirs(path_dir)
        print("Created new directory ", path_dir)
    except FileExistsError: 
        filelist = glob.glob(os.path.join(path_dir, "mapping*"))
        for f in filelist:
            os.remove(f)
        # shutil.rmtree(path_dir)
        # os.makedirs(path_dir)
        print("Reset directory ", path_dir)

def split_training_testing( data, training_length, axis ): 
    """
    Split numpy array on set axis
    Raises exception if split does not result in 2 data chunks (e.g. if training length is too long)
    """
    [training_data, testing_data] = np.split( data, [training_length], axis=axis)
    if testing_data.shape[axis] == 0: 
        raise Exception('training_length ({0}) must be less than size of axis ({2}) in data with size ({1})'.format(training_length, str(data.shape), axis))
    return training_data, testing_data

def load_mat_file( mat_fullpath, var_name='Psi_ts'):
    # loading .mat files, annoying bc the variable name is needed
    # usually we are loading stream function values, so default 
    # var_name to 'Psi_ts', but subject to change
    try:
        if mat_fullpath[-4:] == '.mat': 
            print(mat_fullpath)
            mat = io.loadmat(mat_fullpath[:-4])[var_name]
    except NotImplementedError as e: 
        raise Exception("THIS DOESN'T WORK..!!")
        print(e) 
        import h5py
        with h5py.File(mat_fullpath, 'r') as file:
            mat = np.asarray(file['Psi_ts'])
            mat = mat.reshape(mat.shape[2], mat.shape[1], mat.shape[0])
            print(f'loaded mat file, shape: {mat.shape}')
    return mat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--load_mat_file', '-l')
    
    parser.add_argument('--mat_fullpath', '-f')
    parser.add_argument('--var_name', '-v', required=False)
    args = parser.parse_args()

    if args.mat_fullpath and args.var_name: 
        load_mat_file( args.mat_fullpath, args.var_name )
