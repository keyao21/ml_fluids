import argparse
import os
import numpy as np
from config import *
import util 

def compare_stream_functions(max_iters, sf_filenames):
    """
    sf_filesnames : list of string stream function filenames
        - if both estimated and actual, pass in estimated version first
    """
    sfs = []
    for sf_filename in sf_filenames:
        sf_fullpath = os.path.join( INPUT_PATH_DIR, sf_filename)
        sf = util.load_sf_field(sf_fullpath=sf_fullpath)
        if sf.shape[-1] < max_iters:
            print(f"CANNOT COMPARE STREAM FUNCTIONS: max_iters {max_iters}"
                    " is greater than total time steps in stream function {sf_filename}")
            return
        sfs.append(sf)
    [sfest, sfactual] = sfs
    mses = [ sum(sum((sfest[:,:,i]-sfactual[:,:,i])**2))
            *(1./(sfactual.shape[0]*sfactual.shape[1])) \
                        for i in range(sfactual.shape[-1])]

    data = {}
    for i in range(1, int(np.floor(max_iters/100))):
        upper_i = i*100
        data[upper_i] = np.mean(mses[:upper_i])

    print( data ) 
    return data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=int)
    args = parser.parse_args()

    if args.test == 1:
        max_iters = 2000
        sf_filenames = ["QGds0.03di0.02dm0.00p0.3rs1000sr3.0dens0.1lr0.5insc0.1reg0.1.est", \
                        "QGds0.03di0.02dm0.00p0.3rs1000sr3.0dens0.1lr0.5insc0.1reg0.1.actual"]
        compare_stream_functions(max_iters, sf_filenames)
