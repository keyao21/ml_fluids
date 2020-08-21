import os
from multiprocessing import Pool 
from itertools import product
import random 

def main(): 
    
    resSizes = [1000, 5000]
    spectral_radiuses = [2.0, 3.0]
    num_prcs = len(resSizes)*len(spectral_radiuses)
    
    max_id = 1
    while max_id < num_prcs: 
        max_id *= 10

    unique_ids = random.sample(range(1, max_id), num_prcs)
    processes = ()
    for i,(res, sr) in enumerate(product(resSizes, spectral_radiuses)):
        processes += (f"run_single_qg_experiment.py"
                      " --spectral_radius {sr}"
                      " --resSize {res}"
                      " --id {id}".format(sr=sr,res=res,id=unique_ids[i]),) 
    for p in processes: print(p) 
    pool = Pool(processes=)
    pool.map(run_process, processes)

def run_process(prc): 
    os.system('python {}'.format(prc))



if __name__ == '__main__': 
    main()

