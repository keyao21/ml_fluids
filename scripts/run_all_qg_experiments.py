import os
from multiprocessing import Pool 
from itertools import product

def main(): 
    
    resSizes = [1000, 5000]
    spectral_radiuses = [2.0, 3.0]
    processes = ()
    for (res, sr) in product(resSizes, spectral_radiuses):
        processes += (f"run_single_qg_experiment.py"
                      " --spectral_radius {sr}"
                      " --resSize {res}".format(sr=sr,res=res),) 
    for p in processes: print(p) 
    pool = Pool(processes=len(resSizes)*len(spectral_radiuses))
    pool.map(run_process, processes)

def run_process(prc): 
    os.system('python {}'.format(prc))



if __name__ == '__main__': 
    main()

