import os
from multiprocessing import Pool 
from itertools import product
import random 

def main(): 
    
    resSizes = [5000]
    spectral_radiuses = [1.4]
    training_lengths = [5000]
    init_lengths = [ 4900, 4930, 4950 ]# [3000,4000, 4400,4800]
    ridge_regs = [1]
    densities = [0.5]
    leaking_rates = [0.0]
    input_scalings = [0.1]    

    processes = ()
    for i,(res, sr, training_length, init_length, ridge_reg, density, leaking_rate, input_scaling) in enumerate(
      product(resSizes, spectral_radiuses, training_lengths, init_lengths, ridge_regs, densities, leaking_rates, input_scalings)
      ):
        processes += (f"run_single_qg_experiment.py"
                      " --spectral_radius {sr}"
                      " --resSize {res}"
                      " --training_length {training_length}"
                      " --init_length {init_length}"
                      " --ridge_reg {ridge_reg}"
                      " --density {density}"
                      " --leaking_rate {leaking_rate}"
                      " --input_scaling {input_scaling}"
                      " --id {id}".format(
                            sr=sr,res=res,
                            training_length=max(training_lengths),
                            init_length=init_length,ridge_reg=ridge_reg,
                            density=density,leaking_rate=leaking_rate,
                            input_scaling=input_scaling,id=i),)
        # note: remember that training_length determines the amount of data 
        # chopped off to be used for training, but the init_length determines 
        # the amount of data **skipped** in the chopped off data 
    for p in processes: print(p)
    pool = Pool(processes=len(resSizes)*len(spectral_radiuses))
    pool.map(run_process, processes)

def run_process(prc):
    os.system('python {}'.format(prc))



if __name__ == '__main__':
    main()
