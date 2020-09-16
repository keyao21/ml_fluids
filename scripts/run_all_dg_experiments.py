import os
from multiprocessing import Pool
from itertools import product

def main():

    resSizes = [100, 1000, 5000]
    spectral_radiuses = [0.5, 1.0, 3.0, 5.0]
    training_lengths = [500, 1000, 1500, 2000]
    init_lengths = [ max(training_lengths)-train_len for train_len in training_lengths ]
    ridge_regs = [1e-3,1e-1,0,1,1e1]

    processes = ()
    for i,(res, sr, training_length, init_length, ridge_reg) in enumerate(
      product(resSizes, spectral_radiuses, training_lengths, init_lengths ridge_regs)
      ):
        processes += (f"run_single_dg_experiment.py"
                      " --spectral_radius {sr}"
                      " --resSize {res}"
                      " --training_length {training_length}"
                      " --init_length {init_length}"
                      " --ridge_reg {ridge_reg}"
                      " --id {id}".format(
                            sr=sr,res=res,
                            training_length=max(training_lengths),
                            ridge_reg=ridge_reg,id=i),)
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