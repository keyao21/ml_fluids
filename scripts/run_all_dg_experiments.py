import os
from multiprocessing import Pool
from itertools import product

def main():

    resSizes = [100, 1000, 5000]
    spectral_radiuses = [0.5, 1.0, 3.0, 5.0]
    training_lengths = [500, 1000, 1500, 2000]

    for i,(res, sr, training_length) in enumerate(product(resSizes, spectral_radiuses, training_lengths)):
        processes += (f"run_single_dg_experiment.py"
                      " --spectral_radius {sr}"
                      " --resSize {res}"
                      " --training_length {training_length}"
                      " --id {id}".format(
                            sr=sr,res=res,training_length=training_length,id=i),)
    for p in processes: print(p)
    pool = Pool(processes=len(resSizes)*len(spectral_radiuses))
    pool.map(run_process, processes)

def run_process(prc):
    os.system('python {}'.format(prc))



if __name__ == '__main__':
    main()