import os
from multiprocessing import Pool 
from itertools import product
import random 
from PIL import Image 

def main(): 
    
    resSizes = [1000, 4000]
    spectral_radiuses = [3.0,5.0]
    training_lengths = [3000]
    init_lengths = [ 500, 1000 ]# [3000,4000, 4400,4800]
    ridge_regs = [1, 1e-2]
    densities = [0.5, 0.2]
    leaking_rates = [0.0, 0.5]
    input_scalings = [0.5]    

    processes = ()
    for i,(res, sr, training_length, init_length, ridge_reg, density, leaking_rate, input_scaling) in enumerate(
      product(resSizes, spectral_radiuses, training_lengths, init_lengths, ridge_regs, densities, leaking_rates, input_scalings)
      ):
        processes += (f"run_mesn_single_dg_experiment.py"
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
    # experiment_prefix = 'dgsf_0.1_200_100_0.1_0.2_13000_2.0_id0'
    for p in processes: print(p)
    pool = Pool(processes=len(resSizes)*len(spectral_radiuses))
    pool.map(run_process, processes)
    # switch_to_mlfluids_src_dir(); import util, config 
    # im_list = []
    # for i, _ in enumerate(list(processes)):
    #     im = Image.open(os.path.join('./experiments/', experiment_prefix + '_traj_ts_{0}.compare'.format(i))) 
    #     im_list.append(im)
    # pdf_filename = os.path.join('./experiments/', 'dg_pdf.pdf')
    # im.save(pdf_filename, "PDF", save_all=True, append_images=im_list) 
        
def run_process(prc):
    os.system('python {}'.format(prc))



if __name__ == '__main__':
    main()
