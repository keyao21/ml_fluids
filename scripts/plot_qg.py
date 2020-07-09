from scipy import io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import AxesGrid
from pylab import cm
cmap = cm.get_cmap('YlGnBu_r', 11)
import os 

QG_RAW_DIR_FULLPATH = os.path.abspath('../../quasigeo/')


def load_mat_file(mat_fullpath, var_name='Psi_ts'): 
    mat = io.loadmat(mat_fullpath)[var_name]
    return mat 

def main(): 
    qg_filenames = [
        'QGpsi_64_128_dt0.01_ds0.04_di0.03_dm0.02_pertamp0.2.mat',
        'QGpsi_64_128_dt0.01_ds0.04_di0.05_dm0.03_pertamp0.2.mat',
        'QGpsi_64_128_dt0.01_ds0.04_di0.02_dm0.03_pertamp0.2.mat'
    ]
    qg_datas = []
    for qg_filename in qg_filenames: 
        qg_fullpath = os.path.join(QG_RAW_DIR_FULLPATH, qg_filename)
        qg_data = load_mat_file(mat_fullpath=qg_fullpath)
        qg_datas.append(qg_data)



    fig = plt.figure()
    grid = AxesGrid(fig, 111,
                nrows_ncols=(len(qg_datas), 1),
                axes_pad=(0.06, 0.4),
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.2
                )

    mywriter = animation.FFMpegWriter()
    ims = []
    for i in range(20):
        if (i%100==0): print(i)
        for ax in grid: 
            ax.set_axis_off()
                
        n_ims = []
        for i, qg_data in qg_datas: 
            im = grid[i].imshow(qg_data[i],interpolation='bicubic',cmap=cmap,animated=True,alpha=1.0)
            n_ims.append(im)

        grid[0].cax.colorbar(im)
        ims.append(n_ims)

    # grid[1].cax.colorbar(im1)
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1) 
    plt.show()




if __name__ == '__main__':
    main()







