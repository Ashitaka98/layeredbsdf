
import os
import glob
import imageio
import numpy as np

if __name__ == '__main__':
    #inpath = '/home/lzr/layeredBsdfData/dielectric_new_distribution_renderloss_0_0_-3/'
    #outpath = '/home/lzr/layeredBsdfData/dielectric_new_distribution_renderloss_0_0_-3_npy/'
    inpath = '/home/lzr/layeredBsdfData/rl_back_test_out/'
    outpath = '/home/lzr/layeredBsdfData/rl_back_test_out_npy/'
    if not os.path.exists(outpath):
            os.makedirs(outpath)

    filelist = glob.glob(os.path.join(inpath, '*.exr'))
    print(len(filelist))
    for file in filelist:
        rawname = file[0:len(file)-4]     # remove '.exr'
        rawname = rawname.split('/')[-1]
        with open(file, 'r') as f:
            exr = np.array(imageio.imread(file))
            print(exr.shape)
            #np.save(os.path.join(outpath, rawname), exr)
                                                      