from math import cos, sin
import numpy as np
from time import time
import random as ran

material_table = []

mx_eta = []
mi_eta = []
mx_k = []
mi_k = []
len_eta = []
len_k = []

def readTable(table_path):
    with open(table_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            material_table.append(str(line.split()[0]))


def nor_eta(x, rgb):
    return (x - mi_eta[rgb]) / len_eta[rgb]

def nor_k(x, rgb):
    return (x - mi_k[rgb]) / len_k[rgb]


if __name__ == '__main__':
    from utils import Log
    from mitsuba.core import PluginManager, Properties, Vector3, Spectrum, InterpolatedSpectrum, FileResolver
    from mitsuba.render import Intersection, BSDFSamplingRecord, ETransportMode, EMeasure
    Log('task begin')

    readTable('/home/lzr/Projects/layeredbsdf/pyscript/material_names_table.txt')

    fResolver = FileResolver()

    ls_eta = []
    ls_k = []

    for name in material_table:
        eta = Spectrum(0)
        k = Spectrum(0)
        eta.fromContinuousSpectrum(InterpolatedSpectrum(
            fResolver.resolve("/home/lzr/Projects/layeredbsdf/data/ior/" + name + ".eta.spd")))
        k.fromContinuousSpectrum(InterpolatedSpectrum(
            fResolver.resolve("/home/lzr/Projects/layeredbsdf/data/ior/" + name + ".k.spd")))

        ls_eta.append(list(eta.toLinearRGB()))
        ls_k.append(list(k.toLinearRGB()))

    ls_eta = np.mat(ls_eta)
    ls_k = np.mat(ls_k)


    for i in range(0,3):
        mx_eta.append(ls_eta[:,i].max())
        mi_eta.append(ls_eta[:,i].min())
        mx_k.append(ls_k[:,i].max())
        mi_k.append(ls_k[:,i].min())

        len_eta.append(mx_eta[i] - mi_eta[i])
        len_k.append(mx_k[i] - mi_k[i])

    for i in range(0,3):
        print(mx_eta[i], mi_eta[i])

    
    for name in material_table:
        eta = Spectrum(0)
        k = Spectrum(0)
        eta.fromContinuousSpectrum(InterpolatedSpectrum(
            fResolver.resolve("/home/lzr/Projects/layeredbsdf/data/ior/" + name + ".eta.spd")))
        k.fromContinuousSpectrum(InterpolatedSpectrum(
            fResolver.resolve("/home/lzr/Projects/layeredbsdf/data/ior/" + name + ".k.spd")))
        eta_r, eta_g, eta_b = eta.toLinearRGB()
        k_r, k_g, k_b = k.toLinearRGB()

        outfile = open('/home/lzr/Projects/layeredbsdf/data/ior_rgb_normalized/' + name + '_rgb.txt', 'w')
        outfile.write(str(nor_eta(eta_r, 0)) + ' ' + str(nor_eta(eta_g, 1)) + ' ' + str(nor_eta(eta_b, 2)) + '\n')
        outfile.write(str(nor_k(k_r, 0)) + ' ' + str(nor_k(k_g, 1)) + ' ' + str(nor_k(k_b, 2)) + '\n')
        outfile.close()


        

    

