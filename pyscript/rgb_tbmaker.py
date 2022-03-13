from math import cos, sin
import numpy as np
from time import time
import random as ran

material_table = []

def readTable(table_path):
    with open(table_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            material_table.append(str(line.split()[0]))


if __name__ == '__main__':
    from utils import Log
    from mitsuba.core import PluginManager, Properties, Vector3, Spectrum, InterpolatedSpectrum, FileResolver
    from mitsuba.render import Intersection, BSDFSamplingRecord, ETransportMode, EMeasure
    Log('task begin')

    readTable('/home/lzr/Projects/layeredbsdf/pyscript/material_names_table.txt')

    fResolver = FileResolver()

    for name in material_table:
        eta = Spectrum(0)
        k = Spectrum(0)
        eta.fromContinuousSpectrum(InterpolatedSpectrum(
            fResolver.resolve("/home/lzr/Projects/layeredbsdf/data/ior/" + name + ".eta.spd")))
        k.fromContinuousSpectrum(InterpolatedSpectrum(
            fResolver.resolve("/home/lzr/Projects/layeredbsdf/data/ior/" + name + ".k.spd")))
        eta_r, eta_g, eta_b = eta.toLinearRGB()
        k_r, k_g, k_b = k.toLinearRGB()

        outfile = open('/home/lzr/Projects/layeredbsdf/data/ior_rgb/' + name + '_rgb.txt', 'w')
        outfile.write(str(eta_r) + ' ' + str(eta_g) + ' ' + str(eta_b) + '\n')
        outfile.write(str(k_r) + ' ' + str(k_g) + ' ' + str(k_b) + '\n')
        outfile.close()


        

    

