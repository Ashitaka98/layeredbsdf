from math import cos, sin
import numpy as np
from time import time
import random as ran


class DatasetGenerator:

    dielectric = 0
    conductor = 1

    # each bsdf table is saved as a npy
    # theta, phi is sampled uniformly
    # train_output_dir: directory to save train set
    # test_output_dir: directory to save test set
    # bsdf_number: total bsdf during generation
    # theta_sample_rate: number of samples to sample theta
    # phi_sample_rate:  number of samples to sample phi
    # times_per_sample: samples used to denoise(repeat times per sampling)
    # type: dielectric or conductor
    def __init__(self,
                 train_output_dir,
                 test_output_dir,
                 bsdf_number,
                 theta_sample_rate,
                 phi_sample_rate,
                 times_per_sample,
                 type,
                 debug=False):
        self.train_output_dir = train_output_dir
        self.test_output_dir = test_output_dir
        self.bsdf_number = bsdf_number
        self.theta_sample_rate = theta_sample_rate
        self.phi_sample_rate = phi_sample_rate
        self.times_per_sample = times_per_sample
        self.debug = debug
        self.type = type

        self.pmgr = PluginManager.getInstance()

        samplerProps = Properties('independent')
        samplerProps['sampleCount'] = 128
        self.sampler = self.pmgr.createObject(samplerProps)
        self.sampler.configure()

        self.eta_k_table = []

        # generate uniform solid angle samples
        pi = np.math.pi
        self.solid_angle = []
        for theta_i in range(theta_sample_rate):
            theta = theta_i * 2 * pi / theta_sample_rate
            for phi_i in range(phi_sample_rate):
                if type == DatasetGenerator.dielectric:         # the attr `type` is only used here
                    phi = phi_i * pi / phi_sample_rate
                elif type == DatasetGenerator.conductor:
                    phi = phi_i * 0.5 * pi / phi_sample_rate
                self.solid_angle.append([theta, phi])


    def readTable(self, table_path):
        self.eta_k_table = []
        with open(table_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.eta_k_table.append(list(map(float, line.split())))


    def run(self):
        pi = np.math.pi
        from time import time
        from random import uniform, randint
        from os.path import join
        sigma_t_range = [0, 1, 2, 5]

        for i in range(self.bsdf_number):
            start = time()
            Log('####  round ' + str(i) + ' start')

            # get a random pair of eta-k from the table!
            ranidx = ran.randint(0, len(self.eta_k_table)-1)
            Log('[eta-k] Random index: ' + str(ranidx))
            eta_1, k_1 = self.eta_k_table[ranidx]

            Log('[eta-k] Selected ' + str(eta_1) + '+' + str(k_1) + 'i')



if __name__ == '__main__':
    from utils import Log
    from mitsuba.core import PluginManager, Properties, Vector3, Spectrum
    from mitsuba.render import Intersection, BSDFSamplingRecord, ETransportMode, EMeasure
    Log('task begin')
    task_start = time()
    generator = DatasetGenerator(
        '/home/lzr/layeredBsdfData/conductor_train_000',
        '/home/lzr/layeredBsdfData/conductor_test_000', 300, 25,
        25, 128, DatasetGenerator.conductor, True)

    generator.readTable('/home/lzr/Projects/layeredbsdf/pyscript/eta-k_table.txt')
    Log('[eta-k] Read ' + str(len(generator.eta_k_table)) + ' eta-k pairs')
    generator.run()
    Log('total time: ' + str(time() - task_start) + 's')
