from math import cos, sin
import numpy as np
from time import time
import random as ran
import os


class DatasetGenerator_Conductor:

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
                 texture,
                 train_output_dir,
                 bsdf_number,
                 theta_sample_rate,
                 phi_sample_rate,
                 times_per_sample,
                 debug=False):
        self.tex = texture
        self.train_output_dir = train_output_dir
        self.bsdf_number = bsdf_number
        self.theta_sample_rate = theta_sample_rate
        self.phi_sample_rate = phi_sample_rate
        self.times_per_sample = times_per_sample
        self.debug = debug

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
                phi = phi_i * 0.5 * pi / phi_sample_rate
                self.solid_angle.append([theta, phi])

    def readTable(self, table_path):
        self.material_table = []
        with open(table_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.material_table.append(str(line.split()[0]))

    def run(self):
        pi = np.math.pi
        from time import time
        from random import uniform, randint
        from os.path import join
        width = self.tex.shape[1]
        height = self.tex.shape[0]

        for i in range(self.bsdf_number):
            start = time()
            Log('####  round ' + str(i) + ' start')
            x = randint(0, width - 1)
            y = randint(0, height - 1)
            print(x, y)

            sigma_t = 3 * float(self.tex[y][x][randint(0, 2)])
            print(sigma_t)
            albedo = [0, 0, 0]
            g = 0.0

            # roughness 0.001-1
            alpha_0 = 0.005
            alpha_1 = 0.05

            # normal semi-sphere
            theta_0 = 0
            phi_0 = 0
            theta_1 = 0
            phi_1 = 0

            # ior 1.05-2
            eta_0 = 1.5
            #eta_1 = uniform(1.05, 2)

            # get a random pair of eta-k from the table!
            '''
            ranidx = ran.randint(0, len(self.eta_k_table)-1)
            Log('[eta-k] Random index: ' + str(ranidx))
            eta_1, k_1 = self.eta_k_table[ranidx]

            Log('[eta-k] Selected ' + str(eta_1) + '+' + str(k_1) + 'i')
            '''

            material_1 = 'Hg'
            Log('[Material Preset] Selected ' + str(material_1))

            layered = self.pmgr.create({
                'type':
                'multilayered',
                'bidir':
                True,
                'pdf':
                "bidirStochTRT",
                'stochPdfDepth':
                4,
                'pdfRepetitive':
                1,
                'diffusePdf':
                0.1,
                'maxSurvivalProb':
                1.0,
                'nbLayers':
                2,
                'surface_0': {
                    'type': 'roughdielectric',
                    'distribution': 'ggx',
                    'intIOR': eta_0,
                    'extIOR': 1.0,
                    'alpha': alpha_0
                },
                'normal_0':
                Vector3(
                    cos(theta_0) * sin(phi_0),
                    sin(theta_0) * sin(phi_0), cos(phi_0)),
                'sigmaT_0':
                Spectrum(sigma_t),
                'albedo_0':
                Spectrum(albedo),
                'phase_0': {
                    'type': 'hg',
                    'g': g
                },
                'surface_1': {
                    'type': 'roughconductor',
                    'distribution': 'ggx',
                    #'eta': Spectrum(eta_1),
                    #'k': Spectrum(k_1),
                    'material': material_1,
                    'extEta': eta_0,
                    'alpha': alpha_1
                },
                'normal_1':
                Vector3(
                    cos(theta_1) * sin(phi_1),
                    sin(theta_1) * sin(phi_1), cos(phi_1))
            })
            filename = (str(alpha_0) + '_' + str(theta_0) + '_' + str(phi_0) +
                        '_' + str(eta_0) + '_' + str(sigma_t) + '_' +
                        str(albedo[0]) + '_' + str(albedo[1]) + '_' +
                        str(albedo[2]) + '_' + str(g) + '_' + str(alpha_1) +
                        '_' + str(theta_1) + '_' + str(phi_1) + '_' +
                        str(material_1))

            # generate stritified solid angle samples
            pi = np.math.pi

            its = Intersection()
            table_train = []
            for i1 in range(self.theta_sample_rate):
                for i2 in range(self.phi_sample_rate):
                    for i3 in range(self.theta_sample_rate):
                        for i4 in range(self.phi_sample_rate):
                            theta_i = (i1 + uniform(
                                0, 1)) * 2 * pi / self.theta_sample_rate
                            phi_i = (i2 + uniform(
                                0, 1)) * 0.5 * pi / self.phi_sample_rate
                            theta_o = (i3 + uniform(
                                0, 1)) * 2 * pi / self.theta_sample_rate
                            phi_o = (i4 + uniform(
                                0, 1)) * 0.5 * pi / self.phi_sample_rate
                            x1 = cos(theta_i) * sin(phi_i)
                            y1 = sin(theta_i) * sin(phi_i)
                            z1 = cos(phi_i)
                            x2 = cos(theta_o) * sin(phi_o)
                            y2 = sin(theta_o) * sin(phi_o)
                            z2 = cos(phi_o)
                            wi = Vector3(x1, y1, z1)
                            wo = Vector3(x2, y2, z2)
                            its.wi = wi
                            its.wo = wo
                            bRec = BSDFSamplingRecord(its, self.sampler,
                                                      ETransportMode.ERadiance)
                            bRec.wi = wi
                            bRec.wo = wo
                            accum = Spectrum(0)
                            nan_count = 0
                            for i in range(self.times_per_sample):
                                ret = layered.eval(bRec, EMeasure.ESolidAngle)
                                if np.isnan(ret[0]) or np.isnan(
                                        ret[1]) or np.isnan(ret[2]):
                                    nan_count += 1
                                else:
                                    accum += ret
                            if wo[2] != 0:
                                accum /= abs(wo[2])
                            else:
                                raise Exception('wo_z is zero')
                            if self.debug and nan_count != 0:
                                Log('Sampling theta_i:{:.2f} phi_i:{:.2f} theta_o:{:.2f}, phi_o:{:.2f} | NaN occur {} times '
                                    .format(theta_i, phi_i, theta_o, phi_o,
                                            nan_count))
                            if nan_count == self.times_per_sample:
                                raise Exception(
                                    'all \'eval\' calls return NaN ')
                            accum /= self.times_per_sample - nan_count
                            table_train.append([
                                theta_i, phi_i, theta_o, phi_o, accum[0],
                                accum[1], accum[2]
                            ])

            if not os.path.exists(self.train_output_dir):
                os.makedirs(self.train_output_dir)

            nptable_train = np.array(table_train).astype(np.float32)
            np.save(join(self.train_output_dir, filename), nptable_train)

            Log('time:' + str(time() - start) + 's')


if __name__ == '__main__':
    from utils import Log
    from mitsuba.core import PluginManager, Properties, Vector3, Spectrum
    from mitsuba.render import Intersection, BSDFSamplingRecord, ETransportMode, EMeasure
    import imageio
    exr = np.array(imageio.imread(
        '/home/lzr/Projects/mitsuba/scenes/globe_withoutnormal_ours/textures/sigmat.exr'
    ),
                   dtype=np.float32)
    print(exr.shape)
    Log('task begin')
    task_start = time()
    generator = DatasetGenerator_Conductor(
        exr, '/home/lzr/layeredBsdfData/conductor_globe', 10, 25, 25, 128,
        True)

    generator.readTable(
        '/home/lzr/Projects/layeredbsdf/pyscript/material_names_table.txt')
    Log('[eta-k] Read ' + str(len(generator.material_table)) +
        ' material presets')
    generator.run()
    Log('total time: ' + str(time() - task_start) + 's')
