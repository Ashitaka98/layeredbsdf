class DatasetGenerator:

    dielectric = 0
    conductor = 1

    # each bsdf table is saved as a npy
    # theta, phi is sampled uniformly
    # output_dir: directory to save results
    # bsdf_number: total bsdf during generation
    # theta_sample_rate: number of samples to sample theta
    # phi_sample_rate:  number of samples to sample phi
    # times_per_sample: samples used to denoise(repeat times per sampling)
    # type: dielectric or conductor
    def __init__(self, output_dir, bsdf_number, theta_sample_rate, phi_sample_rate, times_per_sample, type):
        self.output_dir = output_dir
        self.bsdf_number = bsdf_number
        self.theta_sample_rate = theta_sample_rate
        self.phi_sample_rate = phi_sample_rate
        self.times_per_sample = times_per_sample

        self.pmgr = PluginManager.getInstance()

        samplerProps = Properties('independent')
        samplerProps['sampleCount'] = 128
        self.sampler = self.pmgr.createObject(samplerProps)
        self.sampler.configure()

        # generate uniform solid angle samples
        pi = np.math.pi
        self.solid_angle = []
        for theta_i in range(theta_sample_rate):
            theta = theta_i*2*pi/theta_sample_rate
            for phi_i in range(phi_sample_rate):
                if type == DatasetGenerator.dielectric:
                    phi = phi_i * pi / phi_sample_rate
                elif type == DatasetGenerator.conductor:
                    phi = phi_i * 0.5 * pi / phi_sample_rate
                self.solid_angle.append([theta, phi])

    def run(self):
        pi = np.math.pi
        from random import uniform, randint
        from os.path import join
        sigma_t_range = [0.1, 0.5, 1, 3, 5, 10]

        for i in range(self.bsdf_number):
            start = time()
            Log('####  round '+str(i)+' start')
            sigma_t = sigma_t_range[randint(0, len(sigma_t_range)-1)]
            albedo = [uniform(0, 1), uniform(0, 1), uniform(0, 1)]
            g = uniform(-0.99, 0.99)

            # roughness 0.001-1
            alpha_0 = 10**uniform(-3, 0)
            alpha_1 = 10**uniform(-3, 0)

            # normal semi-sphere
            theta_0 = uniform(0, 2*pi)
            phi_0 = uniform(0, 0.5*pi)
            theta_1 = uniform(0, 2*pi)
            phi_1 = uniform(0, 0.5*pi)

            # ior 1-2
            eta_0 = uniform(1, 2)
            eta_1 = uniform(1, 2)

            layered = self.pmgr.create({
                'type': 'multilayered',
                'bidir': True,
                'pdf': "bidirStochTRT",
                'stochPdfDepth': 4,
                'pdfRepetitive': 1,
                'diffusePdf': 0.1,
                'maxSurvivalProb': 1.0,
                'nbLayers': 2,
                'surface_0': {
                        'type': 'roughdielectric',
                        'intIOR': eta_0,
                        'extIOR': 1.0,
                        'alpha': alpha_0
                },
                'normal_0': Vector3(cos(theta_0)*sin(phi_0), sin(theta_0)*sin(phi_0), cos(phi_0)),
                'sigmaT_0': Spectrum(sigma_t),
                'albedo_0': Spectrum(albedo),
                'phase_0': {
                    'type': 'hg',
                    'g': g
                },
                'surface_1': {
                    'type': 'roughdielectric',
                    'intIOR': eta_0*eta_1,
                    'extIOR': eta_0,
                    'alpha': alpha_1
                },
                'normal_1': Vector3(cos(theta_1)*sin(phi_1), sin(theta_1)*sin(phi_1), cos(phi_1))
            })
            filename = (str(alpha_0)+'_'+str(theta_0)+'_'+str(phi_0)+'_' +
                        str(eta_0)+'_'+str(sigma_t)+'_' +
                        str(albedo[0])+'_'+str(albedo[1])
                        + '_'+str(albedo[2])+'_'+str(g)+'_'+str(alpha_1) +
                        '_'+str(theta_1)+'_'+str(phi_1)+'_'+str(eta_1))

            its = Intersection()
            table = []
            for omega_i in self.solid_angle:
                theta_i, phi_i = omega_i
                wi_x = cos(theta_i)*sin(phi_i)
                wi_y = sin(theta_i)*sin(phi_i)
                wi_z = cos(phi_i)
                wi = Vector3(wi_x, wi_y, wi_z)
                its.wi = wi
                for omega_o in self.solid_angle:
                    theta_o, phi_o = omega_o
                    wo_x = cos(theta_o)*sin(phi_o)
                    wo_y = sin(theta_o)*sin(phi_o)
                    wo_z = cos(phi_o)
                    wo = Vector3(wo_x, wo_y, wo_z)
                    bRec = BSDFSamplingRecord(
                        its, self.sampler, ETransportMode.ERadiance)
                    bRec.wi = wi
                    bRec.wo = wo
                    accum = Spectrum(0)
                    for i in range(self.times_per_sample):
                        accum += layered.eval(bRec, EMeasure.ESolidAngle)
                    if wo_z != 0:
                        accum /= abs(wo_z)
                    accum /= self.times_per_sample
                    table.append([theta_i, phi_i, theta_o, phi_o,
                                 accum[0], accum[1], accum[2]])

            nptable = np.array(table)
            np.save(join(self.output_dir, filename), nptable)
            Log('time:'+str(time()-start)+'s')


if __name__ == '__main__':
    from math import cos, sin
    import numpy as np
    from time import time
    from utils import Log
    from mitsuba.core import PluginManager, Properties, Vector3, Spectrum
    from mitsuba.render import BSDF, Intersection, BSDFSamplingRecord, ETransportMode, EMeasure
    Log('task begin')
    task_start = time()
    generator = DatasetGenerator(
        '/home/lzr/layeredBsdfData/dielectric', 300, 25, 25, 128, DatasetGenerator.dielectric)
    generator.run()
    Log('time: '+str(time()-task_start)+'s')
