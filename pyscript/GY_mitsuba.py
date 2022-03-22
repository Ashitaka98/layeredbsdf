from mitsuba.core import PluginManager, Properties, Vector3, Spectrum
from mitsuba.render import Intersection, BSDFSamplingRecord, ETransportMode, EMeasure
import numpy as np
from math import sin, cos


class GY_multilayered:

    dielectric = 0
    conductor = 1

    def __init__(self, filename, sample_times=128, type=dielectric):
        self.sample_times = sample_times
        print(sample_times)
        self.pmgr = PluginManager.getInstance()
        samplerProps = Properties('independent')
        samplerProps['sampleCount'] = 128
        self.sampler = self.pmgr.createObject(samplerProps)
        self.sampler.configure()

        # parse material params from filename
        params = filename.split('/')[-1]
        params = params.split('_')
        alpha_0 = float(params[0])
        theta_0 = float(params[1])
        phi_0 = float(params[2])
        eta_0 = float(params[3])
        sigma_t = float(params[4])
        albedo = [float(x) for x in params[5:8]]
        g = float(params[8])
        alpha_1 = float(params[9])
        theta_1 = float(params[10])
        phi_1 = float(params[11])
        if type == GY_multilayered.dielectric:
            #eta_1 = params[12][0:len(params[12]) - 4]
            eta_1 = params[12]          # filename does not include '.npy'
            eta_1 = float(eta_1)
            GY_multilayered.Log(
                'dielectric: ' + 
                (str(alpha_0) + '_' + str(theta_0) + '_' + str(phi_0) + '_' +
                str(eta_0) + '_' + str(sigma_t) + '_' + str(albedo[0]) + '_' +
                str(albedo[1]) + '_' + str(albedo[2]) + '_' + str(g) + '_' +
                str(alpha_1) + '_' + str(theta_1) + '_' + str(phi_1) + '_' +
                str(eta_1)))

            self.layered = self.pmgr.create({
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
                    'type': 'roughdielectric',
                    'distribution': 'ggx',
                    'intIOR': eta_0 * eta_1,
                    'extIOR': eta_0,
                    'alpha': alpha_1
                },
                'normal_1':
                Vector3(
                    cos(theta_1) * sin(phi_1),
                    sin(theta_1) * sin(phi_1), cos(phi_1))
            })

        elif type == GY_multilayered.conductor:
            material_1 = str(params[12])

            # CAUTION!! Hard Coded! Should be modified if you would change the file name format!
            if len(params) > 13:
                if str(params[13]) != 'palik':
                    GY_multilayered.Log('ERROR! Illegal File Name! ' + params[13])
                material_1 += '_palik'

            GY_multilayered.Log(
                'conductor: ' + 
                (str(alpha_0) + '_' + str(theta_0) + '_' + str(phi_0) + '_' +
                str(eta_0) + '_' + str(sigma_t) + '_' + str(albedo[0]) + '_' +
                str(albedo[1]) + '_' + str(albedo[2]) + '_' + str(g) + '_' +
                str(alpha_1) + '_' + str(theta_1) + '_' + str(phi_1) + '_' +
                str(material_1))
            )

            self.layered = self.pmgr.create({
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
                    'material': material_1,
                    'extEta': eta_0,
                    'alpha': alpha_1
                },
                'normal_1':
                    Vector3(
                        cos(theta_1) * sin(phi_1),
                        sin(theta_1) * sin(phi_1), cos(phi_1))
            })

        else:
            GY_multilayered.Log('ERROR: Illegal type ' + str(type) + ' ! ')


        

    def Log(str):
        print('[GY_mitsuba]: ' + str)

    def eval(self, theta_i, phi_i, theta_o, phi_o, debug=False):
        its = Intersection()
        wi_x = cos(theta_i) * sin(phi_i)
        wi_y = sin(theta_i) * sin(phi_i)
        wi_z = cos(phi_i)
        wi = Vector3(wi_x, wi_y, wi_z)
        its.wi = wi

        wo_x = cos(theta_o) * sin(phi_o)
        wo_y = sin(theta_o) * sin(phi_o)
        wo_z = cos(phi_o)
        wo = Vector3(wo_x, wo_y, wo_z)
        bRec = BSDFSamplingRecord(its, self.sampler, ETransportMode.ERadiance)
        bRec.wi = wi
        bRec.wo = wo
        accum = Spectrum(0)
        nan_count = 0
        for i in range(self.sample_times):
            ret = self.layered.eval(bRec, EMeasure.ESolidAngle)
            if np.isnan(ret[0]) or np.isnan(ret[1]) or np.isnan(ret[2]):
                nan_count += 1
            else:
                accum += ret
        if wo_z != 0:
            accum /= abs(wo_z)
        if debug and nan_count != 0:
            GY_multilayered.Log(
                'Sampling theta_i:{:.2f} phi_i:{:.2f} theta_o:{:.2f}, phi_o:{:.2f} | NaN occur {} times '
                .format(theta_i, phi_i, theta_o, phi_o, nan_count))
        if nan_count > 0.1 * self.sample_times:
            raise Exception('more than 10% \'eval\' calls return NaN ')
        accum /= self.sample_times - nan_count
        return np.array([accum[0], accum[1], accum[2]])
