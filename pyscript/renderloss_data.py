import numpy as np
import argparse
#import tensorflow as tf
from GY_mitsuba import GY_multilayered

#import utils
from renderer import *

parser = argparse.ArgumentParser()
parser.add_argument('--begin_idx',
                    type=int,
                    default=0,
                    help='beginning index of file')
parser.add_argument('--file_num',
                    type=int,
                    default=1,
                    help='how many files to preceed')

FLAGS = parser.parse_args()
BEGIN_IDX = FLAGS.begin_idx
FILE_NUM = FLAGS.file_num


class Render_Loss:
    """
    a vertical projection shading which means vertical parallel primary rays
    an internal cut hemisphere
    """
    def __init__(self, resolution_level, vup=[0, 1, 0], sample_times=128, surface_type=GY_multilayered.dielectric):
        """
        resolution_level: img width = 2**resolution_level
        """
        self.debug = 1
        self.sample_times = sample_times
        self.surface_type = surface_type
    
        #self.bsdf = bsdf
        resolution = 2**np.round(resolution_level)
        self.resolution = resolution
        self.vup = np.array(vup)
        self.wi_list = []  # wi for each illuminant
        self.precomputed_list = []  #precomputed item for each illuminant

        self.filenames = []

        ray_direction = np.array([0, 0, -1])

        # Calculate wo array.       self.wo shape:[rows,2]
        self.wo = []
        for i in range(resolution):
            y = 2 * i / (resolution - 1) - 1            # [-1, 1]
            for j in range(resolution):
                x = 2 * j / (resolution - 1) - 1
                if x * x + y * y < 1:
                    z = np.sqrt(1 - x * x - y * y)
                    normal = np.array([x, y, z])
                    x_ = Renderer.make_unit_vector(np.cross(normal, self.vup))
                    y_ = np.cross(normal, x_)
                    x_ = np.reshape(x_, [3, 1])
                    y_ = np.reshape(y_, [3, 1])
                    z_ = np.reshape(normal, [3, 1])
                    shading_frame = np.concatenate([x_, y_, z_], axis=1)
                    shading_frame_inv = np.linalg.inv(shading_frame)
                    wo = np.matmul(shading_frame_inv, -ray_direction)
                    phi_o = np.arccos(wo[2])
                    theta_o = np.arctan2(wo[1] / np.sin(phi_o),
                                         wo[0] / np.sin(phi_o))
                    if theta_o < 0:
                        theta_o += 2 * np.pi
                    self.wo.append(np.array([theta_o, phi_o]))

        '''self.wo = tf.convert_to_tensor(np.concatenate(self.wo, axis=0))'''


    def add_illuminant(self, illuminant_type, **kwargs):
        resolution = self.resolution
        if illuminant_type == Renderer.ILLUMINANT_TYPE.directional_light:
            intensity = kwargs['intensity']
            light_direction = Renderer.make_unit_vector(
                np.array(kwargs['light_direction']))
        elif illuminant_type == Renderer.ILLUMINANT_TYPE.point_light:
            intensity = kwargs['intensity']
            light_position = np.array(kwargs['light_position'])
        else:
            raise Exception('unexpected illuminant type')

        wi_list = []
        precomputed = []
        ray_direction = np.array([0, 0, -1])
        for i in range(resolution):
            y = 2 * i / (resolution - 1) - 1
            for j in range(resolution):
                x = 2 * j / (resolution - 1) - 1
                if x * x + y * y < 1:
                    z = np.sqrt(1 - x * x - y * y)
                    normal = np.array([x, y, z])
                    x_ = Renderer.make_unit_vector(np.cross(normal, self.vup))
                    y_ = np.cross(normal, x_)
                    x_ = np.reshape(x_, [3, 1])
                    y_ = np.reshape(y_, [3, 1])
                    z_ = np.reshape(normal, [3, 1])
                    shading_frame = np.concatenate([x_, y_, z_], axis=1)
                    shading_frame_inv = np.linalg.inv(shading_frame)
                    if illuminant_type == Renderer.ILLUMINANT_TYPE.directional_light:
                        wi = np.matmul(shading_frame_inv, -light_direction)
                        L = intensity
                    elif illuminant_type == Renderer.ILLUMINANT_TYPE.point_light:
                        hit_point = np.array([x, y, z])
                        wi = np.matmul(
                            shading_frame_inv,
                            Renderer.make_unit_vector(light_position -
                                                      hit_point))
                        dist = np.linalg.norm(light_position - hit_point)
                        L = intensity / dist * dist
                    phi_i = np.arccos(wi[2])
                    theta_i = np.arctan2(wi[1] / np.sin(phi_i),
                                         wi[0] / np.sin(phi_i))
                    if theta_i < 0:
                        theta_i += 2 * np.pi
                        
                    wi_list.append(np.array([theta_i, phi_i]))

                    wo = np.matmul(shading_frame_inv, -ray_direction)
                    phi_o = np.arccos(wo[2])
                    if wi[2] > 0:
                        precomputed.append(
                            np.array([L * np.cos(phi_o)]))
                    else:
                        precomputed.append(np.array([0]))

        self.precomputed = precomputed

        self.wi_list = wi_list


    def render(self, filename):
        # TODO: The rendering result should be $f * precomputed$, in which $f$ is bsdf(theta_i, phi_i, theta_o, phi_o)

        resolution = self.resolution

        GY_bsdf = GY_multilayered(filename,
                                    sample_times=self.sample_times,
                                    type=self.surface_type)

        buf = np.zeros([resolution, resolution, 3], dtype=np.float32)
        cnt = 0

        for i in range(resolution):
            y = 2 * i / (resolution - 1) - 1
            for j in range(resolution):
                x = 2 * j / (resolution - 1) - 1
                if x * x + y * y < 1:
                    # Shading. May use a counter to read precomputed
                    #print(len(self.wi_list))
                    f = GY_bsdf.eval(theta_i=self.wi_list[cnt][0],
                                     phi_i=self.wi_list[cnt][1],
                                     theta_o=self.wo[cnt][0],
                                     phi_o=self.wo[cnt][1],
                                     debug=self.debug)

                    buf[i, j] = f * self.precomputed[cnt]

                    #print(buf[i, j])

                    cnt += 1

                else:
                    # Black
                    pass
                    #buf[i, j] = [0,0,0]
        
        return buf

    
    def read_file_names(self, path):
        import os
        filesList = os.listdir(path)
        for fileName in filesList:
            fileAbsPath = os.path.join(path, fileName)
            if os.path.isdir(fileAbsPath):
                pass
            else:
                tmp = os.path.splitext(fileName)
                rawName = tmp[0]
                extName = tmp[1]
                #print(rawName)
                if extName == ".npy":
                    self.filenames.append(rawName)

    
    def gen_data(self, indir, outdir):
        import imageio
        import os 

        self.read_file_names(indir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if BEGIN_IDX + FILE_NUM > len(self.filenames):
            END_IDX = len(self.filenames)
        else:
            END_IDX = BEGIN_IDX + FILE_NUM

        for i in range(BEGIN_IDX, END_IDX):
            filename = self.filenames[i]
            outfile = os.path.join(outdir, filename + '.exr')
            if not os.path.exists(outfile):
                buf = self.render(filename)
                imageio.imsave(outfile, im=buf)


if __name__ == "__main__":
    '''loss = Render_Loss(None, 5)
    loss.add_illuminant(1)'''

    render_loss = Render_Loss(7, surface_type=GY_multilayered.dielectric)
    render_loss.add_illuminant(Renderer.ILLUMINANT_TYPE.point_light, 
                                intensity=10,
                                light_position=[0, 0, 3])
    render_loss.gen_data('/home/lzr/layeredBsdfData/dielectric_test_new_distribution/', '/home/lzr/layeredBsdfData/dielectric_test_new_distribution_renderloss/')
