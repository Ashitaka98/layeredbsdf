import numpy as np


class Renderer:
    '''
    a simple differentiable renderer for direct illumination   
    '''
    class BSDF_TYPE:
        single_net = 0
        meta_net = 1
        texture = 2
        GY_mitsuba = 3
        lambertian = 4
        phong = 5

    class ILLUMINANT_TYPE:
        point_light = 0
        directional_light = 1
        image_based_lighting = 2

    inv_pi = 1 / np.pi

    def make_unit_vector(v):
        return v / np.linalg.norm(v)

    def __init__(self,
                 bsdf_type,
                 illuminant_type,
                 eye=[0, 0, 1],
                 aspect_ratio=1,
                 vup=[0, 1, 0],
                 lookat=[0, 0, 0],
                 fov=90,
                 cast_shadows=True,
                 debug=False,
                 **kwargs):
        self.camera = Renderer.Camera(eye=eye,
                                      lookat=lookat,
                                      vup=vup,
                                      fov=fov,
                                      aspect_ratio=aspect_ratio)
        self.bsdf_type = bsdf_type
        self.illuminant_type = illuminant_type
        self.cast_shadows = cast_shadows
        self.debug = debug
        if bsdf_type == Renderer.BSDF_TYPE.single_net:
            self.bsdf_net = kwargs['bsdf']
            self.miu = kwargs['miu']
        elif bsdf_type == Renderer.BSDF_TYPE.meta_net:
            self.bsdf_net = kwargs['bsdf']
            self.miu = kwargs['miu']
        elif bsdf_type == Renderer.BSDF_TYPE.lambertian:
            self.albedo = np.array(kwargs['albedo'])
        elif bsdf_type == Renderer.BSDF_TYPE.texture:
            self.texture = kwargs['texture']
            self.theta_sample_rate = kwargs['theta_sample_rate']
            self.phi_sample_rate = kwargs['phi_sample_rate']
            self.is_brdf = False if kwargs.get(
                'is_brdf') is None else kwargs['is_brdf']
        elif bsdf_type == Renderer.BSDF_TYPE.GY_mitsuba:
            from GY_mitsuba import GY_multilayered
            self.filename = kwargs['filename']
            self.sample_times = kwargs['sample_times']

            surface_type_name = kwargs['surface_type']
            if surface_type_name == 'dielectric':
                self.surface_type = GY_multilayered.dielectric 
            elif surface_type_name == 'conductor':
                self.surface_type = GY_multilayered.conductor 
            else:
                raise Exception('unsupported surface type: ' + surface_type_name)

            self.GY_bsdf = GY_multilayered(self.filename,
                                           sample_times=self.sample_times,
                                           type=self.surface_type)
        else:
            raise Exception('unsupported bsdf type')
        if illuminant_type == Renderer.ILLUMINANT_TYPE.point_light:
            self.intensity = kwargs['intensity']
            self.light_position = np.array(kwargs['light_position'])
        elif illuminant_type == Renderer.ILLUMINANT_TYPE.directional_light:
            self.intensity = kwargs['intensity']
            self.light_direction = Renderer.make_unit_vector(
                np.array(kwargs['light_direction']))
        else:
            raise Exception('unsupported illuminant type')

    def bsdf(self, theta_i, phi_i, theta_o, phi_o):
        if theta_i < 0 or theta_o < 0 or phi_i < 0 or phi_o < 0:
            raise Exception('illegal theta,phi range')
        if self.bsdf_type == Renderer.BSDF_TYPE.single_net:
            import tensorflow as tf
            wiwo = tf.constant([theta_i, phi_i, theta_o, phi_o],
                               shape=[1, 1, 4])
            pred = self.bsdf_net(wiwo, training=False)
            pred = tf.squeeze(pred)
            f = (tf.math.exp(pred * tf.math.log(1.0 + self.miu)) -
                 1.0) / self.miu
            return f.numpy()
        if self.bsdf_type == Renderer.BSDF_TYPE.meta_net:
            import tensorflow as tf
            wiwo = tf.constant([theta_i, phi_i, theta_o, phi_o],
                               dtype=tf.float32,
                               shape=[1, 1, 4])
            pred = self.bsdf_net.inference(wiwo)
            pred = tf.squeeze(pred)
            f = (tf.math.exp(pred * tf.math.log(1.0 + self.miu)) -
                 1.0) / self.miu
            return f.numpy()
        if self.bsdf_type == Renderer.BSDF_TYPE.lambertian:
            return self.albedo * self.inv_pi
        if self.bsdf_type == Renderer.BSDF_TYPE.GY_mitsuba:
            return self.GY_bsdf.eval(theta_i=theta_i,
                                     phi_i=phi_i,
                                     theta_o=theta_o,
                                     phi_o=phi_o,
                                     debug=self.debug)
        if self.bsdf_type == Renderer.BSDF_TYPE.texture:
            import utils
            return utils.interpolate_texture(self.texture, theta_i, phi_i,
                                             theta_o, phi_o,
                                             self.theta_sample_rate,
                                             self.phi_sample_rate,
                                             self.is_brdf)
        if self.bsdf_type == Renderer.BSDF_TYPE.phong:
            pass
        return np.array([0, 0, 0])

    def shading(self, hit_point, ray_direction, normal, vup):
        x = Renderer.make_unit_vector(np.cross(normal, vup))
        y = np.cross(normal, x)
        x = np.reshape(x, [3, 1])
        y = np.reshape(y, [3, 1])
        z = np.reshape(normal, [3, 1])
        shading_frame = np.concatenate([x, y, z], axis=1)
        shading_frame_inv = np.linalg.inv(shading_frame)
        wo = np.matmul(shading_frame_inv, -ray_direction)
        # caution theta range should be [0-2*pi], phi range should be [0-pi]
        phi_o = np.arccos(wo[2])
        theta_o = np.arctan2(wo[1] / np.sin(phi_o), wo[0] / np.sin(phi_o))
        if theta_o < 0:
            theta_o += 2 * np.pi
        if self.illuminant_type == Renderer.ILLUMINANT_TYPE.directional_light:
            wi = np.matmul(shading_frame_inv, -self.light_direction)
            L = self.intensity
        elif self.illuminant_type == Renderer.ILLUMINANT_TYPE.point_light:
            wi = np.matmul(
                shading_frame_inv,
                Renderer.make_unit_vector(self.light_position - hit_point))
            dist = np.linalg.norm(self.light_position - hit_point)
            L = self.intensity / dist * dist
        phi_i = np.arccos(wi[2])
        theta_i = np.arctan2(wi[1] / np.sin(phi_i), wi[0] / np.sin(phi_i))
        if theta_i < 0:
            theta_i += 2 * np.pi
        f = self.bsdf(theta_i, phi_i, theta_o, phi_o)
        return f * np.abs(np.cos(phi_o)) * L

    class Ray:
        def __init__(self, origin, direction):
            self.origin = origin
            self.d = direction / np.linalg.norm(direction)

        epsilon = 0.0001

        def point_at_dist(self, dist):
            return self.origin + dist * self.d

        def intersect_sphere(self, center, radius):
            co = self.origin - center
            a = np.dot(self.d, self.d)
            b = np.dot(self.d, co)
            c = np.dot(co, co) - radius * radius
            discriminant = b * b - a * c
            tmin = self.epsilon * radius
            if discriminant > 0:
                tmp = (-b - np.sqrt(discriminant)) / a
                if tmp > tmin:
                    hit_point = self.point_at_dist(tmp)
                    normal = Renderer.make_unit_vector(hit_point - center)
                    return True, hit_point, normal
                tmp = (-b + np.sqrt(discriminant)) / a
                if tmp > tmin:
                    hit_point = self.point_at_dist(tmp)
                    normal = Renderer.make_unit_vector(hit_point - center)
                    return True, hit_point, normal
            return False, None, None

    class Camera:
        def __init__(self, eye, lookat, vup, fov, aspect_ratio):
            self.eye = np.array(eye)
            self.lookat = np.array(lookat)
            self.vup = np.array(vup)
            self.fov = np.deg2rad(fov)
            self.aspect_ratio = aspect_ratio
            self.compute_uvw()

        def compute_uvw(self):
            w = self.eye - self.lookat
            self.w = w / np.linalg.norm(w)
            u = np.cross(self.vup, w)
            self.u = u / np.linalg.norm(u)
            self.v = np.cross(self.w, self.u)

        def generate_ray(self, offset_x, offset_y):
            """
            offset_x,y: [0,1]
            """
            dist = np.linalg.norm(self.lookat - self.eye)
            height = 2 * dist * np.tan(self.fov / 2)
            width = height * self.aspect_ratio
            dest = self.lookat + offset_x * width * self.u + offset_y * height * self.v
            return Renderer.Ray(self.eye, dest - self.eye)

    def render_ball(self, img_w, img_h, spp, output_path):
        """
        render sphere 
        return 
        """
        import imageio
        from random import uniform
        from progressbar import ProgressBar
        progress = ProgressBar()
        buf = np.zeros([img_h, img_w, 3], dtype=np.float32)
        for i in progress(range(img_h)):
            for j in range(img_w):
                # cast primary ray
                buf[i, j, :] = 0
                for k in range(spp):
                    offset_x = (2 * j + 1 - img_w + uniform(-1, 1)) / img_w
                    offset_y = (2 * i + 1 - img_h + uniform(-1, 1)) / img_h
                    ray = self.camera.generate_ray(offset_x=offset_x,
                                                   offset_y=offset_y)
                    hit_anything, hit_point, normal = ray.intersect_sphere(
                        np.array([0, 0, 0]), 0.5)
                    if hit_anything:
                        buf[i, j, :] += self.shading(hit_point=hit_point,
                                                     ray_direction=ray.d,
                                                     normal=normal,
                                                     vup=self.camera.vup)
                    else:
                        buf[i, j, :] += 0
                buf[i, j, :] /= spp
        import os
        dir = os.path.dirname(output_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        imageio.imsave(output_path, im=buf)


if __name__ == '__main__':
    render = Renderer(bsdf_type=Renderer.BSDF_TYPE.lambertian,
                      illuminant_type=Renderer.ILLUMINANT_TYPE.point_light,
                      albedo=[1, 0, 0],
                      intensity=1,
                      light_position=[0, 0, 1])
    render.render_ball(600, 600, spp=1, output_path='./test.png')
