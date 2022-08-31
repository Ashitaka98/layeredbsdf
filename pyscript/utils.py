import numpy as np
from math import sin, cos, sqrt, acos, atan2, pi


def Log(str):
    print('[python]: ' + str)


def whwd_to_wiwo_xyz(whwd):
    theta_h, phi_h, theta_d, phi_d = whwd
    x1 = cos(theta_h) * sin(phi_h)
    y1 = sin(theta_h) * sin(phi_h)
    z1 = cos(phi_h)
    x2 = cos(theta_d) * sin(phi_d)
    y2 = sin(theta_d) * sin(phi_d)
    z2 = cos(phi_d)
    h = np.array([x1, y1, z1])
    d = np.array([x2, y2, z2])
    rot_z = np.array([[cos(theta_h), -sin(theta_h), 0.0],
                      [sin(theta_h), cos(theta_h), 0.0], [0.0, 0.0, 1.0]])
    rot_y = np.array([[cos(phi_h), 0.0, sin(phi_h)], [0.0, 1.0, 0.0],
                      [-sin(phi_h), 0.0, cos(phi_h)]])
    wi = np.squeeze(np.dot(rot_y, d))
    wi = np.squeeze(np.dot(rot_z, wi))
    wo = 2 * np.dot(wi, h) * h - wi
    wi[2] = np.clip(wi[2], -0.9999, 0.9999)
    wo[2] = np.clip(wo[2], -0.9999, 0.9999)
    return wi, wo


def whwd_to_wiwo(whwd):
    theta_h, phi_h, theta_d, phi_d = whwd
    x1 = cos(theta_h) * sin(phi_h)
    y1 = sin(theta_h) * sin(phi_h)
    z1 = cos(phi_h)
    x2 = cos(theta_d) * sin(phi_d)
    y2 = sin(theta_d) * sin(phi_d)
    z2 = cos(phi_d)
    h = np.array([x1, y1, z1])
    d = np.array([x2, y2, z2])
    rot_z = np.array([[cos(theta_h), -sin(theta_h), 0.0],
                      [sin(theta_h), cos(theta_h), 0.0], [0.0, 0.0, 1.0]])
    rot_y = np.array([[cos(phi_h), 0.0, sin(phi_h)], [0.0, 1.0, 0.0],
                      [-sin(phi_h), 0.0, cos(phi_h)]])
    wi = np.squeeze(np.dot(rot_y, d))
    wi = np.squeeze(np.dot(rot_z, wi))
    wo = 2 * np.dot(wi, h) * h - wi
    wi[2] = np.clip(wi[2], -0.9999, 0.9999)
    wo[2] = np.clip(wo[2], -0.9999, 0.9999)
    phi_o = acos(wo[2])
    theta_o = atan2(wo[1], wo[0])
    phi_i = acos(wi[2])
    theta_i = atan2(wi[1], wi[0])
    theta_o += 2 * pi - 0.00001 if theta_o < 0 else 0.0
    theta_i += 2 * pi - 0.00001 if theta_i < 0 else 0.0
    return [theta_i, phi_i, theta_o, phi_o]


def wiwo_to_whwd(wiwo):

    theta_i, phi_i, theta_o, phi_o = wiwo

    x1 = cos(theta_i) * sin(phi_i)
    y1 = sin(theta_i) * sin(phi_i)
    z1 = cos(phi_i)
    x2 = cos(theta_o) * sin(phi_o)
    y2 = sin(theta_o) * sin(phi_o)
    z2 = cos(phi_o)
    hx = x1 + x2
    hy = y1 + y2
    hz = z1 + z2
    norm_h = sqrt(hx * hx + hy * hy + hz * hz + 0.00001)

    hx /= norm_h
    hy /= norm_h
    hz /= norm_h
    phi_h = acos(hz)
    theta_h = atan2(hy, hx)
    theta_h += 2 * pi - 0.00001 if theta_h < 0 else 0.0
    wi = np.array([x1, y1, z1])

    rot_z = np.array([[cos(-theta_h), -sin(-theta_h), 0.0],
                      [sin(-theta_h), cos(-theta_h), 0.0], [0.0, 0.0, 1.0]])
    rot_y = np.array([[cos(-phi_h), 0.0, sin(-phi_h)], [0.0, 1.0, 0.0],
                      [-sin(-phi_h), 0.0, cos(-phi_h)]])

    d = np.squeeze(np.dot(rot_z, wi))
    d = np.squeeze(np.dot(rot_y, d))
    dx = d[0]
    dy = d[1]
    dz = d[2]
    dz = np.clip(dz, -0.9999, 0.9999)
    phi_d = acos(dz)
    theta_d = atan2(dy, dx)
    theta_d += 2 * pi - 0.00001 if theta_d < 0 else 0.0
    return [theta_h, phi_h, theta_d, phi_d]


if __name__ == "__main__":
    wiwo = whwd_to_wiwo([0.1, 0.2, 0.3, 0.4])
    print(wiwo_to_whwd(wiwo))
