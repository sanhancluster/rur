import numpy as np
from scipy.spatial.transform import Rotation

def euler_angle(coo, angles):
    # rotation transform by euler angles (phi-theta-psi)
    rot = Rotation.from_euler('ZXZ', angles)
    return rot.apply(coo)

def get_angles(coo):
    x, y, z = coo[..., 0], coo[..., 1], coo[..., 2]
    phi = np.arctan2(x, y)
    theta = np.arctan2(z, np.sqrt(x**2 + y**2))
    omega = np.zeros_like(theta)
    return np.stack([phi, theta, omega], axis=-1)

def rss(coo, axis=-1):
    # root sum square
    return np.sqrt(ss(coo, axis))

def ss(coo, axis=-1):
    # square sum
    return np.sum(coo ** 2, axis=axis)

def rms(coo, axis=-1):
    # root mean square
    return np.sqrt(np.mean(coo**2, axis=axis))

def rotate(p, angle=0, origin=(0, 0)):
    # rotate transformation of 2d coordinate
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def poly_area(coo):
    # calculate area enclosed by 2d points
    x, y = (*coo.T,)
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))