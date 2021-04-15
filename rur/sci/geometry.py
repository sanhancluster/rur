import numpy as np
from scipy.spatial.transform import Rotation

def euler_angle(coo, angles):
    # rotation transform by euler angles (phi-theta-psi)
    rot = Rotation.from_euler({'Z', 'X', 'Z'}, angles)
    return rot.apply(coo)

def get_angles(coo):
    x, y, z = coo[..., 0], coo[..., 1], coo[..., 2]
    phi = np.arctan2(x, y)
    theta = np.arctan2(z, np.sqrt(x**2 + y**2))
    return phi, theta

def rss(coo, axis=-1):
    # root sum square
    return np.sqrt(ss(coo, axis))

def ss(coo, axis=-1):
    # square sum
    return np.sum(coo ** 2, axis=axis)

def rms(coo, axis=-1):
    # root mean square
    return np.sqrt(np.mean(coo**2, axis=axis))
