from rur import *

def draw_sink_timeline(tl, modes=None):
    if modes is None:
        modes = ['mass', 'velocity', 'density', 'df', 'accretion_rate']

