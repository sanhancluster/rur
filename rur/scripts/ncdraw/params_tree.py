from rur import drawer
import numpy as np
import matplotlib
try:
    import cmasher as cmr
    CMAP = 'cmr.redshift_r'
except:
    v_tuple = tuple(map(int, matplotlib.__version__.split('.')[:2]))
    if v_tuple >= (3, 10):
        CMAP = 'berlin_r'
    else:
        CMAP = 'coolwarm'

REFOUT = 633 # Reference iout, to select galaxy and halo
MAXMASS_CUT = 1e13 # Max Mvir across tree

# Detailed
DETAIL={
    'Mass':dict(
        # Axis
        xscale = 'linear', xlabel='Age of Univ. [Gyr]',
        xmin=None, xmax=None, invertx = False,
        yscale='log', ylabel = r'$M\ [M_\odot]$',
        ymin=None, ymax=None, inverty = False,
        # Text
        tlabel = 'Mass Growth',
        tloc = (0.02, 'left', 0.98, 'top'),
        # Colorbar
        cloc = (0.04, 0.15, 0.63, 0.03) # x, y, width, height
    ),
    'Dist':dict(
        # Axis
        xscale = 'linear', xlabel='Age of Univ. [Gyr]',
        xmin=None, xmax=None, invertx = False,
        yscale='linear', ylabel = r'd [Mpc]',
        ymin=0, ymax=None, inverty = False,
        # Text
        tlabel = 'Distance to Primary',
        tloc = (0.02, 'left', 0.98, 'top'),
        # Colorbar
        cloc = (0.04, 0.15, 0.63, 0.03) # x, y, width, height
    ),
    'DTM':dict(
        # Axis
        xscale = 'log', xlabel = r'$M_*\ [M_\odot]$',
        xmin=None, xmax=None, invertx = False,
        yscale = 'log', ylabel = "DTM",
        ymin=None, ymax=None, inverty = False,
        # Text
        tlabel = "DTM",
        tloc = (0.02, 'left', 0.98, 'top'),
        # Colorbar
        cloc = (0.04, 0.80, 0.63, 0.03), # x, y, width, height
    ),
    'alpha':dict(
        # Axis
        xscale = 'linear', xlabel = r"$\mathrm{[Fe/H]}$",
        xmin=None, xmax=None, invertx = False,
        yscale = 'linear', ylabel = r"$\mathrm{[\alpha/Fe]}$",
        ymin=None, ymax=None, inverty = False,
        # Text
        tlabel = r"$\mathrm{[\alpha/Fe]-[Fe/H]}$",
        tloc = (0.98, 'right', 0.98, 'top'),
        # Colorbar
        cloc = (0.04, 0.15, 0.63, 0.03) # x, y, width, height
    ),
    'O/H':dict(
        # Axis
        xscale = 'log', xlabel = r'$M_*\ [M_\odot]$',
        xmin=None, xmax=None, invertx = False,
        yscale = 'linear', ylabel = r"12 + log(O/H)$_{\rm gas}$",
        ymin=None, ymax=None, inverty = False,
        # Text
        tlabel = 'O/H',
        tloc = (0.02, 'left', 0.98, 'top'),
        # Colorbar
        cloc = (0.30, 0.15, 0.63, 0.03) # x, y, width, height
    ),
    'Cold':dict(
        # Axis
        xscale = 'log', xlabel = r'$M_*\ [M_\odot]$',
        xmin=None, xmax=None, invertx = False,
        yscale = 'log', ylabel = r'$M_{\rm cold,\,R90}\,/\,M_*$',
        ymin=None, ymax=None, inverty = False,
        # Text
        tlabel = 'Cold Gas',
        tloc = (0.98, 'right', 0.98, 'top'),
        # Colorbar
        cloc = (0.04, 0.15, 0.63, 0.03) # x, y, width, height
    ),
    'MBH':dict(
        # Axis
        xscale = 'log', xlabel = r'$M_*\ [M_\odot]$',
        xmin=None, xmax=None, invertx = False,
        yscale = 'log', ylabel = r'$M_{\rm BH}\ [M_\odot]$',
        ymin=1e5, ymax=None, inverty = False,
        # Text
        tlabel = 'MBH',
        tloc = (0.02, 'left', 0.98, 'top'),
        # Colorbar
        cloc = (0.04, 0.75, 0.63, 0.03) # x, y, width, height
    ),
    'Metal':dict(
        # Axis
        xscale = 'log', xlabel = r'$M_*\ [M_\odot]$',
        xmin=None, xmax=None, invertx = False,
        yscale = 'log', ylabel = r'$Z_*$',
        ymin=None, ymax=None, inverty = False,
        # Text
        tlabel = 'Metallicity',
        tloc = (0.02, 'left', 0.98, 'top'),
        # Colorbar
        cloc = (0.30, 0.15, 0.63, 0.03) # x, y, width, height
    ),
    'Size':dict(
        # Axis
        xscale = 'log', xlabel = r'$M_*\ [M_\odot]$',
        xmin=None, xmax=None, invertx = False,
        yscale = 'log', ylabel = r'R$_{50}$ [kpc]',
        ymin=None, ymax=None, inverty = False,
        # Text
        tlabel = 'Size',
        tloc = (0.02, 'left', 0.98, 'top'),
        # Colorbar
        cloc = (0.04, 0.75, 0.63, 0.03) # x, y, width, height
    ),
    'CMD':dict(
        # Axis
        xscale = 'linear', xlabel = r'$M_r$',
        xmin=None, xmax=None, invertx = True,
        yscale = 'linear', ylabel = r'$(g-r)$',
        ymin=None, ymax=None, inverty = False,
        # Text
        tlabel = 'CMD',
        tloc = (0.02, 'left', 0.98, 'top'),
        # Colorbar
        cloc = (0.04, 0.75, 0.63, 0.03) # x, y, width, height
    ),
    'SB':dict(
        # Axis
        xscale = 'linear', xlabel = r'$M_r$',
        xmin=None, xmax=None, invertx = True,
        yscale = 'linear', ylabel = r'$\mu_{e,\,r}$',
        ymin=None, ymax=None, inverty = True,
        # Text
        tlabel = 'Surface Brightness',
        tloc = (0.02, 'left', 0.98, 'top'),
        # Colorbar
        cloc = (0.04, 0.75, 0.63, 0.03) # x, y, width, height
    ),
    'SHMR':dict(
        # Axis
        xscale = 'log', xlabel = r'$M_{\rm vir}\ [M_\odot]$',
        xmin=None, xmax=None, invertx = False,
        yscale = 'log', ylabel = r'$M_*\,/\,M_{\rm vir}$',
        ymin=None, ymax=None, inverty = False,
        # Text
        tlabel = 'SHMR',
        tloc = (0.02, 'left', 0.98, 'top'),
        # Colorbar
        cloc = (0.30, 0.15, 0.63, 0.03) # x, y, width, height
    ),
    'SFMS':dict(
        # Axis
        xscale = 'log', xlabel = r'$M_*\ [M_\odot]$',
        xmin=None, xmax=None, invertx = False,
        yscale = 'log', ylabel = r'SFR [M$_\odot$/yr]',
        ymin=None, ymax=None, inverty = False,
        # Text
        tlabel = 'SFMS',
        tloc = (0.02, 'left', 0.98, 'top'),
        # Colorbar
        cloc = (0.30, 0.15, 0.63, 0.03) # x, y, width, height
    ),
    }
