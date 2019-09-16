import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def testcard(cmap, seed=42):
    '''
    Makes a 4-panel figure showing color map's ramp, and some interesting test plots.
    
    Parameters
    ----------
    cmap : `matplotlib.colors.ListedColormap`
        Instance of a color map object
    seed : int
        Random seed, default=42
        
    Returns
    -------
    fig : matplotlib.Figure
        Instance of a figure object
    '''
    np.random.seed(seed)
    
    x = np.random.uniform(size=256)
    y = np.random.normal(0., 0.1, size=x.size)
    idx = np.abs(y) < 0.2*np.sqrt(x)
    x = x[idx]
    y = y[idx]
    c = y

    fig,axes = plt.subplots(2,2,figsize=(12,12))
    
    axes[0,0].scatter(x, y, c=c, cmap=cmap, edgecolor='#555555', linewidth=1., s=32, 
                      vmin=-0.2, vmax=0.2)
    axes[0,0].set_ylim(-0.5, 0.5)

    X,Y = np.meshgrid(np.linspace(0,1,32), np.linspace(0,1,32))
    C = X
    axes[0,1].pcolormesh(X, Y, C, cmap=cmap)

    X,Y = np.meshgrid(np.linspace(-1,1,128), np.linspace(-1,1,128))
    C = np.arcsin(np.sin(2*(X**2 + Y**2) + np.arctan2(Y, X)))
    axes[1,0].pcolormesh(X, Y, C, cmap=cmap)

    i,j = np.meshgrid(np.arange(512), np.arange(512))
    axes[1,1].pcolormesh(i, j, np.bitwise_xor(i,j), cmap=cmap)
    axes[1,1].set_xlim(0,511)
    axes[1,1].set_ylim(0,511)

    fig.suptitle(str(cmap.name), fontsize=28, y=0.99)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    
    return fig
