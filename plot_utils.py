import argparse
import numpy as np
import tqdm.auto as tqdm
import math
import tempfile
import os
import atexit
import shutil

C = np.array([
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (0, 114, 178),
]) / 255

def import_matplotlib():
    mpldir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, mpldir)
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(mpldir, 0o777 & ~umask)
    os.environ['MPLCONFIGDIR'] = mpldir
    import matplotlib
    if not os.environ.get('DISPLAY'):
        matplotlib.use('PDF')
    import matplotlib.texmanager
    class TexManager(matplotlib.texmanager.TexManager):
        texcache = os.path.join(mpldir, 'tex.cache')

    matplotlib.texmanager.TexManager = TexManager

    pdfmpl = {
        # Use LaTeX to write all text
        "text.usetex": matplotlib.checkdep_usetex(True),
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 25,
        "font.size": 23,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 14,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.figsize": (8, 4),
        "figure.dpi": 100,
        "legend.loc": 'best',
        'axes.titlepad': 20,
        'pdf.use14corefonts': True,
        'ps.useafm': True,
        'lines.linewidth': 3,
        'lines.markersize': 10,
    }
    matplotlib.rcParams.update(pdfmpl)

import_matplotlib()
import matplotlib.pyplot as plt

def format_axes(ax, twinx=False):
    SPINE_COLOR = 'gray'

    if twinx:
        visible_spines = ['bottom', 'right']
        invisible_spines = ['top', 'left']
    else:
        visible_spines = ['bottom', 'left']
        invisible_spines = ['top', 'right']

    for spine in invisible_spines:
        ax.spines[spine].set_visible(False)

    for spine in visible_spines:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')

    if twinx:
        ax.yaxis.set_ticks_position('right')
    else:
        ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax



import matplotlib.pyplot as plt
from matplotlib.ticker import Locator


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))
