
import matplotlib
matplotlib.use('Agg')

try:
    import qball
except:
    import set_qball_path
from qball.sphere import load_sphere
from qball.tools.w1dist import w1_dist
from qball.tools import normalize_odf

import sys, os, pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

import dipy.core.sphere
from dipy.reconst.shm import CsaOdfModel

noise = True
for output_dir in sys.argv[1:]:
    dists_file = os.path.join(output_dir, 'dists.npz')
    dists_plot_file = os.path.join(output_dir, 'plot_dists.pdf')
    data_file = os.path.join(output_dir, 'data.pickle')

    if not os.path.exists(dists_file):
        print("No distance information available.")
        continue

    dists_npz = np.load(open(dists_file, 'rb'))
    data = pickle.load(open(data_file, 'rb'))
    S_data = data['raw'][data['slice']]
    imagedims = S_data.shape[:-1]

    fig = plt.figure()
    subplot_opts = {
        'xticklabels': [],
        'yticklabels': [],
    }
    for i, (distname, dist) in enumerate(dists_npz.items()):
        a = fig.add_subplot(2,2,i+1, **subplot_opts)
        if len(imagedims) == 1:
            d = np.atleast_2d(dist.reshape(imagedims))
        else:
            d = dist.reshape(imagedims).T
        plt.imshow(d, cmap=cm.coolwarm, origin='lower', interpolation='none')
        a.set_title(distname)
    plt.savefig(dists_plot_file)
    #plt.show()
