
from argparse import ArgumentParser
import pickle, os, shutil, sys

parser = ArgumentParser(description="Create test dirs")
parser.add_argument('basename', metavar='BASENAME', type=str)
parser.add_argument('--lbd-min', default=2.40, type=float)
parser.add_argument('--lbd-step', default=0.2, type=float)
parser.add_argument('--count', default=5, type=int)
parsed_args = parser.parse_args()

datadir = parsed_args.basename.rstrip("/")
lbd_min = parsed_args.lbd_min
lbd_step = parsed_args.lbd_step
count = parsed_args.count

params = {
    'base': {
        'sh_order': 6,
        'smooth': 0,
        'min_signal': 0,
        'assume_normed': True,
    },
    'fit': {
        'sphere': None,
        'solver_params': {
            'lbd': 1.5,
            'term_relgap': 1e-05,
            'term_maxiter': 100000,
            'granularity': 5000,
            'step_factor': 0.0001,
            'step_bound': 1.3,
            'dataterm': "W1",
            'use_gpu': True
        },
    },
}

for i in range(count):
    lbd = lbd_min + lbd_step*i
    output_dir = "%s-%.4f" % (datadir, lbd)
    shutil.copytree(datadir, output_dir)
    params_file = os.path.join(output_dir, 'params.pickle')
    params['fit']['solver_params']['lbd'] = lbd
    pickle.dump(params, open(params_file, 'wb'))