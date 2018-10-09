
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
    'solver_name': 'pdhg',
    'solver': {
        'step_factor': 0.01,
        'step_bound': 0.0014,
    },
    'model': {}
}

for i in range(count):
    lbd = lbd_min + lbd_step*i
    output_dir = "%s-%.4f" % (datadir, lbd)
    shutil.copytree(datadir, output_dir)
    params_file = os.path.join(output_dir, 'params.pickle')
    params['model']['lbd'] = lbd
    pickle.dump(params, open(params_file, 'wb'))
