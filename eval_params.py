
import sys, os, pickle

noise = True
for output_dir in sys.argv[1:]:
    params_file = os.path.join(output_dir, 'params.pickle')
    print(output_dir)
    print(pickle.load(open(params_file, 'rb')))