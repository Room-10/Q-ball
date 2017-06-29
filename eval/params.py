
import sys, os, pickle, json

noise = True
for output_dir in sys.argv[1:]:
    params_file = os.path.join(output_dir, 'params.pickle')
    print(output_dir)
    params = pickle.load(open(params_file, 'rb'))
    print(json.dumps(params, sort_keys=True, indent=4))