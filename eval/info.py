
import sys, os, pickle, json
import numpy as np

def print_params(output_dir):
    params_file = os.path.join(output_dir, 'params.pickle')
    if not os.path.exists(params_file):
        print("No parameters set.")
        return
    params = pickle.load(open(params_file, 'rb'))
    print(json.dumps(params, sort_keys=True, indent=4, default=repr))

def print_dists(output_dir):
    dists_file = os.path.join(output_dir, 'dists.npz')
    if not os.path.exists(dists_file):
        print("No distance information available.")
        return
    dists = np.load(open(dists_file, 'rb'))
    noise_dists = {}
    res_dists = {}
    for dname, val in dict(dists).items():
        if dname[:6] == "noise_":
            noise_dists[dname] = val
        else:
            res_dists[dname] = val
    for dname, val in res_dists.items():
        noise_dname = "noise_%s" % dname
        noise_val = noise_dists[noise_dname]
        print("%s: %.5f (min: %.5f, max: %.5f)" % (
            noise_dname, np.sum(noise_val), np.amin(noise_val), np.amax(noise_val)))
        print("%s: %.5f (min: %.5f, max: %.5f)" % (
            dname, np.sum(val), np.amin(val), np.amax(val)))

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Print info.")
    parser.add_argument('basedirs', metavar='BASENAMES', nargs='+', type=str)
    parser.add_argument('--params', action="store_true", default=False)
    parser.add_argument('--dists', action="store_true", default=False)
    parsed_args = parser.parse_args()

    for i,output_dir in enumerate(parsed_args.basedirs):
        print("=> %s" % output_dir)
        printall = not parsed_args.params and not parsed_args.dists
        if printall or parsed_args.params:
            print_params(output_dir)
        if printall or parsed_args.dists:
            print_dists(output_dir)
        if i+1 < len(parsed_args.basedirs):
            print("")
