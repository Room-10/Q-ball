
import sys, os, pickle, json
import numpy as np

try:
    import qball
except:
    import set_qball_path
from qball.sphere import load_sphere
from qball.tools.bounds import compute_bounds

from compute_dists import reconst_f

# Foreground mask for experiment `cross`
cross_mask = np.rot90(np.array([
    [ 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
    [ 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
    [ 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
    [ 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    [ 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [ 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [ 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [ 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [ 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
], dtype=bool), k=3)

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

def print_bounds(output_dir):
    params_file = os.path.join(output_dir, 'params.pickle')
    gtab_file = os.path.join(output_dir, 'gtab.pickle')
    S_data_file = os.path.join(output_dir, 'S_data.np')
    result_file = os.path.join(output_dir, 'result_raw.pickle')

    params = pickle.load(open(params_file, 'rb'))
    gtab = pickle.load(open(gtab_file, 'rb'))
    S_data = np.load(open(S_data_file, 'rb'))
    result = pickle.load(open(result_file, 'rb'))[0]

    upd = result['u2']
    b_vecs = gtab.bvecs[gtab.bvals > 0,...]
    b_sph = load_sphere(vecs=b_vecs.T)
    try:
        conf_lvl = params['fit']['solver_params']['conf_lvl']
    except:
        conf_lvl = float(input("conf_lvl: "))

    fl, fu = compute_bounds(b_sph, S_data, alpha=conf_lvl, mask=cross_mask)

    interv_sizes = fu-fl
    print("Range: %.2f ... %.2f" % (np.amin(fl),np.amax(fu)))
    print("min/avg/max interval size: %.2f ... %.2f ... %.2f" % (
        np.amin(interv_sizes), np.mean(interv_sizes), np.amax(interv_sizes)))

    in_mod_interval = []
    perc = 30
    mod_count = int(120/perc)+1
    for k in range(mod_count):
        fl_mod = fl - k*(perc/100.)*interv_sizes
        fu_mod = fu + k*(perc/100.)*interv_sizes
        idx = (fl_mod < upd) & (upd < fu_mod)
        in_mod_interval.append(np.sum(idx))
    print("  Intervals enlarged by:",
        " ".join(("% 5d%%" % (2*perc*i)) for i in range(mod_count)))
    print("% in enlarged Intervals:",
        " ".join(("% 5.1f%%" % (100*i/upd.size)) for i in in_mod_interval))

def entropy(u, vol):
    u_pos = np.fmax(np.spacing(1),u)
    unif = 0*u + 1./np.sum(vol) # uniform distribution
    return np.einsum('k...,k->...', u_pos*np.log(u_pos/unif), vol)

def print_entropy(output_dir):
    gtab_file = os.path.join(output_dir, 'gtab.pickle')
    result_file = os.path.join(output_dir, 'result_raw.pickle')

    gtab = pickle.load(open(gtab_file, 'rb'))
    result = pickle.load(open(result_file, 'rb'))[0]

    b_vecs = gtab.bvecs[gtab.bvals > 0,...]
    b_sph = load_sphere(vecs=b_vecs.T)
    f_gt, f_noisy = reconst_f(output_dir, b_sph)
    upd = result['u1'].reshape(f_gt.shape[0],-1)

    mask = crossmask.reshape(upd.shape[1:])
    f_gt_fg = f_gt[:,mask]
    f_noisy_fg = f_noisy[:,mask]
    upd_fg = upd[:,mask]
    print("  Ground truth (fg): %.3f" % np.mean(entropy(f_gt_fg, b_sph.b).ravel()))
    print("         Noisy (fg): %.3f" % np.mean(entropy(f_noisy_fg, b_sph.b).ravel()))
    print("Reconstruction (fg): %.3f" % np.mean(entropy(upd_fg, b_sph.b).ravel()))
    print()

    f_gt_bg = f_gt[:,np.logical_not(mask)]
    f_noisy_bg = f_noisy[:,np.logical_not(mask)]
    upd_bg = upd[:,np.logical_not(mask)]
    print("  Ground truth (bg): %.4f" % np.mean(entropy(f_gt_bg, b_sph.b).ravel()))
    print("         Noisy (bg): %.4f" % np.mean(entropy(f_noisy_bg, b_sph.b).ravel()))
    print("Reconstruction (bg): %.4f" % np.mean(entropy(upd_bg, b_sph.b).ravel()))

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Print info.")
    parser.add_argument('basedirs', metavar='BASENAMES', nargs='+', type=str)
    parser.add_argument('--params', action="store_true", default=False)
    parser.add_argument('--dists', action="store_true", default=False)
    parser.add_argument('--bounds', action="store_true", default=False)
    parser.add_argument('--entropy', action="store_true", default=False)
    parsed_args = parser.parse_args()

    for i,output_dir in enumerate(parsed_args.basedirs):
        print("=> %s" % output_dir)
        if not os.path.exists(output_dir):
            print("Directory not found. Skipping...")
            continue
        printall = not parsed_args.params and not parsed_args.dists \
                   and not parsed_args.bounds and not parsed_args.entropy
        if printall or parsed_args.params:
            print_params(output_dir)
        if printall or parsed_args.dists:
            print_dists(output_dir)
        if parsed_args.bounds:
            print_bounds(output_dir)
        if parsed_args.entropy:
            print_entropy(output_dir)
        if i+1 < len(parsed_args.basedirs):
            print("")
