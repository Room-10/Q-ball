
import numpy as np
np.set_printoptions(precision=4,
                    linewidth=200,
                    suppress=True,
                    threshold=10000)

import sys, os, errno, glob, zipfile, pickle, signal
from datetime import datetime
from argparse import ArgumentParser

import logging
logging.basicConfig(
    stream=sys.stdout,
    format="[%(relativeCreated) 8d] %(message)s",
    level=logging.DEBUG
)

import dipy.core.sphere
from dipy.reconst.shm import CsaOdfModel
from dipy.viz import fvtk

class GracefulInterruptHandler(object):
    """ Context manager for handling SIGINT (e.g. if the user presses Ctrl+C)

    Taken from https://gist.github.com/nonZero/2907502

    >>> with GracefulInterruptHandler() as h:
    >>>     for i in xrange(1000):
    >>>         print "..."
    >>>         time.sleep(1)
    >>>         if h.interrupted:
    >>>             print "interrupted!"
    >>>             time.sleep(2)
    >>>             break
    """
    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):
        self.interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)
        signal.signal(self.sig, self.handle)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def handle(self, signum, frame):
        self.release()
        self.interrupted = True

    def release(self):
        if self.released:
            return False
        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True

def add_log_file(logger, output_dir):
    """ Utility function for consistent log file names.

    Args:
        logger : an instance of logging.Logger
        output_dir : path to output directory
    """
    log_file = os.path.join(output_dir, "{}-{}.log".format(
        datetime.now().strftime('%Y%m%d%H%M%S'), logger.name
    ))
    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(
        logging.Formatter(fmt="[%(relativeCreated) 8d] %(message)s")
    )
    fileHandler.setLevel(logging.DEBUG)
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]
    logger.addHandler(fileHandler)

def output_dir_name(label):
    """ Utility function for consistent output dir names.

    Args:
        label : a string that describes the data stored
    Returns:
        path to output directory
    """
    return "./results/{}-{}".format(
        datetime.now().strftime('%Y%m%d%H%M%S'), label
    )

def output_dir_create(output_dir):
    """ Recursively create the given path's directories.

    Args:
        output_dir : some directory path
    """
    try:
        if sys.version_info[0] == 3:
            os.makedirs(output_dir, exist_ok=True)
        else:
            os.makedirs(output_dir)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            print("Can't create directory {}!".format(output_dir))
            raise

def data_from_file(path, format="np"):
    """ Load numpy or pickle data from the given file.

    Args:
        path : path to data file
        format : if "pickle", pickle is used to load the data (else numpy is used)
    Returns:
        restored numpy or pickle data
    """
    try:
        if format == "pickle":
            return pickle.load(open(path, 'rb'))
        else:
            return np.load(open(path, 'rb'))
    except:
        return None

def addDirToZip(zipHandle, path, basePath="", exclude=[]):
    """ From https://stackoverflow.com/a/17020687
    Adding directory given by \a path to opened zip file \a zipHandle

    @param basePath path that will be removed from \a path when adding to archive
    """
    basePath = basePath.rstrip("\\/") + ""
    basePath = basePath.rstrip("\\/")
    for root, dirs, files in os.walk(path):
        if os.path.basename(root) in exclude:
            continue
        # add dir itself (needed for empty dirs
        zipHandle.write(os.path.join(root, "."))
        # add files
        for file in files:
            filePath = os.path.join(root, file)
            inZipPath = filePath.replace(basePath, "", 1).lstrip("\\/")
            #print filePath + " , " + inZipPath
            zipHandle.write(filePath, inZipPath)

def backup_source(output_dir):
    zip_file = os.path.join(output_dir, "{}-source.zip".format(
        datetime.now().strftime('%Y%m%d%H%M%S')
    ))
    zipf = zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED)
    addDirToZip(zipf, "qball", exclude=["__pycache__"])
    addDirToZip(zipf, "eval", exclude=["__pycache__"])
    addDirToZip(zipf, "demos", exclude=["__pycache__"])
    for f in glob.glob('requirements*.txt') + glob.glob('README.md'):
        zipf.write(f)
    zipf.close()

class Experiment(object):
    """ Base class for experiments. """

    def __init__(self, name, args):
        self.name = name

        parser = ArgumentParser(description="See README.md.")
        parser.add_argument('outdir', nargs='?', metavar='OUTPUT_DIR',
                            default=output_dir_name(self.name), type=str,
                            help="Path to output directory. "
                                   + "Existing data will be loaded and used.")
        parser.add_argument('--resume', action="store_true", default=False,
                            help="Continue at last state.")
        parser.add_argument('--batch', action="store_true", default=False,
                            help="Activate batch processing (no plot).")
        parsed_args = parser.parse_args(args)
        self.output_dir = parsed_args.outdir
        self.resume = parsed_args.resume
        self.interactive = not parsed_args.batch

        output_dir_create(self.output_dir)
        add_log_file(logging.getLogger(), self.output_dir)
        backup_source(self.output_dir)
        self.params = {}

    def setup_imagedata(self): pass

    def load_imagedata(self): pass

    def postprocessing(self): pass

    def solve(self): pass

    def run(self):
        # try to load as much data from files as possible
        self.load_imagedata()

        pd_result_file = os.path.join(self.output_dir, 'result_raw.npz')
        details_file = os.path.join(self.output_dir, 'details.pickle')
        params_file = os.path.join(self.output_dir, 'params.pickle')

        self.pd_result = data_from_file(pd_result_file)
        if self.pd_result is not None:
            tpl = ()
            for i in range(20):
                try:
                    tpl += (self.pd_result['arr_%d'%i],)
                except KeyError:
                    break
            self.pd_result = tpl

        params = data_from_file(params_file, format="pickle")
        if params is None:
            pickle.dump(self.params, open(params_file, 'wb'))
        else:
            self.params = params

        if self.pd_result is None or self.resume:
            self.solve()
            np.savez_compressed(open(pd_result_file, 'wb'), *self.pd_result)
            pickle.dump(self.details, open(details_file, 'wb'))
        self.upd = self.pd_result[0].copy()

        self.postprocessing()

    def plot(self): pass

class QBallExperiment(Experiment):
    def __init__(self, name, args):
        Experiment.__init__(self, name, args)
        self.params['fit'] = {}
        self.params['base'] = {
            'sh_order': 6,
            'smooth': 0,
            'min_signal': 0,
            'assume_normed': True
        }

    def load_imagedata(self):
        gtab_file = os.path.join(self.output_dir, 'gtab.pickle')
        S_data_file = os.path.join(self.output_dir, 'S_data.np')
        S_data_orig_file = os.path.join(self.output_dir, 'S_data_orig.np')

        self.gtab = data_from_file(gtab_file, format="pickle")
        self.S_data = data_from_file(S_data_file)
        self.S_data_orig = data_from_file(S_data_orig_file)
        if self.S_data is None:
            self.setup_imagedata()
            pickle.dump(self.gtab, open(gtab_file, 'wb'))
            np.save(open(S_data_file, 'wb'), self.S_data)
            np.save(open(S_data_orig_file, 'wb'), self.S_data_orig)
        self.imagedims = self.S_data.shape[:-1]
        b_vecs = self.gtab.bvecs[self.gtab.bvals > 0,...]
        self.qball_sphere = dipy.core.sphere.Sphere(xyz=b_vecs)

    def solve(self):
        if self.resume and self.pd_result is not None:
            self.continue_at = self.pd_result
            self.params['fit']['solver_params']['continue_at'] = self.continue_at

        self.model = self.Model(self.gtab, **self.params['base'])
        self.model.fit(self.S_data, **self.params['fit'])
        self.pd_result = self.model.solver_state
        self.details = self.model.solver_details

    def postprocessing(self):
        l_labels = self.upd.shape[0]
        imagedims = self.upd.shape[1:]
        self.upd = self.upd.reshape(l_labels, -1)
        self.upd = np.array(self.upd.T, order='C').reshape(imagedims + (l_labels,))

        basemodel = CsaOdfModel(self.gtab, **self.params['base'])
        f = basemodel.fit(self.S_data).odf(self.qball_sphere)
        self.fin = np.clip(f, 0, np.max(f, -1)[..., None])
        f = basemodel.fit(self.S_data_orig).odf(self.qball_sphere)
        self.fin_orig = np.clip(f, 0, np.max(f, -1)[..., None])

    def plot(self):
        n_image = np.prod(self.imagedims)
        d_image = len(self.imagedims)
        l_labels = self.upd.shape[-1]
        logging.info("Plotting results...")

        plot_scale = 1.0
        plot_norm = False

        # set up data to plot, including spacing in the 2d case
        stack = []
        if d_image == 2:
            uniform_odf = np.ones((l_labels,), order='C')/l_labels
            spacing = np.tile(uniform_odf, (self.imagedims[1], 1, 1, 1))
            for i, u in enumerate([self.upd, self.fin, self.fin_orig]):
                stack.append(u[:,:,None,:])
                if i < 2:
                    stack.append(spacing)
        else:
            stack = [u[:,None,None,:] for u in (self.upd, self.fin, self.fin_orig)]
        plotdata = np.concatenate(stack, axis=1)

        if self.interactive:
            # plot self.upd and self.fin as q-ball data sets
            r = fvtk.ren()
            fvtk.add(r, fvtk.sphere_funcs(plotdata, self.qball_sphere, colormap='jet',
                                          norm=plot_norm, scale=plot_scale))
            fvtk.show(r, size=(1024, 768))

        logging.info("Recording plot...")
        imgdata = [
            (self.upd, "upd"),
            (self.fin, "fin"),
            (self.fin_orig, "fin_orig")
        ]
        for img,name in imgdata:
            plotdata2 = img.copy()
            plotdata2.shape = self.imagedims + (1,)*(3-d_image) + (l_labels,)
            r = fvtk.ren()
            fvtk.add(r, fvtk.sphere_funcs(plotdata2, self.qball_sphere, colormap='jet',
                                          norm=plot_norm, scale=plot_scale))
            r.reset_clipping_range()
            fvtk.snapshot(r, size=(1500,1500), offscreen=True,
                          fname=os.path.join(self.output_dir, "plot-"+name+".png"))
