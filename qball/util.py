
import numpy as np
np.set_printoptions(precision=4, linewidth=200, suppress=True, threshold=10000)

import sys, os, errno, glob, zipfile, pickle, signal
from datetime import datetime

import logging
class MyFormatter(logging.Formatter):
    def format(self, record):
        th, rem = divmod(record.relativeCreated/1000.0, 3600)
        tm, ts = divmod(rem, 60)
        record.relStrCreated = "% 2d:%02d:%06.3f" % (int(th),int(tm),ts)
        return super(MyFormatter, self).format(record)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(MyFormatter('[%(relStrCreated)s] %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(ch)

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
    ch = logging.FileHandler(log_file)
    ch.setFormatter(MyFormatter('[%(relStrCreated)s] %(message)s'))
    ch.setLevel(logging.DEBUG)
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]
    logger.addHandler(ch)

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
    for f in glob.glob('requirements*.txt') + glob.glob('README.md') \
             + glob.glob('demo.py'):
        zipf.write(f)
    zipf.close()
