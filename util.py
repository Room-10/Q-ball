
import sys, os, signal

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