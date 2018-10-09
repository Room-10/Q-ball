
from qball.data import QBallData
from qball.data.oned import Data as OnedData
import qball.tools.gen as gen

class Data(OnedData):
    name = "1d-linear"
    def __init__(self, *args):
        QBallData.__init__(self, *args)
        S_data_orig, S_data, self.gtab = gen.synth_unimodals_linear()
        self.raw = S_data[:,None,None,:]
        self.ground_truth = S_data_orig[:,None,None,:]
        self.slice = (slice(None),0,0)
        self.normed = True
