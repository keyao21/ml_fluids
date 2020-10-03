import util
from MESN import MultiEchoStateNetwork
from ESN import EchoStateNetwork
import pdb


train_input_data_fullpath = '../inputs/preprocess/dgsf_0.1_160_80_0.1_0.2_id1.TRAIN'
train_flattened_input_data = util.flatten_time_series( util.load_data( train_input_data_fullpath ) ).transpose()
    

pdb;pdb.set_trace()

