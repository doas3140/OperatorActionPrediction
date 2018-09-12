'''
    Script for plotting metrics in tensorboard
    variables that can be changed:
        * jsonpath - path where json is
        * outpath - output of tensorboard directory
        * class_weights - selected class weights
        * index2name - selected class index2name dict
        * val_dirpath - validation histories directory path
        * test_dirpath - test histories directory path
        ! both histories are computed in model_utils !
'''

import warnings
warnings.filterwarnings('ignore')

from model_utils import load_hist, delete_dir
from metrics import get_metrics_functions
from data import createdata, get_class_weights
from classes import index2action, index2memory, index2splitting, index2xrootd
from keras_tensorboard.keras_tensorboard import plot_all_plots

import os
import numpy as np

seed = 42
np.random.seed(seed)
np.set_printoptions(precision=2)

jsonpath = '../data/history.180618.json'
outpath = '/tmp/tensorboard'

test_dirpath = os.path.join(os.getcwd(),'OP_models/tensorboard_test/')
val_dirpath = os.path.join(os.getcwd(),'OP_models/tensorboard_validation/')

X,X_error_country_tier_positivity,Y_action,Y_memory,Y_splitting,Y_xrootd,Y_sites = createdata(jsonpath)
Y_all = [Y_action,Y_memory,Y_splitting,Y_xrootd]
print('{}: \t\t {}'.format('Y_all array length', len(Y_all)))

action_weights = get_class_weights(Y_action)
memory_weights = get_class_weights(Y_memory)
splitting_weights = get_class_weights(Y_splitting)
xrootd_weights = get_class_weights(Y_xrootd)

class_weights = xrootd_weights
index2name = index2xrootd

print('weights: {}'.format(class_weights))

def folder2modelsdict(dirpath):
    return { f.name[:-7]:load_hist(f.path) for f in os.scandir(dirpath) if f.name[-7:]=='.pickle' }

test_models = folder2modelsdict(test_dirpath)
val_models = folder2modelsdict(val_dirpath)

metrics_functions = get_metrics_functions(class_weights)
delete_dir(outpath)
plot_all_plots( outpath, val_models, metrics_functions, 
                index2class_name=index2name, # test_hists=test_models, 
                kfolded = True, pr_curve = True, notebook = False )