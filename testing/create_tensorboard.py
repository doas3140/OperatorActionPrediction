'''
    Script for plotting metrics in tensorboard
    All predictions and labels from ./testing/test_results/
    folder are plotted w/ metrics from get_metrics_functions

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

from test_utils.metrics import get_metrics_functions
from test_utils.data import createdata, get_class_weights
from test_utils.classes import index2action, index2memory, index2splitting, index2xrootd
from test_utils.keras_tensorboard.keras_tensorboard import plot_all_plots

import os
import numpy as np
import pickle

def load_hist(path):
    ''' loads pickle file from path '''
    return pickle.load(open(path,'rb'))

def delete_dir(dirpath):
    ''' deletes directory and everything inside it '''
    import shutil
    try:
        shutil.rmtree(dirpath)
    except:
        pass

def folder2modelsdict(dirpath):
    ''' gets all pickle files from dirpath and returns dict {filename:filepath, ...} '''
    if not os.path.exists(dirpath): return {}
    return { f.name[:-7]:load_hist(f.path) for f in os.scandir(dirpath) if f.name[-7:]=='.pickle' }

seed = 42
np.random.seed(seed)
np.set_printoptions(precision=2)

jsonpath = './test_data/history.180618.json'
outpath_dir = './tensorboard/'
results_dir = os.path.join('./test_results/')

X,X_error_country_tier_positivity,Y_action,Y_memory,Y_splitting,Y_xrootd,Y_sites = createdata(jsonpath)
Y_all = [Y_action,Y_memory,Y_splitting,Y_xrootd]
print('{}: \t\t {}'.format('Y_all array length', len(Y_all)))

action_weights = get_class_weights(Y_action)
memory_weights = get_class_weights(Y_memory)
splitting_weights = get_class_weights(Y_splitting)
xrootd_weights = get_class_weights(Y_xrootd)

del X,X_error_country_tier_positivity,Y_action,Y_memory,Y_splitting,Y_xrootd,Y_sites

for name,(class_weights,index2name) in zip(['action','splitting','xrootd','memory'], 
                                    zip([action_weights,splitting_weights,xrootd_weights,memory_weights],
                                        [index2action,index2splitting,index2xrootd,index2memory])):
    print('\n\n Creating a tensorboard w/ name: {}'.format(name))
    print('weights: {}'.format(class_weights))
    outpath = os.path.join(outpath_dir, name)
    val_dirpath = os.path.join(results_dir, name)

    val_models = folder2modelsdict(val_dirpath)
    if val_models == {}: print('!!! Didint found any files in {} !!!'.format(name)); continue

    metrics_functions = get_metrics_functions(class_weights)
    delete_dir(outpath)
    plot_all_plots( outpath, val_models, metrics_functions, 
                    index2class_name = index2name,
                    kfolded = True, pr_curve = True, notebook = False )