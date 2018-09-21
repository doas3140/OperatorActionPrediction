'''
    Has two main functions:
     * function createdata(jsonpath), that returns variables:
        X - (9522, 54, 142), where shape=(num_examples, num_errors, num_sites)
        X_error_country_tier_positivity - (9522, 54, 31, 5, 2), where shape=(num_examples, num_errors, num_country, num_tier, num_positivity)
        Y_action - (9522, 3), where shape=(num_examples, num_action)
        Y_memory - (9522, 6), where shape=(num_examples, num_memory)
        Y_splitting - (9522, 3), where shape=(num_examples, num_splitting)
        Y_xrootd - (9522, 3), where shape=(num_examples, num_xrootd)
        Y_sites - (9522, 50, 62), where shape=(num_examples, maximum_length_of_param_site_lists, num_param_site)
     * function get_class_weights(Y), that returns (num_classes,) vector where each number represents a weight for that class
'''

from .classes import index2action,index2memory,index2xrootd,index2splitting,index2param_site
from .classes import index2site,index2error,index2tier,index2country,index2positivity
from .classes import action2index,memory2index,xrootd2index,splitting2index,param_site2index
from .classes import site2index,error2index,tier2index,country2index,positivity2index

from sklearn.model_selection import train_test_split

import numpy as np
import json


def get_class_weights(Y): # Y is one-hot encoded (num_examples,num_classes)
    num_classes = Y.shape[-1]
    class_counts = np.count_nonzero(Y,axis=0)
    total_count = np.sum(class_counts)
    # return (1 / (class_counts*num_classes))*total_count
    return (total_count/num_classes) * (1/class_counts)


def createdata(jsonpath):
    global X, X_error_country_tier_positivity
    global Y_memory,Y_action,Y_splitting,Y_xrootd,Y_sites
    # jsonpath = 'WW/data/history.180618.json'
    j = json.loads( open(jsonpath).read() )
    total_items = len(j)
    X = np.zeros((total_items,len(index2error),len(index2site)))
    X_error_country_tier_positivity = np.zeros((total_items,len(index2error),len(index2country),len(index2tier),len(index2positivity)))
    print('X:\t\t\t\t',X.shape)
    print('X_error*country*tier*positivity:',X_error_country_tier_positivity.shape)
    Y_memory = np.zeros((total_items,len(index2memory)))
    Y_action = np.zeros((total_items,len(index2action)))
    Y_splitting = np.zeros((total_items,len(index2splitting)))
    Y_xrootd = np.zeros((total_items,len(index2xrootd)))
    param_site_max_length = get_max_param_site_length(j)
    Y_sites = np.zeros((total_items,param_site_max_length,len(index2param_site)))
    print('Y_memory:\t\t\t',Y_memory.shape)
    print('Y_action:\t\t\t',Y_action.shape)
    print('Y_splitting:\t\t\t',Y_splitting.shape)
    print('Y_xrootd:\t\t\t',Y_xrootd.shape)
    print('Y_sites:\t\t\t',Y_sites.shape)
    fill_data(j)
    return X,X_error_country_tier_positivity,Y_action,Y_memory,Y_splitting,Y_xrootd,Y_sites
    

def get_max_param_site_length(j):
    biggest_len = 0
    for wf in j.keys():
        sites = get_sites(j[wf])
        if biggest_len < len(sites):
            biggest_len = len(sites)
    return biggest_len


def fill_data(j):
    global X, X_error_country_tier_positivity
    global Y_memory,Y_action,Y_splitting,Y_xrootd,Y_sites
    for i,wf in enumerate(j.keys()):
        for positivity in j[wf]['errors'].keys():
            for error in j[wf]['errors'][positivity].keys():
                for site,nr_count in j[wf]['errors'][positivity][error].items():
                    tier = site[0:2]
                    country = site[3:5]
                    X[i,error2index[error],site2index[site]] = nr_count
                    X_error_country_tier_positivity \
                        [ i, error2index[error], country2index[country], \
                        tier2index[tier], positivity2index[positivity]
                        ] = nr_count 
        action,memory,xrootd,splitting,sites = get_params(j[wf])
        Y_action[i,action2index[action]] = 1
        Y_memory[i,memory2index[memory]] = 1
        Y_xrootd[i,xrootd2index[xrootd]] = 1
        Y_splitting[i,splitting2index[splitting]] = 1
        for k,s in enumerate(sorted(sites)):
            Y_sites[i,k,param_site2index[s]] = 1


def get_params(wf_dict):
    action = get_action(wf_dict)
    sites = get_sites(wf_dict)
    memory = get_memory(wf_dict)
    xrootd = get_xrootd(wf_dict)
    splitting = get_splitting(wf_dict)
    return action,memory,xrootd,splitting,sites


def get_action(wf_dict):
    try:
        return wf_dict['parameters']['action']
    except KeyError:
        return 'key_error'
def get_sites(wf_dict):
    try:
        return wf_dict['parameters']['sites']
    except KeyError:
        return 'key_error'
def get_memory(wf_dict):
    try:
        return wf_dict['parameters']['memory']
    except KeyError:
        return 'key_error'
def get_xrootd(wf_dict):
    try:
        return wf_dict['parameters']['xrootd']
    except KeyError:
        return 'key_error'
def get_splitting(wf_dict):
    try:
        return wf_dict['parameters']['splitting']
    except KeyError:
        return 'key_error'


def splitdata(data,indexes):
    return data[indexes]


