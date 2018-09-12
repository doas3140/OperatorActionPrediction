'''
    *  Contains functions that invoked with or without parameters returns
    a function with inputs (y_true, y_pred), 
    y_true, y_pred shapes are (num_examples, num_classes)
    *  and a function get_metrics_functions, which returns dict of all metrics,
    where key = string name, value = metric function
'''

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, \
                            recall_score, precision_score, confusion_matrix, \
                            mean_squared_error
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import Callback
import keras.backend as K


def accuracy():
    ''' returns acc for each class '''
    def metric(y_true,y_pred):
        num_classes = y_true.shape[-1]
        y_argmax = np.argmax(y_pred,axis=1)
        y_pred = to_categorical(y_argmax,num_classes=num_classes)
        accs = np.zeros(num_classes)
        for i in range(num_classes):
            accs[i] = accuracy_score(y_true[:,i],y_pred[:,i])
        return accs
    return metric


def weighted_accuracy(weights):
    ''' returns weighted accuracy function
        (same number for each class)
    !!! returns the same number for each class,       !!!
    !!! because there are no weighted acc for 1 class !!!
    '''
    def metric(y_true,y_pred):
        W = weights[ np.argmax(y_true,axis=1) ]
        return np.mean( W * (np.argmax(y_true,axis=1) == np.argmax(y_pred,axis=1)) )
    return metric


def roc_auc():
    ''' returns roc auc for each class '''
    def metric(y_true,y_pred):
        return roc_auc_score(y_true,y_pred,average=None)
    return metric


def f1():
    ''' returns f1 for each class '''
    def metric(y_true,y_pred):
        num_classes = y_true.shape[-1]
        y_argmax = np.argmax(y_pred,axis=1)
        y_pred = to_categorical(y_argmax,num_classes=num_classes)
        return f1_score(y_true,y_pred,average=None)
    return metric


def recall():
    ''' returns recall for each class '''
    def metric(y_true,y_pred):
        num_classes = y_true.shape[-1]
        y_argmax = np.argmax(y_pred,axis=1)
        y_pred = to_categorical(y_argmax,num_classes=num_classes)
        return recall_score(y_true,y_pred,average=None)
    return metric


def precision():
    def metric(y_true,y_pred):
        ''' returns precision for each class '''
        num_classes = y_true.shape[-1]
        y_argmax = np.argmax(y_pred,axis=1)
        y_pred = to_categorical(y_argmax,num_classes=num_classes)
        return precision_score(y_true,y_pred,average=None)
    return metric


def normalized_confusion_matrix_and_identity_mse():
    ''' returns mse of normalized confusion matrix and the identity matrix
        (same number for each class) '''
    def metric(y_true,y_pred):
        num_classes = y_true.shape[-1]
        y_pred_argmax = np.argmax(y_pred,axis=1)
        y_true_argmax = np.argmax(y_true,axis=1)
        cm = confusion_matrix(y_true_argmax, y_pred_argmax, labels=np.arange(num_classes)) # returns shape=(num_classes,num_classes)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize confusion matrix
        cm = cm.flatten()
        cm[ np.isnan(cm) ] = 0 # replace nans w/ 0
        identity = np.identity(num_classes).flatten()
        mse = mean_squared_error(cm,identity)
        return mse
    return metric


def cross_entropy(e=1e-07, axis=-1):
    ''' returns cross entropy of all classes
        (same number for each class) '''
    def metric(y_true, y_pred):
        y_pred = np.clip(y_pred, e, 1. - e)
        loss = y_true * np.log(y_pred)
        return - np.mean( np.sum(loss, axis=axis) )
    return metric


def weigthed_cross_entropy(weights, e=1e-07, axis=-1):
    ''' returns weighted cross entropy of all classes 
        (same number for each class) '''
    def metric(y_true, y_pred):
        y_pred = np.clip(y_pred, e, 1. - e)
        loss = y_true * np.log(y_pred) * weights
        return - np.mean( np.sum(loss, axis=axis) )
    return metric


def get_metrics_functions(weights):
    return {
        'accuracy': accuracy(),
        'weighted_acc': weighted_accuracy(weights),
        # 'roc_auc': roc_auc(),
        # 'f1': f1(),
        'recall':recall(),
        'precision':precision(),
        'confusion_mse': normalized_confusion_matrix_and_identity_mse(),
        'CE_loss':cross_entropy(),
        'wCE_loss':weigthed_cross_entropy(weights)
    }