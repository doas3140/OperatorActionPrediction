'''
    XGBoost model class

     - Main variables:
        self.model - keras model
        self.num_classes - number of outputs/classes
        
     - Main functions:
        self.train - trains self.model w/ selected parameters
        self.predict - predicts Y from X with self.model
        self.change_inputs - function that changes inputs before feeding to model
        self.load_model - loads model.h5 file from directory (dirpath)
        self.save_model - saves model.h5 and model.json to directory (dirpath)
        self.evaluation_function - returns function to evaluate when model improves. used for early stopping
'''

import os
import json
import keras
import pickle
import sklearn
import imblearn
import numpy as np
from tqdm import tqdm
import xgboost as xgb

from utils.model_utils import DataPlaceholder, get_class_weights, \
                              imblearn_sample
from utils.model_utils import MultipleMetricsEarlyStopping as EarlyStopping
from utils.metrics import normalized_confusion_matrix_and_identity_mse as confusion_mse
from utils.metrics import accuracy, weighted_accuracy, recall, precision

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, StratifiedKFold


class XGBoostModel():
    def __init__(self, num_classes):
        '''
        @param num_classes - number of outputs/classes
        '''
        self.model = None
        self.num_classes = num_classes


    def change_inputs(self, X, Y=None):
        ''' change inputs before feeding to model
        @param X w/ shape (num_examples, num_error, num_sites) - where each number is integer of number of errors happened at that site
        @param Y w/ shape (num_examples,) - where each number is integer that represents one class
        return DMatrix of X,Y dataset or X dataset
        '''
        X = X.reshape((len(X),-1))
        if Y is not None:
            return xgb.DMatrix(X, label=Y)
        else:
            return xgb.DMatrix(X)


    def train( self, X, Y, max_depth=4, max_epochs=100, seed=42, verbose=0, # training vars
               use_imblearn=False, imblearn_class=SMOTE(random_state=42, ratio=1.0), # imblearn vars
               early_stopping_patience=10, test_split=0.2, # early stopping vars
               testing=False, kfold_function=KFold, kfold_splits=5 ): # testing vars (returns predictions from kfold)
        '''
        @param X w/ shape (num_examples, num_error, num_sites) - where each number is integer of number of errors happened at that site
        @param Y w/ shape (num_examples,) - where each number is integer that represents one class
        @param max_depth - maximum tree depth
        @param max_epochs - maximum learning epochs, if not using kfold to approximate num_epochs: then this is num_epochs used for training
        @param seed - random seed that is used everywhere for reproducability
        @param verbose - if 0: no print output, else: print output
        @param use_imblearn - boolean that decides to use resampling for training or not (from imblearn library)
        @param imblearn_class - class from iblearn library used for resampling (doesn't do anything if use_imblearn = False)
        @param early_stopping_patience - when training doesn't improve for n epochs, then stop. where n is this variable number
        @param test_split - test split used for early stopping
        @param testing - if True: stops training after cross validation and returns predictions of all data across all kfolds
        @param kfold_function - cross validation function from sklearn library (doesn't do anything if testing == False)
        @param kfold_splits - number of cross validation splits, used in kfold_function (doesn't do anything if testing == False) 
        '''

        if verbose != 0: verb_eval = True
        else: verb_eval = False

        train_param = {
            'max_depth': max_depth,  # the maximum depth of each tree
            'eta': 0.3,  # the training step for each iteration
            'silent': 1,  # logging mode - quiet
            'objective': 'multi:softprob',  # error evaluation for multiclass training
            'num_class': self.num_classes # the number of classes that exist in this datset
        }

        if testing:
            enum = enumerate(kfold_function(n_splits=kfold_splits, shuffle=True, random_state=seed).split(X,Y))
            if verbose != 0:
                enum = tqdm(enum, total=kfold_splits, desc='kfold', leave=False, initial=0)
            histories = []
            for i,(index_train, index_valid) in enum:
                X_train, X_val = X[ index_train ], X[ index_valid ]
                y_train, y_val = Y[ index_train ], Y[ index_valid ]
                if use_imblearn:
                    X_train, y_train = imblearn_sample( X_train, y_train, imblearn_class, verbose=verbose )
                dmatrix_train = self.change_inputs(X_train, y_train)
                dmatrix_val = self.change_inputs(X_val, y_val)
                watchlist = [(dmatrix_train, 'train'), (dmatrix_val, 'val')]
                results = {}
                model = xgb.train( train_param, dmatrix_train, max_epochs, 
                                   watchlist, feval = self.evaluation_function(), 
                                   early_stopping_rounds = early_stopping_patience, 
                                   evals_result = results, verbose_eval = verb_eval )
                train_pred = model.predict( dmatrix_train )
                train_labels = keras.utils.to_categorical( y_train )
                val_pred = model.predict( dmatrix_val )
                val_labels = keras.utils.to_categorical( y_val )
                histories.append({ 'pred':train_pred, 'labels':train_labels, 
                                   'val_pred':val_pred, 'val_labels':val_labels })
            return histories
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_split, random_state=seed)
            if use_imblearn:
                X_train, y_train = imblearn_sample( X_train, y_train, imblearn_class, verbose=verbose )
            dmatrix_train = self.change_inputs(X_train, y_train)
            dmatrix_val = self.change_inputs(X_val, y_val)
            watchlist = [(dmatrix_train, 'train'), (dmatrix_val, 'val')]
            results = {}
            self.model = xgb.train( train_param, dmatrix_train, max_epochs, 
                         watchlist, feval = self.evaluation_function(), 
                         early_stopping_rounds = early_stopping_patience, 
                         evals_result = results, verbose_eval = verb_eval )
            return { 'loss': results['train']['merror'], 'val_loss': results['val']['merror'],
                     'main_score': results['val']['confusion_mse'] }

    

    def evaluation_function(self):
        ''' returns function to evaluate when model improves. used for early stopping
        return function(y_pred, y_true)
        '''
        num_classes = self.num_classes
        def evaluation(y_pred,y_true):
            '''
            @param y_pred w/ shape (num_examples, num_classes) - probabilities of each class
            @param y_true w/ shape (num_examples,) - each number represents class index
            '''
            y_true = keras.utils.to_categorical(y_true.get_label())
            y_pred = np.argmax(y_pred,axis=1)
            y_true = np.argmax(y_true,axis=1)
            def confusion_mse(y_true,y_pred):
                cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes)) # returns shape=(num_classes,num_classes)
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize confusion matrix
                cm = cm.flatten()
                cm[ np.isnan(cm) ] = 0 # replace nans w/ 0
                identity = np.identity(num_classes).flatten()
                return mean_squared_error(cm,identity)
            mse = confusion_mse(y_true,y_pred)
            return [('confusion_mse',mse)]
        return evaluation
    

    def predict(self, X, argmax=True):
        ''' predicts Y from X with self.model
        @param X w/ shape (num_examples, num_error, num_sites)
        return y_argmax w/ shape (num_examples,) - each number represents class index
        '''
        X = self.change_inputs(X)
        y_pred = self.model.predict(X) # (num_examples, num_outputs)
        if argmax:
            y_pred = np.argmax(y_pred, axis=-1) # (num_examples,)
        return y_pred

    
    def load_model(self, dirpath):
        ''' loads xgboost.model.pickle file from directory (dirpath)
        @param dirpath - path of directory, from where to load
        '''
        modelpath = os.path.join(dirpath, 'xgboost.model.pickle')
        self.model = pickle.load(open(modelpath, "rb"))


    def save_model(self, dirpath):
        ''' saves xgboost.model.pickle to directory (dirpath)
        @param dirpath - path of directory, where to save
        '''
        modelpath = os.path.join(dirpath, 'xgboost.model.pickle')
        pickle.dump(self.model, open(modelpath, "wb"))




#