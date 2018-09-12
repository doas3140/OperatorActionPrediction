'''
    Main functions:
        - fit_kfold_model:
            1) splits X,Y into kfolds
            2) for each fold trains model, which returns history*
            3) return list (num_kfolds,), where each element is history*
        - fit_test_model:
            1) splits X,Y into train/test
            2) trains on train set, meanwhile evaluating on test set
            3) return keras history of that training
        - evaluate_model: (does both)
            1) splits X,Y into train/test
            2) does fit_kfold on train X,Y
            3) does fit_test_model on test X,Y
            4) returns kfold_history and test_history

    * history in here means = dict({
        'predictions': train set predictions on labels
        'labels': train set labels
        'val_predictions': validation set predictions on labels
        'val_labels': validation set labels
        'test_predictions': test set predictions on labels
        'test_labels': train set labels
        ! all shapes are (num_epochs, num_examples, num_classes) (labels are one-hot encoded) !
        ! there are no val_pred, val_labels in fit_kfold_model !
        ! there are no test_pred, test_labels in fit_test_model !
    })

    Later on these dicts are saved, and metrics are calculated on-the-fly,
    so there is no need to train the same models again
'''

import time
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc, roc_curve
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, TimeDistributed, Masking, Dropout
from keras.layers import Dense, Flatten, MaxPooling2D, Convolution2D
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import pickle
from keras_tqdm import TQDMCallback, TQDMNotebookCallback

class DataPlaceholder():
    def __init__(self):
        class DataItem():
            def __init__(self):
                self.x = None
                self.y = None
        self.train = DataItem()
        self.val = DataItem()
        self.test = DataItem()


def unconcatinate(arr,prev_arr_lengths):
    total_l = 0
    output = []
    for i,l in enumerate(prev_arr_lengths):
        output.append( arr[i][ : , total_l:total_l+l ] )
    return output


def fit_kfold_model(create_model_fun, X, Y, test_split, kfold_function, kfold_splits, epochs, batch_size, \
                    random_seed, multi_outputs=False, notebook=False, upsampling=False):
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        tqdm_callback = TQDMNotebookCallback()
    else:
        from tqdm import tqdm
        tqdm_callback = TQDMCallback()
    if multi_outputs:
        Y_lengths = [y.shape[1] for y in Y]
        Y = np.concatenate(Y,axis=1)
    data = DataPlaceholder()
    X, data.test.x, Y, data.test.y = train_test_split(X, Y, test_size=test_split, shuffle=True, random_state=random_seed)
    histories = []
    enum = enumerate(kfold_function(n_splits=kfold_splits,shuffle=True,random_state=random_seed).split(X))
    for i,(index_train, index_valid) in tqdm(enum,total=kfold_splits,desc='kfold',leave=False,initial=0):
        data.train.x, data.val.x = X[ index_train ], X[ index_valid ]
        data.train.y, data.val.y = Y[ index_train ], Y[ index_valid ]
        model = create_model_fun()
        if multi_outputs:
            data.train.y = unconcatinate(data.train.y,Y_lengths)
            data.val.y = unconcatinate(data.val.y,Y_lengths)
        if upsampling:
            data.train.x, data.train.y = upsample(data.train.x,data.train.y,random_seed=random_seed,verbose=1)
        history = model.fit(
                x = data.train.x, y = data.train.y,
                validation_data = (data.val.x,data.val.y),
                epochs = epochs,
                batch_size = batch_size,
                callbacks = get_cross_validation_callbacks(model,data) + [tqdm_callback],
                verbose = 0
                )
        histories.append(history.history)
    return histories


def fit_test_model(create_model_fun, X, Y, test_split, epochs, batch_size, \
                    random_seed, multi_outputs=False, notebook=False, upsampling=False):
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        tqdm_callback = TQDMNotebookCallback()
    else:
        from tqdm import tqdm
        tqdm_callback = TQDMCallback()
    if multi_outputs:
        Y_lengths = [y.shape[1] for y in Y]
        Y = np.concatenate(Y,axis=1)
    data = DataPlaceholder()
    data.train.x, data.test.x, data.train.y, data.test.y = train_test_split(X, Y, test_size=test_split, shuffle=True, random_state=random_seed)
    model = create_model_fun()
    if multi_outputs:
        data.train.y = unconcatinate(data.train.y,Y_lengths)
        data.test.y = unconcatinate(data.test.y,Y_lengths)
    if upsampling:
        data.train.x, data.train.y = upsample(data.train.x,data.train.y,random_seed=random_seed,verbose=1)
    history = model.fit(
                x = data.train.x, y = data.train.y,
                epochs = epochs,
                batch_size = batch_size,
                callbacks = get_test_callbacks(model,data) + [tqdm_callback],
                verbose = 0
                )
    return history.history


def evaluate_model(create_model_fun, X, Y, test_split, kfold_function, kfold_splits, epochs, batch_size, \
                    random_seed, multi_outputs=False, notebook=False, upsampling=False):
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        tqdm_callback = TQDMNotebookCallback()
    else:
        from tqdm import tqdm
        tqdm_callback = TQDMCallback()
    if multi_outputs:
        Y_lengths = [y.shape[1] for y in Y]
        Y = np.concatenate(Y,axis=1)
    data = DataPlaceholder()
    X, data.test.x, Y, data.test.y = train_test_split(X, Y, test_size=test_split, shuffle=True, random_state=random_seed)
    # cross validation part
    print('Cross Validation on {} Splits'.format(kfold_splits))
    val_histories = []
    enum = enumerate(kfold_function(n_splits=kfold_splits,shuffle=True,random_state=random_seed).split(X))
    for i,(index_train, index_valid) in tqdm(enum,total=kfold_splits,desc='kfold',leave=False,initial=0):
        data.train.x, data.val.x = X[ index_train ], X[ index_valid ]
        data.train.y, data.val.y = Y[ index_train ], Y[ index_valid ]
        model = create_model_fun()
        if multi_outputs:
            data.train.y = unconcatinate(data.train.y,Y_lengths)
            data.val.y = unconcatinate(data.val.y,Y_lengths)
        if upsampling:
            data.train.x, data.train.y = upsample(data.train.x,data.train.y,random_seed=random_seed,verbose=1)
        history = model.fit(
                x = data.train.x, y = data.train.y,
                validation_data = (data.val.x,data.val.y),
                epochs = epochs,
                batch_size = batch_size,
                callbacks = get_cross_validation_callbacks(model,data) + [tqdm_callback],
                verbose = 0
                )
        val_histories.append(history.history)
    # testing part
    print('Testing')
    data.train.x = X
    data.train.y = Y
    if multi_outputs:
        data.train.y = unconcatinate(data.train.y,Y_lengths)
        data.test.y = unconcatinate(data.test.y,Y_lengths)
    if upsampling:
        data.train.x, data.train.y = upsample(data.train.x,data.train.y,random_seed=random_seed,verbose=1)
    history = model.fit(
                x = data.train.x, y = data.train.y,
                epochs = epochs,
                batch_size = batch_size,
                callbacks = get_test_callbacks(model,data) + [tqdm_callback],
                verbose = 0
                )
    test_history = history.history
    return val_histories, test_history


def upsample(X,Y,random_seed=42,verbose=0):
    from imblearn.over_sampling import SMOTE
    from keras.utils import to_categorical
    if verbose != 0: print('Before Upsampling: x:{}, y:{}'.format(X.shape, Y.shape))
    sm = SMOTE(random_state=random_seed, ratio = 1.0)
    X_temp = X.reshape(len(X),54*142)
    Y_temp = np.argmax(Y,axis=1)
    X_res, Y_res = sm.fit_sample(X_temp,Y_temp)
    X = X_res.reshape([len(X_res),54,142])
    Y = to_categorical(Y_res)
    del X_temp, Y_temp, X_res, Y_res
    if verbose != 0: print('After Upsampling: x:{}, y:{}'.format(X.shape, Y.shape))
    return X, Y

def save_hist(path,hist):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    pickle.dump(hist,open(path,'wb'))

def load_hist(path):
    return pickle.load(open(path,'rb'))

def delete_dir(dirpath):
    import shutil
    try:
        shutil.rmtree(dirpath)
    except:
        pass


class PredictData(Callback):
    def __init__(self,model,x,y,log_word):
        self.x = x; self.y = y; 
        self.model = model
        self.log_word = log_word

    def on_epoch_end(self,epoch,logs={}):
        logs[self.log_word+'predictions'] = self.model.predict(self.x)
        logs[self.log_word+'labels'] = self.y


def get_cross_validation_callbacks(model,data):
    return [
        PredictData(model, data.train.x, data.train.y, ''),
        PredictData(model, data.val.x  , data.val.y  , 'val_')
    ]

def get_test_callbacks(model,data):
    return [
        PredictData(model, data.train.x, data.train.y, ''),
        PredictData(model, data.test.x , data.test.y , 'test_')
    ]