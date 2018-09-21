import warnings
warnings.filterwarnings('ignore')
from keras_tensorboard import plot_all_plots

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, auc, roc_curve
import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, TimeDistributed, Masking, Dropout
from keras.layers import Dense, Flatten, MaxPooling2D, Convolution2D
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.callbacks import Callback, LambdaCallback
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.utils import to_categorical

import shutil
import pickle
try:
    shutil.rmtree("/tmp/example")
except:
    pass

seed = 42
np.random.seed(seed)

class DataPlaceholder():
    def __init__(self):
        class DataItem():
            def __init__(self):
                self.x = None
                self.y = None
        self.train = DataItem()
        self.val = DataItem()
        self.test = DataItem()

class PredictData(Callback):
    def __init__(self,model,x,y,log_word):
        self.x = x; self.y = y; 
        self.model = model
        self.log_word = log_word

    def on_epoch_end(self,epoch,logs={}):
        logs[self.log_word+'predictions'] = self.model.predict(self.x)
        logs[self.log_word+'labels'] = self.y

from sklearn.metrics import accuracy_score, roc_auc_score, \
                            f1_score, recall_score, precision_score

def mse(y_true,y_pred):
    return np.mean( (y_true-y_pred)**2, axis=None )

def std(y_true,y_pred):
    return np.std( (y_true-y_pred), axis=None )

metrics_functions = {
    'mse': mse,
    'error_std': std
}

from sklearn.datasets import make_regression
data = DataPlaceholder()
X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
y = (y - y.mean())/y.std()
X_left, data.test.x, y_left, data.test.y = train_test_split(X,y,test_size=0.2)
data.train.x, data.val.x, data.train.y, data.val.y = train_test_split(X_left,y_left,test_size=0.3)

# model 1
model = Sequential()
model.add(Dense(2, activation='relu', input_dim=5))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')

callbacks = [
    PredictData(model, x=data.train.x, y=data.train.y, log_word=''),
    PredictData(model, x=data.val.x  , y=data.val.y  , log_word='val_'),
    PredictData(model, x=data.test.x , y=data.test.y , log_word='test_'),
]
h = model.fit( data.train.x, data.train.y, epochs=30, batch_size=4, verbose=0,
                validation_data=(data.val.x,data.val.y), callbacks=callbacks)

# model 2
model = Sequential()
model.add(Dense(2, activation='relu', input_dim=5))
model.add(Dense(1))
model.compile(loss='mse',optimizer='sgd')

callbacks = [
    PredictData(model, x=data.train.x, y=data.train.y, log_word=''),
    PredictData(model, x=data.val.x  , y=data.val.y  , log_word='val_'),
    PredictData(model, x=data.test.x , y=data.test.y , log_word='test_'),
]
h2 = model.fit( data.train.x, data.train.y, epochs=30, batch_size=4, verbose=0,
                validation_data=(data.val.x,data.val.y), callbacks=callbacks)

models = {
    'model1':h.history,
    'model2':h2.history
}

plot_all_plots('/tmp/example',models,metrics_functions)