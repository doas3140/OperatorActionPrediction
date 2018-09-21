'''
    Creates predictions and labels across kfolds and stores
    them into ./testing/test_results/{Y_name}/ folder

    variables to change:
    * jsonpath - data path (in json format)
'''

from EmbeddingModel import EmbeddingModel
from SimpleModel import SimpleModel
from CNNModel import CNNModel
from XGBoostModel import XGBoostModel

import os
import pickle
import numpy as np

def save_hist(path, hist):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    pickle.dump(hist, open(path,'wb'))
    print('Done!', path)

from testing.test_utils.data import createdata, get_class_weights
from testing.test_utils.classes import index2action, index2memory, index2splitting, index2xrootd
import numpy as np
from sklearn.model_selection import train_test_split

def create_all_models_results_for_tensorboard():
    ''' creates results and stores them into ./testing/test_results/ directory
    '''
    jsonpath = './testing/test_data/history.180618.json'
    X,X_error_country_tier_positivity,Y_action,Y_memory,Y_splitting,Y_xrootd,Y_sites = createdata(jsonpath)
    Y_all = [Y_action,Y_memory,Y_splitting,Y_xrootd]
    print('{}: \t\t {}'.format('Y_all array length', len(Y_all)))

    for Y,Y_name in zip([Y_action,Y_xrootd,Y_splitting,Y_memory],['action','xrootd','splitting','memory']):
        num_classes = Y.shape[-1]
        Y = np.argmax(Y,axis=-1)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        for Model,M_name in zip([SimpleModel,CNNModel,EmbeddingModel],['Simple','CNN','Embed']):
            for use_smote in [False,True]:
                for use_attention in [False,True]:

                    if use_attention == False and use_smote == False:
                        name = 'Simple'
                    if use_attention == True and use_smote == False:
                        if Model == XGBoostModel: continue
                        name = 'Att'
                    if use_attention == False and use_smote == True:
                        name = 'SMOTE'
                    if use_attention == True and use_smote == True:
                        if Model == XGBoostModel: continue
                        name = 'SMOTE_Att'

                    print('./testing/test_results/{}/{}_{}.pickle'.format(Y_name,M_name,name))
                    if os.path.exists('./testing/test_results/{}/{}_{}.pickle'.format(Y_name,M_name,name)):
                        print('file already exists... skipping...'); continue

                    if Model == EmbeddingModel:
                        m = Model(X,num_classes,add_attention=use_attention)
                    elif Model == XGBoostModel:
                        m = Model(num_classes)
                    else:
                        m = Model(num_classes,add_attention=use_attention)
                    hist = m.train( X_train, y_train, verbose=1, use_imblearn=use_smote, max_epochs=200, testing=True )

                    save_hist('./testing/test_results/{}/{}_{}.pickle'.format(Y_name,M_name,name), hist)

import keras

from utils.metrics import normalized_confusion_matrix_and_identity_mse, \
                          recall, precision, conf_matrix
from testing.test_utils.data import createdata, get_class_weights
from testing.test_utils.classes import index2action, index2memory, index2splitting, index2xrootd
import numpy as np
from sklearn.model_selection import train_test_split

def create_test_results():
    ''' splits data into train, test, trains on train data with early stopping
        and saves all results into ./testing/test_results/results.csv
    '''
    with open('./testing/test_results/results.csv','w') as f:
        f.write('model_name,num_trained_epochs,min_score,argmin_score,macro_recall,macro_precision,confusion_mse\n')
    
    def evaluate(y_test,y_pred,num_classes):
        y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)
        y_pred = keras.utils.to_categorical(y_pred, num_classes=num_classes)
        r = recall(average='macro')(y_test,y_pred)
        p = precision(average='macro')(y_test,y_pred)
        n = normalized_confusion_matrix_and_identity_mse()(y_test,y_pred)
        cm = conf_matrix()(y_test,y_pred)
        return r,p,n,cm

    jsonpath = './testing/test_data/history.180618.json'
    X,X_error_country_tier_positivity,Y_action,Y_memory,Y_splitting,Y_xrootd,Y_sites = createdata(jsonpath)
    Y_all = [Y_action,Y_memory,Y_splitting,Y_xrootd]
    print('{}: \t\t {}'.format('Y_all array length', len(Y_all)))

    for Y,Y_name in zip([Y_action,Y_splitting,Y_xrootd,Y_memory],['action','splitting','xrootd','memory']):
        num_classes = Y.shape[-1]
        Y = np.argmax(Y,axis=-1)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        for Model,M_name in zip([SimpleModel,CNNModel,EmbeddingModel,XGBoostModel],['Simple','CNN','Embed','XGBoost']):
            for use_smote in [False,True]:
                for use_attention in [False,True]:

                    if use_attention == False and use_smote == False:
                        name = 'Simple'
                    if use_attention == True and use_smote == False:
                        if Model == XGBoostModel: continue
                        name = 'Att'
                    if use_attention == False and use_smote == True:
                        name = 'SMOTE'
                    if use_attention == True and use_smote == True:
                        if Model == XGBoostModel: continue
                        name = 'SMOTE_Att'
                    
                    print('\n{}/{}_{}.pickle\n'.format(Y_name, M_name, name))
                    
                    if Model == EmbeddingModel:
                        m = Model(X,num_classes,add_attention=use_attention)
                    elif Model == XGBoostModel:
                        m = Model(num_classes)
                    else:
                        m = Model(num_classes,add_attention=use_attention)

                    hist = m.train( X_train, y_train, verbose=1, use_imblearn=use_smote )

                    if use_attention:
                        y_pred = m.predict(X_test)[0]
                    else:
                        y_pred = m.predict(X_test)
                    
                    e = evaluate(y_test,y_pred,num_classes)
                    
                    num_epochs = len(hist['loss'])

                    min_score = np.min(hist['main_score'])
                    argmin = np.argmin(hist['main_score'])

                    with open('./testing/test_results/results.csv','a') as f:
                        f.write('{}/{}_{}.pickle,{},{},{},{:.3},{:.3},{:.3},{}\n'.format(
                            Y_name, M_name, name, str(num_epochs), min_score, argmin, 
                            e[0], e[1], e[2], str( e[3].tolist() )))
                        # to get back from string use eval(str(a.tolist()))

if __name__ == '__main__':
    create_all_models_results_for_tensorboard()
    # create_test_results()