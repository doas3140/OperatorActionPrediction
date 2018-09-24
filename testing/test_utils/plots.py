from .keras_tensorboard.utils import concatinate_kfolds, filter_predictions_and_metrics, history2logs, \
                                     fix_models_predictions_shapes, fix_models_metrics_shapes
from .metrics import recall, precision, conf_matrix, normalized_confusion_matrix_and_identity_mse
import matplotlib.pyplot as plt
import numpy as np
import itertools
import copy

def get_results_mean_and_std(logs,normalize,early_stopping_class):
    y_pred = logs['val']['pred'] # (kfolds,epochs,examples,classes)
    y_true = logs['val']['labels']
    num_kfolds,num_epochs,num_examples,num_classes = y_true.shape
    k_r, k_p, k_mse, k_cm = [], [], [], []
    for k in range(num_kfolds):
        ES = early_stopping_class()
        for e in range(num_epochs):
            y_pred_ = y_pred[k,e]
            y_true_ = y_true[k,e]
            r = recall(average='macro')(y_true_, y_pred_)
            p = precision(average='macro')(y_true_, y_pred_)
            mse = normalized_confusion_matrix_and_identity_mse()(y_true_, y_pred_)
            cm = conf_matrix(normalize=True)(y_true_, y_pred_)
            ES.append(recall=r,precision=p,conf_mse=mse)
            if ES.stop == True:
                break
        k_r.append( r )
        k_p.append( p )
        k_mse.append( mse )
        k_cm.append( cm )
    # np.set_printoptions(precision=2)
    return {
        'recall':{
            'mean':np.mean(k_r, axis=0),
            'std':np.std(k_r, axis=0)
        },
        'precision':{
            'mean':np.mean(k_p, axis=0),
            'std':np.std(k_p, axis=0)
        },
        'confusion_mse':{
            'mean':np.mean(k_mse, axis=0),
            'std':np.std(k_mse, axis=0)
        },
        'confusion_matrix':{
            'mean':np.mean(k_cm, axis=0),
            'std':np.std(k_cm, axis=0)
        }
    }

def plot_cms(mean,std,classes,title,show_scores=True,cmap=plt.cm.Blues):
    '''
    @param mean w/ shape (num_classes,num_classes)
    '''
    plt.imshow(mean, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if show_scores:
        fmt = '.2f'
        thresh = mean.max() / 2.
        for i, j in itertools.product(range(mean.shape[0]), range(mean.shape[1])):
            s = '{:{}}{}{:{}}'.format(mean[i,j], fmt, '±', std[i,j], fmt)
            plt.text(j, i, format(s),
                     horizontalalignment="center",
                     color="white" if mean[i, j] > thresh else "black")
#     plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusion_matrixes(models_hist,classes,early_stopping_class,kfolded=True):
    models_hist = concatinate_kfolds(models_hist)
    models_predictions,models_metrics = filter_predictions_and_metrics(models_hist)
    models_predictions = fix_models_predictions_shapes(models_predictions,kfolded=kfolded)
    fig = plt.figure(figsize=(15, 60), dpi= 80, facecolor='w', edgecolor='k')
    num_models = len(models_predictions)
    for i,(k,v) in enumerate(models_predictions.items()):
        logs = history2logs(v)
        results = get_results_mean_and_std(logs,normalize=True,early_stopping_class=early_stopping_class)
        fig.add_subplot(num_models,2,i*2 + 1)
        plot_cms(results['confusion_matrix']['mean'],results['confusion_matrix']['std'],classes,title=k)
    plt.show()

def plot_models_results(models_hist,classes,early_stopping_class,kfolded=True,plot_title=''):
    models_hist = concatinate_kfolds(models_hist)
    models_predictions,models_metrics = filter_predictions_and_metrics(models_hist)
    models_predictions = fix_models_predictions_shapes(models_predictions,kfolded=kfolded)
    fig = plt.figure(figsize=(15, 8), dpi= 80, facecolor='w', edgecolor='k')
    num_models = len(models_predictions)
    for m,(k,v) in enumerate(models_predictions.items()):
        logs = history2logs(v)
        results = get_results_mean_and_std(logs,normalize=True,early_stopping_class=early_stopping_class)
        results = [ results['confusion_mse'], results['recall'], results['precision'] ]
        num_metrics = len(results)
        x = np.linspace(1,num_models*(num_metrics+2),num_models)
        for i,color in enumerate(['red','blue','green','black','yellow','grey'][:num_metrics]):
            plt.errorbar(x[m]+i, results[i]['mean'], yerr=results[i]['std'], fmt='o', color=color, ecolor=color, elinewidth=3, capsize=5)
    plt.legend(['confusion_mse','recall','precision'])
    plt.title('{}'.format(plot_title))
    plt.xticks(x+1, list(models_hist.keys()))
    plt.show()