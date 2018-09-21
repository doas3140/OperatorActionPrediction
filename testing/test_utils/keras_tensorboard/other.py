from .utils import concatinate_kfolds, filter_predictions_and_metrics, history2logs, \
                                    fix_models_predictions_shapes, fix_models_metrics_shapes
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools


def get_cms_mean_and_std(logs,normalize,epoch_num):
    y_pred = logs['val']['pred'] # (kfolds,epochs,examples,classes)
    y_true = logs['val']['labels']
    num_kfolds,_,_,num_classes = y_true.shape
    y_pred_argmax = np.argmax(y_pred, axis=3)
    y_true_argmax = np.argmax(y_true, axis=3)
    cms = []
    for k in range(num_kfolds):
        y_pred = y_pred_argmax[k,epoch_num]
        y_true = y_true_argmax[k,epoch_num]
        cm = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cms.append( cm )
    cms_mean = np.mean(cms, axis=0)
    cms_std = np.std(cms, axis=0)
    return cms_mean, cms_std

def plot_cms(mean,std,classes,title,show_scores=True,cmap=plt.cm.Blues):
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

def plot_confusion_matrixes(models_hist,classes,epoch_num=20,kfolded=True):
    models_hist = concatinate_kfolds(models_hist)
    models_predictions,models_metrics = filter_predictions_and_metrics(models_hist)
    models_predictions = fix_models_predictions_shapes(models_predictions,kfolded=kfolded)
    fig = plt.figure(figsize=(15, 60), dpi= 80, facecolor='w', edgecolor='k')
    num_models = len(models_predictions)
    for i,(k,v) in enumerate(models_predictions.items()):
        logs = history2logs(v)
        mean, std = get_cms_mean_and_std(logs,normalize=False,epoch_num=epoch_num) # (num_classes,num_classes)
        fig.add_subplot(num_models,2,i*2 + 1)
        plot_cms(mean,std,classes,title=k)
        mean, std = get_cms_mean_and_std(logs,normalize=True,epoch_num=epoch_num) # (num_classes,num_classes)
        fig.add_subplot(num_models,2,i*2 + 2)
        plot_cms(mean,std,classes,title=k)
    plt.show()