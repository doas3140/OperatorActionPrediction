import warnings
warnings.filterwarnings('ignore')


from .utils import concatinate_kfolds, filter_predictions_and_metrics
from .utils import fix_models_metrics_shapes, fix_models_predictions_shapes
from .utils import calculate_models_metrics, append_models_metrics
from .utils import concatinate_predictions

from .metrics import plot_models_metrics
from .tables import plot_models_tables
from .pr_curve import plot_models_pr_curves


def create_metrics_and_predictions(hists,metrics_functions,test_hists=None,kfolded=False,verbose=1,notebook=False):
    ''' 
    inputs:
    hists = { 
            'model1': [ indexes are kfold number
                      0: { this dict is keras_history.history dict
                         'loss':(num_epochs),
                         'val_pred':(num_epochs,num_examples,num_classes),
                         ...
                         },
                      ...
                      ],
            ...
            }
    metrics_functions = { 'metric_name':metric_function w/ inputs (y_true,y_pred), ... }
    outputs: 2 dicts w/ very similar format:
    models_metrics =     {
                         'model1': {
                                   'metric_name1':(num_kfolds,num_epochs,num_classes or 1),
                                   'accuracy2':(num_kfolds,num_epochs,num_classes or 1),
                                   ...
                                   },
                         ...
                         }
    models_predictions = {
                         'model1': {
                                   'pred':(num_kfolds,num_epochs,num_classes or 1),
                                   'labels':(num_kfolds,num_epochs,num_classes or 1),
                                   'val_pred': ... , 'val_labels': ... ,
                                   'test_pred': ... , 'test_labels': ...
                                   },
                         ...
                         }
    '''
    if kfolded:
        if verbose != 0: print('concatinating kfolds...')
        hists = concatinate_kfolds(hists)
    if verbose != 0: print('filtering predictions...')
    models_predictions,models_metrics = filter_predictions_and_metrics(hists)
    if verbose != 0: print('fixing shapes...')
    models_predictions = fix_models_predictions_shapes(models_predictions,kfolded=kfolded)
    models_metrics = fix_models_metrics_shapes(models_metrics,kfolded=kfolded)
    if test_hists is not None:
        test_predictions,test_metrics = filter_predictions_and_metrics(test_hists,throw_away_train=True)
        test_predictions = fix_models_predictions_shapes(test_predictions,kfolded=False)
        models_predictions = concatinate_predictions(models_predictions,test_predictions)
        test_metrics = fix_models_metrics_shapes(test_metrics,kfolded=False)
        models_metrics = concatinate_predictions(models_metrics,test_metrics)
    if verbose != 0: print('calculating metrics...')
    calculated_models_metrics = calculate_models_metrics(models_predictions,metrics_functions,verbose=verbose,notebook=notebook)
    models_metrics = append_models_metrics(models_metrics,calculated_models_metrics)
    return models_metrics, models_predictions


def plot_all_plots(log_path,models_hists,metrics_functions,index2class_name={},test_hists=None,kfolded=False,pr_curve=False,verbose=1,notebook=False):
    ''' 
    inputs:
    models_hists =  { 
                    'model1': [ indexes are kfold number
                            0: { this dict is keras_history.history dict
                                'loss':(num_epochs),
                                'val_pred':(num_epochs,num_examples,num_classes),
                                ...
                                },
                            ...
                            ],
                    ...
                    }
    metrics_functions = { 'metric_name':metric_function w/ inputs (y_true,y_pred), ... }
    '''
    models_metrics,models_predictions = create_metrics_and_predictions(models_hists,metrics_functions,test_hists,kfolded,verbose=verbose,notebook=notebook)
    if verbose != 0: print('\n\n plotting metrics and distributions...')
    plot_models_metrics(log_path,models_metrics,index2class_name)
    if verbose != 0: print('plotting tables...')
    plot_models_tables(log_path,models_metrics,index2class_name)
    if pr_curve:
        if verbose != 0: print('plotting pr curves...')
        plot_models_pr_curves(log_path,models_predictions,index2class_name,filter_logs=['train','test'])
    if verbose != 0: print('Done!')