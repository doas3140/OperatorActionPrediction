import numpy as np

def concatinate_kfolds(kfolded_histories_dict):
    '''
    input:  { 
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
    output: {
            'model1': {
                      'loss':(num_kfolds,num_epochs),
                      'val_pred':(num_kfolds,num_epochs,num_examples,num_classes),
                      ...
                      },
            ...
            }
    '''
    # get all metric names, like 'loss','val_pred',...
    metric_names = []
    for model_name,kfolded_history in kfolded_histories_dict.items():
        num_kfolds = len(kfolded_history)
        for k in range(num_kfolds):
            for metric_name,arr in kfolded_history[k].items():
                if metric_name not in metric_names:
                    metric_names.append(metric_name)
    outDict = {}
    for model_name,kfolded_history in kfolded_histories_dict.items():
        outDict[model_name] = {}
        num_kfolds = len(kfolded_history)
        for metric_name in metric_names:
            concatinated_arr = []
            for k in range(num_kfolds):
                concatinated_arr.append(kfolded_history[k][metric_name])
            outDict[model_name][metric_name] = np.array(concatinated_arr)
    return outDict


def filter_predictions_and_metrics(histories,throw_away_train=False):
    '''
    input:  {
            'model1': {
                      'loss':(num_kfolds,num_epochs),
                      'val_pred':(num_kfolds,num_epochs,num_examples,num_classes),
                      ...
                      },
            ...
            }
    output: 2 dicts with same format as input
    out_pred = {
               'model1': {
                         metrics that ends w/ 'predictions' or 'pred',
                         !!! in output metric_name always ends w/ 'pred' !!!
                         }
               } 
    out_metr = {
               'model1': {
                         all left metrics that are not in out_pred
                         }
               } 
    '''
    out_metr = {}; out_pred = {}; 
    for model_name,hist in histories.items():
        out_metr[model_name] = {}
        out_pred[model_name] = {}
        for metric_name,metric_arr in hist.items():
            if metric_name.endswith('predictions') or metric_name.endswith('pred'):
                metric_name = metric_name.replace('predictions','pred')
                out_pred[model_name][metric_name] = metric_arr
            elif metric_name.endswith('labels'):
                out_pred[model_name][metric_name] = metric_arr
            else:
                if metric_name.startswith('train_'):
                    metric_name = metric_name.replace('train_','')
                    if throw_away_train: continue
                out_metr[model_name][metric_name] = metric_arr
    return out_pred,out_metr


def fix_metrics_shapes(hist,kfolded=False):
    ''' returned predictions arrays are all the same shape (num_kfolds,num_epochs,num_classes) \n
    input:   {
             'metric1': (num_kfolds or 0,num_epochs,num_classes or 0),
             ...
             }
    outnput: {
             'metric1': (num_kfolds,num_epochs,num_examples,num_classes),
             ...
             }
    '''
    out_hist = {}
    if not kfolded:
        for metric_name,metric_arr in hist.items():
            hist[metric_name] = np.expand_dims(metric_arr,axis=0)
    for metric_name,metric_arr in hist.items():
        metric_arr = np.array(metric_arr)
        if True in np.isnan(metric_arr):
            raise Exception('There exists nan values in your metric: {}'.format(metric_name))
        if len(metric_arr.shape) == 3:
            out_hist[metric_name] = metric_arr
        if len(metric_arr.shape) == 2:
            out_hist[metric_name] = np.expand_dims(metric_arr,axis=-1)
    return out_hist

def fix_models_metrics_shapes(models_metrics,kfolded=False):
    ''' does fix_metrics_shapes() across all models '''
    return {model_name:fix_metrics_shapes(hist,kfolded) for model_name,hist in models_metrics.items()}


def fix_predictions_shapes(hist,kfolded=False):
    ''' returned predictions arrays are all the same shape (num_kfolds,num_epochs,num_examples,num_classes)
    input:   {
             'metric1': (num_kfolds or 0,num_epochs,num_examples(can be not equal across epochs),num_classes or 0),
             ...
             }
    outnput: {
             'metric1': (num_kfolds,num_epochs,num_examples,num_classes),
             ...
             }
    '''
    out_hist = {}
    if not kfolded:
        for metric_name,metric_arr in hist.items():
            hist[metric_name] = np.expand_dims(metric_arr,axis=0)
    hist = cut_prediction_examples_axis(hist)
    for metric_name,metric_arr in hist.items():
        if len(metric_arr[0,0].shape) == 1:
            out_hist[metric_name] = np.expand_dims(metric_arr,axis=-1)
        else:
            out_hist[metric_name] = metric_arr
    return out_hist

def fix_models_predictions_shapes(models_predictions,kfolded=False):
    ''' does fix_predictions_shapes() across all models in input: {'model1':hist,...}'''
    return {model_name:fix_predictions_shapes(hist,kfolded) for model_name,hist in models_predictions.items()}


def concatinate_predictions(models_hists,test_hists):
    ''' returned predictions arrays are all the same shape (num_kfolds,num_epochs,num_examples,num_classes)
    inputs:   
    models_hists =       {
                         'model1':  {
                                    'metric1': (num_kfolds,num_epochs,num_examples,num_classes),
                                    ...
                                    },
                         ...
                         }
    test_hists =         {
                         'model1':  {
                                    'metric1': (1,num_epochs,num_examples,num_classes),
                                    ...
                                    },
                         ...
                         }
    output:
    models_hists w/ overwritten metrics
    '''
    # initiate out_hist
    model_names = []
    for hists in [models_hists,test_hists]:
        for model_name in hists.keys():
            if model_name not in model_names:
                model_names.append(model_name)
    out_hist = { model_name:{} for model_name in model_names }
    # append both predictions to out_hist
    for hists in [models_hists,test_hists]:
        for model_name,hist in hists.items():
            for metric_name,metric_arr in hist.items():
                out_hist[model_name][metric_name] = metric_arr
    return out_hist



def cut_prediction_examples_axis(hist):
    ''' cuts num_examples to same number across epochs
    input:   {
             'metric1': (num_kfolds,num_epochs,num_examples(can be not equal across epochs),num_classes or 0),
             ...
             }
    output:  {
             'metric1': (num_kfolds,num_epochs,num_examples(same across epochs),num_classes or 0),
             ...
             }
    '''
    maximum_allowed_examples = get_min_examples(hist)
    out_hist = {}
    for metric_name,metric_arr in hist.items():
        karr = []
        num_kfolds = len(metric_arr)
        for k in range(num_kfolds):
            earr = []
            num_epochs = len(metric_arr[k])
            for e in range(num_epochs):
                earr.append( np.array(metric_arr[k][e][:maximum_allowed_examples]) )
            karr.append(np.array(earr))
        out_hist[metric_name] = np.array(karr)
    return out_hist


def get_min_examples(hist):
    ''' get minimum num_example from input \n
    input:   {
             'metric1': (num_kfolds,num_epochs,num_examples(can be not equal across epochs),num_classes or 0),
             ...
             }
    output:  lowest num_examples number
    '''
    lowest_num = np.inf
    for metric_name,metric_arr in hist.items():
        metric_arr = np.array(metric_arr)
        num_examples = metric_arr[0,0].shape[0]
        if num_examples < lowest_num:
            lowest_num = num_examples
    return lowest_num-1


def calculate_models_metrics(models_predictions,metrics_functions,notebook=False,verbose=1):
    '''
    inputs:
    models_predictions = {
                         'model1': {
                                'loss':(num_kfolds,num_epochs),
                                'val_pred':(num_kfolds,num_epochs,num_examples,num_classes),
                                ...
                                },
                         ...
                         }
    metrics_functions =  {
                         'metric_name':metric_function w/ inputs (y_true,y_pred),
                         # y_true = (num_examples,num_classes or 1), y_pred = (num_examples,num_classes or 1) #
                         ...
                         }
    output: {
            'model1': {
                      'metric_name1':(num_kfolds,num_epochs,num_classes or 1),
                      'accuracy2':(num_kfolds,num_epochs,num_classes or 1),
                      ...
                      },
            ...
            }
    '''
    out_hists = {}
    if verbose != 0:
        if not notebook: from tqdm import tqdm
        else: from tqdm import tqdm_notebook as tqdm
        dict_items = tqdm(models_predictions.items(),total=len(models_predictions),desc='models ',leave=False,initial=0)
    else:
        dict_items = models_predictions.items()
    for model_name,hist in dict_items:
        out_hists[model_name] = {}
        if verbose != 0: hist_items = tqdm(hist.items(),total=len(hist),desc='dataset',leave=False,initial=0)
        else: hist_items = hist.items()
        for pred_name,pred_arr in hist_items:
            if pred_name.startswith('val'): log_name = 'val_'
            elif pred_name.startswith('test'): log_name = 'test_'
            else: log_name = '' # train
            if not pred_name.endswith('labels'):
                # num_kfolds = len(pred_arr)
                num_kfolds,num_epochs,num_examples,num_classes = pred_arr.shape
                if verbose != 0: function_items = tqdm(metrics_functions.items(),total=len(metrics_functions),desc='metrics',leave=False,initial=0)
                else: function_items = metrics_functions.items()
                for metric_name,metric_fun in function_items:
                    out_arr = np.empty((num_kfolds,num_epochs,num_classes))
                    for k in range(num_kfolds):
                        for e in range(num_epochs):
                            y_pred = hist[pred_name][k,e,:,:]
                            label_name = pred_name.replace('pred','labels')
                            y_true = hist[label_name][k,e,:,:]
                            out_arr[k,e] = metric_fun(y_true,y_pred)
                    out_hists[model_name][log_name+metric_name] = out_arr
    return out_hists


def append_models_metrics(models_metrics,models_metrics2append):
    ''' append metrics from models_metrics2append to models_metrics '''
    for model_name,hist in models_metrics2append.items():
        for metric_name,metric_arr in hist.items():
            models_metrics[model_name][metric_name] = metric_arr
    return models_metrics


def history2logs(hist):
    '''
    input:   {
             'loss': (num_kfolds,num_epochs,num_examples,num_classes or 1),
             'val_loss': (num_kfolds,num_epochs,num_examples,num_classes or 1),
             'test_loss': (num_kfolds,num_epochs,num_examples,num_classes or 1),
             ...
             }
    output:  {
             'train': {
                      'loss': (num_kfolds,num_epochs,num_examples,num_classes or 1),
                      'accuracy': (num_kfolds,num_epochs,num_examples,num_classes or 1),
                      ...
                      },
             'val': { 'loss':... , 'accuracy':... , ... },
             'test': { 'loss':... , 'accuracy':... , ... }
             }
    
    '''
    train_logs = {k: v for k, v in hist.items() if not (k.startswith('val_') or k.startswith('test_'))}
    val_logs = {k.replace('val_', ''): v for k, v in hist.items() if k.startswith('val_')}
    test_logs = {k.replace('test_', ''): v for k, v in hist.items() if k.startswith('test_')}
    return {'train':train_logs,'val':val_logs, 'test':test_logs}


def get_metrics_shapes(logs):
    '''
    input:   {
             'train': {
                      'loss': (num_kfolds,num_epochs,num_examples,num_classes or 1),
                      'accuracy': (num_kfolds,num_epochs,num_examples,num_classes or 1),
                      ...
                      },
             'val': { 'loss':... , 'accuracy':... , ... },
             'test': { 'loss':... , 'accuracy':... , ... }
             }
    output:  {
             'metric1':np.array.shape,
             'accuracy2':np.array.shape
             }
    '''
    metric_shapes = {}
    for log in logs.values():
        for metric_name,metric_arr in log.items():
            if metric_name not in metric_shapes:
                metric_shapes[metric_name] = metric_arr.shape
    return metric_shapes

