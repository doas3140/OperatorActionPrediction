import tensorflow as tf
import numpy as np
import os


from .utils import history2logs


def plot_models_tables(log_path,models_metrics,index2class_name={},classes_filter=[],log_name='val',metrics_names=None,epochs_skip=4,epochs_stop=32):
    '''
    inputs:
    log_path = main folder where to log
    models_metrics =     {
                         'model1': {
                                   'metric_name1':(num_kfolds,num_epochs,num_classes or 1),
                                   'accuracy2':(num_kfolds,num_epochs,num_classes or 1),
                                   ...
                                   },
                         ...
                         }
    index2class_name = { 0:'class_name1',... }
    classes_filter = ['class_name1',...], which classes should't be showed
    log_name = 'val' or 'test' or '' (which means train), which dataset to show in tables
    '''
    log_path = os.path.normpath(log_path)
    max_epochs = get_max_epochs(models_metrics)
    models_metrics = pad_histories_epochs(models_metrics,max_epochs)
    classes = [ index2class_name[c] for c in range(len(index2class_name)) ] + ['ALL']
    first_model_history = list(models_metrics.values())[0]
    if metrics_names is None: metrics_names = [ a for a in history2logs(first_model_history)['train'] ]
    model_names = list(models_metrics.keys())
    with tf.Session() as sess:
        placeholder = tf.placeholder(tf.string)
        for c,class_name in enumerate(classes):
            if class_name not in classes_filter:
                if class_name == 'ALL': c = 0 # to not get KeyError in index2class_name
                with tf.summary.FileWriter('{}/tables/{}'.format(log_path,class_name)) as matrix_writer:
                    summary_op = tf.summary.text(class_name,placeholder)
                    if epochs_stop == 0: epochs_stop = max_epochs
                    for e in range(0,epochs_stop,epochs_skip):
                        matrix = get_matrix(models_metrics,metrics_names,model_names,c,e,log_name=log_name,class_name=class_name)
                        summary_txt = sess.run(summary_op,feed_dict={placeholder:matrix})
                        matrix_writer.add_summary(summary_txt,global_step=e)


def get_matrix(models_metrics,metrics_names,model_names,c,e,log_name,class_name):
    ''' fill (num_metrics,num_models) size matrix w/ numbers from models_metrics histories '''
    matrix = get_empty_matrix(metrics_names,model_names)
    # find bigest numbers
    best_val_indexes = get_best_metrics_indexes(models_metrics,metrics_names,model_names,c,e,log_name)
    # put numbers into table
    for i,(model_name,hist) in enumerate(models_metrics.items()):
        for j,metric_name in enumerate(metrics_names):
            metric_arr = history2logs(hist)[log_name][metric_name]
            num_kfolds,num_epochs,num_classes = metric_arr.shape
            if num_classes != 1:
                if class_name != 'ALL':
                    metric_mean = '{:.4f}'.format( np.mean(metric_arr[:,e,c]) )
                else:
                    metric_mean = '{:.4f}'.format( np.mean(metric_arr[:,e,:]) )
            else:
                metric_mean = '{:.4f}'.format( np.mean(metric_arr[:,e,:]) )
            if i == best_val_indexes[j]:
                metric_mean = '**{}**'.format(metric_mean)
            matrix[i+1][j+1] = metric_mean
    return matrix


def get_best_metrics_indexes(models_metrics,metrics_names,model_names,c,e,log_name):
    ''' goes through models_metrics and returns list of best best value indexes
    best refers to bigest, except for loss
    '''
    best_val_indexes = []
    for j,metric_name in enumerate(metrics_names):
        if metric_name == 'loss': best_val = np.inf
        else: best_val = -np.inf
        for i,(model_name,hist) in enumerate(models_metrics.items()):
            metric_arr = history2logs(hist)[log_name][metric_name]
            num_kfolds,num_epochs,num_classes = metric_arr.shape
            if num_classes != 1:
                metric_mean = np.mean(metric_arr[:,e,c])
            else:
                metric_mean = np.mean(metric_arr[:,e,:])
            if metric_name == 'loss': is_better = metric_mean < best_val
            else: is_better = metric_mean > best_val
            if is_better:
                best_val = metric_mean
                best_index = i
        best_val_indexes.append(best_index)
    return best_val_indexes


def pad_histories_epochs(models_metrics,max_epochs):
    ''' returns histories with the same epochs (max_epochs). padded numbers are from last epoch '''
    for model_name,hist in models_metrics.items():
        for metric_name,metric_arr in hist.items():
            num_kfolds,num_epochs,num_classes = metric_arr.shape
            last_epoch_arr = np.reshape( metric_arr[:,-1,:], (num_kfolds,num_classes) )
            models_metrics[model_name][metric_name] = np.empty((num_kfolds,max_epochs,num_classes))
            for e in range(num_epochs):
                models_metrics[model_name][metric_name][:,e,:] = metric_arr[:,e,:]
            for e in range(num_epochs,max_epochs):
                models_metrics[model_name][metric_name][:,e,:] = last_epoch_arr
    return models_metrics


def get_empty_matrix(metrics_names,model_names):
    ''' get empty (num_metrics,num_models) size matrix '''
    string_max_len = 16
    matrix = [ [' '*string_max_len]*(len(metrics_names)+1) ]*(len(model_names)+1)
    matrix = np.array(matrix)
    for i,name in enumerate(model_names):
        matrix[i+1][0] = '**{:^{}.{}}**'.format(str(name),string_max_len-4,string_max_len-4)
    for i,name in enumerate(metrics_names):
        matrix[0][i+1] = '**{:^{}.{}}**'.format(str(name),string_max_len-4,string_max_len-4)
    return matrix


def get_max_epochs(models_metrics):
    ''' returns max_epochs from all histories '''
    bigest_val = 0
    for hist in models_metrics.values():
        logs = history2logs(hist)
        for log in logs.values():
            for metric_arr in log.values():
                num_kfolds,num_epochs,num_classes = metric_arr.shape
                if num_epochs > bigest_val:
                    bigest_val = num_epochs
    return bigest_val