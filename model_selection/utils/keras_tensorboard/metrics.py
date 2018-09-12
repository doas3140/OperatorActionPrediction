
import tensorboard
from tensorboard.plugins.custom_scalar import layout_pb2
import tensorflow as tf
import numpy as np
import os

from .utils import history2logs
from .utils import get_metrics_shapes
from .tensorboard_writer import TensorBoardWriter


def plot_models_metrics(log_path,models_metrics,index2class_name={},classes_filter=[]):
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
    '''
    # plot metrics in log_path+'/metrics/' folder and distributions in log_path+'/distributions/' folder
    for model_name,hist in models_metrics.items():
        logs = history2logs(hist)
        with TensorBoardWriter('{}/metrics/{}'.format(log_path,model_name)) as scalar_writer:
            for log_name,log in logs.items():
                with TensorBoardWriter('{}/distributions/{}_{}'.format(log_path,model_name,log_name)) as distr_writer:
                    for metric_name,metric_arr in log.items():
                        num_kfolds,num_epochs,num_classes = metric_arr.shape
                        classes = [ index2class_name[c] for c in range(num_classes) if num_classes > 1 ] + ['ALL']
                        for c,class_name in enumerate(classes):
                            if class_name not in classes_filter:
                                for e in range(num_epochs):
                                    if class_name == 'ALL': arr = metric_arr[:,e,:]
                                    else: arr = metric_arr[:,e,c]
                                    scalar_writer.log_scalar(np.mean(arr),name='{}_{}/{}'.format(metric_name,class_name,log_name),global_step=e)
                                    distr_writer.log_histogram(arr,name='{}/{}/{}'.format(metric_name,class_name,log_name),global_step=e)
    # add custom layout to merge different datasets (train,val,test) into one graph and all classes to one graph
    with TensorBoardWriter('{}/custom_scalar_layout'.format(log_path)) as custom_layout_writer:
        metric_shapes = get_metrics_shapes( history2logs(list(models_metrics.values())[0]) )
        layout_summary = get_layout_summary(metric_shapes,index2class_name)
        custom_layout_writer.add_summary(layout_summary)


def get_layout_summary(metric_shapes,index2class_name,log_names=['train','val','test']):
    categories = []; concatinated_charts = []
    for metric_name,metric_shape in metric_shapes.items():
        charts = get_metric_charts(metric_name,metric_shape,index2class_name,log_names)
        categories.append( get_layout_category(title=metric_name,charts=charts) )
        tags = [ '{}_ALL/{}'.format(metric_name,log_name) for log_name in log_names ]
        concatinated_charts.append( get_layout_chart(title=metric_name,tags=tags) )
    categories.append( get_layout_category(title='ALL',charts=concatinated_charts) )
    return tensorboard.summary.custom_scalar_pb( layout_pb2.Layout(category=categories) )


def get_metric_charts(metric_name,metric_shape,index2class_name,log_names):
    num_kfolds,num_epochs,num_classes = metric_shape
    if num_classes == 1: classes = ['ALL']
    else: classes = [ index2class_name[c] for c in range(num_classes) ]
    charts = []
    for log_name in log_names:
        tags = [ '{}_{}/{}'.format(metric_name,class_name,log_name) for class_name in classes ]
        charts.append( get_layout_chart(title=log_name,tags=tags) )
    return charts


def get_layout_chart(title,tags):
    return layout_pb2.Chart(
                            title=title,
                            multiline=layout_pb2.MultilineChartContent(
                                    tag=tags, # search for string using regex (similar to .startswith())
                                )
                        )


def get_layout_category(title,charts):
    return layout_pb2.Category(
                        title=title,
                        chart=charts
                    )

