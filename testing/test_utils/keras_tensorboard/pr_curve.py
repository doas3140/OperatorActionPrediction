import numpy as np
import tensorflow as tf
import os

from .tensorboard_writer import TensorBoardWriter
from .utils import history2logs

def plot_models_pr_curves(log_path,models_predictions,index2class_name,skip_epochs=1,num_thresholds=100,filter_logs=['train']):
    '''
    inputs:
    log_path = main folder where to log
    models_predictions = {
                         'model1': {
                                   'pred':(num_kfolds,num_epochs,num_examples,num_classes or 1),
                                   'labels':(num_kfolds,num_epochs,num_examples,num_classes or 1),
                                   'val_pred': ... , 'val_labels': ... ,
                                   'test_pred': ... , 'test_labels': ...
                                   },
                         ...
                         }
    index2class_name = { 0:'class_name1',... }
    classes_filter = ['class_name1',...], which classes should't be showed
    '''
    log_path = os.path.normpath(log_path)
    with tf.Session() as sess:
        label_placeholder = tf.placeholder(tf.bool)
        pred_placeholder = tf.placeholder(tf.float32)
        for model_name,hist in models_predictions.items():
            logs = history2logs(hist)
            for log_name,log in logs.items():
                if log_name not in filter_logs:
                    num_kfolds,num_epochs,num_examples = log['pred'].shape[:3]
                    labels,predictions = concat_kfold_axis(log['labels'],log['pred'])
                    with TensorBoardWriter('{}/pr_curves/{}_{}'.format(log_path,model_name,log_name)) as pr_writer:
                        classes = [c for c in index2class_name.values()]
                        for c,class_name in enumerate(classes):
                            data, update_op = tf.contrib.metrics.precision_recall_at_equal_thresholds(
                                    name=class_name,
                                    predictions=pred_placeholder,
                                    labels=label_placeholder,
                                    num_thresholds=num_thresholds )
                            sess.run(tf.local_variables_initializer())
                            for e in range(0,num_epochs,skip_epochs):
                                feed_dict = {label_placeholder:labels[e,:,c],pred_placeholder:predictions[e,:,c]}
                                sess.run(update_op,feed_dict=feed_dict)
                                pr_writer.log_pr_curve( tp=data.tp.eval(), fp=data.fp.eval(), tn=data.tn.eval(), \
                                                        fn=data.fn.eval(), name=class_name, global_step=e)



def concat_kfold_axis(labels,predictions):
    '''
    inputs:
    labels - (kfolds,epochs,examples,classes)
    predictions - (kfolds,epochs,examples,classes)
    outputs:
    new_labels  - (epochs,kfolds*examples,classes)
    new_predictions - (epochs,kfolds*examples,classes)
    '''
    num_kfolds,num_epochs,num_examples = predictions.shape[:3]
    l = []; p = []; 
    for e in range(num_epochs):
        la = []; pa = []
        for k in range(num_kfolds):
            for n in range(num_examples):
                pa.append(predictions[k,e,n])
                la.append(labels[k,e,n])
        p.append(np.array(pa))
        l.append(np.array(la))
    new_predictions = np.array(p)
    new_labels = np.array(l)
    return new_labels, new_predictions