# Simple Tensorboard Plotter for Keras Models

## Simple Example

### Step 1. train multiple keras models (pseudo code)

```python
from keras_tensorboard.model_utils import PredictData
hyperparam_preds = {}
for h in hyperparameters:
    model = create_model(h)
    callbacks = [
        PredictData(model, x=data.train.x, y=data.train.y, log_word=''),
        PredictData(model, x=data.val.x  , y=data.val.y  , log_word='val_'),
        PredictData(model, x=data.test.x , y=data.test.y , log_word='test_'),
    ]
    keras_hist = model.fit( data.train.x, data.train.y, epochs=30, batch_size=4, verbose=0,
                            validation_data=(data.val.x,data.val.y), callbacks=callbacks)
    hyperparam_preds[h] = keras_hist.history
```

### Step 2. set metrics to plot and evaluate on

```python
from sklearn.metrics import accuracy_score, recall_score, precision_score

def accuracy(average=False):
    ''' returns function that returns acc for each class. shape = (num_classes,)
    '''
    def metric(y_true, y_pred):
        ''' both inputs are not sparse w/ shape (num_examples,num_classes) '''
        num_classes = y_true.shape[-1]
        y_argmax = np.argmax(y_pred, axis=-1)
        y_pred = to_categorical(y_argmax, num_classes=num_classes)
        accs = np.zeros(num_classes)
        for i in range(num_classes):
            accs[i] = accuracy_score(y_true[:,i],y_pred[:,i])
        if average:
            return np.mean(accs)
        else:
            return accs
    return metric

def recall(average=None):
    ''' returns function that returs recall for each class. shape = (num_classes,)
    '''
    def metric(y_true, y_pred):
        ''' both inputs are not sparse w/ shape (num_examples,num_classes) '''
        num_classes = y_true.shape[-1]
        y_argmax = np.argmax(y_pred, axis=-1)
        y_pred = to_categorical(y_argmax, num_classes=num_classes)
        return recall_score(y_true, y_pred, average=average)
    return metric

def precision(average='macro'):
    ''' returns function that returs precision for each class. shape = (num_classes,)
    '''
    def metric(y_true, y_pred):
        ''' both inputs are not sparse w/ shape (num_examples,num_classes) '''
        num_classes = y_true.shape[-1]
        y_argmax = np.argmax(y_pred, axis=-1)
        y_pred = to_categorical(y_argmax, num_classes=num_classes)
        return precision_score(y_true, y_pred, average=average)
    return metric

metrics_functions = {
    'accuracy': accuracy(),
    'recall':recall(),
    'precision':precision()
}
```

### Step 3. plot tensorboard

```python
from keras_tensorboard.keras_tensorboard import plot_all_plots
models = {
    'model1':hyperparam_preds[0],
    'model2':hyperparam_preds[1],
    ...
}
index2class_name = {
    0:'cat',
    1:'dog',
    2:'none'
}
plot_all_plots( '/tmp/tensorboard', models, metrics_functions, 
                index2class_name, pr_curve=True )
```


## Cross Validated Example

### Step 1. train multiple keras models (pseudo code)

```python
hyperparam_preds = {}
for h in hyperparameters:
    model_kfold_results = []
    for i in kfold_cross_validation:
        data = kfolded_data[i]
        model = create_model(h)
        callbacks = [
            PredictData(model, x=data.train.x, y=data.train.y, log_word=''),
            PredictData(model, x=data.val.x  , y=data.val.y  , log_word='val_'),
            PredictData(model, x=data.test.x , y=data.test.y , log_word='test_'),
        ]
        keras_hist = model.fit( data.train.x, data.train.y, epochs=30, batch_size=4, verbose=0,
                                validation_data=(data.val.x,data.val.y), callbacks=callbacks)
        model_kfold_results.append( keras_hist.history )
    hyperparam_preds[h] = model_kfold_results
```
### Step 2. set metrics to plot and evaluate on

> same as in Simple example

### Step 3. plot tensorboard

```python
from keras_tensorboard.keras_tensorboard import plot_all_plots
models = {
    'model1':hyperparam_preds[0],
    'model2':hyperparam_preds[1],
    ...
}
index2class_name = {
    0:'cat',
    1:'dog',
    2:'none'
}
plot_all_plots( '/tmp/tensorboard', models, metrics_functions, 
                index2class_name, kfolded=True, pr_curve=True )
```
