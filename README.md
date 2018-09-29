# Operator Action Prediction

From what errors happened and in which sites predict operator actions: 
 * which `actions` to choose
 * requested `memory`
 * job `splitting`
 * enabling/disabling `xrootd`

For more info about this problem look at [slides](https://github.com/doas3140/OperatorActionPrediction/blob/master/slides.pdf)

For information about recreating results look at [testing directory](https://github.com/doas3140/OperatorActionPrediction/tree/master/testing)

All scripts for recreating results are in [test_models.ipynb](https://github.com/doas3140/OperatorActionPrediction/blob/master/test_models.ipynb)

## Examples

### Training

For each target (`action`,`memory`,`splitting`,`xrootd`) you will have to train a model separately

```python
from SimpleModel import SimpleModel as Model
from CNNModel import CNNModel as Model
from XGBoostModel import XGBoostModel as Model

m = Model(num_classes=3, num_error=54, num_sites=142)
m.train(X, Y, verbose=1)
```

* EmbeddingModel has pretraining procedure, so you initialize it a little bit differently:  
`m = EmbeddingModel(X=X, num_classes=3, num_error=54, num_sites=142)`

Here:
* X - Numpy array `X.shape = (num_examples, num_error, num_sites)`
* Y - Numpy array `Y.shape = (num_examples,)`, where each number represents class index (like in sklearn)

### Predicting

```python
predictions = m.predict(Y)
```

Here:
* predictions - Numpy array `predictions.shape = (num_examples,)` with argmax over predictions, where each number represents class index (like in sklearn)

### Hyperparameter Search (using skopt library)

```python
best_params = m.find_optimal_parameters(X, Y, num_calls=12)
```
After finding optimal parameter, it automatically sets as default parameters.  
So, by running `m.train(X,Y)` you will train on found parameters.

If you want to set parameters manually and train with them:

```python
# set found parameters
m.model_params = best_params

# or set custom parameters
custom_params_for_SimpleModel = {
  'dense_layers':3,
  'dense_units':50,
  'regulizer_value':0.0015,
  'dropout_value':0.015,
  'learning_rate':1e-3
}
m.model_params = custom_params_for_SimpleModel

m.train(X,Y)
```
### Using Model Attention

* Doesn't work with XGBoost Model

```python
m = Model(num_classes=3, num_error=54, num_sites=142, use_attention=True)
predictions, site_attention, error_attention = m.predict(Y)
```

Here:
* predictions - Numpy array `predictions.shape = (num_examples,)` with argmax over predictions, where each number represents class index (like in sklearn)
* sites_attention - Numpy array `sites_attention.shape = (num_sites,)`, where each number is [0,1]
* error_attention - Numpy array `error_attention.shape = (num_error,)`, where each number is [0,1]
* Each [0,1] number represents how much model was focused on that site/error. Outputs are normalized so that all numbers in `sites_attention` sum to 1 (same for `error_attention`).

### Using Resampling (using imblearn library)

```python
from imblearn.over_sampling import SMOTE

m = Model(num_classes=3, num_error=54, num_sites=142)
m.train( X, Y, verbose=1, use_imblearn=True,
         imblearn_class=SMOTE(random_state=42, ratio=1.0))
```

### Saving and Loading Model

```python
path_to_directory = '~/my_model/'

m.save_model(dirpath=path_to_directory)
m.load_model(dirpath=path_to_directory)
```

