
## Recreating results

### Step 1

 - create testing data:
 for creating data look at `test_models.ipynb` section `Creating data (all models, 5 folds, 200 epochs)`.  
 testing data is done for each model on validation set for 200 epochs (and is repeated for 5 folds)
 all predictions and targets are saved into .pickle files in test_results directory 
 
 ### Step 2
 
 - from created data create tensorboard.  
 by running `python create_tensorboard.py` it will go through all .pickle files in `test_results` directory
 and for each target it will create separate tensorboard directories in `testing/tensorboad` directory
 to run tensorboard run this command: `tensorboard --logdir='testing/tensorboard/{target_dir}'`, where `target_dir` is `action`,`memory`,`splitting`,`xrootd`
 
### Step 3 (Optional)

 - to visualize final result plots on Early Stopping in validation dataset: look at `test_models.ipynb` section `Plotting results from created data (mse, recall, precision)`.  
 it will go through results in `testing/test_results` directory, so you won't need to retrain models

### Step 4 (Optional)

 - to visualize confusion matrices on Early Stopping in validation dataset: look at `test_models.ipynb` section `Plotting confusion matrix from created data`.  
 it will go through results in `testing/test_results` directory, so you won't need to retrain models
