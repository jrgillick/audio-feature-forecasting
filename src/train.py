#Example Usage: python train.py --data_path=generated_data --output_path=saved_models

import util, models
from util import *
from features import *
import importlib, os, argparse, copy, time
from collections import defaultdict
import os, sys, pickle, librosa, numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

############################################
################ Parse Args ################
############################################        
parser = argparse.ArgumentParser()

# Path to the output folder to save model checkpoints
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)

args = parser.parse_args()

root_path = args.data_path
output_path = args.output_path

############################################
################ Load Data #################
############################################

print("Loading Data...\n")

train_mixture_datasets = defaultdict(None)
dev_mixture_datasets = defaultdict(None)
test_mixture_datasets = defaultdict(None)

mixture_values = [2,3,6,12,20,30]


# load training data
# FFT
filename = f"{root_path}/train/train_weighted_ffts.pkl"
print(f"Loading Pre-computed features from {filename}...")
with open(filename, "rb") as f:
    train_weighted_ffts = pickle.load(f)

# MFCC
filename = f"{root_path}/train/train_weighted_mfccs.pkl"
print(f"Loading Pre-computed features from {filename}...")
with open(filename, "rb") as f:
    train_weighted_mfccs = pickle.load(f)
    
for m in mixture_values:
    filename = f"{root_path}/train/mixture_data_{m}.pkl"
    print(f"Loading Pre-computed features from {filename}...")
    with open(filename, "rb") as f:
        train_mixture_datasets[m] = pickle.load(f)
        print(f"Contains {len(train_mixture_datasets[m]['components'])} datapoints")
             
# load dev data
# FFT
filename = f"{root_path}/dev/dev_weighted_ffts.pkl"
print(f"Loading Pre-computed features from {filename}...")
with open(filename, "rb") as f:
    dev_weighted_ffts = pickle.load(f)

# MFCC
filename = f"{root_path}/dev/dev_weighted_mfccs.pkl"
print(f"Loading Pre-computed features from {filename}...")
with open(filename, "rb") as f:
    dev_weighted_mfccs = pickle.load(f)
    
for m in mixture_values:
    filename = f"{root_path}/dev/mixture_data_{m}.pkl"
    print(f"Loading Pre-computed features from {filename}...")
    with open(filename, 'rb') as f:
        dev_mixture_datasets[m] = pickle.load(f)
        
# load test data
# FFT
filename = f"{root_path}/test/test_weighted_ffts.pkl"
print(f"Loading Pre-computed features from {filename}...")
with open(filename, "rb") as f:
    test_weighted_ffts = pickle.load(f)
    
# MFCC
filename = f"{root_path}/test/test_weighted_mfccs.pkl"
print(f"Loading Pre-computed features from {filename}...")
with open(filename, "rb") as f:
    test_weighted_mfccs = pickle.load(f)

for m in mixture_values:
    filename = f"{root_path}/test/mixture_data_{m}.pkl"
    print(f"Loading Pre-computed features from {filename}...")
    with open(filename, 'rb') as f:
        test_mixture_datasets[m] = pickle.load(f)
        
# Normalize all fft data
for m in mixture_values:
    d = train_mixture_datasets[m]
    apply_max_norm(d['mix_ffts'])
    d = dev_mixture_datasets[m]
    apply_max_norm(d['mix_ffts'])
    d = test_mixture_datasets[m]
    apply_max_norm(d['mix_ffts'])
    
apply_max_norm(train_weighted_ffts)
apply_max_norm(dev_weighted_ffts)
apply_max_norm(test_weighted_ffts)

# Compute the mean mixture FFT's and MFCC's for all datasets to use as a baseline
#for data in [train_mixture_datasets, dev_mixture_datasets, test_mixture_datasets]:
for data in [train_mixture_datasets, dev_mixture_datasets]:    
    for m in mixture_values:
        data[m]['mean_fft'] = np.mean(data[m]['mix_ffts'], axis=0)
        data[m]['mean_mfcc'] = np.mean(data[m]['mix_mfccs'], axis=0)
        
############################################
################ FFT MODELS ################
############################################

# MLP
# Train MLP to predict FFT from FFT
print("FFT MLP model")
patience = 10
loss_fn = 'mean_squared_error'
fft_mlp_models = defaultdict(None)
os.system(f"mkdir -p {output_path}/fft/mlp")

# Train a separate model for each number of components
for m in mixture_values:
    print("Training model for %d components" % (m))
    
    fft_mlp_models[m] = models.fft_to_fft_model(m, loss_fn=loss_fn)
    
    X_train = get_fft_inputs(train_mixture_datasets[m], train_weighted_ffts)
    y_train = get_fft_mix_targets(train_mixture_datasets[m])
    
    X_dev = get_fft_inputs(dev_mixture_datasets[m], dev_weighted_ffts)
    y_dev = get_fft_mix_targets(dev_mixture_datasets[m])
    
    history = [99999999.]
    patience_level = 0
    still_training=True
    
    epochs = 75
    for e in range(epochs):
        if patience_level < 10:
            res = fft_mlp_models[m].fit(X_train, y_train, validation_data=[X_dev, y_dev],
                                        batch_size=200, epochs=1, verbose=False)
            loss = res.history['loss'][0]
            val_loss = res.history['val_loss'][0]
            print(f"Epoch: {e} ... loss: {loss} ... val loss: {val_loss}", end="\r")

            best_result = min(history)
            if val_loss < best_result:
                patience_level = 0
                fft_mlp_models[m].save(f"{output_path}/fft/mlp/best_{m}.pkl")
            else:
                patience_level += 1
            history.append(val_loss)
        if patience_level >= 10 or e == epochs-1:
            if still_training:
                print(f"Epoch: {e} ... loss: {loss} ... val loss: {best_result}")
            still_training=False
            

print("FFT Ordered LSTM")
# Ordered LSTM
# Train LSTM models to predict FFT from FFT
patience = 10
loss_fn = 'mean_squared_error'
fft_lstm_models_ordered = defaultdict(None)
os.system(f"mkdir -p {output_path}/fft/lstm_ordered")

for m in mixture_values:
    print("Training model for %d components" % (m))
    fft_lstm_models_ordered[m] = models.fft_to_fft_lstm_model(m, loss_fn=loss_fn)
    
    X_train = get_fft_inputs(train_mixture_datasets[m], train_weighted_ffts, flatten=False)
    y_train = get_fft_mix_targets(train_mixture_datasets[m])
    
    X_dev = get_fft_inputs(dev_mixture_datasets[m], dev_weighted_ffts, flatten=False)
    y_dev = get_fft_mix_targets(dev_mixture_datasets[m])
    
    X_train = order_dataset_by_norm(X_train)
    X_dev = order_dataset_by_norm(X_dev)
    
    history = [99999999.]  
    patience_level = 0
    still_training=True
    
    epochs = 75
    for e in range(epochs):
        if patience_level < 10:
            res = fft_lstm_models_ordered[m].fit(X_train, y_train,
                                                 validation_data=[X_dev, y_dev],
                                                 batch_size=200, epochs=1, verbose=False)
            loss = res.history['loss'][0]
            val_loss = res.history['val_loss'][0]
            print(f"Epoch: {e} ... loss: {loss} ... val loss: {val_loss}", end="\r")

            best_result = min(history)
            if val_loss < best_result:
                patience_level = 0
                fft_lstm_models_ordered[m].save(f"{output_path}/fft/lstm_ordered/best_{m}.pkl")
            else:
                patience_level += 1
            history.append(val_loss)
        if patience_level >= 10 or e == epochs-1:
            if still_training:
                print(f"Epoch: {e} ... loss: {loss} ... val loss: {best_result}")
            still_training=False
            

print("FFT Unordered LSTM")
# Unordered LSTM
# Train LSTM models to predict FFT from FFT
patience = 10
loss_fn = 'mean_squared_error'
fft_lstm_models_unordered = defaultdict(None)
os.system(f"mkdir -p {output_path}/fft/lstm_unordered")

for m in mixture_values:
    print("Training model for %d components" % (m))
    fft_lstm_models_unordered[m] = models.fft_to_fft_lstm_model(m, loss_fn=loss_fn)
    
    X_train = get_fft_inputs(train_mixture_datasets[m], train_weighted_ffts, flatten=False)
    y_train = get_fft_mix_targets(train_mixture_datasets[m])
    
    X_dev = get_fft_inputs(dev_mixture_datasets[m], dev_weighted_ffts, flatten=False)
    y_dev = get_fft_mix_targets(dev_mixture_datasets[m])
    
    #X_train = order_dataset_by_norm(X_train)
    #X_dev = order_dataset_by_norm(X_dev)
    
    history = [99999999.]
    patience_level = 0
    still_training=True
    
    epochs = 75
    for e in range(epochs):
        if patience_level < 10:
            res = fft_lstm_models_unordered[m].fit(X_train, y_train,
                                                   validation_data=[X_dev, y_dev],
                                                   batch_size=200, epochs=1, verbose=False)
            shuffle_order(X_train)
        
            loss = res.history['loss'][0]
            val_loss = res.history['val_loss'][0]
            print(f"Epoch: {e} ... loss: {loss} ... val loss: {val_loss}", end="\r")

            best_result = min(history)
            if val_loss < best_result:
                patience_level = 0
                fft_lstm_models_unordered[m].save(f"{output_path}/fft/lstm_unordered/best_{m}.pkl")
            else:
                patience_level += 1
            history.append(val_loss)
        if patience_level >= 10 or e == epochs-1:
            if still_training:
                print(f"Epoch: {e} ... loss: {loss} ... val loss: {best_result}")
            still_training=False


print("FFT Residual LSTM")
# Residual LSTM
# Train LSTM models to predict FFT from FFT
patience = 10
loss_fn = 'mean_squared_error'
fft_lstm_models_residual = defaultdict(None)
os.system(f"mkdir -p {output_path}/fft/lstm_residual")

for m in mixture_values:
    print("Training model for %d components" % (m))
    fft_lstm_models_residual[m] = models.fft_to_fft_residual_inputs_lstm_model(m, loss_fn=loss_fn)
    
    X_train = get_fft_inputs(train_mixture_datasets[m], train_weighted_ffts, flatten=False)
    y_train = get_fft_mix_targets(train_mixture_datasets[m])
    
    X_dev = get_fft_inputs(dev_mixture_datasets[m], dev_weighted_ffts, flatten=False)
    y_dev = get_fft_mix_targets(dev_mixture_datasets[m])
    
    X_train = order_dataset_by_norm(X_train)
    X_dev = order_dataset_by_norm(X_dev)
    
    history = [99999999.]
    patience_level = 0
    still_training = True
    
    epochs = 75
    for e in range(epochs):
        if patience_level < 10:
            res = fft_lstm_models_residual[m].fit(X_train, y_train,
                                                          validation_data=[X_dev, y_dev],
                                                          batch_size=200, epochs=1, verbose=False)
            
            shuffle_order(X_train)
            loss = res.history['loss'][0]
            val_loss = res.history['val_loss'][0]
            print(f"Epoch: {e} ... loss: {loss} ... val loss: {val_loss}", end="\r")

            best_result = min(history)
            if val_loss < best_result:
                patience_level = 0
                fft_lstm_models_residual[m].save(f"{output_path}/fft/lstm_residual/best_{m}.pkl")
            else:
                patience_level += 1
            history.append(val_loss)
        if patience_level >= 10 or e == epochs-1:
            if still_training:
                print(f"Epoch: {e} ... loss: {loss} ... val loss: {best_result}")
            still_training=False
            
############################################
################ MFCC MODELS ###############
############################################

print("MFCC MLP")
#MLP
patience = 10
loss_fn = 'mean_squared_error'
mlp_models_mfcc = defaultdict(None)
os.system(f"mkdir -p {output_path}/mfcc/mlp")

# Train a separate model for each number of components
for m in mixture_values:
    print("Training model for %d components" % (m))
    mlp_models_mfcc[m] = models.mfcc_to_mfcc_model(m, loss_fn=loss_fn)
    
    X_train = get_mfcc_inputs(train_mixture_datasets[m], train_weighted_mfccs)
    y_train = get_mfcc_mix_targets(train_mixture_datasets[m])

    X_dev = get_mfcc_inputs(dev_mixture_datasets[m], dev_weighted_mfccs)
    y_dev = get_mfcc_mix_targets(dev_mixture_datasets[m])
    
    history = [99999999.]  
    patience_level = 0 
    still_training=True
    
    epochs = 75
    for e in range(epochs):
        if patience_level < 10:
            res = mlp_models_mfcc[m].fit(X_train, y_train,
                                         validation_data=[X_dev, y_dev],
                                         batch_size=200, epochs=1, verbose=False)
            
            loss = res.history['loss'][0]
            val_loss = res.history['val_loss'][0]
            print(f"Epoch: {e} ... loss: {loss} ... val loss: {val_loss}", end="\r")

            best_result = min(history)
            if val_loss < best_result:
                patience_level = 0
                mlp_models_mfcc[m].save(f"{output_path}/mfcc/mlp/best_{m}.pkl")
            else:
                patience_level += 1
            history.append(val_loss)
        if patience_level >= 10 or e == epochs-1:
            if still_training:
                print(f"Epoch: {e} ... loss: {loss} ... val loss: {best_result}")
            still_training=False
            
print("MFCC Ordered LSTM")
# Ordered LSTM
# Train LSTM models to predict MFCC from MFCC
patience = 10
loss_fn = 'mean_squared_error'
mfcc_lstm_models_ordered = defaultdict(None)
os.system(f"mkdir -p {output_path}/mfcc/lstm_ordered")

# Train a separate model for each number of components
for m in mixture_values:
    print("Training model for %d components" % (m))
    mfcc_lstm_models_ordered[m] = models.mfcc_to_mfcc_lstm_model(m, loss_fn=loss_fn)
    
    X_train = get_mfcc_inputs(train_mixture_datasets[m], train_weighted_mfccs, flatten=False)
    y_train = get_mfcc_mix_targets(train_mixture_datasets[m])

    X_dev = get_mfcc_inputs(dev_mixture_datasets[m], dev_weighted_mfccs, flatten=False)
    y_dev = get_mfcc_mix_targets(dev_mixture_datasets[m])
    
    X_train = order_dataset_by_norm(X_train)
    X_dev = order_dataset_by_norm(X_dev)
    
    history = [99999999.]  
    patience_level = 0  
    still_training = True

    epochs = 75
    for e in range(epochs):
        if patience_level < 10:
            res = mfcc_lstm_models_ordered[m].fit(X_train, y_train,
                                                  validation_data=[X_dev, y_dev],
                                                  batch_size=200, epochs=1, verbose=False)

            loss = res.history['loss'][0]
            val_loss = res.history['val_loss'][0]
            print(f"Epoch: {e} ... loss: {loss} ... val loss: {val_loss}", end="\r")

            best_result = min(history)
            if val_loss < best_result:
                patience_level = 0
                mfcc_lstm_models_ordered[m].save(f"{output_path}/mfcc/lstm_ordered/best_{m}.pkl")
            else:
                patience_level += 1
            history.append(val_loss)
        if patience_level >= 10 or e == epochs-1:
            if still_training:
                print(f"Epoch: {e} ... loss: {loss} ... val loss: {best_result}")
            still_training=False
            
print("MFCC Unordered LSTM")
# Unordered LSTM
# Train LSTM models to predict MFCC from MFCC
patience = 10
loss_fn = 'mean_squared_error'
mfcc_lstm_models_unordered = defaultdict(None)
os.system(f"mkdir -p {output_path}/mfcc/lstm_unordered")

# Train a separate model for each number of components
for m in mixture_values:
    print("Training model for %d components" % (m))
    mfcc_lstm_models_unordered[m] = models.mfcc_to_mfcc_lstm_model(m, loss_fn=loss_fn)
    
    X_train = get_mfcc_inputs(train_mixture_datasets[m], train_weighted_mfccs, flatten=False)
    y_train = get_mfcc_mix_targets(train_mixture_datasets[m])

    X_dev = get_mfcc_inputs(dev_mixture_datasets[m], dev_weighted_mfccs, flatten=False)
    y_dev = get_mfcc_mix_targets(dev_mixture_datasets[m])

    history = [99999999.]
    patience_level = 0  
    still_training=True

    epochs = 75
    for e in range(epochs):
        if patience_level < 10:
            res = mfcc_lstm_models_unordered[m].fit(X_train, y_train,
                                                    validation_data=[X_dev, y_dev],
                                                    batch_size=200, epochs=1, verbose=False)
            shuffle_order(X_train)
            
            loss = res.history['loss'][0]
            val_loss = res.history['val_loss'][0]
            print(f"Epoch: {e} ... loss: {loss} ... val loss: {val_loss}", end="\r")

            best_result = min(history)
            if val_loss < best_result:
                patience_level = 0
                mfcc_lstm_models_unordered[m].save(f"{output_path}/mfcc/lstm_unordered/best_{m}.pkl")
            else:
                patience_level += 1
            history.append(val_loss)
        if patience_level >= 10 or e == epochs-1:
            if still_training:
                print(f"Epoch: {e} ... loss: {loss} ... val loss: {best_result}")
            still_training=False
            
print("MFCC Residual LSTM")
# Residual LSTM
# Train LSTM Residual models to predict MFCC from MFCC
patience = 10
loss_fn = 'mean_squared_error'
mfcc_lstm_models_residual = defaultdict(None)
os.system(f"mkdir -p {output_path}/mfcc/lstm_residual")

# Train a separate model for each number of components
for m in mixture_values:
    print("Training model for %d components" % (m))
    mfcc_lstm_models_residual[m] = models.mfcc_to_mfcc_residual_inputs_lstm_model(m, loss_fn=loss_fn)
    
    X_train = get_mfcc_inputs(train_mixture_datasets[m], train_weighted_mfccs, flatten=False)
    y_train = get_mfcc_mix_targets(train_mixture_datasets[m])

    X_dev = get_mfcc_inputs(dev_mixture_datasets[m], dev_weighted_mfccs, flatten=False)
    y_dev = get_mfcc_mix_targets(dev_mixture_datasets[m])
    
    X_train = order_dataset_by_norm(X_train)
    X_dev = order_dataset_by_norm(X_dev)
    
    history = [99999999.]  
    patience_level = 0     
    still_training = True

    epochs = 75
    for e in range(epochs):
        if patience_level < 10:
            res = mfcc_lstm_models_residual[m].fit(X_train, y_train,
                                                   validation_data=[X_dev, y_dev],
                                                   batch_size=200, epochs=1, verbose=False)
            shuffle_order(X_train)
            loss = res.history['loss'][0]
            val_loss = res.history['val_loss'][0]
            print(f"Epoch: {e} ... loss: {loss} ... val loss: {val_loss}", end="\r")

            best_result = min(history)
            if val_loss < best_result:
                patience_level = 0
                mfcc_lstm_models_residual[m].save(f"{output_path}/mfcc/lstm_residual/best_{m}.pkl")
            else:
                patience_level += 1
            history.append(val_loss)
        if patience_level >= 10 or e == epochs-1:
            if still_training:
                print(f"Epoch: {e} ... loss: {loss} ... val loss: {best_result}")
            still_training=False
