import numpy as np

def get_mfcc_inputs(dataset, source_ffts, flatten=True):
    datapoints = []
    component_lists = dataset['components']
    for component_indices in component_lists:
        fft_vectors = np.vstack([source_ffts[i] for i in component_indices])
        if flatten:
            fft_vectors = fft_vectors.reshape(-1)
        datapoints.append(fft_vectors)
    return np.array(datapoints)

def get_mfcc_mix_targets(dataset):
    return np.array(dataset['mix_mfccs'])

def get_fft_mix_targets(dataset):
    return np.array(dataset['mix_ffts'])

def get_fft_inputs(dataset, source_ffts, flatten=True):
    datapoints = []
    component_lists = dataset['components']
    for component_indices in component_lists:
        fft_vectors = np.vstack([source_ffts[i] for i in component_indices])
        if flatten:
            fft_vectors = fft_vectors.reshape(-1)
        datapoints.append(fft_vectors)
    return np.array(datapoints)

# shuffles the order of lstm steps in the entire data set
def shuffle_order(X):
    for i in range(len(X)):
        np.random.shuffle(X[i])

# orders by norm for a single 2D matrix
def order_by_norm(X):
    norm_order = np.argsort(np.linalg.norm(X, axis=1))
    new_X = np.zeros_like(X)
    for i in range(len(new_X)):
        new_X[i] = X[norm_order[i]]
    return new_X
        
# orders lstm steps by norm for the entire data set
def order_dataset_by_norm(X):
    new_X = np.zeros_like(X)
    for i in range(len(X)):
        new_X[i] = order_by_norm(X[i])
    return new_X
        
def apply_max_norm(data):
    for i in range(len(data)):
        data[i] = data[i]/max(data[i])
        
# Train all lstm lengths together in one model with padded inputs
def get_padded_mfcc_inputs_and_targets(datasets_hash, source_mfccs):
    max_steps = max(mixture_values)
    padded_datasets = []
    targets = []
    for m in mixture_values:
        # get inputs
        d = datasets_hash[m]
        inputs = copy.deepcopy(get_mfcc_inputs(d, source_mfccs, flatten=False))
        padded = np.zeros((inputs.shape[0],max_steps,inputs.shape[2]))
        padded[:inputs.shape[0], :inputs.shape[1], :inputs.shape[2]] = inputs
        padded_datasets.append(padded)
        
        #get targets
        targets.append(copy.deepcopy(get_mfcc_mix_targets(datasets_hash[m])))
        
    return np.vstack(padded_datasets), np.vstack(targets)

# Train all lstm lengths together in one model with padded inputs
def get_padded_fft_inputs_and_targets(datasets_hash, source_ffts):
    max_steps = max(mixture_values)
    padded_datasets = []
    targets = []
    for m in mixture_values:
        # get inputs
        d = datasets_hash[m]
        inputs = copy.deepcopy(get_fft_inputs(d, source_ffts, flatten=False))
        #padded = np.zeros((inputs.shape[0],max_steps,inputs.shape[2]))
        #padded[:inputs.shape[0], :inputs.shape[1], :inputs.shape[2]] = inputs
        #padded_datasets.append(padded)
        #padded_datasets += list(padded)
        padded_datasets.append(inputs)
        
        #get targets
        targets.append(copy.deepcopy(get_fft_mix_targets(datasets_hash[m])))
        
    #return np.vstack(padded_datasets), np.vstack(targets)
    return padded_datasets, targets

def lstm_batch_generator(datasets, targets, batch_size=100, shuffle_sequence=False):
    # each dataset has the right shape
    while True:
        for d, t in zip(datasets, targets):
            i = 0
            while i < len(d):
                batch_inputs = d[i:i+batch_size]
                if shuffle_sequence:
                    shuffle_order(batch_inputs)
                batch_targets = t[i:i+batch_size]
                #print(batch_inputs[0].shape)
                i += batch_size
                yield(batch_inputs, batch_targets)
            #padded = np.zeros((inputs.shape[0],max_steps,inputs.shape[2]))
            
def lstm_batch_generator_sampling(datasets, targets, batch_size=100, shuffle_sequence=False):
    # each dataset has the right shape
    counter = 0
    # reshuffle each dataset every time this function gets called or after 500 batches
    for d, t in zip(datasets, targets):
        d, t = shuffle(d, t)
    while True:
        counter += 1
        # reshuffle each dataset after 500 batches
        if counter % 500 == 0:
            for d, t in zip(datasets, targets):
                d, t = shuffle(d, t)
        dataset_index = np.random.randint(len(datasets))
        d = datasets[dataset_index]
        t = targets[dataset_index]
        i = np.random.randint(len(d)-batch_size)
        batch_inputs = d[i:i+batch_size]
        if shuffle_sequence:
            shuffle_order(batch_inputs)
        batch_targets = t[i:i+batch_size]
        #print(batch_inputs[0].shape)
        #i += batch_size
        yield(batch_inputs, batch_targets)
        #padded = np.zeros((inputs.shape[0],max_steps,inputs.shape[2]))