import librosa, numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

######### Signal Processing functions

def mix_n_sounds(sounds, coefs=None, parallelize=False):
    if coefs is not None and len(sounds) != len(coefs):
        raise("Need same # of sounds and mixture coefs")
        
    # Zero pad up to length of the longest file
    l = max([len(s) for s in sounds])
    
    if parallelize:
        sounds = Parallel(n_jobs=int(len(sounds)/2))(
            delayed(librosa.util.fix_length)(s, l) for s in sounds)
    
    else:
        sounds = [librosa.util.fix_length(s, l) for s in sounds]
  
    # if mixing coefficients are given, weight the sounds by those coefs
    if coefs is not None:
        weighted_sounds = [sounds[i]*coefs[i] for i in range(len(sounds))]
    else:
        weighted_sounds = sounds
        
    # don't normalize, just divide by N
    return np.sum(weighted_sounds, axis=0)/len(sounds) 

def mean_squared_feature_distance(actual, forecast):    
    return np.sum(np.square(actual-forecast))/len(actual)

def mean_absolute_percentage_error(actual, forecast):
    return np.sum(np.abs ((actual - forecast) / actual)) / len (actual)

def real_stft(y):
    S, phase = librosa.magphase(librosa.stft(y))
    return S

def energy_weighted_fft(y):
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rmse(S=S)
    return np.sum(S*rms, axis=1)/np.sum(rms)
    
def energy_weighted_mfcc(y):
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rmse(S=S)
    mfcc = librosa.feature.mfcc(y, n_mfcc=20)[1:]
    return np.sum(mfcc*rms, axis=1)/np.sum(rms)

def energy_weighted_fft_and_mfcc(y):
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rmse(S=S)
    mfcc = librosa.feature.mfcc(y, n_mfcc=20)[1:]
    weighted_fft = np.sum(S*rms, axis=1)/np.sum(rms)
    weighted_mfcc = np.sum(mfcc*rms, axis=1)/np.sum(rms)
    return weighted_fft, weighted_mfcc

######### Functions for processing lists of files
def unpack_list_of_tuples(list_of_tuples):
    return tuple([list(tup) for tup in list(zip(*list_of_tuples))])

def make_mix(source_signals, n_components, equal_weights=True):
    file_indices = np.random.choice(len(source_signals), n_components, replace=False)
    component_signals = source_signals[file_indices]
    if equal_weights:
        mixture_weights = np.ones(n_components)
    else:
        mixture_weights = np.random.uniform(0,1,n_components)
    mix = mix_n_sounds(component_signals, mixture_weights)
        
    return mix, file_indices, mixture_weights

#def make_mixes(source_signals, n_components, n_files, n_processes, equal_weights=True):
#    results = Parallel(n_jobs=n_processes)(
#        delayed(make_mix)(source_signals, n_components, equal_weights) for i in tqdm(range(n_files)))
#    
#    # returns (mixes, components, mixture_coefs)
#    return unpack_list_of_tuples(results)


def make_mixes(source_signals, n_components, n_files, equal_weights=True):
    mixes = []
    components = []
    mixture_coefs = []
    for i in tqdm(range(n_files)):
        file_indices = np.random.choice(len(source_signals), n_components, replace=False)
        component_signals = source_signals[file_indices]
        if equal_weights:
            mixture_weights = np.ones(n_components)
            mix = mix_n_sounds(component_signals)
        else:
            mixture_weights = np.random.uniform(0,1,n_components)
            mix = mix_n_sounds(component_signals, mixture_weights)

        mixes.append(mix)
        components.append(file_indices)
        mixture_coefs.append(mixture_weights)
    
    return mixes, components, mixture_coefs

# load in a list of wav files, cropping to max_duration
def get_all_signals(files):
    signals = []
    for f in tqdm(files):
        y, sr = librosa.load(f)
        signals.append(y)
    return signals

# librosa.load() but return only the signal, not (y, sr)
def librosa_load_without_sr(f, sr=None):
    return librosa.load(f, sr=sr)[0]

# Runs librosa.load() on a list of files in parallel, returns [y1, y2, ...]
def parallel_load_audio_batch(files, n_processes, sr=None):
    return Parallel(n_jobs=n_processes)(
        delayed(librosa_load_without_sr)(f,sr=sr) for f in tqdm(files))

# Compute STFT and keep the magnitudes for a list of signals
def get_all_specs(signals):
    return [real_stft(signal) for signal in tqdm(signals)]

def get_all_weighted_ffts(signals, n_processes):
    return Parallel(n_jobs=n_processes)(
        delayed(energy_weighted_fft)(signal) for signal in tqdm(signals))
    #return [energy_weighted_fft(signal) for signal in tqdm(signals)]

def get_all_weighted_mfccs(signals, n_processes):
    return Parallel(n_jobs=n_processes)(
        delayed(energy_weighted_mfcc)(signal) for signal in tqdm(signals))
    #return [energy_weighted_mfcc(signal) for signal in tqdm(signals)]
    
def get_all_weighted_ffts_and_mfccs(signals, n_processes):
    results = Parallel(n_jobs=n_processes)(
        delayed(energy_weighted_fft_and_mfcc)(signal) for signal in tqdm(signals))
    ffts, mfccs = unpack_list_of_tuples(results)
    return ffts, mfccs

def get_file_rms(signal):
    return np.sqrt(np.dot(signal, signal))/len(signal)

def get_all_energies(signals):
    return [get_file_rms(signal) for signal in tqdm(signals)]

