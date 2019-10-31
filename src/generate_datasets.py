#python generate_datasets.py --db_path=/data/jrgillick/projects/assisted_orchestration/OrchDB/OrchDB_flat \
#--num_parallel_processes=20

import util
from features import *
import os, sys, pickle, argparse, time, librosa, numpy as np
from collections import defaultdict
from sklearn.utils import shuffle

# Import different progress bar depending on environment
# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


parser = argparse.ArgumentParser()

# Path to the OrchDB database of individually recorded instrument samples
parser.add_argument('--db_path', type=str, required=True)

# Path where the generated dataset will be stored. Default is the generated_data folder.
# Folders named 'train', 'dev', and 'test' will be created inside here, if they don't exist.
parser.add_argument('--generated_dataset_path', type=str, default='generated_data')

# Number of parallel processes to use when pre-computing features.
# Our server has 20 cpu's, so we use '20. 
# This speeds up computation a lot.
parser.add_argument('--num_parallel_processes', type=str, default='1')
    
parser.add_argument('--num_train_datapoints', type=str, default='7500')
parser.add_argument('--num_dev_datapoints', type=str, default='2000')
parser.add_argument('--num_test_datapoints', type=str, default='2000')

args = parser.parse_args()

db_path = args.db_path
generated_dataset_path = args.generated_dataset_path
num_processes = int(args.num_parallel_processes)
    
num_train_datapoints = int(args.num_train_datapoints)
num_dev_datapoints = int(args.num_dev_datapoints)
num_test_datapoints = int(args.num_test_datapoints)


print(f"Using dataset at {db_path}")
print(f"Generating files to {generated_dataset_path}")
print(f"Using {num_processes} parallel processes")

os.system(f"mkdir -p {generated_dataset_path}/train {generated_dataset_path}/dev {generated_dataset_path}/test")

##################################################################
###### Step 1. Load files and make train/dev/test splits. ######
##################################################################

# Load the filepaths from the (OrchDB) Dataset of instrument samples
all_wav_files = librosa.util.find_files(db_path)
print(f"Found {len(all_wav_files)} audio samples")

all_wav_files = shuffle(all_wav_files, random_state=0)

# Make training splits across all files
# Don't partition by different instruments or pitches
split_1 = int(0.8 * len(all_wav_files))
split_2 = int(0.9 * len(all_wav_files))

train_files = np.array(all_wav_files[0:split_1])
test_files = np.array(all_wav_files[split_1:split_2])
dev_files = np.array(all_wav_files[split_2:])

print(f"Using {len(train_files)} samples for training, {len(dev_files)} for dev, and {len(test_files)} for test.")

train_signals = np.array(parallel_load_audio_batch(train_files, n_processes=num_processes))
dev_signals = np.array(parallel_load_audio_batch(dev_files, n_processes=num_processes))
test_signals = np.array(parallel_load_audio_batch(test_files, n_processes=num_processes))

##################################################################
###### Step 2. Precompute features of all source files. ######
##################################################################

print("Computing RMS Energies...")
train_energies = get_all_energies(train_signals)
dev_energies = get_all_energies(dev_signals)
test_energies = get_all_energies(test_signals)

with open(f"{generated_dataset_path}/train/train_energies.pkl", "wb") as f:
    pickle.dump(train_energies, f)

with open(f"{generated_dataset_path}/dev/dev_energies.pkl", "wb") as f:
    pickle.dump(dev_energies, f)

with open(f"{generated_dataset_path}/test/test_energies.pkl", "wb") as f:
    pickle.dump(test_energies, f)

print("Computing Energy-Weighted FFTS...")
train_weighted_ffts = get_all_weighted_ffts(train_signals, n_processes=num_processes)
dev_weighted_ffts = get_all_weighted_ffts(dev_signals, n_processes=num_processes)
test_weighted_ffts = get_all_weighted_ffts(test_signals, n_processes=num_processes)

with open(f"{generated_dataset_path}/train/train_weighted_ffts.pkl", "wb") as f:
    pickle.dump(train_weighted_ffts, f)

with open(f"{generated_dataset_path}/dev/dev_weighted_ffts.pkl", "wb") as f:
    pickle.dump(dev_weighted_ffts, f)

with open(f"{generated_dataset_path}/test/test_weighted_ffts.pkl", "wb") as f:
    pickle.dump(test_weighted_ffts, f)
    
print("Computing Energy-Weighted MFCCS...")
train_weighted_mfccs = get_all_weighted_mfccs(train_signals, n_processes=num_processes)
dev_weighted_mfccs = get_all_weighted_mfccs(dev_signals, n_processes=num_processes)
test_weighted_mfccs = get_all_weighted_mfccs(test_signals, n_processes=num_processes)

with open(f"{generated_dataset_path}/train/train_weighted_mfccs.pkl", "wb") as f:
    pickle.dump(train_weighted_mfccs, f)

with open(f"{generated_dataset_path}/dev/dev_weighted_mfccs.pkl", "wb") as f:
    pickle.dump(dev_weighted_mfccs, f)

with open(f"{generated_dataset_path}/test/test_weighted_mfccs.pkl", "wb") as f:
    pickle.dump(test_weighted_mfccs, f)
    

##################################################################
###### Step 3. Generate Mixtures and Pre-compute Mixture Features. ######
##################################################################    

# Update this list to try with different numbers of notes mixed together
mixture_values = [2,3,6,12,20,30]

print("Generating datasets...")

root_path = generated_dataset_path

for m in mixture_values:
    print(f"Working on M={m}")
    t0 = time.time()
    
    # Train
    print("Mixing audio files...")
    mixes, components, mixture_coefs = make_mixes(train_signals, m, num_train_datapoints)
    #h = {'mixes': mixes, 'components': components, 'mixture_coefs': mixture_coefs}
    h = {'components': components, 'mixture_coefs': mixture_coefs}
    print("Pre-computing FFT and MFCC features...")
    h['mix_ffts'], h['mix_mfccs'] = get_all_weighted_ffts_and_mfccs(mixes, n_processes=20)
    output_path = f"{root_path}/train/mixture_data_{m}.pkl"
    with open(output_path, 'wb') as f: pickle.dump(h, f)
    #train_mixture_datasets[m] = h
    
    # Dev
    print("Mixing audio files...")
    mixes, components, mixture_coefs = make_mixes(dev_signals, m, num_dev_datapoints)
    #h = {'mixes': mixes, 'components': components, 'mixture_coefs': mixture_coefs}
    h = {'components': components, 'mixture_coefs': mixture_coefs}
    print("Pre-computing FFT and MFCC features...")
    h['mix_ffts'], h['mix_mfccs'] = get_all_weighted_ffts_and_mfccs(mixes, n_processes=20)
    output_path = f"{root_path}/dev/mixture_data_{m}.pkl"
    with open(output_path, 'wb') as f: pickle.dump(h, f)
    #dev_mixture_datasets[m] = h

    # Test
    print("Mixing audio files...")
    mixes, components, mixture_coefs = make_mixes(test_signals, m, num_test_datapoints)
    #h = {'mixes': mixes, 'components': components, 'mixture_coefs': mixture_coefs}
    h = {'components': components, 'mixture_coefs': mixture_coefs}
    print("Pre-computing FFT and MFCC features...")
    h['mix_ffts'], h['mix_mfccs'] = get_all_weighted_ffts_and_mfccs(mixes, n_processes=20)
    output_path = f"{root_path}/test/mixture_data_{m}.pkl"
    with open(output_path, 'wb') as f: pickle.dump(h, f)
    #test_mixture_datasets[m] = h
    
    print(f"Finished in {time.time()-t0} seconds.")


