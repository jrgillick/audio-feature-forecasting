# audio-feature-forecasting

This repository contains code for estimating features of mixed sounds using the pre-computed features of individual sound sources. The code accompanies this paper:
*  Jon Gillick, Carmine-Emanuele Cella, and David Bamman, "[Estimating Unobserved Audio Features for Target-Based Orchestration](http://archives.ismir.net/ismir2019/paper/000021.pdf)", ISMIR 2019. 


This code was developed using the [OrchDB](http://www.carminecella.com/orchidea#datasets) dataset of individual instrument samples.  This task of predicting the way that signals mix together can be useful particularly in the context of searching for ways to automatically orchestrate or layer sounds together.

To reproduce the experiments in the paper:

1. Download the [OrchDB](http://www.carminecella.com/orchidea#datasets) dataset.

2. Generate datasets by randomly combining individual notes together using [generate_datasets.py](src/generate_datasets.py). The script generates notes for a few different fixed values of "M" (2,3,6,12,20,20), where M in the number of notes in each mixture. You can change this value in the script. Precomputed features (energy-weight FFT or MFCC) will be computed for each note in the data. To use other features, you can replace the ones defined in [features.py](src/features.py).

`python generate_datasets.py --db_path=OrchDB/OrchDB_flat --generated_dataset_path=generated_data --num_train_datapoints=20000 --num_parallel_processes=<your_number_of_parallel_processes> `

3. Train and save models for prediction using [train.py](src/train.py).

`python train.py --data_path=generated_data --output_path=saved_models`

4. You can generate the plots and results from the paper in the [Analysis](src/Analysis.ipynb) notebook. We've improved the results for the FFT prediction task a bit quite a bit beyond those reported in the paper by generating more data in step 3.
