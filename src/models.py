from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, LSTM, Bidirectional
import keras.optimizers

def mfcc_to_mfcc_model(n_components, loss_fn='mean_squared_error'):
    model = Sequential()
    model.add(Dense(250, use_bias=True,input_dim=19*n_components))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(19, use_bias=True))
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss_fn)
    return model

def mfcc_to_mfcc_model_two_layer(n_components, loss_fn='mean_squared_error'):
    model = Sequential()
    model.add(Dense(450, use_bias=True,input_dim=19*n_components))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(250, use_bias=True,input_dim=19*n_components))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(19, use_bias=True))
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss_fn)
    return model

def mfcc_to_mfcc_lstm_model(n_components, loss_fn='mean_squared_error'):
    model = Sequential()
    model.add(Bidirectional(LSTM(250,input_shape=(n_components,19),return_sequences=False,dropout=0.05),input_shape=(n_components,19)))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Activation("relu"))
    model.add(Dense(19, use_bias=True))
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss_fn)
    return model

def get_mfcc_mix_targets(dataset):
    return np.array(dataset['mix_mfccs'])

def get_mfcc_inputs(dataset, source_mfccs, flatten=True):
    datapoints = []
    component_lists = dataset['components']
    for component_indices in component_lists:
        mfcc_vectors = np.vstack([source_mfccs[i] for i in component_indices])
        if flatten:
            mfcc_vectors = mfcc_vectors.reshape(-1)
        datapoints.append(mfcc_vectors)
    return np.array(datapoints)

def fft_to_fft_model(n_components, loss_fn='mean_squared_error'):
    model = Sequential()
    model.add(Dense(200, use_bias=True,input_dim=1025*n_components))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(1025, use_bias=True))
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss_fn)
    return model

def fft_to_fft_model_two_layer(n_components, loss_fn='mean_squared_error'):
    model = Sequential()
    model.add(Dense(450, use_bias=True,input_dim=1025*n_components))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(200, use_bias=True,input_dim=1025*n_components))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(1025, use_bias=True))
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss_fn)
    return model

def fft_to_fft_lstm_model(n_components, loss_fn='mean_squared_error'):
    model = Sequential()
    model.add(Bidirectional(LSTM(250,input_shape=(n_components,1025),return_sequences=False,dropout=0.5),input_shape=(n_components,1025)))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Activation("relu"))
    model.add(Dense(1025, use_bias=True))
    model.add(Activation("relu"))
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss_fn)
    return model

def fft_to_fft_lstm_model_together(n_components=None, loss_fn='mean_squared_error'):
    model = Sequential()
    #model.add(LSTM(500,input_shape=(None,1025),return_sequences=False,dropout=0.5))
    model.add(Bidirectional(LSTM(250,input_shape=(None,1025),return_sequences=False,dropout=0.35),input_shape=(None,1025)))
    #model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.35))
    model.add(Activation("relu"))
    model.add(Dense(1025, use_bias=True))
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss_fn)
    return model

def fft_to_mfcc_model(n_components, loss_fn='mean_squared_error'):
    model = Sequential()
    model.add(Dense(200, use_bias=True,input_dim=1025*n_components))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(19, use_bias=True))
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss_fn)
    return model

def qwert():
    return 5