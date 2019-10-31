import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

#import tensorflow as tf
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#if tf.test.gpu_device_name():
#    print('GPU found')
#else:
#    print("No GPU found")
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)

import numpy as np
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, LSTM, Bidirectional, Lambda
import keras.optimizers
import keras.backend as K

K.set_session(session)

from keras.models import Model, Input
from keras.layers.merge import add

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
    model.add(Dense(300, use_bias=True,input_dim=1025*n_components))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.8))
    model.add(Dense(1025, use_bias=True))
    model.add(Activation("relu"))
    model.add(Lambda(max_norm))
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss_fn)
    return model

def fft_to_fft_model_two_layer(n_components, loss_fn='mean_squared_error'):
    model = Sequential()
    model.add(Dense(600, use_bias=True,input_dim=1025*n_components))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(300, use_bias=True,input_dim=1025*n_components))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(1025, use_bias=True))
    model.add(Activation("relu"))
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss_fn)
    return model

def clip_to_0_1(x):
    return K.clip(x, 0., 1.)

def max_norm(x):
    max_vals = K.expand_dims(K.max(x, axis=1),1)
    return x / max_vals

"""
def fft_to_fft_lstm_model(n_components, loss_fn='mean_squared_error'):
    model = Sequential()
    model.add(LSTM(300,input_shape=(n_components,1025),return_sequences=False,dropout=0.3))
    #model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))
    #model.add(Activation("relu"))
    model.add(Dense(1025, use_bias=True))
    model.add(Activation("sigmoid"))
    #model.add(Lambda(clip_to_0_1))
    #model.add(Lambda(max_norm))
    #model.add(Activation("relu"))
    #optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss_fn)
    return model
"""

def fft_to_fft_residual_inputs_lstm_model(n_components, loss_fn='mean_squared_error'):
    inputs = Input(shape=(n_components,1025))
    x = LSTM(units=1025, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
    x = add([x, inputs])
    x = LSTM(units=300, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)
    x = Dense(1025, use_bias=True)(x)
    x = Activation('relu')(x)
    x = Lambda(max_norm)(x)
    model = Model(inputs, x)
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss_fn)
    return model

def mfcc_to_mfcc_residual_inputs_lstm_model(n_components, loss_fn='mean_squared_error'):
    inputs = Input(shape=(n_components,19))
    x = LSTM(units=19, return_sequences=True, dropout=0.1)(inputs)
    x = add([x, inputs])
    x = Bidirectional(LSTM(units=250, return_sequences=False, dropout=0.1))(x)
    x = Dense(19, use_bias=True)(x)
    #x = Activation('relu')(x)
    #x = Lambda(max_norm)(x)
    model = Model(inputs, x)
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss_fn)
    return model

def fft_to_fft_lstm_model(n_components, loss_fn='mean_squared_error'):
    inputs = Input(shape=(n_components,1025))
    x = LSTM(units=300, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(inputs)
    #x = keras.layers.BatchNormalization()(x)
    x = Dense(1025, use_bias=True)(x)
    x = Activation('relu')(x)
    x = Lambda(max_norm)(x)
    model = Model(inputs, x)
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



