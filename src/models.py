import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import numpy as np
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, LSTM, Bidirectional, Lambda
from keras.layers.merge import Concatenate, add, multiply, subtract
from keras.models import Model, Input
import keras.optimizers
import keras.backend as K

K.set_session(session)


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

def keras_get_last_lstm_output(x):
    # x shape is (batch_size, n_components, output_dim)
    x = K.reverse(x, axes=1) # put last timestep first
    x = K.permute_dimensions(x, pattern=(1,0,2)) # put time dim first
    x = K.gather(x, indices=[0])
    x = K.permute_dimensions(x, pattern=(1,0,2)) # permute back
    return x

def squeeze1(x):
    return K.squeeze(x, axis=1)

def split1(x):
    return(x[:,:,0:1])

def split2(x):
    return(x[:,:,1:])

def fft_to_fft_lstm_linear_correction(n_components, loss_fn='mean_squared_error'):
    inputs = Input(shape=(n_components,1025))
    x=inputs
    x = LSTM(units=1025, return_sequences=True, dropout=0.05, recurrent_dropout=0.0)(inputs)
    x = add([x, inputs])
    x = LSTM(units=1025, return_sequences=True, dropout=0.05, recurrent_dropout=0.2)(x)
    weighting_lstm = LSTM(units=100, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(x)

    weighting_projection = LSTM(2, return_sequences=True)(weighting_lstm)
    weighting_projection = Activation('softmax')(weighting_projection)

    lstm_weighting = Lambda(split1)(weighting_projection)
    input_weighting = Lambda(split2)(weighting_projection)
    
    input_weighting = multiply([input_weighting,inputs])
    
    lstm_weighting = multiply([lstm_weighting,x])
    outputs = add([input_weighting, lstm_weighting])
    
    last_output = Lambda(keras_get_last_lstm_output)(outputs)
    last_output = Lambda(squeeze1)(last_output)
    outputs = Activation('relu')(last_output)
    outputs = Lambda(max_norm)(last_output)
    model = Model(inputs, outputs)
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss_fn)
    return model

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
    model.add(Bidirectional(LSTM(250,input_shape=(None,1025),return_sequences=False,dropout=0.35),input_shape=(None,1025)))
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



