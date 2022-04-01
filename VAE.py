import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, concatenate, Add
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, LeakyReLU
from tensorflow.keras.activations import swish, elu, softmax
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse, binary_crossentropy, mae

import matplotlib.pyplot as plt
import matplotlib.cm, matplotlib.colors


import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


DIR = 'outputs/'

# -- load data
age = np.load('eICU_age.npy')
age = age.astype('float32')


 # -- normalize data
def normalize(data):
    
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    data = (data - mean)/std
    
    return data

# -- denormalize data
def denormalize(data):
    
    mean = np.mean(age, axis = 0)
    std = np.std(age, axis = 0)
    data = data*std + mean
    
    return data

# -- Global variables

intermediate_dim = 64
latent_dim = 4
image_size = 1
original_dim = 1
epochs = 300
batch_size = 128


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1] # Returns the shape of tensor or variable as a tuple of int or None entries.
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# VAE model = encoder + decoder
# build encoder model
def encoder_model(inputs):
    x = Dense(512, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder, z_mean, z_log_var


# build decoder model
def decoder_model():
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(16, activation='relu')(latent_inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(original_dim)(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder


if __name__ == "__main__":


    x_train = age
    input_shape = (original_dim, )
    inputs = Input(shape=input_shape, name='encoder_input')
    encoder, z_mean, z_log_var = encoder_model(inputs)

    decoder = decoder_model()
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')


    reconstruction_loss = mae(inputs, outputs)
    # reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    opt = Adam(lr=1e-3)
    vae.compile(optimizer=opt)
    
    vae.fit(x_train, epochs=epochs, batch_size=batch_size)
    latent = encoder.predict(x_train)[2]
    np.save(DIR+'latent.npy', latent)
    