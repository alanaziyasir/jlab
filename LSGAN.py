# -- original implementation of lsgan can be found here: https://github.com/eriklindernoren/Keras-GAN/blob/master/lsgan/lsgan.py


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, concatenate, Add
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, LeakyReLU
from tensorflow.keras.activations import swish, elu, softmax
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from matplotlib.legend_handler import HandlerLine2D
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import pylab as pyy

import sys
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# -- outputs directory
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

class LSGAN():
    def __init__(self):
        self.img_shape = (1,)
        self.latent_dim = 100

        optimizer = Adam(0.00001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        generator_noise = Input(shape=(self.latent_dim,))
        img = self.generator(generator_noise)
#         MMD = self.make_MMD()

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(generator_noise, valid )
        # (!!!) Optimize w.r.t. MSE loss instead of crossentropy
#         MMD_loss = self.MMD_loss()
        self.combined.compile(loss='mse', optimizer=optimizer) 
        

    def build_generator(self):
        noise = Input(shape=(100,))
        
        rate = 0.01        
        
        x1 = Dense(512)(noise)
        x1 = BatchNormalization(momentum=0.8)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        
        
        x1 = Dense(512)(x1)
        x1 = BatchNormalization(momentum=0.8)(x1)
        x1 = LeakyReLU(alpha=0.2) (x1)
        
        
        x1 = Dense(512)(x1)
        x1 = BatchNormalization(momentum=0.8)(x1)
        x1 = LeakyReLU(alpha=0.2) (x1)
    
        output = Dense(1)(x1)
        generator = Model(inputs=noise, outputs=output)
        generator.summary()
        return(generator)


    def build_discriminator(self):
        rate = 0.01
        
        vis = Input(shape=(1,))
        
        x1 = Dense(512)(vis)
        #x1 = BatchNormalization(momentum=0.8)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        
        
        x1 = Dense(512)(x1)
        #x1 = BatchNormalization(momentum=0.8)(x1)
        x1 = LeakyReLU(alpha=0.2) (x1)
        
        
        x1 = Dense(512)(x1)
        #x1 = BatchNormalization(momentum=0.8)(x1)
        x1 = LeakyReLU(alpha=0.2) (x1)
        



        
        # (!!!) No softmax
        output = Dense(1)(x1)

        discriminator = Model(inputs=vis, outputs=output)
        discriminator.summary()
        return(discriminator)
    
    

    def train(self, epochs, batch_size=128, sample_interval=50):
        

                  
        print(X_train.shape)



        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

#         self.generator.load_weights('ls_generator.h5')
#         self.discriminator.load_weights('ls_discriminator.h5')
        
        dloss=[]
        gloss=[]
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, size = (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            #valid_y = np.array([1] * batch_size)
            # Train the generator
            #g_loss = self.combined.train_on_batch(noise, [valid_y, imgs])

            # Plot the progress
            #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            #loss_gen.append(g_loss[0])
            #loss_dis.append(d_loss[0])
            

            # If at save interval => save generated image samples
            dloss=np.append(dloss,d_loss[0])
            gloss=np.append(gloss,g_loss)
            if epoch % sample_interval == 0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                self.sample_images(epoch, dloss, gloss)
                gan.generator.save_weights(DIR+'generator.h5')
                gan.discriminator.save_weights(DIR+'discriminator.h5')
                
                
    def sample_images(self, epoch, dloss, gloss):

        SAMPLE_SIZE = age.shape[0]
        noise = np.random.normal(0, 1, size = (SAMPLE_SIZE, 100))
        
    

        results = self.generator.predict(noise, batch_size = 1000)
        results = denormalize(results)

        plt.hist(age, bins=100, histtype='step')
        plt.hist(np.round(results), bins=100, histtype='step')
        #plt.show()
        plt.savefig(DIR+'results_'+str(epoch)+'.png')

        
        nrows,ncols=1,1
        fig = pyy.figure(figsize=(ncols*7,nrows*5))

        ax=pyy.subplot(nrows,ncols,1)
        ax.plot(range(1,len(gloss)+1),gloss,label=r'$\rm generator$')
        ax.plot(range(1,len(dloss)+1),dloss,label=r'$\rm discriminator$')
        ax.semilogy()
        ax.semilogx()
        ax.legend(fontsize=20)
        ax.set_ylabel(r'$\rm Loss$',size=20)
        ax.set_xlabel(r'$\rm epochs$',size=20)
        ax.tick_params(axis='both', which='both', labelsize=15,direction='in')
        plt.savefig(DIR+'loss_'+str(epoch)+'.png')


if __name__ == "__main__":
    
    X_train = age
    gan = LSGAN()
    gan.train(epochs=100000, batch_size=2000, sample_interval=5000)    