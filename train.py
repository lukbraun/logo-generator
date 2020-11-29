import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, Input, Conv2DTranspose, Reshape, Conv2D, MaxPool2D, Flatten
from keras.models import Model, Sequential
from keras.activations import sigmoid
from keras.datasets import mnist
from keras.optimizers import Adam
from tqdm import tqdm
from os import listdir
from functools import reduce
from PIL import Image
# %matplotlib inline

TEST_PATH = "train_and_test/test/"
TRAIN_PATH = "train_and_test/train/"
# Maxmum Number of Images per company
COMPANY_COUNT_MAX = 2
HEIGHT_AND_WIDTH = 64

def load_data_from_path(path):
    """
    Load all Images from a given Path.
    The Path must be structured like so:
    dataset/(train|test)/Category/Company/image
    """
    imgs = list()
    category = list()
    for directory in listdir(path):
        company_count = 0
        for company in listdir(path + directory):
            if company_count > COMPANY_COUNT_MAX:
                break
            for img in listdir(path + directory + "/" + company):
                im = Image.open(path + directory + "/" + company + "/" + img)
                # 256 * 256 to 128 * 238
                resizedImage = im.resize((HEIGHT_AND_WIDTH, HEIGHT_AND_WIDTH));
                imgs.append(np.asarray(resizedImage))
                category.append(directory)
            company_count = company_count + 1
    return (np.array(imgs), category)

def load_data():
    (train_data, train_labels) = load_data_from_path(TRAIN_PATH)
    return (train_data, train_labels)

def check_categories(train_labels):
    """
    Inspects how many Items per category there are
    """
    def count(acc, element):
        if element in acc:
            acc[element] = acc[element] + 1
        else:
            acc[element] = 1
        return acc
    print(reduce(count, train_labels, {}))

def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

##############
# Generator  #
##############

def create_generator():
    generator = Sequential()
    nodes = HEIGHT_AND_WIDTH ** 2 * 4
    generator.add(Dense(nodes, input_dim=16))
    generator.add(Reshape((HEIGHT_AND_WIDTH,HEIGHT_AND_WIDTH,4)))

    generator.add(Conv2DTranspose(HEIGHT_AND_WIDTH, (4,4), strides=(2,2), activation="relu"))
    generator.add(Conv2DTranspose(HEIGHT_AND_WIDTH, (4,4), strides=(2,2), activation="relu"))
    generator.add(Conv2D(3,(136,136),activation="sigmoid"))
    # generator.add(Conv2D(3,(132,132),activation="sigmoid"))
    generator.add(MaxPool2D(2, padding='same'))

    generator.compile(loss='binary_crossentropy',optimizer=adam_optimizer())
    return generator

#################
# Discriminator #
#################

def create_discriminator():
    discriminator = Sequential()
    discriminator.add(Input(shape=(HEIGHT_AND_WIDTH,HEIGHT_AND_WIDTH,3)))
    discriminator.add(Conv2D(HEIGHT_AND_WIDTH/4, (3, 3)))
    discriminator.add(MaxPool2D(2, padding='same'))
    discriminator.add(Conv2D(HEIGHT_AND_WIDTH/4, (3, 3)))
    discriminator.add(MaxPool2D(2, padding='same'))
    discriminator.add(Flatten())
    discriminator.add(Dense(1,activation="sigmoid"))

    discriminator.compile(loss='categorical_crossentropy', optimizer=adam_optimizer())
    return discriminator

#######
# GAN #
#######

def create_gan(dis,gen):
    dis.trainable = False
    gan_input = Input(shape=(16,))
    x = gen(gan_input)
    gan_output = dis(x)
    gan = Model(inputs=gan_input,outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer="adam")
    return gan


def plot_generated_images(epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):
    noise = np.random.normal(0,1,[batch_size, 16])
    generated_images = generator.predict(noise)
    print(generated_images.shape)
    generated_images = generated_images.reshape(100,28,28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image-%d.png' %epoch)
    print("saved")

def training(epochs=1, batch_size=128):
    (x_train, x_label) = load_data()
    batch_count = x_train.shape[0] / batch_size

    generator = create_generator()
    discriminator = create_discriminator()
    discriminator.summary()
    generator.summary()
    gan = create_gan(gen=generator, dis=discriminator)

    for e in range(1, epochs+1):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
            noise = np.random.normal(0,1,[batch_size, 16])
            generated_images = generator.predict(noise)
            image_batch = x_train[np.random.randint(low=0,high=x_train.shape[0],size=batch_size)]
            x = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2*batch_size)
            y_dis[:batch_size] = 0.9
            discriminator.trainable = True
            discriminator.train_on_batch(x, y_dis)
            noise = np.random.normal(0,1, [batch_size, 16])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)
        if e == 1 or e % 20 == 0:
            plot_generated_images(e, generator)

training(20,1)
