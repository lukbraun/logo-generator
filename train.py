import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from tqdm import tqdm
from os import listdir
from functools import reduce
from PIL import Image
# %matplotlib inline

TRAIN_PATH = "train_and_test/train/"
# Maxmum Number of Images per company
COMPANY_COUNT_MAX = 10

def load_data(path):
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
                imgs.append(np.asarray(Image.open(path + directory + "/" + company + "/" + img)))
                category.append(directory)
            company_count = company_count + 1
    return (np.array(imgs), category)

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

(train_data, train_labels) = load_data(TRAIN_PATH)
check_categories(train_labels)
print(train_data.shape)

def create_generator():
    generator = Sequential()
    generator.add(Dense(units=256, input_dim=100))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=256*256*3, activation='tanh'))
    generator.add(LeakyReLU(0.2))

    generator.compile(loss='binary_crossentropy')

    return generator

g = create_generator()
g.summary()
