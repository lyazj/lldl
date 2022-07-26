#!/usr/bin/env python3

from tensorflow.keras.datasets import mnist
import pickle

data = mnist.load_data()
pickle.dump(data, open("mnist.pickle", "wb"))
