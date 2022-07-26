#!/usr/bin/env python3

import pickle

data = pickle.load(open("mnist.pickle", "rb"))
data = [item.tobytes() for item in (*data[0], *data[1])]
assert len(data[0]) == 60000 * 28 * 28
assert len(data[1]) == 60000
assert len(data[2]) == 10000 * 28 * 28
assert len(data[3]) == 10000
with open("mnist.bin", "wb") as mnist:
    for item in data:
        mnist.write(item)
