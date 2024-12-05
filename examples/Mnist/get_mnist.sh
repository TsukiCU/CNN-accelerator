#!/bin/bash

if [ ! -d "data" ]; then
    mkdir data/
fi

# Will run into 403 Forbidden for some reasons. Fuck.

wget https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

if [ ! -e "*.gz" ]; then
    echo -e "\e[1;31mWe've run into some trouble downloading. Using another source...\n\e[0m"
    wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
    wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
fi

if [ -e "t10k-images-idx3-ubyte.gz" ]; then
    gunzip t10k-images-idx3-ubyte.gz
    mv t10k-images-idx3-ubyte data/
fi

if [ -e "t10k-labels-idx1-ubyte.gz" ]; then
    gunzip t10k-labels-idx1-ubyte.gz
    mv t10k-labels-idx1-ubyte data/
fi

if [ -e "train-images-idx3-ubyte.gz" ]; then
    gunzip train-images-idx3-ubyte.gz
    mv train-images-idx3-ubyte data/
fi

if [ -e "train-labels-idx1-ubyte.gz" ]; then
    gunzip train-labels-idx1-ubyte.gz
    mv train-labels-idx1-ubyte data/
fi