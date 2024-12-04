#!/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -O iris.csv
python3 convert_iris.py
sudo rm iris.csv
if [ ! -d "data" ]; then
    mkdir data/
fi
mv iris.txt data/

