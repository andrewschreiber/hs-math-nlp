#!/bin/bash

wget https://transfer.sh/12N6Ki/hs-math-nlp.zip
unzip hs-math-nlp.zip

cd hs-math-nlp
wget https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz

tar xvzf mathematics_dataset-v1.0.tar.gz

pip install --user tensorboard tensorboardX

python training.py
