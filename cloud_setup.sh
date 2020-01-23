#!/bin/bash

# Assumes the latest github version is synced to the bucket

gsutil cp gs://math-checkpoints-data/hs-math-nlp-master.zip math.zip

unzip math.zip

cd hs-math-nlp-master

gsutil cp gs://math-checkpoints-data/mathematics_dataset-v1.0.tar.gz dataset.zip

tar xvzf dataset.zip

pip install --user tensorboard tensorboardX

python training.py
