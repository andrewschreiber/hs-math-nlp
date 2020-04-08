#!/bin/bash

# Assumes the latest github version is synced to the bucket

echo "Running startup script"

wget https://github.com/andrewschreiber/hs-math-nlp/archive/master.zip

unzip master.zip

cd hs-math-nlp-master && ls

# gsutil cp gs://math-checkpoints-data/mathematics_dataset-v1.0.tar.gz dataset.zip

# tar xvzf dataset.zip

# pip install --user tensorboard tensorboardX

# python3 training.py

python --version

python3 --version

python3 -u gce/print.py

echo "Completed startup script"