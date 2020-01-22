#!/bin/bash

wget https://transfer.sh/12N6Ki/hs-math-nlp.zip
unzip hs-math-nlp.zip

cd hs-math-nlp
wget https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz

tar xvzf mathematics_dataset-v1.0.tar.gz

pip install --user tensorboard tensorboardX

python training.py



# Attempt 2
# Assumes the latest github version is synced to the bucket

gsutil cp gs://math-checkpoints-data/hs-math-nlp-master.zip math.zip

unzip math.zip

cd hs-math-nlp-master

gsutil cp gs://math-checkpoints-data/mathematics_dataset-v1.0.tar.gz dataset.zip

tar xvzf dataset.zip

pip install --user tensorboard tensorboardX

python training.py
