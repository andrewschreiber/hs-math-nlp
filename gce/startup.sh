#!/bin/bash

echo "Running startup script"

# gsutil cp gs://math-checkpoints-data/mathematics_dataset-v1.0.tar.gz dataset.zip

# tar xvzf dataset.zip

# pip install --user tensorboard tensorboardX

# python3 training.py

cd /hs-math-nlp-master

python --version

python gce/print.py

echo "try unbuffered"

python -u gce/print.py

echo "Completed startup script"