#!/bin/bash

print("Running startup script")

# gsutil cp gs://math-checkpoints-data/mathematics_dataset-v1.0.tar.gz dataset.zip

# tar xvzf dataset.zip

# pip install --user tensorboard tensorboardX

# python3 training.py

python --version

python3 --version

pip freeze

python3 -u gce/print.py

echo "Completed startup script"