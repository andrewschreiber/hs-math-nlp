#!/bin/bash

echo "Running startup script"

cd /hs-math-nlp-master

pip install --user tensorboard tensorboardX

python training.py

echo "Completed startup script"