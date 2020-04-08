#!/bin/bash

echo "Running startup script"

# cd /home/andrew_schreiber1/hs-math-nlp-master

cd /hs-math-nlp-master

pip install --user tensorboard tensorboardX

echo "Start training"

python training.py

echo "Completed startup script"