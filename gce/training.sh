#!/bin/bash

echo "~~~~~~~ Running startup.sh ~~~~~~~"

cd /home/andrew_schreiber1/hs-math-nlp-master

pip install --user tensorboard tensorboardX

python --version
echo "~~~~~~~ Start training ~~~~~~~"

nohup python -u training.py &

echo "~~~~~~~ Completed startup script ~~~~~~~"