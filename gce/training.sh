#!/bin/bash

echo "~~~~~~~ Running training.sh ~~~~~~~"

cd /home/andrew_schreiber1/hs-math-nlp-master

sudo apt-get install -y at
pip install --user tensorboard tensorboardX

python --version
echo "~~~~~~~ Start training ~~~~~~~"


# Use 'at' to allow the python execution to run async
# Needed because preemption will insta-kill the startup script
# nohup still runs under startup script

echo "python training.py" | at now

echo "~~~~~~~ Completed startup script ~~~~~~~"