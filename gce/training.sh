#!/bin/bash

echo "~~~~~~~ Running startup.sh ~~~~~~~"

cd /home/andrew_schreiber1/hs-math-nlp-master

sudo apt-get install at
pip install --user tensorboard tensorboardX

python --version
echo "~~~~~~~ Start training ~~~~~~~"

# Use 'at' to allow the python execution to run async
# Key because preemption will insta-kill the startup script
# nohup does not work

echo "python training.py" | at now

echo "~~~~~~~ Completed startup script ~~~~~~~"