#!/bin/bash

echo "~~~~~~~ Running training.sh ~~~~~~~"

cd /home/andrew_schreiber1/hs-math-nlp-master

pip install --user tensorboard tensorboardX

python --version
echo "~~~~~~~ Start training ~~~~~~~"
tty


# Use 'at' to allow the python execution to run async
# Needed because preemption will insta-kill the startup script
# nohup still runs under startup script

echo "python training.py" >  /dev/ttyS0 | at now

echo "~~~~~~~ Completed startup script ~~~~~~~"