#!/bin/bash

echo "~~~~~~~ Running startup.sh ~~~~~~~"

cd /home/andrew_schreiber1/hs-math-nlp-master

pip install --user tensorboard tensorboardX

python --version
echo "~~~~~~~ Start training ~~~~~~~"

# Nohup + & allow the python execution to run async
# Key because preemption will insta-kill the startup script
# nohup python training.py &

echo "python training.py" | at now + 1 minute

echo "~~~~~~~ Completed startup script ~~~~~~~"