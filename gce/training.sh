#!/bin/bash

echo "~~~~~~~ Running training.sh ~~~~~~~"

cd /home/andrew_schreiber1/hs-math-nlp-master

pip install --user tensorboard tensorboardX

python --version
echo "~~~~~~~ Start training ~~~~~~~"

# Use 'at' to allow the python execution to run async
# Needed because preemption will insta-kill the startup script
# nohup still runs under startup script


# Will write logs to /var/spool/mail/andrew_schreiber1
# Also /var/mail/andrew_schreiber1
# echo "python training.py" | at now

# /dev/ttyS0 is blocked via permissions on GCP, but /dev/ttyS1 can be written to
# Logs to serial 2. Connect via: 
# gcloud compute connect-to-serial-port $INSTANCE_NAME --port 2

echo "python training.py >> /dev/ttyS1" | at now

sleep 1

echo "python3 -m tensorboard.main --logdir=runs" | at now

echo "~~~~~~~ Completed startup script ~~~~~~~"