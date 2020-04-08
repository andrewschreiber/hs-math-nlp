#!/bin/bash

echo "Running prestartup script..."       

echo "Sleep to wait for creation of user directory"
sleep 45

cd /home/andrew_schreiber1

wget https://github.com/andrewschreiber/hs-math-nlp/archive/master.zip

unzip master.zip

sudo chmod -R 777 hs-math-nlp-master

cd hs-math-nlp-master

# gsutil cp gs://math-checkpoints-data/mathematics_dataset-v1.0.tar.gz dataset.zip

# For speed
gsutil cp gs://math-checkpoints-data/mini_mathematics_dataset-v1.0.tar.gz dataset.zip

tar xvzf dataset.zip

su - andrew_schreiber1 -c '/home/andrew_schreiber1/hs-math-nlp-master/gce/startup.sh'
