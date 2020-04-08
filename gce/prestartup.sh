#!/bin/bash

echo "Running prestartup script..."       

sleep 20

cd /home/andrew_schreiber1

wget https://github.com/andrewschreiber/hs-math-nlp/archive/master.zip

unzip master.zip

cd hs-math-nlp-master && ls

gsutil cp gs://math-checkpoints-data/mathematics_dataset-v1.0.tar.gz dataset.zip

tar xvzf dataset.zip

chmod +x gce/startup.sh

cd .. 

sudo chmod -R 755 hs-math-nlp-master

su - andrew_schreiber1 -c '/home/andrew_schreiber1/hs-math-nlp-master/gce/startup.sh'
