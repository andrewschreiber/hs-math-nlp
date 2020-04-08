#!/bin/bash

echo "Running prestartup script..."       

wget https://github.com/andrewschreiber/hs-math-nlp/archive/master.zip

unzip master.zip

cd hs-math-nlp-master && ls

gsutil cp gs://math-checkpoints-data/mathematics_dataset-v1.0.tar.gz dataset.zip

tar xvzf dataset.zip

chmod +x gce/startup.sh
mkdir runs
mkdir checkpoints

su - andrew_schreiber1 -c '/hs-math-nlp-master/gce/startup.sh'
