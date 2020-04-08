#!/bin/bash

echo "Running prestartup script..."
echo "....................................."
wget https://github.com/andrewschreiber/hs-math-nlp/archive/master.zip

unzip master.zip

cd hs-math-nlp-master && ls

echo "Sleeping for 20s for install"
sleep 20

chmod +x gce/startup.sh
bash gce/startup.sh