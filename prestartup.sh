#!/bin/bash

echo "Running prestartup script"
wget https://github.com/andrewschreiber/hs-math-nlp/archive/master.zip

unzip master.zip

cd hs-math-nlp-master && ls

echo "Sleeping for 60s"
sleep 60

chmod +x gce/startup.sh
bash gce/startup.sh