#!/bin/bash

echo "Running prestartup script..."
echo "....................................."
wget https://github.com/andrewschreiber/hs-math-nlp/archive/master.zip

unzip master.zip

cd hs-math-nlp-master && ls

# echo "Sleeping for 10s for install"
# sleep 60

chmod +x gce/startup.sh

su - andrew_schreiber1 -c '/hs-math-nlp-master/gce/startup.sh'

# andrew_schreiber1
# bash gce/startup.sh