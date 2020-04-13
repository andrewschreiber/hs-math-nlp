#!/bin/bash

echo "~~~~~~~ Running startup.sh ~~~~~~~~"       
# TODO: Figure out how to get proper permissions for writing to /dev/ttyS0
# sudo adduser andrew_schreiber1 dialout
# sudo usermod -a -G dialout andrew_schreiber1
# sudo chmod +x /dev/ttyS0

# TODO: Find better way than favorable race condition to do this.
echo "Sleeping to wait for creation of user directory"
sleep 5

sudo apt-get install -y at

# So we can change serial port

# Important to put things in the correct folder and chmod for permissions
cd /home/andrew_schreiber1

wget https://github.com/andrewschreiber/hs-math-nlp/archive/master.zip

unzip master.zip

sudo chmod -R 777 hs-math-nlp-master

cd hs-math-nlp-master

# gsutil cp gs://math-checkpoints-data/mathematics_dataset-v1.0.tar.gz dataset.zip
 
# 10kb dataset for faster testing
gsutil cp gs://math-checkpoints-data/mini_mathematics_dataset-v1.0.tar.gz dataset.zip

tar xvzf dataset.zip

# We need to use the user account, as root does not have imaged python packages
su - andrew_schreiber1 -c '/home/andrew_schreiber1/hs-math-nlp-master/gce/training.sh'
