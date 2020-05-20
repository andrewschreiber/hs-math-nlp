#!/bin/bash
echo "~~~~~~~ Running startup.sh ~~~~~~~~"       

DIRECTORY=/home/andrew_schreiber1

if [ ! -d "$DIRECTORY" ]; then
  # Critical to sleep here, or user-related tasks will fizzle/race
  echo "Sleeping to wait for creation of user directory"
  sleep 1

  # For running async and logging
  sudo apt-get install -y at
  sudo adduser andrew_schreiber1 dialout

  cd $DIRECTORY
  wget https://github.com/andrewschreiber/hs-math-nlp/archive/master.zip
  unzip master.zip
  sudo chmod -R 777 hs-math-nlp-master
  cd hs-math-nlp-master

  # In startup, pytorch-latest-gpu updates metadata to use its shutdown script
  IMAGE_SHUTDOWN_SCRIPT=/opt/deeplearning/bin/shutdown_script.sh
  echo "Appending gce/shutdown.sh to $IMAGE_SHUTDOWN_SCRIPT..."

  # Append our shutdown script to it
  sudo cat gce/shutdown.sh >> $IMAGE_SHUTDOWN_SCRIPT

  # Full dataset
  gsutil cp gs://math-checkpoints-data/mathematics_dataset-v1.0.tar.gz dataset.zip
  
  # 10kb dataset for faster testing
  # gsutil cp gs://math-checkpoints-data/mini_mathematics_dataset-v1.0.tar.gz dataset.zip

  tar xvzf dataset.zip
else
  echo "$DIRECTORY already exists, skipping to training"
fi

# We need to use the user account, as root does not have imaged python packages

# su - andrew_schreiber1 -c '/home/andrew_schreiber1/hs-math-nlp-master/gce/training.sh'
