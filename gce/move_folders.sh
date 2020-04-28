#!/bin/bash

rm -rf m2

mv hs-math-nlp-master m2

rm master.zip

wget https://github.com/andrewschreiber/hs-math-nlp/archive/master.zip

unzip master.zip

sudo chmod -R 777 hs-math-nlp-master

mv m2/runs hs-math-nlp-master

mv m2/checkpoints hs-math-nlp-master

mv m2/mathematics_dataset-v1.0 hs-math-nlp-master

cd hs-math-nlp-master

ls
