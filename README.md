# Reproduction of 'Analysing Mathematical Reasoning Abilities of Neural Models' Saxton et. al. 2019

## Setup
1) Clone the repo

2) Set up a Python 3.7 environment

3) Enter the repo folder and run `pip install -r requirements.txt`

4) Optional: download the full 2GB dataset (https://console.cloud.google.com/storage/browser/_details/mathematics-dataset/mathematics_dataset-v1.0.tar.gz) and unzip in the project root directory.

## Usage
If you're running on a non CUDA machine (i.e. a laptop), these commands will run with reduced batch size and dataset size for faster testing/debugging.

### Training
To train a model you must select from Vanilla Transformer, Simple LSTM, and Attentional LSTM. Options `transformer`, `simLSTM`, and `attLSTM`, respectively.

For example:
`python training.py -m transformer`

### Benchmarking
To run performance benchmarks on the Transformer run
`python benchmark.py`

### Visualization
Tensorboard logs are saved to the `runs` folder.

`tensorboard --logdir runs`

## Colaborators 

- Andrew Schreiber 
- Taylor Kulp-McDowall

## Links

- Arxiv paper: https://arxiv.org/abs/1904.01557
- ICLR 2019 Open Review: https://openreview.net/forum?id=H1gR5iR5FX
- Dataset Code from paper: https://github.com/deepmind/mathematics_dataset
- PyTorch helpers: https://github.com/mandubian/pytorch_math_dataset

## Relevant papers

- Machine Learning Projects for Iterated Distillation and Amplification: https://owainevans.github.io/pdfs/evans_ida_projects.pdf
- Do NLP Models Know Numbers? Probing Numeracy in Embeddings: https://arxiv.org/pdf/1909.07940.pdf
- AllenNLP Interpret:
A Framework for Explaining Predictions of NLP Models: https://arxiv.org/pdf/1909.09251.pdf
- Attending to Mathematical Language with Transformers: https://arxiv.org/abs/1812.02825
- All citing papers: https://scholar.google.com/scholar?um=1&ie=UTF-8&lr&cites=5177820928273150256

