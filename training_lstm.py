"""
  code by Tae Hwan Jung(Jeff Jung) @graykode
"""
import numpy as np
from tensorboard_utils import Tensorboard
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import model_process
from math_dataset import (
    lstm_batch_collate_fn,
    MathDatasetManager,
    FullDatasetManager,
)

from math_dataset import VOCAB_SZ, MAX_QUESTION_SZ, MAX_ANSWER_SZ

dtype = torch.FloatTensor

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
# cudnn.benchmark = True

# Uni-LSTM(Attention) Parameters
max_elements = 100
num_hidden = 2048
max_batches = 1
num_workers = 2

exp_name = "math_test"
unique_id = "02-24-2020"

tb = Tensorboard(exp_name, unique_name=unique_id)

ds_train = FullDatasetManager(
    "./mathematics_dataset-v1.0", max_elements=10, deterministic=True,
)
train_loader = torch.utils.data.DataLoader(
    ds_train,
    batch_size=16,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=lstm_batch_collate_fn,
)


# Define Model Architecture
class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        self.lstm = nn.LSTM(VOCAB_SZ, num_hidden, 1)

        self.tgt_word_prj = nn.Linear(2048, VOCAB_SZ, bias=False)
        self.q_to_a = nn.Linear(162, 31)

        # nn.init.xavier_normal_(self.tgt_word_prj.weight)

    def forward(self, batch_qs, batch_qs_pos, batch_as, batch_as_pos):
        # To Do: Change this input forward pass to match inputs
        batch_size = len(batch_qs)
        batch_qs = torch.transpose(batch_qs, 0, 1)
        batch_qs = torch.nn.functional.one_hot(batch_qs, VOCAB_SZ)

        batch_as = batch_as[:, 1:]
        batch_as = torch.transpose(batch_as, 0, 1)
        batch_as = torch.nn.functional.one_hot(batch_as, VOCAB_SZ)
        batch_as = batch_as.float()

        hidden_state = Variable(
            torch.zeros(1, batch_size, num_hidden, dtype=torch.cuda.FloatTensor)
        )
        cell_state = Variable(torch.zeros(1, batch_size, num_hidden, dtype=torch.float))
        batch_qs = batch_qs.float()  # (162, 16, 95)

        output_seq = torch.empty((MAX_ANSWER_SZ - 1, 16, VOCAB_SZ))

        # Input question phase
        for t in range(MAX_QUESTION_SZ):
            outputs, (hidden_state, cell_state) = self.lstm(
                batch_qs[t].unsqueeze(0), (hidden_state, cell_state)
            )
        # Extra 15 Computational Steps
        dummy_input = torch.zeros(1, batch_size, VOCAB_SZ, dtype=torch.cuda.FloatTensor)
        for t in range(15):
            outputs_junk, (hidden_state, cell_state) = self.lstm(
                dummy_input, (hidden_state, cell_state)
            )
        # Answer generation phase, need to input correct answer as hidden/cell state, find what to put in input
        for t in range(MAX_ANSWER_SZ - 1):
            if t == 0:
                output_seq[t] = self.tgt_word_prj(outputs)
                char = output_seq[t].clone().unsqueeze(0)
                outputs, (hidden_state, cell_state) = self.lstm(
                    char, (hidden_state, cell_state)
                )
            else:
                output_seq[t] = self.tgt_word_prj(outputs)
                outputs, (hidden_state, cell_state) = self.lstm(
                    batch_as[t].unsqueeze(0), (hidden_state, cell_state)
                )

        # seq_logit_q_length = output_seq.permute(2, 1, 0)
        # seq_logit_a_length = self.q_to_a(seq_logit_q_length)
        # seq_logit_a_length = seq_logit_a_length.permute(2, 1, 0)
        # batch_ans_size = seq_logit_a_length.size(2)  # batch_size x answer_length
        # seq_logit_new = seq_logit_a_length.reshape(-1, batch_ans_size)

        # return seq_logit_new
        batch_ans_size = output_seq.size(2)  # batch_size x answer_length
        return output_seq.reshape(-1, batch_ans_size)


model = TextLSTM()


# Specify optimizations algs
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    model = model.cuda()

 model = model.to(device)

# model_process.train(
#     exp_name=exp_name,
#     unique_id=unique_id,
#     model=model,
#     training_data=train_loader,
#     optimizer=optimizer,
#     device=device,
#     epochs=1,
#     tb=tb,
#     max_batches=max_batches,
#     validation_data=None,
# )
# TODO:
# Rejigger weight loss calculation as it is different from transformer.
model_process.train(
    name=f"{exp_name}_{unique_id}",
    model=model,
    training_data=train_loader,
    optimizer=optimizer,
    device=device,
    epochs=1000,  # Not relevant, will get ended before this due to max_b
    tb=tb,
)

# Training (Old School)
# for epoch in range(1000):
#     optimizer.zero_grad()

#     output = model(input_batch)
#     loss = criterion(output, target_batch)
#     if (epoch + 1) % 100 == 0:
#         print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

#     loss.backward()
#     optimizer.step()

# inputs = [sen[:3] for sen in seq_data]

# predict = model(input_batch).data.max(1, keepdim=True)[1]
# print(inputs, '->', [number_dict[n.item()] for n in predict.squeeze()])
