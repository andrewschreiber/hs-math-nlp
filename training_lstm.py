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
        self.lstm = nn.LSTM(VOCAB_SZ, num_hidden,1)
        self.tgt_word_prj = nn.Linear(2048, VOCAB_SZ , bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)
        # self.W = nn.Parameter(
        #     torch.randn([num_hidden, VOCAB_SZ]).type(dtype)
        # )
        # self.b = nn.Parameter(torch.randn([VOCAB_SZ]).type(dtype))

    def forward(self, batch_qs, batch_qs_pos, batch_as, batch_as_pos):
        # To Do: Change this input forward pass to match inputs
        batch_size = len(batch_qs)
        batch_qs = torch.transpose(batch_qs, 0, 1)
        batch_qs = torch.nn.functional.one_hot(batch_qs, VOCAB_SZ)
        # hidden_state = Variable(
        #     torch.zeros(1, batch_size, num_hidden)
        # )  # [num_layers(=1) * num_directions(=1), batch_size, num_hidden]
        # cell_state = Variable(
        #     torch.zeros(1, batch_size, num_hidden)
        # )  # [num_layers(=1) * num_directions(=1), batch_size, num_hidden]
        hidden_state = Variable(torch.zeros(1, batch_size, num_hidden, dtype=torch.float))
        cell_state = Variable(torch.zeros(1, batch_size, num_hidden, dtype=torch.float)) 
        batch_qs = batch_qs.float()
        outputs, (_, _) = self.lstm(batch_qs, (hidden_state, cell_state))
        # outputs = outputs[-1]  # [batch_size, num_hidden]
        # model = torch.mm(outputs, self.W) + self.b  # model : [batch_size, n_class]
        # return model
        seq_logit = self.tgt_word_prj(outputs) * 1.0
        return seq_logit.view(-1)



model = TextLSTM()
# Specify optimizations algs
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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