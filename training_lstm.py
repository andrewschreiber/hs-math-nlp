'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import numpy as np
from tensorboard_utils import Tensorboard
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import model_process
from math_dataset import (
    question_answer_to_position_batch_collate_fn,
    MathDatasetManager,
    FullDatasetManager,
)

dtype = torch.FloatTensor

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True

# Uni-LSTM(Attention) Parameters
max_sentence_length = 50 
max_elements = 100
n_step = 1
num_hidden = 128
max_batches = 1
num_workers = 0

exp_name = "math_test"
unique_id = "02-24-2020"

tb = Tensorboard(exp_name, unique_name=unique_id)

mdsmgr = MathDatasetManager("./mathematics_dataset-v1.0")
ds_train = mdsmgr.build_dataset_from_module("algebra", "linear_1d", "train-easy", max_elements=max_elements)
# ds_train = FullDatasetManager(
#         "./mathematics_dataset-v1.0", max_elements=max_elements
#     )

train_loader = torch.utils.data.DataLoader(ds_train, batch_size=1,
                        shuffle=True, num_workers=num_workers)    

#Define Model Architecture 

class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=max_sentence_length, hidden_size=num_hidden)
        self.W = nn.Parameter(torch.randn([num_hidden, max_sentence_length]).type(dtype))
        self.b = nn.Parameter(torch.randn([max_sentence_length]).type(dtype))
        self.out = nn.Linear(1)

    def forward(self, X):
        input = X.transpose(0, 1)  # X : [n_step, batch_size, n_class]

        hidden_state = Variable(torch.zeros(1, len(X), num_hidden))   # [num_layers(=1) * num_directions(=1), batch_size, num_hidden]
        cell_state = Variable(torch.zeros(1, len(X), num_hidden))     # [num_layers(=1) * num_directions(=1), batch_size, num_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, num_hidden]
        model = torch.mm(outputs, self.W) + self.b  # model : [batch_size, n_class]
        return model

model = TextLSTM()
# Specify optimizations algs 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model_process.train(
        exp_name=exp_name,
        unique_id=unique_id,
        model=model,
        training_data=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=1,
        tb=tb,
        max_batches=max_batches,
        validation_data=None,
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