
'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
  Reference : https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py
'''
import numpy as np
from tensorboard_utils import Tensorboard
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
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
embedding_dim = 2
max_sentence_length = 50 
max_elements = 100
n_step = 1
num_hidden = 128
max_batches = 1
num_workers = 0

exp_name = "math_test"
unique_id = "02-24-2020"

# 3 words sentences (=sequence_length is 3)

mdsmgr = MathDatasetManager("./mathematics_dataset-v1.0")
ds_train = mdsmgr.build_dataset_from_module("algebra", "linear_1d", "train-easy", max_elements=max_elements)

train_loader = torch.utils.data.DataLoader(ds_train, batch_size=1024,
                        shuffle=True, num_workers=0) 
    
tb = Tensorboard(exp_name, unique_name=unique_id)

class UniLSTM_Attention(nn.Module):
    def __init__(self):
        super(UniLSTM_Attention, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, num_hidden, bidirectional=False)
        self.out = nn.Linear(1)

    # lstm_output : [batch_size, n_step, num_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, num_hidden * 2, 1)   # hidden : [batch_size, num_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, num_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, num_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy() # context : [batch_size, num_hidden * num_directions(=2)]

    def forward(self, X):
        input = self.embedding(X) # input : [batch_size, len_seq, embedding_dim]
        input = input.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(torch.zeros(1*2, len(X), num_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, num_hidden]
        cell_state = Variable(torch.zeros(1*2, len(X), num_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, num_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, num_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, num_hidden        ]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention # model : [batch_size, num_classes], attention : [batch_size, n_step]

model = UniLSTM_Attention()

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

# # Training
# for epoch in range(5000):
#     optimizer.zero_grad()
#     output, attention = model(input_batch)
#     loss = criterion(output, target_batch)
#     if (epoch + 1) % 1000 == 0:
#         print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

#     loss.backward()
#     optimizer.step()

# # Test
# test_text = 'sorry hate you'
# tests = [np.asarray([word_dict[n] for n in test_text.split()])]
# test_batch = Variable(torch.LongTensor(tests))

# # Predict
# predict, _ = model(test_batch)
# predict = predict.data.max(1, keepdim=True)[1]
# if predict[0][0] == 0:
#     print(test_text,"is Bad Mean...")
# else:
#     print(test_text,"is Good Mean!!")
    