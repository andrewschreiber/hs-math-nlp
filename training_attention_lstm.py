
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
#cudnn.benchmark = True

# Uni-LSTM(Attention) Parameters
embedding_dim = 2
max_sentence_length = 50
max_elements = 100
n_step = 1
decoder_num_hidden = 2048
encoder_num_hidden = 512
max_batches = 1
num_workers = 0

exp_name = "math_test"
unique_id = "06-08-2020"

# 3 words sentences (=sequence_length is 3)

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

# TODO:
# Find a way to split the hidden state of the encoder into key value pairs
# Find a way to iteratively (for loop) apply attention to the key value pairs
# Input them into the decoder with the shifted answer
# Return new query pairs as well as an answer
# Verify our attention function is correct

# We're in a good state here and don't need to incorporate more example code
# Pre-compile state, will have to play with the matrix shapes.



class UniLSTM_Attention(nn.Module):
    def __init__(self):
        super(UniLSTM_Attention, self).__init__()

        self.embedding = nn.Embedding(VOCAB_SZ, embedding_dim)
        self.decoding_lstm = nn.LSTM(embedding_dim, decoder_num_hidden, bidirectional=False)
        self.encoding_lstm = nn.LSTM(embedding_dim, encoder_num_hidden, bidirectional=False)
        self.encoder_fc = nn.Linear(encoder_num_hidden, decoder_num_hidden)
        self.decoder_fc = nn.Linear(decoder_num_hidden, VOCAB_SZ)

    # lstm_output : [batch_size, n_step, num_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, num_hidden * 2, 1)   # hidden : [batch_size, num_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, num_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, num_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()  # context : [batch_size, num_hidden * num_directions(=2)]

    def forward(self, batch_qs, batch_qs_pos, batch_as, batch_as_pos):
        batch_size = len(batch_qs)
        batch_qs = torch.transpose(batch_qs, 0, 1)
        batch_qs = torch.nn.functional.one_hot(batch_qs, VOCAB_SZ)
        input = self.embedding(batch_qs)  # input : [batch_size, len_seq, embedding_dim]
        input = input.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(
            torch.zeros(1, batch_size, encoder_num_hidden, dtype=torch.float)
        )
        cell_state = Variable(torch.zeros(1, batch_size, encoder_num_hidden, dtype=torch.float))

        output, hidden = self.encoding_lstm(
            input, (hidden_state, cell_state)
        )

        hidden_state = Variable(
            torch.zeros(1, batch_size, decoder_num_hidden, dtype=torch.float)
        )
        cell_state = Variable(torch.zeros(1, batch_size, decoder_num_hidden, dtype=torch.float))

        output = output.permute(1, 0, 2)  # output : [batch_size, len_seq, num_hidden        ]
        attn_output, attention = self.attention_net(output, hidden)

        hidden_state = Variable(
            torch.zeros(1, batch_size, decoder_num_hidden, dtype=torch.float)
        )
        cell_state = Variable(torch.zeros(1, batch_size, decoder_num_hidden, dtype=torch.float))

        decoder_input = torch.cat((answer_shift, attn_output), dim=2)

        decoder_out = self.decoding_lstm(decoder_input, (hidden_state, cell_state))
        return self.decoder_fc(decoder_out)


model = UniLSTM_Attention()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model_process.train(
    name=f"{exp_name}_{unique_id}",
    model=model,
    training_data=train_loader,
    optimizer=optimizer,
    device=device,
    epochs=1000,  # Not relevant, will get ended before this due to max_b
    tb=tb,
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
