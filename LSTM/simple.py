import torch
import torch.nn as nn
from torch.autograd import Variable
from math_dataset import VOCAB_SZ, MAX_QUESTION_SZ, MAX_ANSWER_SZ


class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.num_hidden = 2048

        self.lstm = nn.LSTM(VOCAB_SZ, self.num_hidden, 1)
        self.tgt_word_prj = nn.Linear(self.num_hidden, VOCAB_SZ, bias=False)
        self.q_to_a = nn.Linear(162, 31)

        # nn.init.xavier_normal_(self.tgt_word_prj.weight)

    def forward(self, batch_qs, batch_qs_pos, batch_as, batch_as_pos):
        # To Do: Change this input forward pass to match inputs
        batch_size = len(batch_qs)
        batch_qs = torch.transpose(batch_qs, 0, 1)
        batch_qs = nn.functional.one_hot(batch_qs, VOCAB_SZ)

        batch_as = batch_as[:, 1:]
        batch_as = torch.transpose(batch_as, 0, 1)
        batch_as = nn.functional.one_hot(batch_as, VOCAB_SZ)
        batch_as = batch_as.float()

        batch_qs = batch_qs.float()  # (162, 16, 95)

        hidden_state = Variable(
            torch.zeros(1, batch_size, self.num_hidden, dtype=torch.float)
        )
        cell_state = Variable(
            torch.zeros(1, batch_size, self.num_hidden, dtype=torch.float)
        )
        output_seq = torch.empty((MAX_ANSWER_SZ - 1, 16, VOCAB_SZ))
        thinking_input = torch.zeros(1, batch_size, VOCAB_SZ, dtype=torch.float)

        if torch.cuda.is_available():
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()
            output_seq = output_seq.cuda()
            thinking_input = thinking_input.cuda()
            batch_as = batch_as.cuda()
            batch_qs = batch_qs.cuda()

        # Input question phase
        for t in range(MAX_QUESTION_SZ):
            outputs, (hidden_state, cell_state) = self.lstm(
                batch_qs[t].unsqueeze(0), (hidden_state, cell_state)
            )
        # Extra 15 Computational Steps
        for t in range(15):
            outputs_junk, (hidden_state, cell_state) = self.lstm(
                thinking_input, (hidden_state, cell_state)
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
