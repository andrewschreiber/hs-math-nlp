import torch
import torch.nn as nn
from torch.autograd import Variable


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_sz, max_answer_sz, max_question_sz, batch_size):
        super(SimpleLSTM, self).__init__()
        self.num_hidden = 2048
        self.vocab_sz = vocab_sz
        self.max_answer_sz = max_answer_sz
        self.max_question_sz = max_question_sz

        self.lstm = nn.LSTM(vocab_sz, self.num_hidden, 1)
        self.tgt_word_prj = nn.Linear(self.num_hidden, vocab_sz, bias=False)
        self.q_to_a = nn.Linear(max_question_sz, max_answer_sz - 1)

        # nn.init.xavier_normal_(self.tgt_word_prj.weight)

    def forward(self, batch_qs, batch_qs_pos, batch_as, batch_as_pos):
        batch_size = len(batch_qs)
        batch_qs = torch.transpose(batch_qs, 0, 1)
        batch_qs = nn.functional.one_hot(batch_qs, self.vocab_sz)

        batch_as = batch_as[:, 1:]
        batch_as = torch.transpose(batch_as, 0, 1)
        batch_as = nn.functional.one_hot(batch_as, self.vocab_sz)
        batch_as = batch_as.float()

        batch_qs = batch_qs.float()  # (max_q_sz, batch_sz, vocab_sz)

        hidden_state = Variable(
            torch.zeros(1, batch_size, self.num_hidden, dtype=torch.float)
        )
        cell_state = Variable(
            torch.zeros(1, batch_size, self.num_hidden, dtype=torch.float)
        )
        output_seq = torch.empty((self.max_answer_sz - 1, batch_size, self.vocab_sz))
        thinking_input = torch.zeros(1, batch_size, self.vocab_sz, dtype=torch.float)

        if torch.cuda.is_available():
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()
            output_seq = output_seq.cuda()
            thinking_input = thinking_input.cuda()
            batch_as = batch_as.cuda()
            batch_qs = batch_qs.cuda()

        # Input question phase
        self.lstm.flatten_parameters()
        for t in range(self.max_question_sz):
            outputs, (hidden_state, cell_state) = self.lstm(
                batch_qs[t].unsqueeze(0), (hidden_state, cell_state)
            )
        # Extra 15 Computational Steps
        for t in range(15):
            outputs_junk, (hidden_state, cell_state) = self.lstm(
                thinking_input, (hidden_state, cell_state)
            )
        # Answer generation phase, need to input correct answer as hidden/cell state, find what to put in input
        for t in range(self.max_answer_sz - 1):
            if t == 0:
                out = self.tgt_word_prj(outputs)
                output_seq[t] = out
                char = output_seq[t].clone().unsqueeze(0)
                outputs, (hidden_state, cell_state) = self.lstm(
                    char, (hidden_state, cell_state)
                )
            else:
                output_seq[t] = self.tgt_word_prj(outputs)
                outputs, (hidden_state, cell_state) = self.lstm(
                    batch_as[t].unsqueeze(0), (hidden_state, cell_state)
                )

        batch_ans_size = output_seq.size(2)  # batch_size x answer_length
        return output_seq.reshape(-1, batch_ans_size)
