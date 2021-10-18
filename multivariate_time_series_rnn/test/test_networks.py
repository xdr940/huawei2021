import unittest
import numpy as np
import os
import torch
import torch.nn as nn
from networks import getLSTM,getDecoder
class TestNetworks(unittest.TestCase):

    def test_myrnn(self):
        batch_size = 1
        seq_len = 5
        fin_num = 10
        fout_num=6
        D = 1  # not bidirectional
        hidden_size = 20
        num_layers = 6

        encoder = getLSTM(input_fnum=fin_num,hidden_size=hidden_size)
        decoder = getDecoder(hidden_size=hidden_size,output_fnum=fout_num)

        input = torch.randn(batch_size,seq_len, fin_num)

        f ,_= encoder(input)
        # s,b,h = f.size()
        # f = f.view(s*b,h)
        out = decoder(f)

        print('ok')

    def testother(self):
        batch_size = 8
        seq_len = 5
        feature_num = 10
        D = 1  # not bidirectional
        hidden_size = 20
        num_layers = 2

        rnn = nn.LSTM(input_size=feature_num, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        input = torch.randn(batch_size, seq_len,
                            feature_num)  # sequence length, batch size,Hin = input size= featrue nums

        h0 = torch.randn(D * num_layers, batch_size, hidden_size)
        c0 = torch.randn(D * num_layers, batch_size, hidden_size)
        out, (hn, cn) = rnn(input, (h0, c0))

        print(out, hn, cn)


