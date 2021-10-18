import torch.nn as nn
import torch
class SeqDecoder(nn.Module):
    def __init__(self, hidden_size,output_fnum,seq_length):
        super().__init__()
        self.ln1 = nn.Linear(in_features = hidden_size,out_features=output_fnum)
        self.bn1 = nn.BatchNorm1d(seq_length)#SEQ LENGTH
        self.relu1 = nn.ReLU(inplace=True)

        self.ln2 = nn.Linear(in_features=output_fnum, out_features=output_fnum)
        self.bn2 = nn.BatchNorm1d(seq_length)
        self.relu2 = nn.ReLU(inplace=True)


        # torch.nn.init.kaiming_normal_(self.linear.weight)


    def forward(self,x):
        y = self.ln1(x)#b,6,40 -> b,6,12
        y = self.bn1(y)#b,6,12 ->
        y = self.relu1(y)

        # y = self.ln2(y)
        # y = self.bn2(y)
        # y = self.relu2(y)

        prediction = y.mean(dim=1)

        return prediction


def getDecoder(hidden_size,output_fnum,seq_length):
    return SeqDecoder(hidden_size=hidden_size,output_fnum=output_fnum,seq_length=seq_length)
