
import torch.nn as nn
import torch
class NNwork(nn.Module):
    def __init__(self,input_fnum,seq_length,num_layers=4,batch_size=64,device='cuda'):
        super().__init__()
        self.num_layers=num_layers
        self.batch_size = batch_size
        self.device = device
        

        self.bn = nn.BatchNorm1d(seq_length)# seq length
    def init_weights(self):
        for m in self.modules():
            torch.nn.init.kaiming_normal_(m.weight)
        # self.linear = nn.Linear(in_features = hidden_size,out_features=output_fnum)

        # self.hn = torch.zeros(num_layers,batch_size,hidden_size),#h0
        # self.cn = torch.zeros(num_layers,batch_size,hidden_size),#h0

        # self.hidden_cell = (torch.zeros(num_layers,batch_size,hidden_size).to(device),torch.zeros(num_layers,batch_size,hidden_size).to(device))

        # self.lstm.cuda()
        # self.linear.cuda()
    def forward(self,input,hidden_cell):
        if hidden_cell==None:
            hidden_cell = self.hidden_cell = (
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device),

            )

        lstm_out, ret_hidden_cell= self.lstm(input,hidden_cell)
        lstm_out = self.bn(lstm_out)
        # prediction = self.linear(lstm_out.view(len(input),-1))


        return lstm_out,ret_hidden_cell