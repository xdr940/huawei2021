

import torch.nn as nn
import torch
class LSTM(nn.Module):
    def __init__(self,
                 input_fnum,
                 hidden_size,
                 output_fnum,
                 num_layers
                 ):
        super().__init__()

        self.input_fnum = input_fnum
        self.hidden_size = hidden_size
        self.output_fnum = output_fnum



        self.lstm = nn.LSTM(
            input_size=input_fnum,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_fnum)  #输出层

        for p in self.lstm.parameters():                    #对RNN层的参数做初始化
            nn.init.normal_(p, mean=0.0, std=0.001)



    def forward(self,input,hn,cn):


        out, (hn,cn)= self.lstm(input,(hn,cn))
        b, seq_l, _ = out.shape
        # 因为要把输出传给线性层处理，这里将batch和seq_len维度打平，再把batch=1添加到最前面的维度（为了和y做MSE）
        # out = out.view(-1,self.hidden_size)    #[batch=1,seq_len,hidden_len]->[seq_len,hidden_len]
        out = out.reshape([b * seq_l, self.hidden_size])

        out = self.linear(out)  # [seq_len,hidden_len]->[seq_len,feature_len=1]
        _, out_num = out.shape

        out = out.reshape([b, seq_l, out_num])




        return out,(hn,cn)
def getLSTM(input_fnum,hidden_size,output_fnum,num_layers):
    return LSTM(
        input_fnum=input_fnum,
        hidden_size=hidden_size,
        output_fnum=output_fnum,
        num_layers=num_layers
    )











class EnDe(nn.Module):
    def __init__(self,input_fnum,output_fnum,hidden_size,num_layers=4,batch_size=1,device='cuda'):
        super().__init__()
        self.num_layers=num_layers
        self.batch_size = batch_size
        self.device = device

        self.hidden_size = hidden_size


        self.lstm = nn.LSTM(input_size=input_fnum,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.linear = nn.Linear(in_features = hidden_size,out_features=output_fnum)

        # self.hn = torch.zeros(num_layers,batch_size,hidden_size),#h0
        # self.cn = torch.zeros(num_layers,batch_size,hidden_size),#h0


        self.lstm.cuda()
        self.linear.cuda()
    def forward(self,input):
        self.hidden_cell = (
            torch.zeros(self.num_layers,self.batch_size,self.hidden_size).to(self.device),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device),

        )

        lstm_out, self.hidden_cell= self.lstm(input,self.hidden_cell)

        prediction = self.linear(lstm_out)
        prediction = prediction.mean(dim=1)


        return prediction
def getEnDe(input_fnum,output_fnum,hidden_size):
    return EnDe(
        input_fnum=input_fnum,
        hidden_size=hidden_size,
        output_fnum=output_fnum

    )

