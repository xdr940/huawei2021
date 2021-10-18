
import torch.nn as nn
import torch
class RNN(nn.Module):

    def __init__(self,input_fnum,hidden_size,output_fnum,num_layers):
        super(RNN, self).__init__()
        self.input_fnum = input_fnum
        self.hidden_size = hidden_size
        self.output_fnum=output_fnum


        self.rnn = nn.RNN(
            input_size=input_fnum,                         #feature_len=1
            hidden_size=hidden_size,                       #隐藏记忆单元尺寸hidden_len
            num_layers=num_layers,                                 #层数
            batch_first=True,                              #在传入数据时,按照[batch,seq_len,feature_len]的格式
        )


        self.linear = nn.Linear(hidden_size, output_fnum)  #输出层
        for p in self.rnn.parameters():                    #对RNN层的参数做初始化
            nn.init.normal_(p, mean=0.0, std=0.001)

    def forward(self, x, hidden_prev):
        '''
        x：一次性输入所有样本所有时刻的值(batch,seq_len,feature_len)
        hidden_prev：第一个时刻空间上所有层的记忆单元(batch,num_layer,hidden_len)
        输出out(batch,seq_len,hidden_len)和hidden_prev(batch,num_layer,hidden_len)
        '''
        out, hidden_prev = self.rnn(x, hidden_prev)
        b,seq_l,_ = out.shape
        #因为要把输出传给线性层处理，这里将batch和seq_len维度打平，再把batch=1添加到最前面的维度（为了和y做MSE）
        # out = out.view(-1,self.hidden_size)    #[batch=1,seq_len,hidden_len]->[seq_len,hidden_len]
        out = out.reshape([b*seq_l,self.hidden_size])

        out = self.linear(out)             #[seq_len,hidden_len]->[seq_len,feature_len=1]
        _,out_num = out.shape

        out = out.reshape([b,seq_l,out_num])

        # out = out.unsqueeze(dim=0)         #[seq_len,feature_len=1]->[batch=1,seq_len,feature_len=1]
        return out, hidden_prev
def getRNN(input_fnum,hidden_size,output_fnum,num_layers):
    return RNN(
        input_fnum=input_fnum,
        hidden_size=hidden_size,
        output_fnum=output_fnum,
        num_layers = num_layers
    )