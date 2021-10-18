import  numpy as np
import  torch
import  torch.nn as nn
import  torch.optim as optim
from    matplotlib import pyplot as plt


input_size = 1
hidden_size = 16
output_size = 1
num_layers = 1

class Net(nn.Module):

    def __init__(self, ):
        super(Net, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,                         #feature_len=1
            hidden_size=hidden_size,                       #隐藏记忆单元尺寸hidden_len
            num_layers=num_layers,                                  #层数
            batch_first=True,                              #在传入数据时,按照[batch,seq_len,feature_len]的格式
        )
        for p in self.rnn.parameters():                    #对RNN层的参数做初始化
            nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size)  #输出层


    def forward(self, x, hidden_prev):
        '''
        x：一次性输入所有样本所有时刻的值(batch,seq_len,feature_len)
        hidden_prev：第一个时刻空间上所有层的记忆单元(batch,num_layer,hidden_len)
        输出out(batch,seq_len,hidden_len)和hidden_prev(batch,num_layer,hidden_len)
        '''
        out, hidden_prev = self.rnn(x, hidden_prev)
        #因为要把输出传给线性层处理，这里将batch和seq_len维度打平，再把batch=1添加到最前面的维度（为了和y做MSE）
        out = out.view(-1, hidden_size)    #[batch=1,seq_len,hidden_len]->[seq_len,hidden_len]
        out = self.linear(out)             #[seq_len,hidden_len]->[seq_len,feature_len=1]
        out = out.unsqueeze(dim=0)         #[seq_len,feature_len=1]->[batch=1,seq_len,feature_len=1]
        return out, hidden_prev


#训练过程
lr=0.01

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)

hidden_prev = torch.zeros(1, num_layers, hidden_size)     #初始化记忆单元h0[batch,num_layer,hidden_len]
num_time_steps = 50                              #区间内取多少样本点

for iter in range(6000):
    start = np.random.randint(3, size=1)[0]                            #在0~3之间随机取开始的时刻点
    time_steps = np.linspace(start, start + 10, num_time_steps)        #在[start,start+10]区间均匀地取num_points个点
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)                             #[num_time_steps,]->[num_points,1]
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1) #输入前49个点(seq_len=49)，即下标0~48[batch, seq_len, feature_len]
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)  #预测后49个点，即下标1~49
    #以上步骤生成(x,y)数据对

    output, hidden_prev = model(x, hidden_prev)       #喂入模型得到输出
    hidden_prev = hidden_prev.detach()

    loss = criterion(output, y)                       #计算MSE损失
    model.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 1000 == 0:
        print("Iteration: {} loss {}".format(iter, loss.item()))


#测试过程
#先用同样的方式生成一组数据x,y
start = np.random.randint(3, size=1)[0]
time_steps = np.linspace(start, start + 10, num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

predictions = []

input = x[:, 0, :]                      #取seq_len里面第0号数据
input = input.view(1, 1, 1)             #input：[1,1,1]
for _ in range(x.shape[1]):             #迭代seq_len次

    pred, hidden_prev = model(input, hidden_prev)
    input = pred                        #预测出的(下一个点的)序列pred当成输入(或者直接写成input, hidden_prev = model(input, hidden_prev))
    predictions.append(pred.detach().numpy().ravel()[0])


x = x.data.numpy()
y = y.data.numpy()
plt.plot(time_steps[:-1], x.ravel())

plt.scatter(time_steps[:-1], x.ravel(), c='r')     #x值
plt.scatter(time_steps[1:], y.ravel(), c='y')     #y值
plt.scatter(time_steps[1:], predictions, c='b')    #y的预测值
plt.show()