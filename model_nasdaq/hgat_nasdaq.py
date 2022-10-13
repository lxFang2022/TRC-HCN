from torch_geometric import nn
import torch
import torch.nn.functional as F
import torch.nn
import numpy as np
from scipy import sparse
from torch_geometric import utils
import math
class Attention(torch.nn.Module):

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = torch.nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = torch.nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.tanh = torch.nn.Tanh()
        self.ae = torch.nn.Parameter(torch.FloatTensor(1026,1,1))
        self.ab = torch.nn.Parameter(torch.FloatTensor(1026,1,1))
        self.reset_parameters()

    def reset_parameters(self):
        for i in self.named_parameters():
            if len(i[1].size()) == 1:
                std = 1.0 / math.sqrt(i[1].size(0))
                torch.nn.init.uniform_(i[1], -std, std)
            else:
                torch.nn.init.xavier_normal_(i[1])

    def forward(self, query, context):
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)


        mix = attention_weights*(context.permute(0,2,1))

        delta_t = torch.flip(torch.arange(0, query_len), [0]).type(torch.float32).to('cuda')
        delta_t = delta_t.repeat(1026,1).reshape(1026,1,query_len)
        bt = torch.exp(-1*self.ab * delta_t)
        term_2 = F.relu(self.ae * mix * bt)
        mix = torch.sum(term_2+mix, -1).unsqueeze(1)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

class gru(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(gru, self).__init__()
        self.gru1 = torch.nn.GRU(input_size = input_size, hidden_size=hidden_size, batch_first=True)
        self.reset_parameters()

    def reset_parameters(self):
        for i in self.named_parameters():
            if len(i[1].size()) == 1:
                std = 1.0 / math.sqrt(i[1].size(0))
                torch.nn.init.uniform_(i[1], -std, std)
            else:
                torch.nn.init.xavier_normal_(i[1])
    def forward(self, inputs):
        full, last  = self.gru1(inputs)
        return full,last



class HGAT(torch.nn.Module):
    def __init__(self, tickers, T):
        super(HGAT, self).__init__()
        self.T = T-1
        self.tickers = tickers
        self.grup = gru(5,64)  #or lstm
        self.attention = Attention(64) #32

        self.hatt1 = nn.HypergraphConv(64, 64, use_attention=False, heads=4, concat=False, negative_slope=0.2, dropout=0.5, bias=True)
        self.hatt2 = nn.HypergraphConv(64, 64, use_attention=False, heads=1, concat=False, negative_slope=0.2, dropout=0.5, bias=True)

        self.liear = torch.nn.Linear(128, 1)
        self.liear2 = torch.nn.Linear(20, 32)
        self.softmax = torch.nn.Softmax(dim=0)
        self.relu = torch.nn.LeakyReLU(0.2)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(64, self.T))
        self.w2 = torch.nn.Parameter(torch.FloatTensor(self.T, 64))
        self.c = torch.tensor(1).float()
        self.a = torch.nn.Parameter(torch.FloatTensor(self.T,1,1))
        self.reset_parameters()

    def reset_parameters(self):
        for i in self.named_parameters():
            if len(i[1].size()) == 1:
                std = 1.0 / math.sqrt(i[1].size(0))
                torch.nn.init.uniform_(i[1], -std, std)
            else:
                torch.nn.init.xavier_normal_(i[1])

    def diff(self,a):
        x = torch.zeros(len(a)-1,len(a[0]),len(a[0][0]))
        for i in range(len(a)-1):
            x[i] = a[i+1] - a[i]
        sub = x[0]
        sub = sub.reshape(1,len(x[0]),len(x[0][0]))
        x = x.reshape(len(x),1,len(x[0]),len(x[0][0]))
        for i in range(len(a)-2):
            sub = torch.cat((sub,x[i+1]),dim=0)
        return sub

    def forward(self, price_input,  hyp_input_T, hyp_input):
        context, query = self.grup(price_input.cuda())
        query = query.reshape(1026, 1, 64)
        output, weights = self.attention(query, context)
        output = output.reshape((1026, 64))

        # Local
        x1 = []
        x2 = []
        for i in range(price_input.shape[1]):
            x1.append(F.leaky_relu(self.hatt1(output, hyp_input_T[i].cuda()), 0.2))
            x2.append(F.leaky_relu(self.hatt2(x1[i], hyp_input_T[i].cuda()), 0.2))
        sub_graph = self.diff(x2).cuda()
        b = torch.mean(sub_graph,dim=(1,2),keepdim=True)
        U = self.c.div(self.c + self.a * (b - sub_graph))
        z = ((U * sub_graph).div(U)).sum(axis=(1,2))
        wattn = self.softmax(torch.matmul(self.w2, self.relu(torch.matmul(self.w1, z.float().cuda()))))
        concat = wattn.unsqueeze(dim=1).unsqueeze(dim=1) * sub_graph
        concat = torch.stack((concat[0,:,:],concat[2,:,:]),0)
        xx1 = torch.sum(concat, 0)

        # Global
        x = F.leaky_relu(self.hatt1(output, hyp_input), 0.2)
        x = F.leaky_relu(self.hatt2(x, hyp_input), 0.2)
        oput = torch.cat((x, xx1), 1)

        return F.leaky_relu(self.liear(oput))  # x
