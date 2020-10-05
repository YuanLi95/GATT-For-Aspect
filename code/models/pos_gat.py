import torch
import torch.nn as nn
import torch.nn.functional as F
# import  torch.functional as F
import math
from layers.dynamic_rnn import DynamicLSTM

from torch.autograd import Variable
import numpy as np
import  time

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.relu = nn.ReLU()


    def forward(self, input, adj,flage=True):
        h = torch.matmul(input, self.W)
        batch_size = h.size()[0]
        token_lenth = h.size()[1]

        # h.repeat_interleave(repeats=token_lenth, dim=2).view(batch_size, token_lenth * token_lenth, -1)    [bacth_szie,token_len*token_len,embedding_dem]

        a_input = torch.cat([h.repeat_interleave(repeats=token_lenth,dim=2).view(batch_size,token_lenth * token_lenth, -1), h.repeat_interleave(token_lenth, dim=0).view(batch_size,token_lenth * token_lenth, -1)], dim=2).view(batch_size,token_lenth,-1, 2 * self.out_features)  # 这里让每两个节点的向量都连接在一起遍历一次得到 bacth* N * N * (2 * out_features)大小的矩阵

        # a_input = torch.cat([h.repeat(1, 1, token_lenth).view(batch_size, token_lenth * token_lenth, -1), h.repeat(1, token_lenth, 1)], dim=2).view(batch_size, token_lenth, -1, 2 * self.out_features)
        # print(a_input.shape)
        e = self.leakyrelu(torch.matmul(a_input, self .a).squeeze(3))

        e = self.relu(torch.matmul(a_input, self.a).squeeze(3))
        # print(e.shape)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        # print(attention.shape)

        attention = torch.softmax(attention, dim=2)             # 这里是一个非线性变换，将有权重的变得更趋近于1，没权重的为0

        attention = F.dropout(attention, self.dropout, training=self.training)
        # if flage ==False:   #False为测试模式
        #     print(adj)
        #     time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        #     numpy_attention = attention.cpu().numpy()
        #     np.save("./attention_numpy/numpy_attention_{0}.npy".format(time_str), numpy_attention)
        #     exit()
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class PoS_GAT(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(PoS_GAT, self).__init__()
        self.opt = opt
        if opt.use_bert ==False:
            self.embed = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float))
        self.hid_dim = opt.hidden_dim

        self.text_lstm = DynamicLSTM(
            opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True,rnn_type="LSTM")
        # self.text_lstmasp = DynamicLSTM(
        #     opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gat1 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)
        self.gat2 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)

        # self.gc22 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        self.gat3 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)
        self.gat4 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)
        self.gat5 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)
        self.gat6 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)
        self.gat7 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)
        self.gat8 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)
        self.gat9 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)
        self.gat10 = GraphAttentionLayer(opt.hidden_dim, opt.hidden_dim)
        self.conv1 = nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 3, padding=1)
        # self.poool = nn.MaxPool1d(9)
        # self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        # self.fc1 = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)

        self.fc2 = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

        # self.fc2 = nn.Linear(opt.hidden_dim*3, opt.polarities_dim)
        # self.fc4 = nn.Linear(opt.hidden_dim*4, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.5)
        # self.batch_nor = nn.BatchNorm1d(90)

        self.layer_norm1 = torch.nn.LayerNorm(opt.hidden_dim, eps=1e-12)
        self.layer_norm2 = torch.nn.LayerNorm(opt.hidden_dim, eps=1e-12)
        self.layer_norm3 = torch.nn.LayerNorm(opt.hidden_dim, eps=1e-12)
        # self.layer_norm1 = torch.nn.LayerNorm(opt.hidden_dim*2, eps=1e-12)

        # self.layer_norm2 = torch.nn.LayerNorm(opt.hidden_dim*3, eps=1e-12)

    # def speech_weight(self, x, aspect_double_idx, text_len, aspect_len, seq_len,speech_list):
    #     batch_size = x.shape[0]
    #     print(seq_len)
    #     tol_len = x.shape[1]  # sl+cl
    #     aspect_double_idx = aspect_double_idx.cpu().numpy()
    #     text_len = text_len.cpu().numpy()
    #     aspect_len = aspect_len.cpu().numpy()
    #     weight = [[] for i in range(batch_size)]
    #     for i in range(batch_size):
    #         # weight for text
    #         context_len = text_len[i] - aspect_len[i]
    #         for j in range(aspect_double_idx[i, 0]):
    #             weight[i].append((speech_list[j]))
    #         for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
    #             weight[i].append(0)
    #         for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
    #             weight[i].append((speech_list[j]))
    #         for j in range(text_len[i], seq_len):
    #             weight[i].append(0)
    #         # # weight for concept_mod
    #         # for j in range(seq_len, seq_len + concept_mod_len[i]):
    #         #     weight[i].append(1)
    #         # for j in range(seq_len + concept_mod_len[i], tol_len):
    #         #     weight[i].append(0)
    #     weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
    #     return weight * x  # 根据词性获得不同权重


    def position_weight(self, x, aspect_double_idx, text_len, aspect_len, seq_len):
        all_weights = []
        # for ii in range(len(x)):
        batch_size = x.shape[0]
        tol_len = x.shape[1]  # sl+cl
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        # concept_mod_len = concept_mod_len.cpu().numpy()

        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            # weight for text
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(
                    1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(
                    1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        # print((weight*x).shape)
        # print((x).shape)
        # print((weight).shape)
        # all_weights.append(weight * x[ii])
        return weight * x  # 根据上下文的位置关系获得不同权重的


    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask * x  # 只保留aspect word 的dependency

    def kmax_pooling(self, x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def get_state(self, bsz):
        if True:
            return Variable(torch.rand(bsz, self.hid_dim)).cuda()
        else:
            return Variable(torch.zeros(bsz, self.hid_dim))

    def forward(self, inputs):
        [text_indices, aspect_indices, left_indices, adj,speech_list],flage = inputs

        text_len = torch.sum(text_indices != 0, dim=-1)
        # concept_mod_len = torch.sum(concept_mod != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat(
            [left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)  # 获得了 aspect的 两个左右下标

        text = self.embed(text_indices)
        # text = self.batch_nor(text)
        text = self.text_embed_dropout(text)


        # textasp = self.embed(aspect_indices)
        # text = self.batch_nor(text)
        # textasp = self.text_embed_dropout(textasp)

        # text_out, _ = self.text_lstm(text)


        text_out, _ = self.text_lstm(text, text_len)

        # text_outasp, _ = self.text_lstm(textasp,aspect_len)


        batch_size = text_out.shape[0]
        seq_len = text_out.shape[1]
        hidden_size = text_out.shape[2] // 2
        # print(text_out.shape)
        text_out = text_out.reshape(batch_size, seq_len, hidden_size, -1).mean(dim=-1)
        # print(text_out.shape)
        # concept_mod = self.embed(concept_mod)
        # x = torch.cat([text_out, concept_mod], dim=1)
        x = text_out
        okk = x


        x_speech =  self.position_weight(text_out, aspect_double_idx, text_len, aspect_len, seq_len)
        #no position
        # x_speech =  text_out

        # x_conv = 0.5(x_conv_2_1+x_conv_3_1)

        x_conv = F.relu(self.conv1(x_speech.transpose(1, 2)))
        x_conv = F.relu(self.conv2(
            self.position_weight(x_conv.transpose(1, 2), aspect_double_idx, text_len, aspect_len, seq_len).transpose(1,2)))
        # x_conv = F.relu(self.conv1(text.transpose(1, 2)))
        # x_conv = F.relu(self.conv2(x_conv))



        # no position
        # x_conv = F.relu(self.conv2(x_conv))

        x_position_out =self.position_weight(x, aspect_double_idx, text_len, aspect_len, seq_len,)
        # no position
        # x_position_out = x

        # x_speech_out = self.speech_weight(x_position_out,aspect_double_idx, text_len, aspect_len, seq_len,speech_list)
        x = torch.relu(self.gat1(x_position_out , adj))

        #no position
        x =  self.position_weight(x,aspect_double_idx, text_len, aspect_len, seq_len)
        # x =x

        x2 = torch.relu((self.gat2(x, adj)))
        x3 = torch.relu(self.gat3(x, adj))
        x4 = torch.relu(self.gat4(x, adj))
        x5 = torch.relu(self.gat5(x, adj))
        x6 = torch.relu(self.gat6(x, adj))
        # x7 = torch.relu(self.gat7(x, adj))
        # x8 = torch.relu(self.gat8(x, adj))
        # x9 = torch.relu(self.gat9(x, adj))
        # x10  = torch.relu(self.gat10(x, adj))

        # x_graph = 0.25*(x2+x3+x4+x5)
        x_graph =  0.2*(x2 + x3 + x4 + x5 +x6)


        # graph_mask = x66 + x55 + x44 + x33 + x22
        # print(graph_mask.shape)
        graph_mask = self.mask(x_graph,aspect_double_idx)
        #no mask


        #
        hop = 1
        #hop = 3
        lambdaa = 0.01
        # gat_liner_list = [self.gat_liner1, self.gat_liner2, self.gat_liner3,self.gat_liner3,self.gat_liner4,self.gat_liner5]
        for i in range(hop):
            alpha_mat = torch.matmul(graph_mask, text_out.transpose(1, 2))
            if i == hop - 1:
                alpha = torch.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
                if flage == False:  # False为测试模式
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    gat_attention = alpha.cpu().numpy()
                    print(gat_attention.shape)
                    np.save("./attention_numpy/gat_attention_{0}.npy".format(time_str), gat_attention)
                a1 = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x hidden_dim
            else:
                # gat_liner = gat_liner_list[i]
                alpha = torch.softmax(alpha_mat, dim=2)
                a1 = torch.matmul(alpha, text_out).squeeze(1)
                # graph_mask = lambdaa*torch.sigmoid(a1)+graph_mask
                graph_mask = lambdaa * self.layer_norm1(torch.sigmoid(a1)) + graph_mask
                # graph_mask = self.layer_norm1(torch.sigmoid(gat_liner(a1))) + graph_mask

        # calculate hidden state attention
        text_speet_state = self.position_weight(text_out, aspect_double_idx, text_len, aspect_len, seq_len,)
        text_out_mask = self.mask(text_speet_state , aspect_double_idx)
        #no position
        # text_out_mask = text_out
        # text_out_mask =self.mask(text_out, aspect_double_idx)

        # no mask

        # #对照试验
        # for i in range(hop):
        #     alpha_mat_text = torch.matmul(text_out_mask, text_out.transpose(1, 2))
        #     if i == hop - 1:
        #         alpha_text = torch.softmax(alpha_mat_text.sum(1, keepdim=True), dim=2)
        #         a3 = torch.matmul(alpha_text, text_out).squeeze(1)
        #     else:
        #         # alpha_text = torch.softmax(alpha_mat_text, dim=2)
        #         alpha_text = torch.softmax(alpha_mat_text, dim=2)
        #         a3 = torch.matmul(alpha_text, text_out).squeeze(1)
        #         text_out_mask = lambdaa * self.layer_norm2(torch.sigmoid(a3)) + text_out_mask
        # # # text_liner_list = [self.text_liner1, self.text_liner2, self.text_liner3,self.text_liner4,self.text_liner5]
        # for i in range(hop):
        #     alpha_mat_text = torch.matmul(text_out_mask, text_out.transpose(1, 2))
        #     if i == hop - 1:
        #         alpha_text = torch.softmax(alpha_mat_text.sum(1, keepdim=True), dim=2)
        #         a2 = torch.matmul(alpha_text, text_out).squeeze(1)
        #     else:
        #         # text_liner = text_liner_list[i]
        #         # alpha_text = torch.softmax(alpha_mat_text, dim=2)
        #         alpha_text = torch.softmax(alpha_mat_text,dim=2)
        #         a2 = torch.matmul(alpha_text, text_out).squeeze(1)
        #         text_out_mask = lambdaa * self.layer_norm2(torch.sigmoid(a2)) + text_out_mask
        #         # text_out_mask =  self.layer_norm2(torch.sigmoid(text_liner(a2))) + text_out_mask


        # calculate CNN attention
        # no mask
        # conv_liner_list = [self.conv_liner1, self.conv_liner2, self.conv_liner3,self.conv_liner4,self.conv_liner5]
        x_conv = self.mask(x_conv.transpose(1, 2), aspect_double_idx)
        # no mask
        # x_conv = x_conv.transpose(1, 2)
        for i in range(hop):
            alpha_mat_x_conv = torch.matmul(x_conv, text_out.transpose(1, 2))
            if i == hop - 1:
                alpha_x_conv = torch.softmax(alpha_mat_x_conv.sum(1, keepdim=True), dim=2)
                if flage == False:  # False为测试模式
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    conv_attention = alpha_x_conv.cpu().numpy()
                    np.save("./attention_numpy/conv_attention_{0}.npy".format(time_str), conv_attention)
                a3 = torch.matmul(alpha_x_conv, x_conv).squeeze(1)  # batch_size x hidden_dim
            else:
                # conv_liner = conv_liner_list[i]
                alpha_x_conv = torch.softmax(alpha_mat_x_conv, dim=2)
                # alpha_x_conv = alpha_mat_x_conv
                a3 = torch.matmul(alpha_x_conv, text_out).squeeze(1)  #
                x_conv = lambdaa * self.layer_norm3(torch.sigmoid(a3)) + x_conv
                # x_conv = self.layer_norm3(torch.sigmoid(conv_liner(a3))) + x_conv


        fnout = torch.cat((a1, a3), 1)
        if self.opt.use_lstm_attention:
            # output = self.fc(fnout)
            output = self.fc2(fnout)
        else:
            output = self.fc(fnout)
        return output

