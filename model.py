import math

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl import load_graphs
from torch import LongTensor as LT
from utils import gen_sparse_A, gen_local_graph, haversine_distance, load_dict_from_pkl
from torch.nn import Parameter
from dgl.nn.pytorch.conv import GATConv
import torch as t
class LSTM(nn.Module):
    def __init__(self, features, layers, userNum, poiNum, catNum, dev):
        super(LSTM, self).__init__()
        global device
        device = dev

        self.user_emb = nn.Embedding(userNum, features)
        self.poi_emb = nn.Embedding(poiNum, features)
        self.cat_emb = nn.Embedding(catNum, features)
        self.tod_emb = nn.Embedding(24, features)
        self.dow_emb = nn.Embedding(7, features)

        self.lstm = nn.LSTM(features, features, layers, batch_first=True)
        self.linear = nn.Linear(features, poiNum)

        self.layers = layers
        self.features = features

    def forward(self, user, poi, cat, lat, lon, tod, dow, unixtime):
        # [B,L,D]
        inputs = self.poi_emb(poi)  # , self.cat_emb(x[...,2]), self.tod_emb(x[...,-2]), self.dow_emb(x[...,-1])

        h0 = torch.zeros(self.layers, user.shape[0], self.features).to(device)
        c0 = torch.zeros(self.layers, user.shape[0], self.features).to(device)

        outputs, (hn, cn) = self.lstm(inputs, (h0, c0))
        pre = self.linear(outputs[:, -1, :])

        return pre, pre


class RNN(nn.Module):
    def __init__(self, features, layers, userNum, poiNum, catNum, dev):
        super(RNN, self).__init__()
        global device
        device = dev

        self.user_emb = nn.Embedding(userNum, features)
        self.poi_emb = nn.Embedding(poiNum, features)
        self.cat_emb = nn.Embedding(catNum, features)
        self.tod_emb = nn.Embedding(24, features)
        self.dow_emb = nn.Embedding(7, features)

        self.lstm = nn.RNN(features, features, layers, batch_first=True)
        self.linear = nn.Linear(features, poiNum)

        self.layers = layers
        self.features = features

    def forward(self, user, poi, cat, lat, lon, tod, dow, unixtime):
        # [B,L,D]
        inputs = self.poi_emb(poi)  # + self.cat_emb(cat) + self.tod_emb(tod) + self.dow_emb(dow) self.user_emb(user) +
        #          = user + poi + cat + tod + dow

        h0 = torch.zeros(self.layers, user.shape[0], self.features).to(device)

        outputs, hn = self.lstm(inputs, h0)
        pre = self.linear(outputs[:, -1, :])

        return pre, pre


from math import pi


def distance(lat1, lon1, lat2, lon2):
    p = pi / 180
    a = 0.5 - torch.cos((lat2 - lat1) * p) / 2 + torch.cos(lat1 * p) * torch.cos(lat2 * p) * (
            1 - torch.cos((lon2 - lon1) * p)) / 2
    return 12742 * torch.asin(torch.sqrt(a))


class STGN_Module(nn.Module):
    def __init__(self, features):
        super(STGN_Module, self).__init__()
        self.ilinear = nn.Linear(2 * features, features)
        self.clinear = nn.Linear(2 * features, features)

        self.t1linear1 = nn.Linear(features, features)
        self.t1linear2 = nn.Linear(features, features)
        self.t2linear1 = nn.Linear(features, features)
        self.t2linear2 = nn.Linear(features, features)
        self.d1linear1 = nn.Linear(features, features)
        self.d1linear2 = nn.Linear(features, features)
        self.d2linear1 = nn.Linear(features, features)
        self.d2linear2 = nn.Linear(features, features)

        self.olinear1 = nn.Linear(2 * features, features)
        self.olinear2 = nn.Linear(features, features)
        self.olinear3 = nn.Linear(features, features)

    def forward(self, x, h, c, d, t):
        i = torch.sigmoid(self.ilinear(torch.cat([x, h], -1)))
        candi_c = torch.tanh(self.clinear(torch.cat([x, h], -1)))

        self.t1linear2.weight.data = self.t1linear2.weight.data.clamp(max=0)
        t1 = torch.sigmoid(self.t1linear1(x) + torch.sigmoid(self.t1linear2(t)))
        t2 = torch.sigmoid(self.t2linear1(x) + torch.sigmoid(self.t2linear2(t)))

        self.d1linear2.weight.data = self.d1linear2.weight.data.clamp(max=0)
        d1 = torch.sigmoid(self.d1linear1(x) + torch.sigmoid(self.d1linear2(d)))
        d2 = torch.sigmoid(self.d2linear1(x) + torch.sigmoid(self.d2linear2(d)))

        hat_c = (1 - i * t1 * d1) * c + i * t1 * d1 * candi_c
        out_c = (i - i) * c + i * t2 * d2 * candi_c

        o = torch.sigmoid(self.olinear1(torch.cat([x, h], -1)) + self.olinear2(t) + self.olinear3(d))

        out_h = o * torch.tanh(hat_c)

        return out_h, out_c


class STGN(nn.Module):
    def __init__(self, features, layers, userNum, poiNum, catNum, dev):
        super(STGN, self).__init__()
        global device
        device = dev

        self.user_emb = nn.Embedding(userNum, features)
        self.poi_emb = nn.Embedding(poiNum, features)
        self.cat_emb = nn.Embedding(catNum, features)
        self.tod_emb = nn.Embedding(24, features)
        self.dow_emb = nn.Embedding(7, features)

        self.gru = STGN_Module(features)
        self.fcd = nn.Linear(1, features)
        self.fct = nn.Linear(1, features)
        self.linear = nn.Linear(features, poiNum)

        self.layers = layers
        self.features = features

    def forward(self, user, poi, cat, lat, lon, tod, dow, unixtime):
        # [B,L,D]
        inputs = self.poi_emb(poi)

        delta_t = [torch.zeros(user.shape[0], 1, 1).to(device)]
        delta_d = [torch.zeros(user.shape[0], 1, 1).to(device)]

        for l in range(1, user.shape[1]):
            time_interval = torch.abs(unixtime[:, l:l + 1].unsqueeze(-1) - unixtime[:, l - 1:l].unsqueeze(-1))
            delta_t.append(time_interval)
            dis_interval = distance(lat[:, l - 1:l].unsqueeze(-1), lon[:, l - 1:l].unsqueeze(-1),
                                    lat[:, l:l + 1].unsqueeze(-1), lat[:, l:l + 1].unsqueeze(-1))
            delta_d.append(dis_interval)

        delta_d = torch.cat(delta_d, 1).float()
        delta_t = torch.cat(delta_t, 1)

        delta_d = (delta_d - delta_d.min()) / (delta_d.max() - delta_d.min())
        delta_t = (delta_t - delta_t.min()) / (delta_t.max() - delta_t.min())

        delta_d = self.fcd(delta_d)
        delta_t = self.fct(delta_t)
        h0 = torch.zeros(self.layers, user.shape[0], self.features).to(device)
        c0 = torch.zeros(self.layers, user.shape[0], self.features).to(device)

        h, c = self.gru(inputs[:, 0, :], h0[0], c0[0], delta_d[:, 0, :], delta_t[:, 0, :])

        for l in range(1, user.shape[1]):
            h, c = self.gru(inputs[:, l, :], h, c, delta_d[:, l, :], delta_t[:, l, :])

        pre = self.linear(h)

        return pre, pre


class Flashback(nn.Module):
    def __init__(self, features, layers, userNum, poiNum, catNum, dev):
        super(Flashback, self).__init__()
        global device
        device = dev

        self.user_emb = nn.Embedding(userNum, features)
        self.poi_emb = nn.Embedding(poiNum, features)
        self.cat_emb = nn.Embedding(catNum, features)
        self.tod_emb = nn.Embedding(24, features)
        self.dow_emb = nn.Embedding(7, features)

        self.rnn = nn.RNN(features, features, layers, batch_first=True)
        self.linear = nn.Linear(features, poiNum)

        self.layers = layers
        self.features = features

    def forward(self, user, poi, cat, lat, lon, tod, dow, unixtime):
        # [B,L,D]
        inputs = self.user_emb(user) + self.poi_emb(poi)  # + self.cat_emb(cat) + self.tod_emb(tod) + self.dow_emb(dow)

        h0 = torch.zeros(self.layers, user.shape[0], self.features).to(device)

        outputs, hn = self.rnn(inputs, h0)

        outs = torch.zeros(outputs.shape).to(device)
        for i in range(outputs.shape[1]):
            sum_weight = torch.zeros(user.shape[0], 1).to(device)
            for j in range(i + 1):
                time_interval = torch.abs(unixtime[:, i:i + 1] - unixtime[:, j:j + 1])

                dis_interval = distance(lat[:, j:j + 1], lon[:, j:j + 1], lat[:, i:i + 1], lon[:, i:i + 1])

                weight = (1 + torch.cos(2 * pi * time_interval)) / 2 * torch.exp(-0.01 * time_interval) * torch.exp(
                    -100 * dis_interval) + 1e-10

                outs[:, i] += outputs[:, j] * weight
                sum_weight += weight

            outs[:, i] /= sum_weight

        pre = self.linear(outs[:, -1, :])

        return pre, pre


class STAN(nn.Module):
    def __init__(self, features, layers, userNum, poiNum, catNum, distance_matrix, dev):
        super(STAN, self).__init__()
        global device
        device = dev

        self.user_emb = nn.Embedding(userNum, features)
        self.poi_emb = nn.Embedding(poiNum, features)
        self.cat_emb = nn.Embedding(catNum, features)
        self.tod_emb = nn.Embedding(24, features)
        self.dow_emb = nn.Embedding(7, features)

        self.emb_sl = nn.Embedding(2, features)
        self.emb_su = nn.Embedding(2, features)
        self.emb_tl = nn.Embedding(2, features)
        self.emb_tu = nn.Embedding(2, features)

        self.sq = nn.Linear(features, features)
        self.sk = nn.Linear(features, features)
        self.sv = nn.Linear(features, features)

        self.value = nn.Linear(25, 1)

        self.layers = layers
        self.features = features
        self.poiNum = poiNum
        self.distance_matrix = torch.from_numpy(distance_matrix).to(device)

    def forward(self, user, poi, cat, lat, lon, tod, dow, unixtime):
        # [B,L,D]
        inputs = self.user_emb(user) + self.poi_emb(poi) + self.tod_emb(tod) + self.dow_emb(dow)

        delta_s = torch.zeros(user.shape[0], user.shape[1], user.shape[1]).to(device)
        delta_t = torch.zeros(user.shape[0], user.shape[1], user.shape[1]).to(device)
        for i in range(user.shape[1]):
            for j in range(user.shape[1]):
                delta_s[:, i, j] = distance(lat[:, i], lon[:, i], lat[:, j], lon[:, j])
                delta_t[:, i, j] = torch.abs(unixtime[:, i] - unixtime[:, j])
        emb = torch.ones_like(delta_s, dtype=torch.long).to(device)
        esl, esu, etl, etu = self.emb_sl(emb), self.emb_su(emb), self.emb_tl(emb), self.emb_tu(emb)
        su, sl, tu, tl = delta_s.max(), delta_s.min(), delta_t.max(), delta_t.min()
        vsl, vsu, vtl, vtu = (delta_s - sl).unsqueeze(-1).expand(-1, -1, -1, self.features), \
            (su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.features), \
            (delta_t - tl).unsqueeze(-1).expand(-1, -1, -1, self.features), \
            (tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.features)
        #         space_dis = (esl*vsu+esu*vsl) / (su-sl)
        #         time_dis = (etl*vtu+etu*vtl) / (tu-tl)
        #         delta = space_dis + time_dis
        space_dis = delta_s / (su - sl)
        time_dis = delta_t / (tu - tl)
        delta = (torch.exp(-space_dis) + torch.exp(-time_dis)) / 2
        #         print(su,sl,tu,tl)
        #         print(delta.max(), delta.min())

        time_interval = torch.zeros(user.shape[0], user.shape[1]).to(device)
        for i in range(1, user.shape[1]):
            time_interval[:, i] = delta_t[:, i, i - 1]
        interval_t = time_interval.unsqueeze(-1).expand(-1, -1, self.poiNum)
        interval_s = torch.zeros_like(interval_t, dtype=torch.float32).to(device)
        emb1 = torch.ones_like(interval_t, dtype=torch.long).to(device)
        for i in range(user.shape[0]):
            interval_s[i] = torch.index_select(self.distance_matrix, 0, poi[i, :])
        esl1, esu1, etl1, etu1 = self.emb_sl(emb1), self.emb_su(emb1), self.emb_tl(emb1), self.emb_tu(emb1)
        su1, sl1, tu1, tl1 = interval_s.max(), interval_s.min(), interval_t.max(), interval_t.min()
        vsl1, vsu1, vtl1, vtu1 = (interval_s - sl1).unsqueeze(-1).expand(-1, -1, -1, self.features), \
            (su1 - interval_s).unsqueeze(-1).expand(-1, -1, -1, self.features), \
            (interval_t - tl1).unsqueeze(-1).expand(-1, -1, -1, self.features), \
            (tu1 - interval_t).unsqueeze(-1).expand(-1, -1, -1, self.features)
        #         space_inter = (esl1*vsu1+esu1*vsl1) / (su1-sl1)
        #         time_inter = (etl1*vtu1+etu1*vtl1) / (tu1-tl1)
        #         interval = space_inter #+ time_inter
        space_inter = interval_s / (su1 - sl1)
        time_inter = interval_t / (tu1 - tl1)
        interval = (torch.exp(-space_inter) + torch.exp(-time_inter)) / 2
        #         print(su1,sl1,tu1,tl1)
        #         print(interval.max(), interval.min())

        self_attn = torch.matmul(self.sq(inputs), self.sk(inputs).transpose(-2, -1)) + delta
        self_attn = F.softmax(self_attn, -1)
        self_attn_out = torch.matmul(self_attn, self.sv(inputs))

        candidates = torch.linspace(0, self.poiNum - 1, self.poiNum).long().to(device)
        candidates = candidates.unsqueeze(0).expand(user.shape[0], -1)
        candidates = self.poi_emb(candidates)
        attn = torch.matmul(candidates, self_attn_out.transpose(-2, -1)) * interval.transpose(-2, -1)

        pre = self.value(attn).squeeze(-1)

        return pre, pre


class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i + 1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L - 1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x





# class SIGIR1(nn.Module):
#     def __init__(self, userNum, poiNum, catNum, dev):
#         super(SIGIR1, self).__init__()
#         global device
#         device = dev
#
#         self.user_emb = nn.Embedding(userNum, 64)
#         self.poi_emb = nn.Embedding(poiNum, 64)
#         self.cat_emb = nn.Embedding(catNum, 64)
#         self.tod_emb = nn.Embedding(24, 64)
#         self.dow_emb = nn.Embedding(7, 64)
#
#         self.poiNum = poiNum
#         self.catNum = catNum
#
#         self.catatt = Attention(64, 8, 8)
#         self.catcrossatt = Attention(64, 8, 8)
#
#         self.poiatt = Attention(64, 8, 8)
#         self.poicrossatt = Attention(64, 8, 8)
#
#         self.catatt1 = Attention(64, 8, 8)
#         self.poiatt1 = Attention(64, 8, 8)
#
#         self.prepoiatt = AAttention(64, 8, 8)
#         self.precatatt = Attention(64, 8, 8)
#
#         self.endpoi = nn.Linear(64, poiNum)
#
#         self.endcat = nn.Sequential(
#             nn.Linear(64, 8),
#             nn.ReLU(),
#             nn.Linear(8, 1),
#         )
#
#     def forward(self, x):
#         # [B,L,D]
#         user, poi, cat, tod, dow = self.user_emb(x[..., 0]), self.poi_emb(x[..., 1]), self.cat_emb(
#             x[..., 2]), self.tod_emb(x[..., -2]), self.dow_emb(x[..., -1])
#         poip, catp = self.poi_emb(torch.arange(self.poiNum).to(device)).unsqueeze(0), self.cat_emb(
#             torch.arange(self.catNum).to(device)).unsqueeze(0)
#         poip, catp = poip.repeat(x.shape[0], 1, 1), catp.repeat(x.shape[0], 1, 1)
#
#         poi_inputs = user + poi + tod + dow
#         #         cat_inputs = user + cat + tod + dow
#
#         #         cat_inputs = self.catatt(cat_inputs, cat_inputs, cat_inputs)
#         poi_inputs = self.poiatt(poi_inputs, poi_inputs, poi_inputs)
#
#         #         cat_inputs = self.catcrossatt(cat_inputs, poi_inputs, poi_inputs)
#         #         poi_inputs = self.poicrossatt(poi_inputs, cat_inputs, cat_inputs)
#
#         #         cat_inputs = self.catatt1(cat_inputs, cat_inputs, cat_inputs)
#         #         poi_inputs = self.poiatt1(poi_inputs, poi_inputs, poi_inputs)
#
#         #         prepoi = self.prepoiatt(poip, poi_inputs, poi_inputs)
#         #         precat = self.precatatt(catp, cat_inputs, cat_inputs)
#
#         prepoi = self.endpoi(poi_inputs[:, -1, :]).squeeze(-1)
#         #         precat = self.endcat(precat).squeeze(-1)
#
#         return prepoi, prepoi


class Attention(nn.Module):
    def __init__(self, features):
        super(Attention, self).__init__()
        self.query = nn.Linear(features, features)
        self.key = nn.Linear(features, features)
        self.value = nn.Linear(features, features)

    def forward(self, q, k, v):
        #         print(q.shape, k.shape)
        attn = torch.matmul(self.query(q), self.key(k).transpose(-2, -1))
        attn = torch.softmax(attn, -1)
        out = torch.matmul(attn, self.value(v)) + q
        return out


class SIGIR(nn.Module):
    def __init__(self, heads, features, layers, userNum, poiNum, catNum, distance_matrix, dev):
        super(SIGIR, self).__init__()
        global device
        device = dev

        self.user_emb = nn.Embedding(userNum, features)
        self.poi_emb = nn.Embedding(poiNum, features)
        self.cat_emb = nn.Embedding(catNum, features)
        self.tod_emb = nn.Embedding(24, features)
        self.dow_emb = nn.Embedding(7, features)
        self.user_embc = nn.Embedding(userNum, features)
        self.tod_embc = nn.Embedding(24, features)
        self.dow_embc = nn.Embedding(7, features)
        #         self.user_emba = nn.Embedding(userNum, features)
        #         self.poi_emba = nn.Embedding(poiNum, features)
        #         self.cat_emba = nn.Embedding(catNum, features)

        #         self.sq = nn.Linear(features, features)
        #         self.sk = nn.Linear(features, features)
        #         self.sv = nn.Linear(features, features)
        #         self.sqc = nn.Linear(features, features)
        #         self.skc = nn.Linear(features, features)
        #         self.svc = nn.Linear(features, features)

        #         self.poi_attention = nn.ModuleList([Attention(features) for i in range(layers)])
        #         self.cat_attention = nn.ModuleList([Attention(features) for i in range(layers)])
        #         self.cross_poi_attention = nn.ModuleList([Attention(features) for i in range(layers)])
        #         self.cross_cat_attention = nn.ModuleList([Attention(features) for i in range(layers)])
        self.poi_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), features, heads) for i in range(layers)])
        self.cat_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), features, heads) for i in range(layers)])
        self.cross_poi_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), features, heads) for i in range(layers)])
        self.cross_cat_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), features, heads) for i in range(layers)])
        #         self.spatial_attention = nn.ModuleList([Attention(features) for i in range(layers)])
        #         AutoCorrelationLayer(AutoCorrelation(),features,4)

        self.value = nn.Linear(25, 1)
        # self.valuec = nn.Linear(25, 1)
        # self.value = nn.Linear(10, 1)
        # self.valuec = nn.Linear(10, 1)
        self.layers = layers
        self.features = features
        self.poiNum = poiNum
        self.catNum = catNum
        self.distance_matrix = torch.from_numpy(distance_matrix).to(device)

    def forward(self, user, poi, cat, lat, lon, tod, dow, unixtime):  # tod 天  dow 星期， unixtime ?
        # user: 32*25 poi:32*25 cat :....

        inputs = self.user_emb(user) + self.poi_emb(poi) + self.tod_emb(tod) + self.dow_emb(dow)
        inputs_cat = self.user_embc(user) + self.cat_emb(cat) + self.tod_embc(tod) + self.dow_embc(dow)

        interval_dis = torch.zeros(user.shape[0], user.shape[1], self.poiNum).to(device)
        for i in range(user.shape[0]):
            interval_dis[i] = torch.index_select(self.distance_matrix, 0, poi[i, :])
        interval = torch.exp(-interval_dis / (interval_dis.max() - interval_dis.min()))  # 为什么interval_dis 取反

        outputs, outputs_cat = inputs, inputs_cat
        for i in range(self.layers):
            outputs = self.poi_attention[i](outputs, outputs, outputs)  # Query ,Key, value
            outputs_cat = self.cat_attention[i](outputs_cat, outputs_cat, outputs_cat)
            outputs = self.cross_poi_attention[i](outputs, outputs_cat, outputs_cat)
            outputs_cat = self.cross_cat_attention[i](outputs_cat, outputs, outputs)

        candidates = torch.linspace(0, self.poiNum - 1, self.poiNum).long().to(device)
        candidates = candidates.unsqueeze(0).expand(user.shape[0], -1)
        candidates = self.poi_emb(candidates)
        attn = torch.matmul(candidates, outputs.transpose(-2, -1)) * interval.transpose(-2, -1)  # 论文对不上

        candidates_cat = torch.linspace(0, self.catNum - 1, self.catNum).long().to(device)
        candidates_cat = candidates_cat.unsqueeze(0).expand(user.shape[0], -1)
        candidates_cat = self.cat_emb(candidates_cat)
        attn_cat = torch.matmul(candidates_cat, outputs_cat.transpose(-2, -1))

        pre_poi = self.value(attn).squeeze(-1)
        pre_cat = self.value(attn_cat).squeeze(-1)

        return pre_poi, pre_cat


class AutoCorrelation(nn.Module):
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))  # 这里不明白
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):  # 这里没看明白
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)  # rfft   # 之后再看
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)  # 不明白

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):  # n_heads 是什么参数？ 1
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values
        )
        out = out.view(B, L, -1)

        return self.out_projection(out)


# MSTHN model
# coding=utf-8
"""
@author: Yantong Lai
@description: Local spatial-temproal enhanced GNN and global hypergraph neural network in MSTHN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTemporalEnhancedLayer(nn.Module):
    """Spatial-temporal enhanced layer"""

    def __init__(self, num_users, num_pois, seq_len, emb_dim, num_heads, dropout, device):
        super(SpatialTemporalEnhancedLayer, self).__init__()

        self.num_users = num_users
        self.num_pois = num_pois
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.device = device

        self.pos_embeddings = nn.Embedding(seq_len + 1, emb_dim, padding_idx=0)
        self.fc_geo = nn.Linear(emb_dim, emb_dim, bias=True, device=device)
        self.multi_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout, batch_first=True, device=device)
        self.weight = nn.Parameter(torch.Tensor(emb_dim, emb_dim))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, G, nodes_embeds, batch_users_seqs, batch_users_seqs_masks, batch_users_geo_adjs,
                batch_users_indices):
        batch_size = batch_users_seqs.size(0)
        batch_users_geo_adjs = batch_users_geo_adjs.float()

        # generate sequence embeddings  poi embedding
        batch_seqs_embeds = nodes_embeds[batch_users_seqs]  # 没弄懂（刚弄懂）

        # generate position embeddings  # 位置编码的作用？
        batch_seqs_pos = torch.arange(1, self.seq_len + 1, device=self.device)
        batch_seqs_pos = batch_seqs_pos.repeat(batch_size, 1)
        # batch_seqs_pos = torch.multiply(batch_seqs_pos, batch_users_seqs_masks)
        batch_seqs_pos_embs = self.pos_embeddings(batch_seqs_pos)

        # generate geographical embeddings  去掉 几何 cat embedding
        batch_seqs_geo_embeds = batch_users_geo_adjs.matmul(batch_seqs_embeds)
        batch_seqs_geo_embeds = torch.relu(self.fc_geo(batch_seqs_geo_embeds))  # 加全连接层

        # multi-head attention
        batch_seqs_total_embeds = batch_seqs_embeds + batch_seqs_pos_embs + batch_seqs_geo_embeds
        batch_seqs_mha, batch_seqs_mha_weight = self.multi_attn(batch_seqs_total_embeds, batch_seqs_total_embeds,
                                                                batch_seqs_total_embeds)
        batch_users_embeds = torch.mean(batch_seqs_mha, dim=1)

        nodes_embeds = nodes_embeds.clone()  # ?
        nodes_embeds[batch_users_indices] = batch_users_embeds

        # graph convolutional
        lconv_nodes_embeds = torch.sparse.mm(G, nodes_embeds[:-1])
        nodes_embeds[:-1] = lconv_nodes_embeds

        return nodes_embeds


class SpatialTemporalEnhancedLayer2(nn.Module):
    """Spatial-temporal enhanced layer"""

    def __init__(self, num_users, num_cats, seq_len, emb_dim, num_heads, dropout, device):
        super(SpatialTemporalEnhancedLayer2, self).__init__()

        self.num_users = num_users
        self.num_pois = num_cats
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.device = device

        self.pos_embeddings = nn.Embedding(seq_len + 1, emb_dim, padding_idx=0)
        self.fc_geo = nn.Linear(emb_dim, emb_dim, bias=True, device=device)
        # 多头注意机制
        self.multi_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout, batch_first=True, device=device)
        self.weight = nn.Parameter(torch.Tensor(emb_dim, emb_dim))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, CG, nodes_embeds, batch_users_c_seqs, batch_users_c_seqs_masks,
                batch_users_indices):
        batch_size = batch_users_c_seqs.size(0)

        # generate sequence embeddings  poi embedding
        batch_seqs_embeds = nodes_embeds[batch_users_c_seqs]  # 没弄懂（刚弄懂）

        # generate position embeddings  # 位置编码的作用？
        batch_seqs_pos = torch.arange(1, self.seq_len + 1, device=self.device)
        batch_seqs_pos = batch_seqs_pos.repeat(batch_size, 1)
        batch_seqs_pos = torch.multiply(batch_seqs_pos, batch_users_c_seqs_masks)
        batch_seqs_pos_embs = self.pos_embeddings(batch_seqs_pos)

        # generate geographical embeddings  去掉 几何 cat embedding
        # batch_seqs_geo_embeds = batch_users_geo_adjs.matmul(batch_seqs_embeds)
        # batch_seqs_geo_embeds = torch.relu(self.fc_geo(batch_seqs_geo_embeds))  # 加全连接层

        # multi-head attention
        # batch_seqs_total_embeds = batch_seqs_embeds + batch_seqs_pos_embs + batch_seqs_geo_embeds
        batch_seqs_total_embeds = batch_seqs_embeds + batch_seqs_pos_embs
        batch_seqs_mha, batch_seqs_mha_weight = self.multi_attn(batch_seqs_total_embeds, batch_seqs_total_embeds,
                                                                batch_seqs_total_embeds)
        batch_users_embeds = torch.mean(batch_seqs_mha, dim=1)

        nodes_embeds = nodes_embeds.clone()  # ?
        nodes_embeds[batch_users_indices] = batch_users_embeds

        # graph convolutional
        lconv_nodes_embeds = torch.sparse.mm(CG, nodes_embeds[:-1])
        nodes_embeds[:-1] = lconv_nodes_embeds

        return nodes_embeds


class LocalSpatialTemporalGraph(nn.Module):
    """Local spatial-temporal enhanced graph neural network module"""

    def __init__(self, num_layers, num_users, num_pois, seq_len, emb_dim, num_heads, dropout, device):
        super(LocalSpatialTemporalGraph, self).__init__()

        self.num_layers = num_layers
        self.num_users = num_users
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.device = device
        self.spatial_temporal_layer = SpatialTemporalEnhancedLayer(num_users, num_pois, seq_len, emb_dim, num_heads,
                                                                   dropout, device)

    def forward(self, G, nodes_embeds, batch_users_seqs, batch_users_seqs_masks, batch_users_geo_adjs,
                batch_users_indices):
        nodes_embedding = [nodes_embeds]
        for layer in range(self.num_layers):
            nodes_embeds = self.spatial_temporal_layer(G, nodes_embeds, batch_users_seqs, batch_users_seqs_masks,
                                                       batch_users_geo_adjs, batch_users_indices)
            # nodes_embeds = F.dropout(nodes_embeds, self.dropout)
            nodes_embedding.append(nodes_embeds)

        nodes_embeds_tensor = torch.stack(nodes_embedding)
        final_nodes_embeds = torch.mean(nodes_embeds_tensor, dim=0)

        return final_nodes_embeds


class LocalSpatialTemporalGraph2(nn.Module):
    """Local spatial-temporal enhanced graph neural network module"""

    def __init__(self, num_layers, num_users, num_cats, seq_len, emb_dim, num_heads, dropout, device):
        super(LocalSpatialTemporalGraph2, self).__init__()

        self.num_layers = num_layers
        self.num_users = num_users
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.device = device
        self.spatial_temporal_layer = SpatialTemporalEnhancedLayer2(num_users, num_cats, seq_len, emb_dim, num_heads,
                                                                    dropout, device)

    def forward(self, CG, nodes_embeds, batch_users_c_seqs, batch_users_c_seqs_masks,
                batch_users_indices):
        nodes_embedding = [nodes_embeds]
        for layer in range(self.num_layers):
            nodes_embeds = self.spatial_temporal_layer(CG, nodes_embeds, batch_users_c_seqs, batch_users_c_seqs_masks,
                                                       batch_users_indices)
            # nodes_embeds = F.dropout(nodes_embeds, self.dropout)
            nodes_embedding.append(nodes_embeds)

        nodes_embeds_tensor = torch.stack(nodes_embedding)
        final_nodes_embeds = torch.mean(nodes_embeds_tensor, dim=0)

        return final_nodes_embeds


class HypergraphConvolutionalNetwork(nn.Module):
    """Hypergraph convolutional network"""

    def __init__(self, emb_dim, num_layers, num_users, dropout, device):
        super(HypergraphConvolutionalNetwork, self).__init__()

        self.num_layers = num_layers
        self.num_users = num_users
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.device = device

    def forward(self, x, HG):
        item_embedding = [x]
        for layer in range(self.num_layers):
            x = torch.sparse.mm(HG, x)  # 不懂
            # x = F.dropout(x, self.dropout)
            item_embedding.append(x)

        item_embedding_tensor = torch.stack(item_embedding)
        final_item_embedding = torch.mean(item_embedding_tensor, dim=0)

        return final_item_embedding


class MSTHN(nn.Module):
    """Our proposed Multi-view Spatial-Temporal Enhanced Hypergraph Network (MSTHN)"""

    def __init__(self, num_local_layer, num_global_layer, num_users, num_pois, seq_len, emb_dim, num_heads, dropout,
                 device):
        super(MSTHN, self).__init__()

        self.num_nodes = num_users + num_pois + 1
        self.num_users = num_users
        self.num_pois = num_pois
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.padding_idx = num_users + num_pois
        self.device = device

        self.nodes_embeddings = nn.Embedding(self.num_nodes, emb_dim, padding_idx=self.padding_idx)  # 4670*128
        self.pos_embeddings = nn.Embedding(seq_len + 1, emb_dim, padding_idx=0)  # 疑惑 seq_len 100  padding_idx=0
        self.w_1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_dim, 1))
        self.glu1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.glu2 = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        # local graph and global hypergraph
        self.local_graph = LocalSpatialTemporalGraph(num_local_layer, num_users, num_pois, seq_len, emb_dim, num_heads,
                                                     dropout, device)
        self.global_hyg = HypergraphConvolutionalNetwork(emb_dim, num_global_layer, num_users, dropout, device)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def user_temporal_pref_augmentation(self, node_embedding, session_len, reversed_sess_item, mask):
        """user temporal preference augmentation"""
        batch_size = session_len.size(0)
        seq_h = node_embedding[reversed_sess_item]
        hs = torch.div(torch.sum(seq_h, 1), session_len.unsqueeze(-1))

        batch_seqs_pos = torch.arange(1, self.seq_len + 1, device=self.device)
        batch_seqs_pos = batch_seqs_pos.repeat(batch_size, 1)
        batch_seqs_pos = torch.multiply(batch_seqs_pos, mask)
        pos_emb = self.pos_embeddings(batch_seqs_pos)

        hs = hs.unsqueeze(1).repeat(1, self.seq_len, 1)
        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))  # hs xu*  nh xiu*
        beta = torch.matmul(nh, self.w_2)  # betai :ai
        select = torch.sum(beta * seq_h, 1)

        return select

    def forward(self, G, HG, batch_users_seqs, batch_users_seqs_masks, batch_users_geo_adjs, batch_users_indices,
                batch_seqs_lens, batch_users_rev_seqs):
        nodes_embeds = self.nodes_embeddings.weight

        local_nodes_embs = self.local_graph(G, nodes_embeds, batch_users_seqs, batch_users_seqs_masks,
                                            batch_users_geo_adjs, batch_users_indices)
        local_batch_users_embs = local_nodes_embs[batch_users_indices]
        local_pois_embs = local_nodes_embs[self.num_users: -1, :]

        global_pois_embs = self.global_hyg(nodes_embeds[self.num_users: -1, :], HG)

        pois_embs = local_pois_embs + global_pois_embs

        # 种类embs,local global ;  crossModal Auto_Correlaton

        fusion_nodes_embs = torch.cat([local_nodes_embs[:self.num_users], pois_embs], dim=0)
        fusion_nodes_embs = torch.cat([fusion_nodes_embs, torch.zeros(size=(1, self.emb_dim), device=self.device)],
                                      dim=0)
        batch_users_embs = self.user_temporal_pref_augmentation(fusion_nodes_embs, batch_seqs_lens,
                                                                batch_users_rev_seqs, batch_users_seqs_masks)
        batch_users_embs = batch_users_embs + local_batch_users_embs

        return batch_users_embs, pois_embs


class SpatialTemporalEnhancedLayer3(nn.Module):
    """Spatial-temporal enhanced layer"""

    def __init__(self, num_users, num_pois, seq_len, emb_dim, num_heads, dropout, device):
        super(SpatialTemporalEnhancedLayer3, self).__init__()

        self.num_users = num_users
        self.num_pois = num_pois
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.device = device

        self.pos_embeddings = nn.Embedding(seq_len + 1, emb_dim, padding_idx=0)
        self.fc_geo = nn.Linear(emb_dim, emb_dim, bias=True, device=device)
        self.multi_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout, batch_first=True, device=device)
        self.weight = nn.Parameter(torch.Tensor(emb_dim, emb_dim))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, G, nodes_embeds, batch_users_seqs, batch_users_geo_adjs,
                batch_users_indices):
        batch_size = batch_users_seqs.size(0)
        # dim=batch_users_seqs.size(1)
        batch_users_geo_adjs = batch_users_geo_adjs.float()  # 32*25*7725 32*25*128

        # generate sequence embeddings  poi embedding
        batch_seqs_embeds = nodes_embeds[batch_users_seqs]  # 没弄懂（刚弄懂）

        # generate position embeddings  # 位置编码的作用？
        batch_seqs_pos = torch.arange(1, self.seq_len + 1, device=self.device)
        batch_seqs_pos = batch_seqs_pos.repeat(batch_size, 1)
        # batch_seqs_pos = torch.multiply(batch_seqs_pos, batch_users_seqs_masks)
        batch_seqs_pos_embs = self.pos_embeddings(batch_seqs_pos)

        # generate geographical embeddings  去掉 几何 cat embedding batch_seqs_geo_embeds:
        batch_seqs_geo_embeds = batch_users_geo_adjs.matmul(batch_seqs_embeds)

        batch_seqs_geo_embeds = torch.relu(self.fc_geo(batch_seqs_geo_embeds))  # 加全连接层

        # multi-head attention  batch_seqs_embeds: 32,25,128, batch_seqs_pos: 32,25,128  batch_seqs_geo_embs:
        batch_seqs_total_embeds = batch_seqs_embeds + batch_seqs_pos_embs + batch_seqs_geo_embeds
        batch_seqs_mha, batch_seqs_mha_weight = self.multi_attn(batch_seqs_total_embeds, batch_seqs_total_embeds,
                                                                batch_seqs_total_embeds)
        batch_users_embeds = torch.mean(batch_seqs_mha, dim=1)

        nodes_embeds = nodes_embeds.clone()  # ?
        nodes_embeds[batch_users_indices] = batch_users_embeds

        # graph convolutional
        lconv_nodes_embeds = torch.sparse.mm(G, nodes_embeds[:-1])
        nodes_embeds[:-1] = lconv_nodes_embeds

        return nodes_embeds


class LocalSpatialTemporalGraph3(nn.Module):
    """Local spatial-temporal enhanced graph neural network module"""

    def __init__(self, num_layers, num_users, num_pois, seq_len, emb_dim, num_heads, dropout, device):
        super(LocalSpatialTemporalGraph3, self).__init__()

        self.num_layers = num_layers
        self.num_users = num_users
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.device = device
        self.spatial_temporal_layer = SpatialTemporalEnhancedLayer3(num_users, num_pois, seq_len, emb_dim, num_heads,
                                                                    dropout, device)

    def forward(self, G, nodes_embeds, batch_users_seqs, batch_users_geo_adjs,
                batch_users_indices):
        nodes_embedding = [nodes_embeds]
        for layer in range(self.num_layers):
            nodes_embeds = self.spatial_temporal_layer(G, nodes_embeds, batch_users_seqs,
                                                       batch_users_geo_adjs, batch_users_indices)
            # nodes_embeds = F.dropout(nodes_embeds, self.dropout)
            nodes_embedding.append(nodes_embeds)

        nodes_embeds_tensor = torch.stack(nodes_embedding)
        final_nodes_embeds = torch.mean(nodes_embeds_tensor, dim=0)

        return final_nodes_embeds


class SIGIR2(nn.Module):
    def __init__(self, heads, features, layers, userNum, poiNum, catNum, distance_matrix, dev, dropout=0.3):
        super(SIGIR2, self).__init__()
        global device
        device = dev

        # MSTHN
        self.userNum = userNum
        self.num_nodes = userNum + poiNum + 1
        self.padding_idx = userNum + poiNum
        self.nodes_embeddings = nn.Embedding(self.num_nodes, features, padding_idx=self.padding_idx)
        self.local_graph = LocalSpatialTemporalGraph3(num_layers=1, num_users=userNum, num_pois=poiNum, seq_len=25,
                                                      emb_dim=features, num_heads=8,
                                                      dropout=dropout, device=device)

        self.user_emb = nn.Embedding(userNum, features)
        self.poi_emb = nn.Embedding(poiNum, features)
        self.cat_emb = nn.Embedding(catNum, features)
        self.tod_emb = nn.Embedding(24, features)
        self.dow_emb = nn.Embedding(7, features)
        self.user_embc = nn.Embedding(userNum, features)
        self.tod_embc = nn.Embedding(24, features)
        self.dow_embc = nn.Embedding(7, features)
        #         self.user_emba = nn.Embedding(userNum, features)
        #         self.poi_emba = nn.Embedding(poiNum, features)
        #         self.cat_emba = nn.Embedding(catNum, features)

        #         self.sq = nn.Linear(features, features)
        #         self.sk = nn.Linear(features, features)
        #         self.sv = nn.Linear(features, features)
        #         self.sqc = nn.Linear(features, features)
        #         self.skc = nn.Linear(features, features)
        #         self.svc = nn.Linear(features, features)

        #         self.poi_attention = nn.ModuleList([Attention(features) for i in range(layers)])
        #         self.cat_attention = nn.ModuleList([Attention(features) for i in range(layers)])
        #         self.cross_poi_attention = nn.ModuleList([Attention(features) for i in range(layers)])
        #         self.cross_cat_attention = nn.ModuleList([Attention(features) for i in range(layers)])
        self.poi_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), features, heads) for i in range(layers)])
        self.cat_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), features, heads) for i in range(layers)])
        self.cross_poi_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), features, heads) for i in range(layers)])
        self.cross_cat_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), features, heads) for i in range(layers)])
        #         self.spatial_attention = nn.ModuleList([Attention(features) for i in range(layers)])
        #         AutoCorrelationLayer(AutoCorrelation(),features,4)

        self.value = nn.Linear(25, 1)
        # self.valuec = nn.Linear(25, 1)
        # self.value = nn.Linear(10, 1)
        # self.valuec = nn.Linear(10, 1)
        self.layers = layers
        self.features = features
        self.poiNum = poiNum
        self.catNum = catNum
        self.distance_matrix = torch.from_numpy(distance_matrix).to(device)

    def forward(self, G, user, poi, cat, lat, lon, tod, dow, unixtime):  # tod 天  dow 星期， unixtime ?
        # user: 32*25 poi:32*25 cat :....

        interval_dis = torch.zeros(user.shape[0], user.shape[1], self.poiNum).to(device)
        interval_dis1 = torch.zeros(user.shape[0], self.poiNum, user.shape[1]).to(device)
        # interval_dis1 = torch.zeros(user.shape[0],  self.poiNum).to(device)
        # interval_dis2 = torch.zeros(user.shape[0],  self.poiNum).to(device)

        for i in range(user.shape[0]):
            interval_dis[i] = torch.index_select(self.distance_matrix, 0, poi[i, :])
        interval = torch.exp(-interval_dis / (interval_dis.max() - interval_dis.min()))  # 为什么interval_dis 取反

        user1 = user[:, -1]
        # poi1 = poi[:, -1]
        for i in range(user.shape[0]):
            interval_dis1[i] = torch.index_select(self.distance_matrix, 1, poi[i, :])
        interval1 = torch.exp(-interval_dis1 / (interval_dis1.max() - interval_dis1.min()))
        # interval1=torch.transpose(interval1, dim0=interval1.shape[-2], dim1=interval1.shape[-1])
        # for i in range(user.shape[0]):
        #     interval_dis2[i] = torch.index_select(self.distance_matrix, 0, poi1[i])
        # interval2 = torch.exp(-interval_dis2 / (interval_dis2.max() - interval_dis2.min()))

        interval_s = torch.matmul(interval, interval1)

        # 增加
        nodes_embs = self.nodes_embeddings.weight
        # user1=user[:,-1]
        # poi1=poi[:,-1]
        local_nodes_embs = self.local_graph(G, nodes_embs, poi, interval_s, user1)
        local_batch_users_embs = local_nodes_embs[user]
        local_pois_embs = local_nodes_embs[poi]
        # user_temp = user.float().reshape(-1, user.shape[-2], user.shape[-1])
        # poi_temp = poi.float().reshape(-1, poi.shape[-2], poi.shape[-1])
        # local_user = torch.matmul(user_temp.float(), local_batch_users_embs)
        # local_poi = torch.matmul(poi_temp.float(), local_pois_embs)
        # self.user_emb.weight.data = local_batch_users_embs
        # self.user_emb.weight.data=local_pois_embs

        inputs = local_batch_users_embs + local_pois_embs + self.tod_emb(tod) + self.dow_emb(
            dow)
        # inputs=local_batch_users_embs+local_pois_embs+self.tod_emb(tod)+self.dow_emb(dow)
        inputs_cat = self.user_embc(user) + self.cat_emb(cat) + self.tod_embc(tod) + self.dow_embc(dow)

        outputs, outputs_cat = inputs, inputs_cat
        for i in range(self.layers):
            outputs = self.poi_attention[i](outputs, outputs, outputs)  # Query ,Key, value
            outputs_cat = self.cat_attention[i](outputs_cat, outputs_cat, outputs_cat)
            outputs = self.cross_poi_attention[i](outputs, outputs_cat, outputs_cat)
            outputs_cat = self.cross_cat_attention[i](outputs_cat, outputs, outputs)

        candidates = torch.linspace(0, self.poiNum - 1, self.poiNum).long().to(device)
        candidates = candidates.unsqueeze(0).expand(user.shape[0], -1)
        candidates = self.poi_emb(candidates)
        attn = torch.matmul(candidates, outputs.transpose(-2, -1)) * interval.transpose(-2, -1)  # 论文对不上

        candidates_cat = torch.linspace(0, self.catNum - 1, self.catNum).long().to(device)
        candidates_cat = candidates_cat.unsqueeze(0).expand(user.shape[0], -1)
        candidates_cat = self.cat_emb(candidates_cat)
        attn_cat = torch.matmul(candidates_cat, outputs_cat.transpose(-2, -1))

        pre_poi = self.value(attn).squeeze(-1)
        pre_cat = self.value(attn_cat).squeeze(-1)

        return pre_poi, pre_cat


# class CategoryEmbeddings(nn.Module):
#     def __init__(self, num_cats, embedding_dim):
#         super(CategoryEmbeddings, self).__init__()
#
#         self.cat_embedding = nn.Embedding(
#             num_embeddings=num_cats,
#             embedding_dim=embedding_dim,
#         )
#
#     def forward(self, cat_idx):
#         embed = self.cat_embedding(cat_idx)
#         return embed

# NewModel
class MSTHN2(nn.Module):
    """Our proposed Multi-view Spatial-Temporal Enhanced Hypergraph Network (MSTHN)"""

    def __init__(self, num_local_layer, num_global_layer, num_users, num_pois, num_cats, seq_len, emb_dim, num_heads,
                 dropout, device, layers=1):
        super(MSTHN2, self).__init__()
        self.layers = layers

        self.num_nodes = num_users + num_pois + 1
        self.num_users = num_users
        self.num_pois = num_pois
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.padding_idx = num_users + num_pois
        # cat
        self.num_c_nodes = num_users + num_cats + 1
        self.padding_idx_c = num_users + num_cats
        self.num_cats = num_cats

        self.device = device

        self.nodes_embeddings = nn.Embedding(self.num_nodes, emb_dim, padding_idx=self.padding_idx)  # 4670*128
        # cats_embeddings
        # self.cats_embeddings = nn.Embedding(self.num_cats, emb_dim, padding_idx=self.padding_idx_c)
        self.nodes_c_embeddings = nn.Embedding(self.num_c_nodes, emb_dim, padding_idx=self.padding_idx_c)

        self.pos_embeddings = nn.Embedding(seq_len + 1, emb_dim, padding_idx=0)  # 疑惑 seq_len 100  padding_idx=0
        self.w_1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_dim, 1))
        self.glu1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.glu2 = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        # local graph and global hypergraph
        self.local_graph = LocalSpatialTemporalGraph(num_local_layer, num_users, num_pois, seq_len, emb_dim, num_heads,
                                                     dropout, device)
        self.global_hyg = HypergraphConvolutionalNetwork(emb_dim, num_global_layer, num_users, dropout, device)

        # c local graph
        self.local_c_graph = LocalSpatialTemporalGraph2(num_local_layer, num_users, num_cats, seq_len, emb_dim,
                                                        num_heads,
                                                        dropout, device)

        self.poi_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), emb_dim, 1) for i in range(1)])
        self.cat_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), emb_dim, 1) for i in range(1)])
        self.cross_poi_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), emb_dim, 1) for i in range(1)])
        self.cross_cat_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), emb_dim, 1) for i in range(1)])

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def user_temporal_pref_augmentation(self, node_embedding, session_len, reversed_sess_item, mask):
        """user temporal preference augmentation"""
        batch_size = session_len.size(0)
        seq_h = node_embedding[reversed_sess_item]
        hs = torch.div(torch.sum(seq_h, 1), session_len.unsqueeze(-1))

        batch_seqs_pos = torch.arange(1, self.seq_len + 1, device=self.device)
        batch_seqs_pos = batch_seqs_pos.repeat(batch_size, 1)
        batch_seqs_pos = torch.multiply(batch_seqs_pos, mask)
        pos_emb = self.pos_embeddings(batch_seqs_pos)

        hs = hs.unsqueeze(1).repeat(1, self.seq_len, 1)
        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))  # hs xu*  nh xiu*
        beta = torch.matmul(nh, self.w_2)  # betai :ai
        select = torch.sum(beta * seq_h, 1)

        return select

    def forward(self, G, CG, HG, batch_users_seqs, batch_users_seqs_masks, batch_users_c_seqs, batch_users_c_seqs_masks,
                batch_users_geo_adjs, batch_users_indices,
                batch_seqs_lens, batch_users_rev_seqs):
        nodes_embeds = self.nodes_embeddings.weight
        # cat
        nodes_c_embeds = self.nodes_c_embeddings.weight

        local_nodes_embs = self.local_graph(G, nodes_embeds, batch_users_seqs, batch_users_seqs_masks,
                                            batch_users_geo_adjs, batch_users_indices)

        # cat
        local_c_nodes_embs = self.local_c_graph(CG, nodes_c_embeds, batch_users_c_seqs, batch_users_c_seqs_masks,
                                                batch_users_indices)

        local_batch_users_embs = local_nodes_embs[batch_users_indices]
        local_pois_embs = local_nodes_embs[self.num_users: -1, :]  # 7725*128 c:1121*128

        # local cat users
        local_batch_users_c_embs = local_c_nodes_embs[batch_users_indices]
        local_c_embs = local_c_nodes_embs[self.num_users:-1, :]

        global_pois_embs = self.global_hyg(nodes_embeds[self.num_users: -1, :], HG)

        pois_embs = local_pois_embs + global_pois_embs
        print(batch_users_indices.shape[0])
        # pois_embs_temp = torch.unsqueeze(pois_embs, 0).repeat(batch_users_indices.shape[0],1,1)
        # local_c_embs_temp = torch.unsqueeze(local_c_embs, 0).repeat(batch_users_indices.shape[0],1,1)

        # print(pois_embs_temp.shape(),local_c_embs_temp.shape())

        # outputs, outputs_cat = pois_embs_temp, local_c_embs_temp
        # for i in range(self.layers):
        #     outputs = self.poi_attention[i](outputs, outputs, outputs)  # Query ,Key, value
        #     outputs_cat = self.cat_attention[i](outputs_cat, outputs_cat, outputs_cat)
        #     outputs = self.cross_poi_attention[i](outputs, outputs_cat, outputs_cat)
        #     outputs_cat = self.cross_cat_attention[i](outputs_cat, outputs, outputs)

        # pois_embs, local_c_embs = torch.mean(outputs,dim=0),torch.mean(outputs_cat,dim=0)

        fusion_nodes_embs = torch.cat([local_nodes_embs[:self.num_users], pois_embs], dim=0)
        fusion_nodes_embs = torch.cat([fusion_nodes_embs, torch.zeros(size=(1, self.emb_dim), device=self.device)],
                                      dim=0)
        batch_users_embs = self.user_temporal_pref_augmentation(fusion_nodes_embs, batch_seqs_lens,
                                                                batch_users_rev_seqs, batch_users_seqs_masks)
        batch_users_embs = batch_users_embs + local_batch_users_embs

        return batch_users_embs, pois_embs, local_batch_users_c_embs, local_c_embs


# GETNext
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x


class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.leaky_relu(x)
        return x


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec1(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec1, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(7, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(7, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


class Time2Vec2(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec2, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(24, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(24, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        return {"e": F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata["z"] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop("h")

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge="cat"):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == "cat":
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super(GatedFusion, self).__init__()
        self.hid_dim = d_model
        self.HS_fc = nn.Linear(self.hid_dim, self.hid_dim)
        self.HT_fc = nn.Linear(self.hid_dim, self.hid_dim)

    def forward(self, HS, HT):
        '''
        gated fusion
        '''
        XS = F.leaky_relu(self.HS_fc(HS))
        XT = F.leaky_relu(self.HT_fc(HT))
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.multiply(z, HS), torch.multiply(1 - z, HT))
        return H

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)


    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h

class PTRoadGraphEmbedding(nn.Module):
    def __init__(self, features: int, layers: int, dropout: float, h: int):
        super(PTRoadGraphEmbedding, self).__init__()
        self.features = features
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            GATConv(
            in_feats=features,
            out_feats=features // h,
            num_heads=h,
            feat_drop=0.0,
            attn_drop=0.0,
            residual=False,
            activation=F.leaky_relu if i + 1 < layers else None
            ) for i in range(layers)
        ])

    def forward(self, g, x):
        for _, gat in enumerate(self.layers):
            x = gat(g, x)
            x = x.reshape(-1, self.features)
        x = self.dropout(x)
        g.ndata['x'] = x
        # if 'w' in g.ndata:
        #     return dgl.mean_nodes(g, 'x', weight='w'), g
        # else:
        #     return dgl.mean_nodes(g, 'x'), g
        return x, g



class SIGIR_A_G(nn.Module):
    def __init__(self, heads, features, layers, userNum, poiNum, catNum, distance_matrix,
                 poi_embed_model, user_embed_model, time_embed_model1,
                 time_embed_model2, cat_embed_model, X, A,
                 poi_nearby_list, dev,arg):
        super(SIGIR_A_G, self).__init__()
        self.device = dev
        self.X = X
        self.A = A
        self.user_emb = nn.Embedding(userNum, features)
        self.poi_emb = nn.Embedding(poiNum, features)
        self.poi_embed_model = poi_embed_model
        self.cat_emb = nn.Embedding(catNum, features)
        self.tod_emb = nn.Embedding(24, features)
        self.dow_emb = nn.Embedding(7, features)
        self.user_embc = nn.Embedding(userNum, features)
        self.tod_embc = nn.Embedding(24, features)
        self.dow_embc = nn.Embedding(7, features)


        self.poi_nearby_list = torch.from_numpy(np.array(poi_nearby_list)).to(dev, dtype=torch.long)
        # self.user_nearby_list = torch.from_numpy(np.array(user_nearby_list)).to(dev, dtype=torch.long)
        # self.cat_nearby_list = torch.from_numpy(np.array(cat_nearby_list)).to(dev, dtype=torch.long)
        # self.tod_nearby_list = torch.from_numpy(np.array(tod_nearby_list)).to(dev, dtype=torch.long)
        # self.dow_nearby_list = torch.from_numpy(np.array(dow_nearby_list)).to(dev, dtype=torch.long)


        # # GAT
        # G_dgl = load_graphs("./data.bin")
        # self.GAT = GAT(G_dgl[0][0].to(dev), in_dim=features, hidden_dim=features,
        #                       out_dim=features, num_heads=2)

        # sub_graph
        sub_graphs = load_dict_from_pkl('./config//NY/sub_graphs.pkl')
        # self.GatedFusion1=GatedFusion(features)
        # self.GatedFusion2=GatedFusion(features)
        # self.GatedFusion3=GatedFusion(features)
        # self.GatedFusion4=GatedFusion(features)

        self.sub_graphs= dgl.add_self_loop(dgl.batch(sub_graphs).to(dev))
        #
        self.layers_sub = nn.ModuleList([
            PTRoadGraphEmbedding(
               features=features,layers=layers,dropout=0.5,h=heads
            ) for i in range(layers)
        ])




        # self.g = load_graphs("./data.bin")
        # # self.g=dgl.add_self_loop(G_dgl)
        # self.gatconv=GATConv( features, features,num_heads=3)
        # GETNext
        # self.poi_embeddings = poi_embeddings
        # self.node_attn_model = node_attn_model


        self.user_embed_model = user_embed_model
        self.time_embed_model1 = time_embed_model1
        self.time_embed_model2 = time_embed_model2
        self.cat_embed_model = cat_embed_model
        # self.embed_fuse_model1 = embed_fuse_model1
        # self.embed_fuse_model2 = embed_fuse_model2


        ## HMT_GRN
        self.dropout=nn.Dropout(p=arg['dropout'])
        self.padding_idx = 0

        self.geoHashEmbed2 = nn.Embedding(len(arg['index2geoHash_2']), arg['embedding_dim'],
                                          padding_idx=self.padding_idx)
        self.geoHashEmbed3 = nn.Embedding(len(arg['index2geoHash_3']), arg['embedding_dim'],
                                          padding_idx=self.padding_idx)
        self.geoHashEmbed4 = nn.Embedding(len(arg['index2geoHash_4']), arg['embedding_dim'],
                                          padding_idx=self.padding_idx)
        self.geoHashEmbed5 = nn.Embedding(len(arg['index2geoHash_5']), arg['embedding_dim'],
                                          padding_idx=self.padding_idx)
        self.geoHashEmbed6 = nn.Embedding(len(arg['index2geoHash_6']), arg['embedding_dim'],
                                          padding_idx=self.padding_idx)
        self.userEmbed = nn.Embedding(arg['numUsers'], arg['userEmbed_dim'], padding_idx=self.padding_idx)

        self.nextGeoHash2Dense = nn.Linear(256, len(arg['geohash2Index_2']),
                                           bias=True)
        print(arg['geohash2Index_2'])
        self.nextGeoHash3Dense = nn.Linear(256, len(arg['geohash2Index_3']),
                                           bias=True)
        print(arg['geohash2Index_3'])

        self.nextGeoHash4Dense = nn.Linear(256, len(arg['geohash2Index_4']),
                                           bias=True)

        self.nextGeoHash5Dense = nn.Linear(256, len(arg['geohash2Index_5']),
                                           bias=True)

        self.nextGeoHash6Dense = nn.Linear(256, len(arg['geohash2Index_6']),
                                           bias=True)


        self.poi_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), features, heads) for i in range(layers)])
        self.cat_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), features, heads) for i in range(layers)])
        self.cross_poi_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), features, heads) for i in range(layers)])
        self.cross_cat_attention = nn.ModuleList(
            [AutoCorrelationLayer(AutoCorrelation(), features, heads) for i in range(layers)])

        self.value = nn.Linear(25, 1)

        self.layers = layers
        self.features = features
        self.poiNum = poiNum
        self.catNum = catNum
        self.distance_matrix = torch.from_numpy(distance_matrix).to(self.device)
        self.AutoCorrelationLayer_P = AutoCorrelationLayer(AutoCorrelation(), features* self.poi_nearby_list.shape[-1] ,
                                                           heads)
        self.AutoCorrelationLayer_C = AutoCorrelationLayer(AutoCorrelation(), features * self.poi_nearby_list.shape[-1],
                                                           heads)
        self.LinearP = nn.Linear(features* self.poi_nearby_list.shape[-1] , features)
        self.LinearC = nn.Linear(features * self.poi_nearby_list.shape[-1], features)

    def forward(self, user, poi, cat, lat, lon, tod, dow, unixtime,input_hash,arg,mode='train'):  # tod 天  dow 星期， unixtime ?
        # user: 32*25 poi:32*25 cat :....

        # 7725*128
        # if mode=="train":
        x_geoHash2, x_geoHash3, x_geoHash4, x_geoHash5, x_geoHash6 = input_hash
        numTimeSteps=poi.shape[1]
        # else:
        #     x_geoHash2, x_geoHash3, x_geoHash4, x_geoHash5, x_geoHash6 = input_hash
        #     # x_geoHash2, x_geoHash3, x_geoHash4, x_geoHash5, x_geoHash6 =  LT(
        #     #     x_geoHash2).cuda(), LT(x_geoHash3).cuda(), \
        #     #     LT(x_geoHash4).cuda(), LT(x_geoHash5).cuda(), LT(x_geoHash6).cuda()
        #     numTimeSteps=poi.shape[1]

        batchSize = poi.shape[0]

        x_geoHash_embed2 = self.dropout(self.geoHashEmbed2(x_geoHash2))
        x_geoHash_embed3 = self.dropout(self.geoHashEmbed3(x_geoHash3))
        x_geoHash_embed4 = self.dropout(self.geoHashEmbed4(x_geoHash4))
        x_geoHash_embed5 = self.dropout(self.geoHashEmbed5(x_geoHash5))
        x_geoHash_embed6 = self.dropout(self.geoHashEmbed6(x_geoHash6))


        nearby_pois = self.poi_nearby_list[poi]

        # poi_graph_index=np.array(poi.cpu(),np.int)
        # sub_graphs = self.sub_graphs[poi_graph_index]
        # sub_graph = [element for sub_list in sub_graphs for element in sub_list ]
        # sub_graph = dgl.batch(sub_graph).to(self.device)
        # # sub_graph = dgl.add_self_loop(sub_graph).to(self.device)
        # nearby_users = self.user_nearby_list[user]
        # nearby_cats = self.cat_nearby_list[cat]
        # nearby_tods = self.tod_nearby_list[tod]
        # nearby_dows = self.dow_nearby_list[dow]

        all_poi_embeddings = self.poi_embed_model(self.X, self.A)

        # nearby_cats_emb = self.cat_emb(nearby_cats)
        # poi_embeddings = self.GAT(poi_embeddings)
        nearby_pois_emb1 = all_poi_embeddings[nearby_pois]  # B T N F
        selected_poi_embeddings = torch.index_select(all_poi_embeddings, index=self.sub_graphs.ndata[dgl.NID], dim=0 )
        for _, gat in enumerate(self.layers_sub):
            selected_poi_embeddings, _ = gat(self.sub_graphs, selected_poi_embeddings)

        # # nearby_pois_emb = selected_poi_embeddings.reshape((nearby_pois_emb1.shape[0],nearby_pois_emb1.shape[1],nearby_pois_emb1.shape[-1]))
        # # # # poi_index=torch.flatten(poi)
        #
        # selected_poi_embeddings = selected_poi_embeddings.reshape((self.poiNum,nearby_pois.shape[-1],-1))
        nearby_pois = nearby_pois.reshape(-1)
        # # nearby_pois_emb = torch.index_select(selected_poi_embeddings, index=poi_index, dim=0)
        nearby_pois_emb = torch.index_select(selected_poi_embeddings, index=nearby_pois, dim=0)
        # #
        nearby_pois_emb = torch.reshape(nearby_pois_emb, (nearby_pois_emb1.shape[0], nearby_pois_emb1.shape[1], -1))
        nearby_pois_emb= self.AutoCorrelationLayer_P(nearby_pois_emb, nearby_pois_emb, nearby_pois_emb)
        nearby_pois_emb = self.LinearP(nearby_pois_emb )


        nearby_pois_emb1 = torch.reshape(nearby_pois_emb1, (nearby_pois_emb1.shape[0], nearby_pois_emb1.shape[1], -1))
        nearby_pois_emb1 = self.AutoCorrelationLayer_C(nearby_pois_emb1, nearby_pois_emb1, nearby_pois_emb1)
        nearby_pois_emb1 = self.LinearC(nearby_pois_emb1)

        nearby_pois_emb=nearby_pois_emb+nearby_pois_emb1

        nextGeoHashlogits_2 = self.nextGeoHash2Dense(t.cat((nearby_pois_emb, x_geoHash_embed2), dim=2))
        nextGeoHashlogits_3 = self.nextGeoHash3Dense(t.cat((nearby_pois_emb, x_geoHash_embed3), dim=2))
        nextGeoHashlogits_4 = self.nextGeoHash4Dense(t.cat((nearby_pois_emb, x_geoHash_embed4), dim=2))
        nextGeoHashlogits_5 = self.nextGeoHash5Dense(t.cat((nearby_pois_emb, x_geoHash_embed5), dim=2))
        nextGeoHashlogits_6 = self.nextGeoHash6Dense(t.cat((nearby_pois_emb, x_geoHash_embed6), dim=2))



        nextGeoHashlogits_2 = nextGeoHashlogits_2.view(batchSize, numTimeSteps, len(arg['geohash2Index_2']))
        nextGeoHashlogits_3 = nextGeoHashlogits_3.view(batchSize, numTimeSteps, len(arg['geohash2Index_3']))
        nextGeoHashlogits_4 = nextGeoHashlogits_4.view(batchSize, numTimeSteps, len(arg['geohash2Index_4']))
        nextGeoHashlogits_5 = nextGeoHashlogits_5.view(batchSize, numTimeSteps, len(arg['geohash2Index_5']))
        nextGeoHashlogits_6 = nextGeoHashlogits_6.view(batchSize, numTimeSteps, len(arg['geohash2Index_6']))

        nextgeohashPred_2 = F.log_softmax(nextGeoHashlogits_2, dim=2)
        nextgeohashPred_3 = F.log_softmax(nextGeoHashlogits_3, dim=2)
        nextgeohashPred_4 = F.log_softmax(nextGeoHashlogits_4, dim=2)
        nextgeohashPred_5 = F.log_softmax(nextGeoHashlogits_5, dim=2)
        nextgeohashPred_6 = F.log_softmax(nextGeoHashlogits_6, dim=2)

               # nearby_pois_emb=nearby_pois_emb1
        # nearby_pois_emb = self.GatedFusion(nearby_pois_emb1 , nearby_pois_emb)


        # nearby_pois_emb = self.AutoCorrelationLayer_P(nearby_pois_emb, nearby_pois_emb, nearby_pois_emb)
        #
        # nearby_pois_emb = self.LinearP(nearby_pois_emb)

        # nearby_cats_emb = torch.reshape(nearby_cats_emb, (nearby_cats_emb.shape[0], nearby_cats_emb.shape[1], -1))
        # nearby_cats_emb = self.AutoCorrelationLayer_C(nearby_cats_emb, nearby_cats_emb, nearby_cats_emb)
        #
        # nearby_cats_emb = self.LinearC(nearby_cats_emb)

        # 25*25*128
        # input1=self.GatedFusion1(self.user_embed_model(user),self.tod_emb(tod) + self.dow_emb(dow))
        # inputs=self.GatedFusion2(input1,nearby_pois_emb)
        #
        # input2=self.GatedFusion3(self.cat_emb(cat),self.tod_embc(tod) + self.dow_embc(dow))
        # inputs_cat=self.GatedFusion4(input2,self.user_embc(user))
        # inputs_cat = self.user_embc(user) + self.cat_emb(cat) + \
        #              self.tod_embc(tod) + self.dow_embc(dow)

        # # 25*25*128
        inputs = self.user_embed_model(user) + nearby_pois_emb + \
                 self.tod_emb(tod) + self.dow_emb(dow)
        inputs_cat = self.user_embc(user) + self.cat_emb(cat) + \
                     self.tod_embc(tod) + self.dow_embc(dow)


        # inputs_cat = self.user_embc(user) + nearby_cats_emb + \
        #              self.tod_embc(tod) + self.dow_embc(dow)

        # inputs = self.user_embed_model(nearby_users) + nearby_pois_emb + \
        #          self.tod_emb(nearby_tods) + self.dow_emb(nearby_dows)
        # inputs_cat = self.user_embc(nearby_users) + self.cat_emb(nearby_cats) + \
        #              self.tod_embc(nearby_tods) + self.dow_embc(nearby_dows)

        # inputs = torch.reshape(inputs, (inputs.shape[0], inputs.shape[1], -1))
        # inputs_cat = torch.reshape(inputs_cat, (inputs_cat.shape[0], inputs_cat.shape[1], -1))

        interval_dis = torch.zeros(user.shape[0], user.shape[1], self.poiNum).to(self.device)
        for i in range(user.shape[0]):
            interval_dis[i] = torch.index_select(self.distance_matrix, 0, poi[i, :])
        interval = torch.exp(-interval_dis / (interval_dis.max() - interval_dis.min()))  # 为什么interval_dis 取反

        outputs, outputs_cat = inputs, inputs_cat
        for i in range(self.layers):
            outputs = self.poi_attention[i](outputs, outputs, outputs)  # Query ,Key, value
            outputs_cat = self.cat_attention[i](outputs_cat, outputs_cat, outputs_cat)
            outputs = self.cross_poi_attention[i](outputs, outputs_cat, outputs_cat)
            outputs_cat = self.cross_cat_attention[i](outputs_cat, outputs, outputs)

        # outputs = self.LinearP(outputs)
        # outputs_cat = self.LinearC(outputs_cat)

        candidates = torch.linspace(0, self.poiNum - 1, self.poiNum).long().to(self.device)
        candidates = candidates.unsqueeze(0).expand(user.shape[0], -1)
        candidates = self.poi_emb(candidates)
        attn = torch.matmul(candidates, outputs.transpose(-2, -1)) * interval.transpose(-2, -1)  # 论文对不上

        candidates_cat = torch.linspace(0, self.catNum - 1, self.catNum).long().to(self.device)
        candidates_cat = candidates_cat.unsqueeze(0).expand(user.shape[0], -1)
        candidates_cat = self.cat_emb(candidates_cat)
        attn_cat = torch.matmul(candidates_cat, outputs_cat.transpose(-2, -1))


        pre_poi = self.value(attn).squeeze(-1)
        pre_cat = self.value(attn_cat).squeeze(-1)

        if mode == 'train':


            return  pre_poi,pre_cat, nextgeohashPred_2, nextgeohashPred_3, nextgeohashPred_4, nextgeohashPred_5, nextgeohashPred_6


        elif mode == 'test':


            nextgeohashPred_2_test = F.softmax(nextGeoHashlogits_2, dim=2)
            nextgeohashPred_3_test = F.softmax(nextGeoHashlogits_3, dim=2)
            nextgeohashPred_4_test = F.softmax(nextGeoHashlogits_4, dim=2)
            nextgeohashPred_5_test = F.softmax(nextGeoHashlogits_5, dim=2)
            nextgeohashPred_6_test = F.softmax(nextGeoHashlogits_6, dim=2)
            return pre_poi,pre_cat, nextgeohashPred_2_test, nextgeohashPred_3_test, nextgeohashPred_4_test, nextgeohashPred_5_test, nextgeohashPred_6_test


