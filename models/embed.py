import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.dnn import Dnn_Embedding

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        # print('x:', x.shape)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        # print('Tokenembedding', x.shape)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)


class ConvLayer_fp(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer_fp, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=1,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = x.transpose(1,2)
        return x

class Conv1d_ext(nn.Module):
    def __init__(self):
        super(Conv1d_ext, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        #self.imgConv = nn.Conv1d(in_channels=6, out_channels=6,kernel_size=3, padding=padding, padding_mode='circular')
        #self.lastConv = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=3, padding=padding, padding_mode='circular')
        self.imgConv = ConvLayer_fp(c_in=4)
        #self.lastConv = ConvLayer_fp(c_in=20)
        self.euiConv =ConvLayer_fp(c_in=1)
        self.fireConv = ConvLayer_fp(c_in=1)
        self.poiConv = ConvLayer_fp(c_in=13)
        self.tempConv = ConvLayer_fp(c_in=3)
        self.taxiConv = ConvLayer_fp(c_in=2)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        #print('x:', x.shape)
        eui = x[:,:,:1]
        temp = x[:,:,1:4]
        poi = x[:, :, 4:17]
        taxi = x[:,:,17:19]
        img = x[:,:,-5:-1]
        fire = x[:,:,-1:]
        #last = torch.cat([x[:,:,:19],x[:,:,-1:]],dim=2)
        #img = self.imgConv(img.permute(0, 2, 1)).transpose(1,2)
        #last = self.lastConv(last.permute(0, 2, 1)).transpose(1,2)
        eui = self.euiConv(eui)
        temp = self.tempConv(temp)
        poi = self.poiConv(poi)
        taxi = self.taxiConv(taxi)
        img = self.imgConv(img)
        fire = self.fireConv(fire)
        #last = self.lastConv(last)
        x = torch.cat([eui, temp, poi, taxi, img, fire],dim=2)
        #print('Conv1d_ext', x.shape)
        return x


class DataEmbedding_en(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_en, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        self.img_embedding = Dnn_Embedding(4, d_model)
        self.poi_embedding = Dnn_Embedding(13, d_model)
        self.eui_embedding = Dnn_Embedding(1, d_model)

        self.le = nn.Linear(1, 36, bias=False)
        self.lp = nn.Linear(1, 36, bias=False)
        self.li = nn.Linear(1, 36, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, poi, img, eui):
        eui_embeded = self.eui_embedding(eui) #.unsqueeze(1).repeat(1, x.shape[1], 1)
        eui_embeded = eui_embeded.unsqueeze(1).permute(0,2,1)
        eui_embeded = self.le(eui_embeded)
        eui_embeded = eui_embeded.permute(0,2,1)

        img_embeded = self.img_embedding(img).unsqueeze(1).permute(0,2,1) #.repeat(1, x.shape[1], 1)
        img_embeded = self.li(img_embeded).permute(0,2,1)

        poi_embeded = self.poi_embedding(poi).unsqueeze(1).permute(0,2,1) #.repeat(1, x.shape[1], 1)
        poi_embeded = self.lp(poi_embeded).permute(0,2,1)

        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark) + eui_embeded + img_embeded + poi_embeded

        return self.dropout(x)

class DataEmbedding_dn(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_dn, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        self.img_embedding = Dnn_Embedding(4, d_model)
        self.poi_embedding = Dnn_Embedding(13, d_model)
        self.eui_embedding = Dnn_Embedding(1, d_model)

        self.le = nn.Linear(1, 21, bias=False)
        self.lp = nn.Linear(1, 21, bias=False)
        self.li = nn.Linear(1, 21, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, poi, img, eui):
        eui_embeded = self.eui_embedding(eui)  # .unsqueeze(1).repeat(1, x.shape[1], 1)
        eui_embeded = eui_embeded.unsqueeze(1).permute(0, 2, 1)
        eui_embeded = self.le(eui_embeded)
        eui_embeded = eui_embeded.permute(0, 2, 1)

        img_embeded = self.img_embedding(img).unsqueeze(1).permute(0, 2, 1)  # .repeat(1, x.shape[1], 1)
        img_embeded = self.li(img_embeded).permute(0, 2, 1)

        poi_embeded = self.poi_embedding(poi).unsqueeze(1).permute(0, 2, 1)  # .repeat(1, x.shape[1], 1)
        poi_embeded = self.lp(poi_embeded).permute(0, 2, 1)

        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark) + eui_embeded + img_embeded + poi_embeded

        return self.dropout(x)

class DataEmbeddingX_en(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbeddingX_en, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        self.xgboost_embedding = Dnn_Embedding(1, d_model)
        self.lx = nn.Linear(1, 36, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, xgboost):
        xgboost_embeded = self.xgboost_embedding(xgboost) #.unsqueeze(1).repeat(1, x.shape[1], 1)
        print('start:')
        print(xgboost_embeded.shape)
        xgboost_embeded = xgboost_embeded.unsqueeze(1).permute(0,2,1)
        print(xgboost_embeded.shape)
        xgboost_embeded = self.lx(xgboost_embeded)
        print(xgboost_embeded.shape)
        xgboost_embeded = xgboost_embeded.permute(0,2,1)
        print(xgboost_embeded.shape)

        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark) + xgboost_embeded
        print('x.shape:', x.shape)
        return self.dropout(x)

class DataEmbeddingX_dn(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbeddingX_dn, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        self.xgboost_embedding = Dnn_Embedding(1, d_model)
        self.lx = nn.Linear(1, 21, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, xgboost):
        xgboost_embeded = self.xgboost_embedding(xgboost)  # .unsqueeze(1).repeat(1, x.shape[1], 1)
        xgboost_embeded = xgboost_embeded.unsqueeze(1).permute(0, 2, 1)
        xgboost_embeded = self.lx(xgboost_embeded)
        xgboost_embeded = xgboost_embeded.permute(0, 2, 1)

        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark) + xgboost_embeded

        return self.dropout(x)