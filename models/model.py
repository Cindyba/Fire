import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as Xgboost
from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding, Conv1d_ext, DataEmbedding_en, DataEmbedding_dn, DataEmbeddingX_en, DataEmbeddingX_dn
from models.dnn import Dnn_Net

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        #conv1d feature extraction
        self.conv1d_ext = Conv1d_ext()

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        #self.sigmoidlayer = nn.Sigmoid() #分类
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc = torch.cat([x_enc[:, :, 1:5], x_enc[:, :, 18:20], x_enc[:, :, -1:]], dim=2)
        # 拼接动态数据
        #x_enc = self.conv1d_ext(x_enc)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        x_dec = torch.cat([x_dec[:, :, 1:5], x_dec[:, :, 18:20], x_dec[:, :, -1:]], dim=2)
        # 拼接动态数据
        #x_dec = self.conv1d_ext(x_dec)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        #dec_out = self.sigmoidlayer(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


class Informer_Dnn(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer_Dnn, self).__init__()
        self.Dnn_img = Dnn_Net(4, 1024)
        self.Dnn_poi = Dnn_Net(13, 1024)
        self.Dnn_eui = Dnn_Net(1, 1024)
        self.informer = Informer(enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor, d_model, n_heads, e_layers, d_layers, d_ff, dropout, attn, embed, freq, activation,
                output_attention, distil, mix,device)
        self.projection = nn.Linear(4, c_out, bias=True)
        self.activate = nn.Sigmoid()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_img = x_enc[:, 0, -5:-1].squeeze(dim=1)
        x_poi = x_enc[:, 0, 5:18].squeeze(dim=1)
        x_eui = x_enc[:, 0, :1]
        y_img = self.Dnn_img(x_img)
        y_poi = self.Dnn_poi(x_poi)
        y_eui = self.Dnn_eui(x_eui)
        x_enc = torch.cat([x_enc[:, :, 1:5], x_enc[:,:,18:20], x_enc[:,:,-1:]], dim=2)
        x_dec = torch.cat([x_dec[:, :, 1:5], x_dec[:,:,18:20], x_dec[:,:,-1:]], dim=2)
        y_informer = self.informer(x_enc, x_mark_enc, x_dec, x_mark_dec)
        #print(y_informer.shape, y_poi.shape, y_img.shape)
        lin = torch.cat([y_informer.squeeze(dim=1), y_poi, y_img, y_eui], dim=1)
        #lin = torch.cat([y_informer.squeeze(dim=1), y_poi], dim=1)
        #print(lin.shape)
        informer_out = self.projection(lin)
        #print(y_eui.shape)
        return informer_out.unsqueeze(dim=1), y_eui.unsqueeze(dim=1), y_informer.unsqueeze(dim=1), y_poi.unsqueeze(dim=1), y_img.unsqueeze(dim=1)


class Informer_Xgboost(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer_Xgboost, self).__init__()
        self.Dnn_img = Dnn_Net(4, 1024)
        self.Dnn_poi = Dnn_Net(13, 1024)
        self.Dnn_eui = Dnn_Net(1, 1024)
        self.informer = Informer(enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor, d_model, n_heads, e_layers, d_layers, d_ff, dropout, attn, embed, freq, activation,
                output_attention, distil, mix,device)
        self.projection = nn.Linear(4, c_out, bias=True)
        self.model_fusion = nn.Linear(2, c_out, bias=True)
        self.activate = nn.Sigmoid()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, xgboost_output):
        x_img = x_enc[:, 0, -5:-1].squeeze(dim=1)
        x_poi = x_enc[:, 0, 4:17].squeeze(dim=1)
        x_eui = x_enc[:, 0, :1]
        y_img = self.Dnn_img(x_img)
        y_poi = self.Dnn_poi(x_poi)
        y_eui = self.Dnn_eui(x_eui)
        x_enc = torch.cat([x_enc[:, :, 1:4], x_enc[:,:,17:19], x_enc[:,:,-1:]], dim=2)
        x_dec = torch.cat([x_dec[:, :, 1:4], x_dec[:,:,17:19], x_dec[:,:,-1:]], dim=2)
        y_informer = self.informer(x_enc, x_mark_enc, x_dec, x_mark_dec)
        #print(y_informer.shape, y_poi.shape, y_img.shape)
        lin = torch.cat([y_informer.squeeze(dim=1), y_poi, y_img, y_eui], dim=1)
        #lin = torch.cat([y_informer.squeeze(dim=1), y_poi], dim=1)
        #print(lin.shape)
        informer_out = self.projection(lin)
        xgboost_output = xgboost_output.unsqueeze(dim=1)
        x = torch.cat([informer_out, xgboost_output], dim=1)
        weight = self.model_fusion(x)
        weight = self.activate(weight)
        out = torch.mul(xgboost_output, weight) + torch.mul(informer_out, 1-weight)
        informer_out = informer_out.unsqueeze(dim=1)
        out = out.unsqueeze(dim=1)
        #out = (y_informer.squeeze(dim=1) + y_poi + y_img).unsqueeze(dim=1)
        #print(out.shape)
        return out, informer_out, y_eui, y_informer, y_poi, y_img


class Informer_(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Informer_, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # conv1d feature extraction
        self.conv1d_ext = Conv1d_ext()

        # Encoding
        self.enc_embedding = DataEmbedding_en(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding_dn(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        # self.sigmoidlayer = nn.Sigmoid() #分类

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc = self.conv1d_ext(x_enc)
        img = x_enc[:, 0, -5:-1].squeeze(dim=1)
        poi = x_enc[:, 0, 5:18].squeeze(dim=1)
        eui = x_enc[:, 0, :1]
        x_enc = torch.cat([x_enc[:, :, 1:5], x_enc[:, :, 18:20], x_enc[:, :, -1:]], dim=2)

        enc_out = self.enc_embedding(x_enc, x_mark_enc, poi, img, eui)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # x_dec = self.conv1d_ext(x_dec)
        x_dec = torch.cat([x_dec[:, :, 1:5], x_dec[:,:,18:20], x_dec[:,:,-1:]], dim=2)
        dec_out = self.dec_embedding(x_dec, x_mark_dec, poi, img, eui)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        # dec_out = self.sigmoidlayer(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

class Informer_embeddingX(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Informer_embeddingX, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # conv1d feature extraction
        self.conv1d_ext = Conv1d_ext()

        # Encoding
        self.enc_embedding = DataEmbeddingX_en(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbeddingX_dn(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        # self.sigmoidlayer = nn.Sigmoid() #分类

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, xgboost_input,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc = torch.cat([x_enc[:, :, 1:5], x_enc[:, :, 18:20], x_enc[:, :, -1:]], dim=2)

        enc_out = self.enc_embedding(x_enc, x_mark_enc, xgboost_input.unsqueeze(1))
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # x_dec = self.conv1d_ext(x_dec)
        x_dec = torch.cat([x_dec[:, :, 1:5], x_dec[:,:,18:20], x_dec[:,:,-1:]], dim=2)
        dec_out = self.dec_embedding(x_dec, x_mark_dec, xgboost_input.unsqueeze(1))
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        # dec_out = self.sigmoidlayer(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

class DynamicWeighting(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(DynamicWeighting, self).__init__()
        self.informer = Informer_(enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor, d_model, n_heads, e_layers, d_layers, d_ff, dropout, attn, embed, freq, activation,
                output_attention, distil, mix,device)
        self.model_fusion = nn.Linear(2, c_out, bias=True)
        self.activate = nn.Sigmoid()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, xgboost_output):
        informer_output = self.informer(x_enc, x_mark_enc, x_dec, x_mark_dec).squeeze(dim=1)
        xgboost_output = xgboost_output.unsqueeze(dim=1)
        x = torch.cat([informer_output, xgboost_output], dim=1)

        weight = self.model_fusion(x)
        weight = self.activate(weight)
        out = torch.mul(xgboost_output, weight) + torch.mul(informer_output, 1-weight)

        informer_out = informer_output.unsqueeze(dim=1)
        out = out.unsqueeze(dim=1)
        #out = (y_informer.squeeze(dim=1) + y_poi + y_img).unsqueeze(dim=1)
        #print(out.shape)
        return out, informer_out, weight

class DynamicWeighting_Embedding_Xgboost(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(DynamicWeighting_Embedding_Xgboost, self).__init__()
        self.informer = Informer_embeddingX(enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor, d_model, n_heads, e_layers, d_layers, d_ff, dropout, attn, embed, freq, activation,
                output_attention, distil, mix,device)
        self.model_fusion = nn.Linear(2, c_out, bias=True)
        self.activate = nn.Sigmoid()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, xgboost_output):
        informer_output = self.informer(x_enc, x_mark_enc, x_dec, x_mark_dec, xgboost_output).squeeze(dim=1)
        xgboost_output = xgboost_output.unsqueeze(dim=1)
        x = torch.cat([informer_output, xgboost_output], dim=1)

        weight = self.model_fusion(x)
        weight = self.activate(weight)
        out = torch.mul(xgboost_output, weight) + torch.mul(informer_output, 1-weight)

        informer_out = informer_output.unsqueeze(dim=1)
        out = out.unsqueeze(dim=1)
        weight = weight.unsqueeze(dim=1)
        return out, informer_out, weight

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.model_fusion = nn.Linear(2, 2, bias=False)
        self.linear = nn.Linear(2, 1, bias=True)
        self.activate = nn.Softmax()

    def forward(self, xgboost_output, informer_output):
        informer_output = informer_output.view(-1, 1)
        xgboost_output = xgboost_output.view(-1, 1)
        x = torch.cat([xgboost_output, informer_output], dim=1)
        fusion_output = self.model_fusion(x)
        out = self.activate(fusion_output)
        out = self.linear(out * x)
        return out.unsqueeze(dim=1)

class Res_Embedding_Xgboost(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Res_Embedding_Xgboost, self).__init__()
        self.informer = Informer_embeddingX(enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor, d_model, n_heads, e_layers, d_layers, d_ff, dropout, attn, embed, freq, activation,
                output_attention, distil, mix,device)
        self.model_fusion = nn.Linear(1, c_out, bias=True)
        self.activate = nn.Sigmoid()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, xgboost_output):
        informer_output = self.informer(x_enc, x_mark_enc, x_dec, x_mark_dec, xgboost_output).squeeze(dim=1)
        xgboost_output = xgboost_output.unsqueeze(dim=1)

        weight = self.model_fusion(informer_output)
        weight = self.activate(weight)
        out = torch.mul(informer_output, weight) + xgboost_output

        informer_out = informer_output.unsqueeze(dim=1)
        out = out.unsqueeze(dim=1)

        return out

class DynamicWeighting_Only(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(DynamicWeighting_Only, self).__init__()
        self.informer = Informer(enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor, d_model, n_heads, e_layers, d_layers, d_ff, dropout, attn, embed, freq, activation,
                output_attention, distil, mix,device)
        self.model_fusion = nn.Linear(2, c_out, bias=True)
        self.activate = nn.Sigmoid()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, xgboost_output):
        informer_output = self.informer(x_enc, x_mark_enc, x_dec, x_mark_dec).squeeze(dim=1)
        xgboost_output = xgboost_output.unsqueeze(dim=1)
        x = torch.cat([informer_output, xgboost_output], dim=1)

        weight = self.model_fusion(x)
        weight = self.activate(weight)
        out = torch.mul(xgboost_output, weight) + torch.mul(informer_output, 1-weight)

        informer_out = informer_output.unsqueeze(dim=1)
        out = out.unsqueeze(dim=1)
        return out, informer_out