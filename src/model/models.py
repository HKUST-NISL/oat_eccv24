from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import numpy as np
from .positionalEncoding import *
#UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 27, 28, 29, 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import torch.nn.functional as F
#from .transformerLightning import PositionalEncoding, VisualPositionalEncoding, TokenEmbedding
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SPPLayer(nn.Module):
    def __init__(self, num_levels=4, pool_type='max'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        previous_conv_size = [h, w]
        for i in range(self.num_levels):
            level = 2 ** i
            h_kernel = int(math.ceil(previous_conv_size[0] / level))
            w_kernel = int(math.ceil(previous_conv_size[1] / level))
            w_pad1 = int(math.floor((w_kernel * level - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil((w_kernel * level - previous_conv_size[1]) / 2))
            h_pad1 = int(math.floor((h_kernel * level - previous_conv_size[0]) / 2))
            h_pad2 = int(math.ceil((h_kernel * level - previous_conv_size[0]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * level - previous_conv_size[1]) and \
                   h_pad1 + h_pad2 == (h_kernel * level - previous_conv_size[0])

            padded_input = F.pad(input=x, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                                 mode='constant', value=0)
            if self.pool_type == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif self.pool_type == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")

            tensor = pool(padded_input)
            tensor = torch.flatten(tensor, start_dim=1, end_dim=-1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1) #32,128,512
        return x

class PositionalEncodingOri(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncodingOri, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)#(1,N,d)

    def forward(self, x):
        # x(B,N,d)
        return x + self.pos_table[:, :x.size(1)].clone().detach()

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class VisualPositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(VisualPositionalEncoding, self).__init__()
        pos_embedding = nn.Parameter(torch.randn(maxlen, emb_size))
        self.pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        # self.register_buffer('visual_pos_embedding', pos_embedding) # NOTICE: not learned, it's deterministic

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class CNNEmbedding(nn.Module):
    def __init__(self, outputSize, spp):
        super(CNNEmbedding, self).__init__()
        self.spp = spp
        self.cnn1 = nn.Sequential(nn.Conv2d(3, 16, (5, 5)), nn.ReLU(), nn.MaxPool2d(5))
        self.cnn2 = nn.Sequential(nn.Conv2d(16, 32, (3, 3)), nn.ReLU())
        if spp == 0:
            self.cnn2 = nn.Sequential(nn.Conv2d(16, 32, (3, 3)), nn.ReLU(), nn.MaxPool2d(3))
            self.fc = nn.Linear(1440, outputSize)
        elif spp == 2:
            self.sppLayer = SPPLayer(num_levels=2)
            self.fc = nn.Linear(160, outputSize)
        elif spp == 3:
            self.sppLayer = SPPLayer(num_levels=3)
            self.fc = nn.Linear(672, outputSize)
        elif spp == 4:
            self.sppLayer = SPPLayer(num_levels=4)
            self.fc = nn.Linear(2720, outputSize)

        #nn.init.kaiming_normal_(self.fc.weight, mode='fan_in',
        #                        nonlinearity='leaky_relu')

    def forward(self, x: Tensor, patch_in_batch):
        if patch_in_batch:
            # 4, 28, 150, 93, 3
            b, l = x.size()[0], x.size()[1]
            w, h = x.size()[2], x.size()[3]
            input = x.contiguous().view(-1, w, h, 3).permute(0, 3, 1, 2)
            output = self.cnn1(input) # b, 16, 29, 17
            output = self.cnn2(output) # 112, 32, 9, 5
            if self.spp == 0:
                outputs = torch.flatten(output, start_dim=1, end_dim=-1)
            else:
                outputs = self.sppLayer(output)
            outputs = self.fc(outputs)
            return outputs.view(b, l, -1)
        else:
            b, l = len(x), x[0].size()[0]
            # w, h = x[0].size()[1], x[0].size()[2]
            outputs = []
            for i in range(b):
                #print(i)
                tokens = x[i]
                input = tokens.permute(0, 3, 1, 2)
                output = self.cnn1(input)  # b, 3, 96, 128 to b, 16, 18, 24
                output = self.cnn2(output)
                output = self.sppLayer(output)
                outputs.append(output)
            outputs = torch.stack(outputs)
            outputs = self.fc(outputs)
            return outputs.view(b, l, -1).transpose(0, 1)


def mag(x):
    return math.sqrt(sum(i ** 2 for i in x))


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 #src_vocab_size: int,
                 tgt_vocab_size: int,
                 input_dimension: int,
                 dim_feedforward: int,
                 #posOption: int,
                 functionChoice: str,
                 alpha: float,
                 changeX: str,
                 CAVersion: int,
                 CA_head: int,
                 CA_d_k: int,
                 spp: int,
                 PE_path: int,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.generator = nn.Linear(emb_size, tgt_vocab_size).float()

        self.onedpositional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)
        self.pe_embed = int(emb_size/2)
        self.functionChoice = functionChoice
        if functionChoice == 'original_update':
            self.threedSin_wine = getSinPositional(3, int(emb_size / 2), functionChoice,
                                                   alpha, 'wine', changeX=changeX)  # 1, 3, 9, 2, 256
            self.threedSin_yogurt = getSinPositional(3, int(emb_size / 2), functionChoice,
                                                     alpha, 'yogurt', changeX=changeX)
            # finetune original embedding
            self.threedSin_wine = finetune_embedding(self.threedSin_wine, 'wine')
            self.threedSin_yogurt = finetune_embedding(self.threedSin_yogurt, 'yogurt')
        elif functionChoice == 'learned':
            try:
                pe = np.load(PE_path) # 11, 86
            except:
                print('Learned PE is not found. It is generated by running positionalEncodingLearned.py.')
                quit()
            # norm to mag=1
            '''for i in range(pe.shape[0]):
                magnitude = mag(pe[i])
                pe[i] /= magnitude'''
            pe = torch.from_numpy(pe).to(DEVICE)

            extra_token = torch.zeros((4, pe.size()[1])).to(DEVICE)
            self.pe = torch.cat((pe, extra_token), dim=0)
        else:
            self.threedSin = getSinPositional(3, int(emb_size / 2), functionChoice, alpha, 'all', changeX=changeX)


        self.cnn_embedding = CNNEmbedding(int(emb_size/2), spp)
        self.LinearEmbedding = nn.Linear(input_dimension, int(emb_size/2))

        if CAVersion == 3:
            self.cross_attentions = nn.ModuleList()
            self.CA_head = CA_head
            for _ in range(CA_head):
                self.cross_attentions.append(Cross_Attention(emb_size, CA_d_k))
            self.readout = nn.Linear(CA_head, 1)

        self.CAVersion = CAVersion

    def getCNNFeature(self, src_img: Tensor):
        with torch.no_grad():
            src_cnn_emb = self.cnn_embedding(src_img).transpose(0, 1)
        return src_cnn_emb

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_img: Tensor,
                tgt_img: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor,
                dataset=None,
                patch_in_batch=True):
        if self.functionChoice != 'original_update' and self.functionChoice != 'learned':
            threed_pe = self.threedSin
        elif self.functionChoice != 'learned':
            if dataset is None:
                #threed_pe = self.threedSin
                shelf_len = src.size()[0]
                if shelf_len == 23:
                    threed_pe = self.threedSin_wine
                elif shelf_len == 28:
                    threed_pe = self.threedSin_yogurt
            elif dataset == 0:
                threed_pe = self.threedSin_yogurt
            elif dataset == 1:
                threed_pe = self.threedSin_wine
        src_cnn_emb = self.cnn_embedding(src_img, patch_in_batch).transpose(0, 1) #28, 4, 256
        if self.functionChoice != 'learned':
            src_pos_emb = calculate3DPositional(threed_pe, src).to(DEVICE)
        else:
            src_pos_emb = calculate3DPositional_learned(self.pe, src, self.pe_embed).to(DEVICE)

        src_emb = torch.cat((src_cnn_emb, src_pos_emb), dim=2) #28, 1, 384(256+128)
        #src_emb = self.positional_encoding(src_emb) #CHANGE: use positional encoding as well

        tgt_cnn_emb = self.cnn_embedding(tgt_img, patch_in_batch).transpose(0, 1)  # 28, 4, 256
        #tgt_pos_emb = self.tgt_tok_emb(trg)  # 28, 4, 256

        #tgt_pos_emb = self.LinearEmbedding(trg)
        if self.functionChoice != 'learned':
            tgt_pos_emb = calculate3DPositional(threed_pe, trg).to(DEVICE)
        else:
            tgt_pos_emb = calculate3DPositional_learned(self.pe, trg, self.pe_embed).to(DEVICE)

        tgt_emb = torch.cat((tgt_cnn_emb, tgt_pos_emb), dim=2)
        tgt_emb = self.onedpositional_encoding(tgt_emb)

        encoder_out = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        decoder_out = self.transformer_decoder(tgt_emb, encoder_out, tgt_mask, None, tgt_padding_mask,
                                               memory_key_padding_mask)
        if self.CAVersion == 3:
            att_matrices = []
            for layer in self.cross_attentions:
                att = layer(encoder_out, decoder_out)
                att_matrices.append(att)
            out = torch.cat(att_matrices, dim=-1)
            if self.CA_head != 1:
                out = self.readout(out)
            return out.squeeze(-1)  # 8, 2, 29
        else:
            return self.generator(decoder_out)

    def encode(self, src: Tensor, src_mask: Tensor):
        return 0

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return 0


class Cross_Attention(nn.Module):
    def __init__(self, emb_size, d_k):
        super(Cross_Attention, self).__init__()
        self.d_k = d_k
        self.decode_fc = nn.Linear(emb_size, emb_size)
        self.encode_fc = nn.Linear(emb_size, emb_size)
        self.placeholder = torch.zeros((1, 1, 1)).to(DEVICE)

    def forward(self, encoder_out, decoder_out):
        encoder_out = self.encode_fc(encoder_out).permute(1, 0, 2)  # 2, 28, 512
        decoder_out = self.decode_fc(decoder_out).permute(1, 2, 0)  # 2, 512, 8
        out = torch.bmm(encoder_out, decoder_out).permute(2, 0, 1)  # b, 28, 8
        placeholder = self.placeholder.repeat(out.size()[0], out.size()[1], 1)
        out = torch.cat((out, placeholder), dim=-1) / (self.d_k ** 0.5)
        return out.unsqueeze(-1)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, PAD_IDX):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask





