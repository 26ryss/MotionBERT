import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class ActionHeadClassification(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_classes=60, num_joints=17, hidden_dim=2048):
        super(ActionHeadClassification, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat = self.fc2(feat)
        return feat

class ActionHeadEmbed(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_joints=17, hidden_dim=2048):
        super(ActionHeadEmbed, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = F.normalize(feat, dim=-1)
        return feat

class ActionNet(nn.Module):
    def __init__(self, backbone, dim_rep=512, dropout_ratio=0., hidden_dim=2048, num_joints=17):
        super(ActionNet, self).__init__()
        self.backbone = backbone
        self.feat_J = num_joints
        self.head = ActionHeadEmbed(dropout_ratio=dropout_ratio, dim_rep=dim_rep, hidden_dim=hidden_dim, num_joints=num_joints)

    def forward(self, x):
        '''
            Input: (N, M x T x 17 x 3)
        '''
        N, M, T, J, C = x.shape
        x = x.reshape(N*M, T, J, C)
        feat = self.backbone.get_representation(x)
        feat = feat.reshape([N, M, T, self.feat_J, -1])      # (N, M, T, J, C)
        out = self.head(feat)
        return out

class TokenDrop(nn.Module):
    """For a batch of tokens indices, randomly replace a non-specical token.

    Args:
        prob (float): probability of dropping a token
        blank_token (int): index for the blank token
        num_special (int): Number of special tokens, assumed to be at the start of the vocab
    """

    def __init__(self, prob=0.1, blank_token=1, eos_token=102):
        self.prob = prob
        self.eos_token = eos_token
        self.blank_token = blank_token

    def __call__(self, sample):
        # Randomly sample a bernoulli distribution with p=prob
        # to create a mask where 1 means we will replace that token
        mask = torch.bernoulli(self.prob * torch.ones_like(sample)).long()

        # only replace if the token is not the eos token
        can_drop = (~(sample == self.eos_token)).long()
        mask = mask * can_drop

        # Do not replace the sos tokens
        mask[:, 0] = torch.zeros_like(mask[:, 0]).long()

        replace_with = (self.blank_token * torch.ones_like(sample)).long()

        sample_out = (1 - mask) * sample + mask * replace_with

        return sample_out


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Define a decoder module for the Transformer architecture
class Decoder(nn.Module):
    def __init__(self, num_emb, hidden_size=128, num_layers=3, num_heads=4):
        super(Decoder, self).__init__()

        # Create an embedding layer for tokens
        self.embedding = nn.Embedding(num_emb, hidden_size)
        # Initialize the embedding weights
        self.embedding.weight.data = 0.001 * self.embedding.weight.data

        # Initialize sinusoidal positional embeddings
        self.pos_emb = SinusoidalPosEmb(hidden_size)

        # Create multiple transformer blocks as layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size,
                                                    nhead=num_heads,
                                                    dim_feedforward=hidden_size*4,
                                                    dropout=0.1,
                                                    batch_first=True)

        self.decoder_layers = nn.TransformerDecoder(decoder_layer, num_layers)

        # Define a linear layer for output prediction
        self.fc_out = nn.Linear(hidden_size, num_emb)

    def forward(self, input_seq, encoder_output, input_padding_mask=None,
                encoder_padding_mask=None):
        # Embed the input sequence
        input_embs = self.embedding(input_seq)
        bs, l, h = input_embs.shape

        # Add positional embeddings to the input embeddings
        seq_indx = torch.arange(l, device=input_seq.device)
        pos_emb = self.pos_emb(seq_indx).reshape(1, l, h).expand(bs, l, h)
        embs = input_embs + pos_emb

        causal_mask = torch.triu(torch.ones(l, l, device=input_seq.device), 1).bool()

        output = self.decoder_layers(embs, encoder_output, tgt_mask=causal_mask,
                                     tgt_key_padding_mask=input_padding_mask,
                                     memory_key_padding_mask=encoder_padding_mask)

        return self.fc_out(output)

class EncoderDecoder(nn.Module):
    def __init__(self, backbone, dim_rep=512, dropout_ratio=0., enc_hidden_dim=2048, num_joints=17, vocab_size=60, num_layers=3, num_heads=4):
        super(EncoderDecoder, self).__init__()
        self.backbone = backbone
        self.encoder = ActionNet(backbone, dim_rep, dropout_ratio, enc_hidden_dim, num_joints) # outputs enc_hiddem_dim vector
        self.decoder = Decoder(vocab_size, enc_hidden_dim, num_layers, num_heads) # outputs vocab_size vector

    def forward(self, motion, target_seq, padding_mask):
        bool_padding_mask = padding_mask == 0
        encode_seq = self.encoder(motion).unsqueeze(1)
        encoder_padding_mask = torch.zeros(encode_seq.size(0), 1, dtype=torch.bool, device=encode_seq.device)
        output = self.decoder(
            target_seq,
            encode_seq,
            input_padding_mask=bool_padding_mask,
            encoder_padding_mask=encoder_padding_mask
        )
        return output
