import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model

    def forward(self):
        # print('Positional Encoding')
        pos = torch.arange(self.max_len, dtype=torch.float).unsqueeze(1)  # (max_len,1)
        _i = torch.arange(self.d_model, dtype=torch.float).unsqueeze(0)  # (1,d_model)

        _i = 1 / torch.pow(torch.tensor(10000.0), (2 * (_i // 2)) / self.d_model)
        angles = pos * _i  # (max_len * d_model)

        angles[:, 0::2] = torch.sin(angles[:, 0::2])  # even indices
        angles[:, 1::2] = torch.cos(angles[:, 1::2])  # odd indices

        return angles[:self.max_len].unsqueeze(0)  # convert it to (1,max_len, d_model) so it can be added to a batch


def scaled_dot_product_attention(q, k, v, s_mask=None):
    d_k = q.shape[-1]
    scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if s_mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + s_mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    output = torch.matmul(attention, v)
    return output, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, _input_dim, d_model, _num_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert d_model % _num_heads == 0
        self.num_heads = _num_heads
        self.input_dim = _input_dim
        self.d_model = d_model
        self.head_dim = d_model // _num_heads
        # This Represents multiplying input embeddings with W_q, W_k, W_v with will result then in Q, K, V matrices
        # This is a normal Dense Layer like tf.
        # Input matrix is seq_len * input_dim, W_q is output_dim * input_dim.
        # The same for W_k and W_v
        # Linear Layer = x * W_T + b → W: weights, b: bias, x: input
        # so W_q, W_k, W_v will be input_dim * output_dim which will allow us to preform matmul.
        # For efficiency, we stack all the weight matrices together
        # in one big matrix of size (3 * output_dim) * embedding_size
        # when apply matmul this will result of q, k, v all stacked together
        self.qkv_layer = nn.Linear(_input_dim, 3 * d_model)
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, mha_x, self_mask=None):
        seq_len = mha_x.shape[1]
        batch_size = mha_x.shape[0]
        qkv = self.qkv_layer(mha_x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product_attention(q, k, v, self_mask)
        values = values.permute(0, 2, 1, 3)
        output = values.reshape(batch_size, seq_len, self.head_dim * self.num_heads)
        output = self.output_layer(output)
        return output


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, _num_heads=8):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % _num_heads == 0
        self.num_heads = _num_heads
        self.d_model = d_model
        self.head_dim = d_model // _num_heads
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, mhca_input, encoder_output, cross_mask=None):
        decoder_seq_len = mhca_input.shape[1]
        encoder_seq_len = encoder_output.shape[1]
        batch_size = mhca_input.shape[0]
        kv = self.kv_layer(encoder_output)
        q = self.q_layer(mhca_input)

        kv = kv.reshape(batch_size, encoder_seq_len, self.num_heads, 2 * self.head_dim)
        kv = kv.permute(0, 2, 1, 3)

        q = q.reshape(batch_size, decoder_seq_len, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        k, v = kv.chunk(2, dim=-1)
        values, attention = scaled_dot_product_attention(q, k, v, cross_mask)
        values = values.permute(0, 2, 1, 3)
        output = values.reshape(batch_size, decoder_seq_len, self.head_dim * self.num_heads)
        output = self.output_layer(output)
        return output


class LayerNorm(nn.Module):
    def __init__(self, normalization_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.normalization_shape = normalization_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalization_shape))
        self.beta = nn.Parameter(torch.zeros(normalization_shape))

    def forward(self, layer):
        dim = [-(i + 1) for i in range(len(self.normalization_shape))]
        mean = layer.mean(dim=dim, keepdim=True)
        std = (layer.var(dim=dim, keepdim=True) + self.eps).sqrt()
        y = (layer - mean) / std
        return self.gamma * y + self.beta


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, pos_feed_input):
        pos_feed_input = self.linear1(pos_feed_input)
        pos_feed_input = self.gelu(pos_feed_input)
        pos_feed_input = self.dropout(pos_feed_input)
        pos_feed_input = self.linear2(pos_feed_input)
        return pos_feed_input


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, d_model, num_heads)
        self.layer_norm = LayerNorm([d_model], eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, dropout)

    def forward(self, encoder_layer_input, mask=None):
        encoder_layer_residual = encoder_layer_input
        encoder_layer_input = self.self_attn(encoder_layer_input, mask)
        encoder_layer_input = self.dropout(encoder_layer_input)
        encoder_layer_input = self.layer_norm(encoder_layer_input + encoder_layer_residual)
        encoder_layer_residual = encoder_layer_input
        encoder_layer_input = self.ffn(encoder_layer_input)
        encoder_layer_input = self.dropout(encoder_layer_input)
        encoder_layer_input = self.layer_norm(encoder_layer_input + encoder_layer_residual)
        return encoder_layer_input


class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        sequential_encoder_x, mask = inputs
        for module in self._modules.values():
            sequential_encoder_x = module(sequential_encoder_x, mask)
        return sequential_encoder_x


class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout, num_layers):
        super(Encoder, self).__init__()
        self.layers = SequentialEncoder(
            *[EncoderLayer(d_model, num_heads, ffn_hidden, dropout) for _ in range(num_layers)])

    def forward(self, encoder_input, encoder_self_attention_mask):
        return self.layers(encoder_input, encoder_self_attention_mask)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, d_model, num_heads)
        self.layer_norm = LayerNorm([d_model], eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, dropout)
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads)

    def forward(self, decoder_layer_input, encoder_output, self_attn_mask, cross_attn_mask):
        decoder_layer_residual = decoder_layer_input
        decoder_layer_input = self.self_attn(decoder_layer_input, self_attn_mask)
        decoder_layer_input = self.dropout(decoder_layer_input)
        decoder_layer_input = self.layer_norm(decoder_layer_input + decoder_layer_residual)

        decoder_layer_residual = decoder_layer_input
        decoder_layer_input = self.cross_attn(decoder_layer_input, encoder_output, cross_attn_mask)
        decoder_layer_input = self.dropout(decoder_layer_input)
        decoder_layer_input = self.layer_norm(decoder_layer_input + decoder_layer_residual)

        decoder_layer_residual = decoder_layer_input
        decoder_layer_input = self.ffn(decoder_layer_input)
        decoder_layer_input = self.dropout(decoder_layer_input)
        decoder_layer_input = self.layer_norm(decoder_layer_input + decoder_layer_residual)
        return decoder_layer_input


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        sequential_decoder_x, encoder_output, self_mask, cross_mask = inputs
        for module in self._modules.values():
            sequential_decoder_x = module(sequential_decoder_x, encoder_output, self_mask, cross_mask)
        return sequential_decoder_x


class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout, num_layers):
        super(Decoder, self).__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, num_heads, ffn_hidden, dropout)
                                          for _ in range(num_layers)])

    def forward(self, decoder_input, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        return self.layers(decoder_input, encoder_output, self_attn_mask, cross_attn_mask)


class Transformer(nn.Module):
    def __init__(self,
                 d_model,
                 ffn_hidden,
                 num_heads,
                 drop_prob,
                 num_encoder_layers,
                 num_decoder_layers
                 ):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_encoder_layers)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_decoder_layers)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self,
                src,
                tgt,
                encoder_self_attention_mask=None,
                decoder_self_attention_mask=None,
                decoder_cross_attention_mask=None):
        src = self.encoder(src, encoder_self_attention_mask)
        out = self.decoder(tgt,src, decoder_self_attention_mask, decoder_cross_attention_mask)
        return out
