from typing import Optional
import torch
import torch.nn
import torch.nn.functional as F
from .transformer import  ActivationFunction, TransformerEncoderLayer, TransformerDecoderLayer, \
                         Transformer
import framework
from .multi_head_attention import MultiHeadAttention, AttentionMask


class UniversalTransformerEncoder(torch.nn.Module):
    def __init__(self, layer, n_layers: int, *args, **kwargs):
        super().__init__()
        self.layer = layer(*args, **kwargs)
        self.set_n_layers(n_layers)

    def set_n_layers(self, n_layers: int):
        self.layers = [self.layer] * n_layers

    def forward(self, data: torch.Tensor, *args, **kwargs):
        for l in self.layers:
            data = l(data, *args, **kwargs)
        return data


# class UniversalTransformerDecoder(TransformerDecoderBase):
#     def __init__(self, layer, n_layers: int, d_model: int, *args, **kwargs):
#         super().__init__(d_model)
#         self.layer = layer(d_model, *args, **kwargs)
#         self.set_n_layers(n_layers)

#     def set_n_layers(self, n_layers: int):
#         self.layers = [self.layer] * n_layers

#     def forward(self, data: torch.Tensor, *args, **kwargs):
#         for l in self.layers:
#             data = l(data, *args, **kwargs)
#         return data


def UniversalTransformerEncoderWithLayer(layer=TransformerEncoderLayer):
    return lambda *args, **kwargs: UniversalTransformerEncoder(layer, *args, **kwargs)


# def UniversalTransformerDecoderWithLayer(layer=TransformerDecoderLayer):
#     return lambda *args, **kwargs: UniversalTransformerDecoder(layer, *args, **kwargs)


# class UniversalTransformer(Transformer):
#     def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
#                  num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
#                  activation: ActivationFunction = F.relu, attention_dropout: float = 0.0):

#         super().__init__(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation,
#                          UniversalTransformerEncoderWithLayer(),
#                          UniversalTransformerDecoderWithLayer(), attention_dropout)

class UTEncoder(torch.nn.Module):
        
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                dim_feedforward: int = 2048, dropout: float = 0.1,
                activation: ActivationFunction = F.relu, encoder_layer=UniversalTransformerEncoderWithLayer,
                attention_dropout: float = 0, n_input_tokens: int = 300):

        super().__init__()
        self.pos = framework.layers.PositionalEncoding(d_model, max_len=100, batch_first=True,
                                        scale= 1.0)
        self.register_buffer('int_seq', torch.arange(100, dtype=torch.long))

        self.encoder = encoder_layer()(n_layers = num_encoder_layers, d_model = d_model, nhead = nhead, dim_feedforward = dim_feedforward,
                                    dropout = dropout, activation = activation, attention_dropout = attention_dropout)
        
        self.embedding = torch.nn.Embedding(n_input_tokens + 3, d_model)

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[: max_len] >= len.unsqueeze(-1)

    def forward(self, input_ids: torch.Tensor, src_len: torch.Tensor, attention_mask: Optional[AttentionMask] = None):

        src = self.pos(self.embedding(input_ids.long()), 0)
        in_len_mask = self.generate_len_mask(src.shape[1], src_len)
        memory = self.encoder(src, AttentionMask(in_len_mask, None))
        return memory