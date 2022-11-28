from typing import Callable, Optional, Dict, Any
import torch
import torch.nn
import torch.nn.functional as F
# from .universal_transformer import UniversalTransformerEncoderWithLayer, UniversalTransformerDecoderWithLayer
from .multi_head_relative_pos_attention import FixedRelativeMultiheadAttention, AttentionMask
from .multi_head_attention import MultiHeadAttention
from layers.layer_with_visualization import LayerWithVisualization
from layers.transformer.multi_head_relative_pos_attention import FixedRelativeMultiheadAttentionBase
from layers.regularized_layer import RegularizedLayer
import framework
import numpy as np
import math
from .ndr_geometric import NDRGeometric

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]

class UniversalTransformerRandomLayerEncoder(torch.nn.Module):
    def __init__(self, layer, n_layers: int , n_extra: int = 0, n_test: Optional[int] = None, *args, **kwargs):
        super().__init__()
        self.layer = layer(*args, **kwargs)
        self.n_extra = n_extra
        self.n_layers = n_layers
        self.n_test = n_test

    def set_n_layers(self, n_layers: int):
        self.layers = [self.layer] * n_layers

    def forward(self, data: torch.Tensor, *args, **kwargs):
        self.set_n_layers(np.random.randint(self.n_layers, self.n_extra + self.n_layers + 1) if self.training else \
                          (self.n_test or self.n_layers))
        for l in self.layers:
            data = l(data, *args, **kwargs)
        return data


def UniversalTransformerRandomLayerEncoderWithLayer(layer):
    return lambda *args, **kwargs: UniversalTransformerRandomLayerEncoder(layer, *args, **kwargs)


class NDRResidual(RegularizedLayer, LayerWithVisualization):
    def __init__(self, d_model: int, nhead: int, dropout: float, scalar_gate: bool = False,
                abs_gate:bool = True, attention_dropout=0, p_gate_drop=0, **kwargs):
        super().__init__()
        self.plot_cache = []

        self.reg_loss = 0

        self.att = FixedRelativeMultiheadAttention(d_model, nhead, dropout=attention_dropout, absolute_gate=abs_gate)

        self.p1 = torch.nn.Linear(d_model, d_model*4)
        self.p2 = torch.nn.Linear(d_model*4, d_model)


        self.g1 = torch.nn.Linear(d_model, d_model)
        self.g2 = torch.nn.Linear(d_model, 1 if scalar_gate else d_model)

        self.n1 = torch.nn.LayerNorm(d_model)
        self.nmerge = torch.nn.LayerNorm(d_model)

        self.drop = torch.nn.Dropout(dropout)

        self.g2.bias.data.fill_(-3)

        self.p_gate_drop = p_gate_drop

        self.reset_parameters()

    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None) -> torch.Tensor:
        
        input = self.att(src, src, mask)

        net = self.nmerge(src + self.drop(input))

        mid = self.drop(torch.relu(self.p1(net)))
        # proj = torch.relu(self.p2(mid))
        proj = self.p2(mid)
        # proj = self.n1(proj) #* self.scale
        proj = torch.tanh(proj)

        gate = self.g2(self.drop(torch.relu(self.g1(net))))
        bgate = torch.sigmoid(gate)
        # bgate = torch.softmax(gate, -2)

        if self.training and self.p_gate_drop>0:
            bgate = bgate.masked_fill(torch.rand(*bgate.shape[:-1], 1, device=bgate.device, dtype=bgate.dtype) < self.p_gate_drop, 0)

        if self.visualization_enabled:
            self.plot_cache.append(bgate[0])

        src = src * (1-bgate) + proj * bgate


        # self.add_reg(lambda: 0.0000001 * ((bgate+0.2).clamp(max=1)).sum() / bgate.shape[0])
        # self.add_reg(lambda: 0.000001 * (bgate * (1-bgate)).sum() / bgate.shape[0])

        return src

    def plot(self, options: Dict[str, Any]) -> Dict[str, Any]:
        r = {}
        if self.visualization_enabled:
            r["gate"] = framework.visualize.plot.AnimatedHeatmap(
                        torch.stack(self.plot_cache, 0).transpose(1,2),
                        ylabel="dest", xlabel="src", textval=False, x_marks=options.get("steplabel"))
            self.plot_cache.clear()

        return r


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.p1.weight, gain=torch.nn.init.calculate_gain('relu'))
        # torch.nn.init.xavier_uniform_(self.p2.weight)  

        torch.nn.init.xavier_uniform_(self.g1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.g2.weight, gain=torch.nn.init.calculate_gain('sigmoid'))



class NDREncoder(torch.nn.Module):
        
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                dim_feedforward: int = 2048, dropout: float = 0.1,
                activation: ActivationFunction = F.relu, encoder_layer=UniversalTransformerRandomLayerEncoderWithLayer,
                attention_dropout: float = 0, attention_type: str = 'regular', n_input_tokens: int = 300):

        super().__init__()
        self.pos = framework.layers.PositionalEncoding(d_model, max_len=100, batch_first=True,
                                        scale= 1.0)
        self.register_buffer('int_seq', torch.arange(100, dtype=torch.long))

        self.encoder = encoder_layer(layer = NDRGeometric if attention_type == 'geometric' else NDRResidual)(n_layers = num_encoder_layers, d_model = d_model, nhead = nhead, dim_feedforward = dim_feedforward,
                                    dropout = dropout, activation = activation, attention_dropout = attention_dropout)
        
        self.embedding = torch.nn.Embedding(n_input_tokens + 3, d_model)

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[: max_len] >= len.unsqueeze(-1)

    def forward(self, input_ids: torch.Tensor, src_len: torch.Tensor, attention_mask: Optional[AttentionMask] = None):

        src = self.pos(self.embedding(input_ids.long()), 0)
        in_len_mask = self.generate_len_mask(src.shape[1], src_len)
        memory = self.encoder(src, AttentionMask(in_len_mask, None))
        return memory
