from typing import Optional, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class SigLIPConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    layer_norm_eps: int = 1e-6
    attention_dropout: int = 0.0
    num_image_tokens: int = None

