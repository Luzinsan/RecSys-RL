import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    Общий GRU-энкодер + две головы:
    - policy_head: logits по действиям
    - value_head: одиночное значение состояния
    """
    def __init__(
        self,
        num_products: int,
        product_embedding_dim: int,
        gru_hidden_size: int,
        padding_idx: int,
        num_brands: int,
        brand_embedding_dim: int,
        num_holidays: int,
        holiday_embedding_dim: int,
        num_numerical_features: int,
        intermediate_layer_size: int
    ):
        super().__init__()
        # эмбеддинги
        self.product_embedding = nn.Embedding(num_products, product_embedding_dim, padding_idx=padding_idx)
        self.brand_embedding   = nn.Embedding(num_brands, brand_embedding_dim)
        self.holiday_embedding = nn.Embedding(num_holidays, holiday_embedding_dim)
        # GRU
        self.gru = nn.GRU(product_embedding_dim, gru_hidden_size, batch_first=True)
        # промежуточный слой
        combined = gru_hidden_size + num_numerical_features + brand_embedding_dim + holiday_embedding_dim
        self.inter = nn.Linear(combined, intermediate_layer_size)
        # policy & value
        self.policy_head = nn.Linear(intermediate_layer_size, num_products)
        self.value_head  = nn.Linear(intermediate_layer_size, 1)

    def forward(self,
                state_history: torch.Tensor,
                lengths:       torch.Tensor,
                num_feats:     torch.Tensor,
                brand_idx:     torch.Tensor,
                hol_idx:       torch.Tensor):
        # GRU
        lengths_cpu = torch.clamp(lengths.cpu(), min=1)
        emb = self.product_embedding(state_history)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        h = h_n.squeeze(0)                      # [B, hidden]
        # категориальные
        b = self.brand_embedding(brand_idx)     # [B, brand_dim]
        h_emb = self.holiday_embedding(hol_idx) # [B, hol_dim]
        # собрать состояние
        x = torch.cat([h, num_feats, b, h_emb], dim=1)
        x = F.relu(self.inter(x))
        logits = self.policy_head(x)            # [B, num_products]
        value  = self.value_head(x).squeeze(-1) # [B]
        return logits, value
