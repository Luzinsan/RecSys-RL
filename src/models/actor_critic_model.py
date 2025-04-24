import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    Enhanced Actor-Critic model with:
    - Layer normalization
    - Dropout for regularization
    - Improved feature processing
    - Separate policy and value networks
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
        intermediate_layer_size: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        # Embeddings
        self.product_embedding = nn.Embedding(num_products, product_embedding_dim, padding_idx=padding_idx)
        self.brand_embedding = nn.Embedding(num_brands, brand_embedding_dim)
        self.holiday_embedding = nn.Embedding(num_holidays, holiday_embedding_dim)
        
        # GRU with layer normalization
        self.gru = nn.GRU(product_embedding_dim, gru_hidden_size, batch_first=True)
        self.gru_norm = nn.LayerNorm(gru_hidden_size)
        
        # Feature processing
        self.feature_norm = nn.LayerNorm(num_numerical_features)
        self.brand_norm = nn.LayerNorm(brand_embedding_dim)
        self.holiday_norm = nn.LayerNorm(holiday_embedding_dim)
        
        # Shared layers
        combined_size = gru_hidden_size + num_numerical_features + brand_embedding_dim + holiday_embedding_dim
        self.shared_layer = nn.Sequential(
            nn.Linear(combined_size, intermediate_layer_size),
            nn.LayerNorm(intermediate_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(intermediate_layer_size, intermediate_layer_size // 2),
            nn.LayerNorm(intermediate_layer_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_layer_size // 2, num_products)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(intermediate_layer_size, intermediate_layer_size // 2),
            nn.LayerNorm(intermediate_layer_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_layer_size // 2, 1)
        )

    def forward(self,
                state_history: torch.Tensor,
                lengths: torch.Tensor,
                num_feats: torch.Tensor,
                brand_idx: torch.Tensor,
                hol_idx: torch.Tensor):
        # Process product history through GRU
        lengths_cpu = torch.clamp(lengths.cpu(), min=1)
        emb = self.product_embedding(state_history)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        h = self.gru_norm(h_n.squeeze(0))
        
        # Process categorical features
        b = self.brand_norm(self.brand_embedding(brand_idx))
        h_emb = self.holiday_norm(self.holiday_embedding(hol_idx))
        num_feats = self.feature_norm(num_feats)
        
        # Combine features
        x = torch.cat([h, num_feats, b, h_emb], dim=1)
        x = self.shared_layer(x)
        
        # Get policy and value outputs
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        
        return logits, value
