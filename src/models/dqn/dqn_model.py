from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torch import nn
import torch


class DQNRecommender(nn.Module):
    def __init__(self, num_products, product_embedding_dim, gru_hidden_size, padding_idx,
                 num_brands, brand_embedding_dim, num_holidays, holiday_embedding_dim,
                 num_numerical_features, intermediate_layer_size):
        super().__init__()
        self.padding_idx = padding_idx

        
        self.product_embedding = nn.Embedding(num_products, product_embedding_dim, padding_idx=padding_idx)
        self.brand_embedding = nn.Embedding(num_brands, brand_embedding_dim)
        self.holiday_embedding = nn.Embedding(num_holidays, holiday_embedding_dim)

        
        self.gru = nn.GRU(product_embedding_dim, gru_hidden_size, batch_first=True)

        
        combined_feature_size = (
            gru_hidden_size +
            num_numerical_features +
            brand_embedding_dim +
            holiday_embedding_dim
        )
        self.intermediate_layer = nn.Linear(combined_feature_size, intermediate_layer_size)
        self.out_layer = nn.Linear(intermediate_layer_size, num_products)


    def forward(self, state_history, lengths, state_numerical_features, state_brand_idx, state_holiday_idx):
        
        lengths = lengths.to(state_history.device)
        lengths_cpu = torch.clamp(lengths, min=1).cpu() 

        product_embedded = self.product_embedding(state_history)
        packed_embedded = pack_padded_sequence(
            product_embedded, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.gru(packed_embedded)
        last_hidden_state = hidden.squeeze(0) # [batch_size, gru_hidden_size]

        
        brand_embedded = self.brand_embedding(state_brand_idx) # [batch_size, brand_embedding_dim]
        holiday_embedded = self.holiday_embedding(state_holiday_idx) # [batch_size, holiday_embedding_dim]

        
        combined_state = torch.cat([
            last_hidden_state,
            state_numerical_features,
            brand_embedded,
            holiday_embedded
        ], dim=1) 

        
        hidden_after_concat = F.relu(self.intermediate_layer(combined_state))

        
        q_values = self.out_layer(hidden_after_concat) # [batch_size, num_products]
        return q_values