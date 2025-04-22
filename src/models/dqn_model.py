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

        # Embedding слои
        self.product_embedding = nn.Embedding(num_products, product_embedding_dim, padding_idx=padding_idx)
        self.brand_embedding = nn.Embedding(num_brands, brand_embedding_dim)
        self.holiday_embedding = nn.Embedding(num_holidays, holiday_embedding_dim)

        # GRU для истории продуктов
        self.gru = nn.GRU(product_embedding_dim, gru_hidden_size, batch_first=True)

        # Рассчитываем общий размер вектора состояния после конкатенации
        combined_feature_size = (
            gru_hidden_size +
            num_numerical_features +
            brand_embedding_dim +
            holiday_embedding_dim
        )
        self.intermediate_layer = nn.Linear(combined_feature_size, intermediate_layer_size)
        self.out_layer = nn.Linear(intermediate_layer_size, num_products)


    def forward(self, state_history, lengths, state_numerical_features, state_brand_idx, state_holiday_idx):
        # 1. Обработка истории продуктов через GRU
        lengths = lengths.to(state_history.device)
        lengths_cpu = torch.clamp(lengths, min=1).cpu() # Для pack_padded_sequence

        product_embedded = self.product_embedding(state_history)
        packed_embedded = pack_padded_sequence(
            product_embedded, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.gru(packed_embedded)
        last_hidden_state = hidden.squeeze(0) # [batch_size, gru_hidden_size]

        # 2. Получение эмбеддингов для категориальных фичей
        brand_embedded = self.brand_embedding(state_brand_idx) # [batch_size, brand_embedding_dim]
        holiday_embedded = self.holiday_embedding(state_holiday_idx) # [batch_size, holiday_embedding_dim]

        # 3. Конкатенация всех компонент состояния
        combined_state = torch.cat([
            last_hidden_state,
            state_numerical_features,
            brand_embedded,
            holiday_embedded
        ], dim=1) # Конкатенируем по оси признаков

        # 4. Пропускаем через новый слой + ReLU
        hidden_after_concat = F.relu(self.intermediate_layer(combined_state))

        # 5. Пропускаем через финальный слой для получения Q-значений
        q_values = self.out_layer(hidden_after_concat) # [batch_size, num_products]
        return q_values