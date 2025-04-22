import torch
from torch.utils.data import Dataset
import numpy as np
from collections import deque


class SessionTransitionDataset(Dataset):
    def __init__(self, 
                 df, 
                 numerical_feature_cols, 
                 categorical_feature_cols, 
                 max_history_length, 
                 padding_idx, 
                 reward_map, 
                 default_reward, 
                 min_history_length=1):
        
        self.numerical_feature_cols = numerical_feature_cols
        self.categorical_feature_cols = categorical_feature_cols
        self.num_numerical_features = len(numerical_feature_cols)
        self.max_history_length = max_history_length
        self.padding_idx = padding_idx
        self.reward_map = reward_map
        self.default_reward = default_reward
        self.min_history_length = min_history_length

        df[self.numerical_feature_cols] = df[self.numerical_feature_cols].fillna(0.0)
        self.transitions = self._prepare_transitions(df)

    def _create_user_transitions(self, user_df):
        """
        Обрабатывает DataFrame одного пользователя и создает список переходов.
        """
        user_transitions = []
        user_history = deque(maxlen=self.max_history_length)

        product_indices = user_df['product_id_idx'].to_numpy()
        event_types = user_df['event_type'].to_numpy()
        session_ids = user_df['user_session'].to_numpy()
        # <<< Извлекаем числовые фичи >>>
        numerical_features = user_df[self.numerical_feature_cols].to_numpy(dtype=np.float32)
        # <<< Извлекаем категориальные индексы >>>
        brand_indices = user_df['brand'].to_numpy(dtype=np.int64)
        holiday_indices = user_df['holiday_name'].to_numpy(dtype=np.int64)

        last_session_id = None
        # <<< Храним последнюю известную числовую и категориальную инфу >>>
        last_valid_numerical_features = np.zeros(self.num_numerical_features, dtype=np.float32)
        last_valid_brand_idx = 0 # Используем 0 как дефолтный/неизвестный индекс
        last_valid_holiday_idx = 0

        for i in range(len(user_df)):
            current_product_idx = product_indices[i]
            current_event_type = event_types[i]
            current_session_id = session_ids[i]
            # <<< Текущие числовые фичи и категориальные индексы >>>
            current_numerical_features = numerical_features[i]
            current_brand_idx = brand_indices[i]
            current_holiday_idx = holiday_indices[i]

            is_done = (last_session_id is not None and current_session_id != last_session_id) or (i == len(user_df) - 1)

            # Состояние ДО (`state_t`)
            current_history_list = list(user_history)
            state_len = len(current_history_list)
            state_history_padded = np.full(self.max_history_length, self.padding_idx, dtype=np.int64)
            if state_len > 0:
                state_history_padded[-state_len:] = current_history_list

            # <<< Фичи и индексы для state_t (с предыдущего шага) >>>
            state_numerical_features_np = last_valid_numerical_features.copy()
            state_brand_idx_np = last_valid_brand_idx
            state_holiday_idx_np = last_valid_holiday_idx

            # Действие и награда (текущий шаг i)
            action = current_product_idx
            reward = self.reward_map.get(current_event_type, self.default_reward)

            # Состояние ПОСЛЕ (`state_{t+1}`)
            temp_next_history_list = current_history_list + [action]
            next_state_len = min(len(temp_next_history_list), self.max_history_length)
            next_state_history_padded = np.full(self.max_history_length, self.padding_idx, dtype=np.int64)
            actual_next_history = temp_next_history_list[-next_state_len:]
            if next_state_len > 0:
                 next_state_history_padded[-next_state_len:] = actual_next_history

            # <<< Фичи и индексы для next_state_{t+1} (текущий шаг i) >>>
            next_state_numerical_features_np = current_numerical_features.copy()
            next_state_brand_idx_np = current_brand_idx
            next_state_holiday_idx_np = current_holiday_idx

            if state_len >= self.min_history_length:
                 user_transitions.append({
                    'state_history': torch.from_numpy(state_history_padded),
                    'state_length': torch.tensor(state_len, dtype=torch.long),
                    'state_numerical_features': torch.from_numpy(state_numerical_features_np), # <<< Числовые фичи состояния
                    'state_brand_idx': torch.tensor(state_brand_idx_np, dtype=torch.long), # <<< Индекс бренда состояния
                    'state_holiday_idx': torch.tensor(state_holiday_idx_np, dtype=torch.long), # <<< Индекс праздника состояния
                    'action': torch.tensor(action, dtype=torch.long),
                    'reward': torch.tensor(reward, dtype=torch.float32),
                    'next_state_history': torch.from_numpy(next_state_history_padded),
                    'next_state_length': torch.tensor(next_state_len, dtype=torch.long),
                    'next_state_numerical_features': torch.from_numpy(next_state_numerical_features_np), # <<< Числовые фичи след. состояния
                    'next_state_brand_idx': torch.tensor(next_state_brand_idx_np, dtype=torch.long), # <<< Индекс бренда след. состояния
                    'next_state_holiday_idx': torch.tensor(next_state_holiday_idx_np, dtype=torch.long), # <<< Индекс праздника след. состояния
                    'done': torch.tensor(is_done, dtype=torch.bool)
                 })

            # Обновляем историю и "последние известные" значения для след. итерации
            user_history.append(current_product_idx)
            last_valid_numerical_features = current_numerical_features
            last_valid_brand_idx = current_brand_idx
            last_valid_holiday_idx = current_holiday_idx
            last_session_id = current_session_id

        return user_transitions

    def _prepare_transitions(self, df):
        df_sorted = df.sort_values(by=['user_id', 'event_time'])
        all_user_transitions = df_sorted.groupby(
            'user_id', 
            group_keys=False).apply(
                lambda x: self._create_user_transitions(x.copy()),
                include_groups=False
        )
        return [transition for user_list 
                in all_user_transitions 
                for transition in user_list]


    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx]


