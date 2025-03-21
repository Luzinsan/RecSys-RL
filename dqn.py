import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import optuna

# Конфигурация по умолчанию (будут перезаписаны Optuna)
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_RECOMMENDATIONS = 5
STATE_SIZE = 16 # hidden_size

# Создание фичей
def create_features(df):
    # Фичи пользователя
    user_stats = df.groupby('user_id')['event_type'].agg(
        total_views=lambda x: (x == 'view').sum(),
        total_carts=lambda x: (x == 'cart').sum(),
        total_purchases=lambda x: (x == 'purchase').sum()
    ).reset_index()

    # Фичи продукта
    product_stats = df.groupby('product_id').agg(
        product_views=('event_type', lambda x: (x == 'view').sum()),
        product_purchases=('event_type', lambda x: (x == 'purchase').sum()),
        avg_price=('price', 'mean')
    ).reset_index()

    # Фичи категорий
    category_stats = df.groupby('category_code').agg(
        category_views=('event_type', lambda x: (x == 'view').sum()),
        avg_category_price=('price', 'mean')
    ).reset_index()

    # Объединение фичей
    df = df.merge(user_stats, on='user_id', how='left')
    df = df.merge(product_stats, on='product_id', how='left')
    df = df.merge(category_stats, on='category_code', how='left')

    # Временные фичи сессии
    df['session_event_count'] = df.groupby('user_session').cumcount() + 1

    return df
# Предобработка данных
def preprocess_data(df, MIN_SESSION_SIZE):
    # Удаление сессий с 1 событием
    session_sizes = df.groupby('user_session').size()
    valid_sessions = session_sizes[session_sizes >= MIN_SESSION_SIZE].index # Changed to >=
    df = df[df['user_session']
    .isin(valid_sessions)]

    # Удаление аномальных сессий (длительность > 1 дня)
    session_times = df.groupby('user_session')['event_time'].agg(['min', 'max'])
    session_times['duration'] = (session_times['max'] - session_times['min']).dt.total_seconds()
    valid_sessions = session_times[session_times['duration'] <= 86400].index
    df = df[df['user_session'].isin(valid_sessions)]

    # Заполнение пропусков
    df['category_code'] = df['category_code'].cat.add_categories(['unknown']).fillna('unknown')
    df['brand'] = df['brand'].cat.add_categories(['unknown']).fillna('unknown')

    # Временные фичи
    df['hour'] = df['event_time'].dt.hour
    df['day_of_week'] = df['event_time'].dt.dayofweek
    df['day'] = df['event_time'].dt.day

    return df

class SessionDataset(Dataset):
        def __init__(self, sessions, min_history_length, num_features, cat_features):
            self.sessions = sessions
            self.min_history_length = min_history_length
            self.num_features = num_features
            self.cat_features = cat_features 

        def __len__(self):
            return len(self.sessions)

        def __getitem__(self, idx):
            session = self.sessions[idx][1]
            states = []
            next_states = []
            histories = []
            actions = []
            rewards = []

            # Накопление истории
            history = []
            for i in range(len(session)-1):
                current = session.iloc[i]
                next_ = session.iloc[i+1]

                # Сохраняем данные только для шагов с историей
                if i >= self.min_history_length:
                    states.append(torch.FloatTensor(current[self.num_features + self.cat_features].values.astype('float')))
                    next_states.append(torch.FloatTensor(next_[self.num_features + self.cat_features].values.astype('float')))
                    histories.append(torch.LongTensor(history.copy()))
                    actions.append(current['product_id'])
                    rewards.append(next_['reward'])

                # Обновляем историю для следующего шага
                history.append(current['product_id'])

            return {
                'states': torch.stack(states),        # [seq_len-1, features]
                'next_states': torch.stack(next_states),
                'histories': torch.nn.utils.rnn.pad_sequence(histories, batch_first=True, padding_value=0),  # [seq_len-1, history_len]
                'actions': torch.LongTensor(actions),
                'rewards': torch.FloatTensor(rewards),
                'lengths': torch.LongTensor([len(h) for h in histories])
            }

class DQN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(DQN, self).__init__()
            self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
            self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )

        def forward(self, state, history, lengths):
            # Эмбеддинг с динамическим паддингом
            hist_emb = self.embedding(history)  # [B, S, H]

            # Attention с масками
            key_padding_mask = (history == 0)  # [B, S]
            attn_out, _ = self.attention(
                hist_emb,
                hist_emb,
                hist_emb,
                key_padding_mask=key_padding_mask
            )  # [B, S, H]

            # GRU с учетом реальной длины
            packed = nn.utils.rnn.pack_padded_sequence(attn_out, lengths.cpu(), batch_first=True, enforce_sorted=False)
            gru_out, _ = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)

            # Извлекаем последние действительные состояния
            last_indices = lengths - 1
            context = gru_out[torch.arange(gru_out.size(0)), last_indices]

            combined = torch.cat([state, context], dim=1)
            return self.fc(combined)

class ReplayBuffer:
    def __init__(self, capacity, max_history_length):
        self.buffer = deque(maxlen=capacity)
        self.max_history_length = max_history_length

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        # Динамический паддинг с ограничением максимальной длины
        histories = [s['history'] for s in samples]
        padded_histories = torch.nn.utils.rnn.pad_sequence(
            histories,
            batch_first=True,
            padding_value=0
        )[:, :self.max_history_length]  # Обрезка до максимальной длины

        # Сборка батча
        return {
            'states': torch.stack([s['state'] for s in samples]).to(DEVICE),
            'next_states': torch.stack([s['next_state'] for s in samples]).to(DEVICE),
            'histories': padded_histories.to(DEVICE),
            'actions': torch.stack([s['action'] for s in samples]).to(DEVICE),
            'rewards': torch.stack([s['reward'] for s in samples]).to(DEVICE),
            'lengths': torch.stack([torch.clamp(s['length'],
                                                                    max=self.max_history_length)
                                                        for s in samples]).to(DEVICE)
        }

    def __len__(self):
        return len(self.buffer)


def evaluate(model, dataset, device, PENALTY, top_k=3): # принимать dataset как аргумент
    model.eval()
    total_reward = 0
    total_hits = 0
    total_items = 0
    total_penalty = 0

    with torch.no_grad():
        for session in tqdm(dataset, desc='Evaluating'): # использовать переданный dataset
            states = session['states'].to(device)
            actions = session['actions'].to(device)
            rewards = session['rewards'].to(device)
            histories = session['histories'].to(device)
            lengths = session['lengths'].to(device)

            # Прогноз с учетом реальной истории
            q_values = model(states, histories, lengths)
            _, top_preds = torch.topk(q_values, top_k, dim=1)

            # Расчет метрик
            hits = torch.any(top_preds == actions.unsqueeze(1), dim=1)
            penalties = (~hits).sum().item() * PENALTY
            total_penalty += penalties
            total_hits += hits.sum().item()
            total_items += len(actions)
            total_reward += (rewards * hits.float()).sum().item()

    accuracy = total_hits / total_items
    avg_reward = total_reward / len(dataset)

    return accuracy, avg_reward


def objective(trial):
    # Гиперпараметры для оптимизации
    MIN_SESSION_SIZE = trial.suggest_int('min_session_size', 2, 20)
    MAX_HISTORY_LENGTH = trial.suggest_int('max_history_length', 5, 50)
    PENALTY = -0.1
    
    LEARNING_RATE = trial.suggest_float('learning_rate', 0.000005, 0.0005, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.99)
    batch_size = trial.suggest_int('batch_size', 2, 500)
    epochs_per_trial = 2 # epochs per trial, for faster optimization, consider increasing for better evaluation per trial

    # Загрузка данных
    df = pd.read_csv(
        './datasets/ecommerce-events-history-in-electronics-store/events.csv',
        parse_dates=['event_time'],
        dtype={
            'event_type': 'category',
            'product_id': 'int32',
            'category_id': 'int64',
            'category_code': 'category',
            'brand': 'category',
            'price': 'float32',
            'user_id': 'int64',
            'user_session': 'string'
        },
        nrows=10000
    )
    
    df = create_features(df)
    df = preprocess_data(df, MIN_SESSION_SIZE+2)
    # Кодирование категориальных переменных
    cat_features = ['category_code', 'brand', 'product_id']
    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Нормализация числовых фичей
    num_features = ['price', 'total_views', 'total_carts', 'total_purchases',
                    'product_views', 'product_purchases', 'avg_price',
                    'category_views', 'avg_category_price', 'session_event_count']

    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])

    # Определение наград
    REWARD_MAP = {'view': 0.1, 'cart': 1.0, 'purchase': 5.0}
    df['reward'] = df['event_type'].map(REWARD_MAP)

    INPUT_SIZE = len(num_features + cat_features) + STATE_SIZE
    OUTPUT_SIZE = df['product_id'].nunique()

    
    # Разделение на train/test (and validation - suggested improvement)
    sessions = list(df.groupby('user_session'))
    train_sessions, temp_sessions = train_test_split(
        sessions, test_size=0.3, random_state=RANDOM_SEED # Split into train and temp (val+test)
    )
    val_sessions, test_sessions = train_test_split(
        temp_sessions, test_size=0.5, random_state=RANDOM_SEED # Split temp into val and test
    )


    train_dataset = SessionDataset(train_sessions, MIN_SESSION_SIZE, num_features, cat_features)
    val_dataset = SessionDataset(val_sessions, MIN_SESSION_SIZE, num_features, cat_features) # Use validation dataset in objective
    test_dataset = SessionDataset(test_sessions, MIN_SESSION_SIZE, num_features, cat_features)

    buffer = ReplayBuffer(10000, MAX_HISTORY_LENGTH)

   

    # Тренировочный цикл
    epoch_losses = []
    rewards_history = []

    model = DQN(
        input_size=INPUT_SIZE,
        hidden_size=STATE_SIZE,
        output_size=OUTPUT_SIZE
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()


    for epoch in range(epochs_per_trial): # Use epochs_per_trial here
        model.train()
        epoch_rewards = []

        for session in tqdm(train_dataset, desc=f'Epoch {epoch+1}'):
            states = session['states'].to(DEVICE)
            next_states = session['next_states'].to(DEVICE)
            histories = session['histories'].to(DEVICE)
            actions = session['actions'].to(DEVICE)
            rewards = session['rewards'].to(DEVICE)
            lengths = session['lengths'].to(DEVICE)

            # Прогноз Q-значений
            q_values = model(states, histories, lengths) # [batch_size, num_actions]
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()

            # Целевые Q-значения
            with torch.no_grad():
                next_q_values = model(next_states, histories, lengths)
                max_next_q = next_q_values.max(1)[0]
                target_q = rewards + gamma * max_next_q

            # Обновление буфера
            for i in range(len(states)):
                buffer.push(dict(
                    state=states[i].cpu().detach(),
                    next_state=next_states[i].cpu().detach(),
                    history=histories[i].cpu().detach(),
                    action=actions[i].cpu().detach(),
                    reward=rewards[i].cpu().detach(),
                    length=lengths[i].cpu().detach()
                ))

            # Обучение с учетом длин историй
            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size)
                state_b, next_state_b, history_b, action_b, reward_b, lengths_b = batch.values()

                # Вычисление потерь
                q_values = model(state_b, history_b, lengths_b)
                current_q = q_values.gather(1, action_b.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_q_values = model(next_state_b, history_b, lengths_b)
                    max_next_q = next_q_values.max(1)[0]
                    target_q = reward_b + gamma * max_next_q

                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_rewards.append(rewards.mean().item())


    val_accuracy, _ = evaluate(model, val_dataset, DEVICE, PENALTY) # Evaluate on validation set
    return val_accuracy # Return validation accuracy for optimization


if __name__ == '__main__':
    study = optuna.create_study(storage='sqlite:///rl_project.db', direction='maximize') # maximize accuracy
    study.optimize(objective, n_trials=10000) # Increased n_trials for more thorough search

    print("Best trial:")
    trial = study.best_trial

    print("  Validation Accuracy: {}".format(trial.value)) # Indicate it's validation accuracy

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_params = trial.params

    # --- Retrain best model with best hyperparameters on full training data and evaluate on test set ---
    print("\nRetraining best model with found hyperparameters and evaluating on test set...")

    # Use best_params to define hyperparameters for retraining
    best_MIN_SESSION_SIZE = best_params['min_session_size']
    best_MAX_HISTORY_LENGTH = best_params['max_history_length']
    best_PENALTY = best_params['penalty']
    best_NROWS = best_params['nrows']
    best_LEARNING_RATE = best_params['learning_rate']
    best_gamma = best_params['gamma']
    best_batch_size = best_params['batch_size']
    best_epochs_retrain = 5 # Define epochs for retraining, can be more than in objective


    # Re-run data loading and preprocessing with best hyperparameters
    df_best = pd.read_csv(
        './datasets/ecommerce-events-history-in-electronics-store/events.csv',
        parse_dates=['event_time'],
        dtype={
            'event_type': 'category',
            'product_id': 'int32',
            'category_id': 'int64',
            'category_code': 'category',
            'brand': 'category',
            'price': 'float32',
            'user_id': 'int64',
            'user_session': 'string'
        },
        nrows=best_NROWS
    )
    df_best = create_features(df_best)
    df_best = preprocess_data(df_best, best_MIN_SESSION_SIZE+2) # preprocess_data function is already defined
    cat_features = ['category_code', 'brand', 'product_id']
    for col in cat_features:
        le = LabelEncoder()
        df_best[col] = le.fit_transform(df_best[col].astype(str))
    num_features = ['price', 'total_views', 'total_carts', 'total_purchases',
                    'product_views', 'product_purchases', 'avg_price',
                    'category_views', 'avg_category_price', 'session_event_count']
    scaler = StandardScaler()
    df_best[num_features] = scaler.fit_transform(df_best[num_features])
    REWARD_MAP = {'view': 0.1, 'cart': 1.0, 'purchase': 5.0}
    df_best['reward'] = df_best['event_type'].map(REWARD_MAP)
    OUTPUT_SIZE_BEST = df_best['product_id'].nunique() # Recalculate OUTPUT_SIZE
    INPUT_SIZE_BEST = len(num_features + cat_features) + STATE_SIZE # INPUT_SIZE remains the same if STATE_SIZE is fixed


    sessions_best = list(df_best.groupby('user_session'))
    train_sessions_best, test_sessions_best = train_test_split(
        sessions_best, test_size=0.2, random_state=RANDOM_SEED
    )
    train_dataset_best = SessionDataset(train_sessions_best, min_history_length=best_MIN_SESSION_SIZE - 1, num_features=num_features, cat_features=cat_features) # Use best MIN_SESSION_SIZE
    test_dataset_best = SessionDataset(test_sessions_best, min_history_length=best_MIN_SESSION_SIZE - 1, num_features=num_features, cat_features=cat_features) # Use best MIN_SESSION_SIZE


    best_model = DQN(
        input_size=INPUT_SIZE_BEST,
        hidden_size=STATE_SIZE,
        output_size=OUTPUT_SIZE_BEST
    ).to(DEVICE)
    best_optimizer = optim.AdamW(best_model.parameters(), lr=best_LEARNING_RATE) # Use best LEARNING_RATE
    best_criterion = nn.MSELoss()
    best_buffer = ReplayBuffer(10000, max_history_length=best_MAX_HISTORY_LENGTH) # Use best MAX_HISTORY_LENGTH

    for epoch in range(best_epochs_retrain): # Retrain for more epochs
        best_model.train()
        for session in tqdm(train_dataset_best, desc=f'Retrain Epoch {epoch+1}/{best_epochs_retrain}'):
            states = session['states'].to(DEVICE)
            next_states = session['next_states'].to(DEVICE)
            histories = session['histories'].to(DEVICE)
            actions = session['actions'].to(DEVICE)
            rewards = session['rewards'].to(DEVICE)
            lengths = session['lengths'].to(DEVICE)

            # Прогноз Q-значений
            q_values = best_model(states, histories, lengths)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()

            # Целевые Q-значения
            with torch.no_grad():
                next_q_values = best_model(next_states, histories, lengths)
                max_next_q = next_q_values.max(1)[0]
                target_q = rewards + best_gamma * max_next_q # Use best gamma

            # Обновление буфера
            for i in range(len(states)):
                best_buffer.push(dict(
                    state=states[i].cpu().detach(),
                    next_state=next_states[i].cpu().detach(),
                    history=histories[i].cpu().detach(),
                    action=actions[i].cpu().detach(),
                    reward=rewards[i].cpu().detach(),
                    length=lengths[i].cpu().detach()
                ))

            # Обучение
            if len(best_buffer) > best_batch_size: # Use best batch_size
                batch = best_buffer.sample(best_batch_size)
                state_b, next_state_b, history_b, action_b, reward_b, lengths_b = batch.values()

                q_values = best_model(state_b, history_b, lengths_b)
                current_q = q_values.gather(1, action_b.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_q_values = best_model(next_state_b, history_b, lengths_b)
                    max_next_q = next_q_values.max(1)[0]
                    target_q = reward_b + best_gamma * max_next_q

                loss = best_criterion(current_q, target_q)
                best_optimizer.zero_grad()
                loss.backward()
                best_optimizer.step()

    test_accuracy_best, test_reward_best = evaluate(best_model, test_dataset_best, DEVICE, top_k=N_RECOMMENDATIONS) # Evaluate best model on test set

    print(f'\nTest Accuracy@{N_RECOMMENDATIONS} of best model: {test_accuracy_best:.4f}')
    print(f'Test Average Reward of best model: {test_reward_best:.2f}')

    # Save best model
    torch.save(best_model.state_dict(), 'models/best_model_optuna.pth')
    print(f'Best model with Optuna optimized hyperparameters saved to models/best_model_optuna.pth')
