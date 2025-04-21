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

# Configuration
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_RECOMMENDATIONS = 5
MIN_SESSION_SIZE = 3
MAX_HISTORY_LENGTH = 10
PENALTY = -0.01
LEARNING_RATE = 0.00001
GAMMA = 0.99
BATCH_SIZE = 32
EPOCHS = 10
MEMORY_SIZE = 10000

# Reward mapping
REWARD_MAP = {
    'view': 0.1,
    'cart': 1.0,
    'purchase': 5.0
}

class EcommerceDataset(Dataset):
    def __init__(self, sessions, scaler=None, label_encoders=None):
        self.sessions = sessions
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.max_session_length = max(len(session) for session in sessions)
        
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        session = self.sessions[idx]
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        # Pad session to max length
        padded_session = pd.concat([
            session,
            pd.DataFrame([session.iloc[-1]] * (self.max_session_length - len(session)))
        ])
        
        for i in range(self.max_session_length - 1):
            # Current state
            state = self._get_state(padded_session, i)
            states.append(state)
            
            # Action (product_id)
            action = padded_session.iloc[i]['product_id']
            actions.append(action)
            
            # Reward
            reward = REWARD_MAP[padded_session.iloc[i]['event_type']]
            rewards.append(reward)
            
            # Next state
            next_state = self._get_state(padded_session, i + 1)
            next_states.append(next_state)
            
            # Done flag
            done = 1 if i >= len(session) - 2 else 0
            dones.append(done)
        
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.LongTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'next_states': torch.FloatTensor(next_states),
            'dones': torch.FloatTensor(dones)
        }
    
    def _get_state(self, session, idx):
        """Extract state features for a given session index"""
        current_event = session.iloc[idx]
        
        # Basic features
        state = [
            current_event['price'],
            current_event['session_event_num'],
            current_event['user_views_before'],
            current_event['user_carts_before'],
            current_event['user_purchases_before'],
            current_event['product_views_before'],
            current_event['product_purchases_before'],
            current_event['product_avg_price'],
            current_event['category_views_before'],
            current_event['category_avg_price']
        ]
        
        # Add history features
        history = session.iloc[max(0, idx - MAX_HISTORY_LENGTH):idx]
        history_features = []
        for _, event in history.iterrows():
            history_features.extend([
                event['price'],
                event['product_id'],
                event['category_id']
            ])
        
        # Pad history if needed
        while len(history_features) < MAX_HISTORY_LENGTH * 3:
            history_features.extend([0, 0, 0])
        
        state.extend(history_features[:MAX_HISTORY_LENGTH * 3])
        return state

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Q-value prediction
        self.q_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        q_values = self.q_network(features)
        return q_values

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Initialize networks
        self.policy_net = DQN(state_size, hidden_size, action_size).to(DEVICE)
        self.target_net = DQN(state_size, hidden_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        # Initialize memory
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        # Initialize scaler and encoders
        self.scaler = StandardScaler()
        self.label_encoders = {
            'product_id': LabelEncoder(),
            'category_id': LabelEncoder()
        }
        
        # Action mapping
        self.action_to_product = {}
        self.product_to_action = {}
    
    def _get_state(self, events, idx):
        """Extract state features for a given session index"""
        if isinstance(events, list):
            current_event = events[idx]
        else:
            current_event = events.iloc[idx]
        
        # Basic features
        state = [
            current_event['price'],
            current_event['session_event_num'],
            current_event['user_views_before'],
            current_event['user_carts_before'],
            current_event['user_purchases_before'],
            current_event['product_views_before'],
            current_event['product_purchases_before'],
            current_event['product_avg_price'],
            current_event['category_views_before'],
            current_event['category_avg_price']
        ]
        
        # Add history features
        if isinstance(events, list):
            history = events[max(0, idx - MAX_HISTORY_LENGTH):idx]
            history_features = []
            for event in history:
                history_features.extend([
                    event['price'],
                    event['product_id'],
                    event['category_id']
                ])
        else:
            history = events.iloc[max(0, idx - MAX_HISTORY_LENGTH):idx]
            history_features = []
            for _, event in history.iterrows():
                history_features.extend([
                    event['price'],
                    event['product_id'],
                    event['category_id']
                ])
        
        # Pad history if needed
        while len(history_features) < MAX_HISTORY_LENGTH * 3:
            history_features.extend([0, 0, 0])
        
        state.extend(history_features[:MAX_HISTORY_LENGTH * 3])
        return state
    
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        # Filter sessions
        session_sizes = df.groupby('user_session').size()
        valid_sessions = session_sizes[session_sizes >= MIN_SESSION_SIZE].index
        df = df[df['user_session'].isin(valid_sessions)]
        
        # Sort by session and event time
        df = df.sort_values(['user_session', 'event_time'])
        
        # Encode categorical features
        df['product_id'] = self.label_encoders['product_id'].fit_transform(df['product_id'])
        df['category_id'] = self.label_encoders['category_id'].fit_transform(df['category_id'])
        
        # Create action mapping
        unique_products = df['product_id'].unique()
        self.action_to_product = {i: product for i, product in enumerate(unique_products)}
        self.product_to_action = {product: i for i, product in enumerate(unique_products)}
        
        # Update action size
        self.action_size = len(unique_products)
        self.policy_net = DQN(self.state_size, self.hidden_size, self.action_size).to(DEVICE)
        self.target_net = DQN(self.state_size, self.hidden_size, self.action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Scale numerical features
        numerical_features = [
            'price', 'session_event_num', 'user_views_before',
            'user_carts_before', 'user_purchases_before',
            'product_views_before', 'product_purchases_before',
            'product_avg_price', 'category_views_before',
            'category_avg_price'
        ]
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        return df
    
    def train(self, df, num_epochs=EPOCHS):
        """Train the DQN agent"""
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Split into sessions
        sessions = [group for _, group in df.groupby('user_session')]
        
        # Create dataset
        dataset = EcommerceDataset(sessions, self.scaler, self.label_encoders)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                # Move batch to device
                states = batch['states'].to(DEVICE)
                actions = batch['actions'].to(DEVICE)
                rewards = batch['rewards'].to(DEVICE)
                next_states = batch['next_states'].to(DEVICE)
                dones = batch['dones'].to(DEVICE)
                
                # Reshape tensors to handle sequences
                batch_size, seq_length = states.shape[0], states.shape[1]
                states = states.view(-1, states.shape[-1])
                next_states = next_states.view(-1, next_states.shape[-1])
                actions = actions.view(-1)
                rewards = rewards.view(-1)
                dones = dones.view(-1)
                
                # Map product IDs to action indices
                mapped_actions = torch.zeros_like(actions)
                for i, action in enumerate(actions):
                    mapped_actions[i] = self.product_to_action[action.item()]
                actions = mapped_actions.to(DEVICE)
                
                # Get current Q values
                current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
                
                # Get next Q values from target network
                with torch.no_grad():
                    next_q_values = self.target_net(next_states).max(1)[0]
                
                # Compute target Q values
                target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
                
                # Compute loss
                loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Update target network
            if epoch % 5 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}')
    
    def predict(self, state):
        """Predict Q-values for a given state"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state)
            return q_values.cpu().numpy()[0]
    
    def get_recommendations(self, state, k=N_RECOMMENDATIONS):
        """Get top-k recommendations for a given state"""
        q_values = self.predict(state)
        # Get top-k indices, but ensure they're within our action space
        valid_indices = np.argsort(q_values)[-len(self.action_to_product):][::-1]
        top_k_indices = valid_indices[:k]
        
        # Map action indices back to product IDs, handling any invalid indices
        recommendations = []
        for idx in top_k_indices:
            if idx in self.action_to_product:
                recommendations.append(self.action_to_product[idx])
            else:
                # If index is invalid, use a random valid product
                recommendations.append(random.choice(list(self.action_to_product.values())))
        
        return recommendations

def main():
    # Load synthetic data
    df = pd.read_csv('datasets/synthetic_test_data.csv')
    
    # Initialize agent
    state_size = 10 + MAX_HISTORY_LENGTH * 3  # Basic features + history
    action_size = len(df['product_id'].unique())
    agent = DQNAgent(state_size, action_size)
    
    # Train agent
    agent.train(df)
    
    # Save model
    torch.save(agent.policy_net.state_dict(), 'dqn_model.pth')

if __name__ == "__main__":
    main()
