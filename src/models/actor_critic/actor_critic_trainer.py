import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

class ActorCriticTrainer:
    """
    Actor-Critic trainer with:
    - Policy gradient for action selection
    - Value function for state estimation
    - Entropy regularization for exploration
    - Gradient clipping for stability
    """
    def __init__(self,
                 model: nn.Module,
                 optimizer_cls,
                 lr: float,
                 gamma: float = 0.99,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 device: torch.device = None):
        self.model = model
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.optimizer = optimizer_cls(model.parameters(), lr=lr)
        self.device = device or torch.device('cpu')
        self.model.to(self.device)

    def train_epoch(self, dataloader):
        self.model.train()
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        
        for batch in tqdm(dataloader, desc="Training epoch"):
            b = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            inp = (b['state_history'], b['state_length'],
                   b['state_numerical_features'], b['state_brand_idx'], b['state_holiday_idx'])
            logits, values = self.model(*inp)
            
            # Get action distribution and entropy
            dist = Categorical(logits=logits)
            logp = dist.log_prob(b['action'])
            entropy = dist.entropy().mean()
            
            # Get next state values
            with torch.no_grad():
                inp2 = (b['next_state_history'], b['next_state_length'],
                        b['next_state_numerical_features'], b['next_state_brand_idx'], b['next_state_holiday_idx'])
                _, next_values = self.model(*inp2)
                td_target = b['reward'] + self.gamma * next_values * (~b['done'])
            
            # Compute advantages and losses
            advantages = td_target - values
            policy_loss = -(logp * advantages.detach()).mean()
            value_loss = advantages.pow(2).mean()
            
            # Total loss with entropy regularization
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update statistics
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += entropy.item()
        
        # Average statistics
        n_batches = len(dataloader)
        return {
            'policy_loss': policy_loss_sum / n_batches,
            'value_loss': value_loss_sum / n_batches,
            'entropy': entropy_sum / n_batches
        }

    def evaluate(self, dataloader):
        self.model.eval()
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                b = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                inp = (b['state_history'], b['state_length'],
                       b['state_numerical_features'], b['state_brand_idx'], b['state_holiday_idx'])
                logits, values = self.model(*inp)
                
                # Get action distribution
                dist = Categorical(logits=logits)
                logp = dist.log_prob(b['action'])
                
                # Get next state values
                inp2 = (b['next_state_history'], b['next_state_length'],
                        b['next_state_numerical_features'], b['next_state_brand_idx'], b['next_state_holiday_idx'])
                _, next_values = self.model(*inp2)
                td_target = b['reward'] + self.gamma * next_values * (~b['done'])
                
                # Compute advantages and losses
                advantages = td_target - values
                policy_loss = -(logp * advantages).mean()
                value_loss = advantages.pow(2).mean()
                
                # Update statistics
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
        
        # Average statistics
        n_batches = len(dataloader)
        return {
            'policy_loss': policy_loss_sum / n_batches,
            'value_loss': value_loss_sum / n_batches
        }
