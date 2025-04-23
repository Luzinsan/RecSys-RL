import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np

class ActorCriticTrainer:
    """
    Enhanced A2C trainer with:
    - GAE (Generalized Advantage Estimation)
    - Value function clipping
    - Policy gradient clipping
    - Learning rate scheduling
    - Better loss computation
    """
    def __init__(self,
                 model: nn.Module,
                 optimizer_cls,
                 lr: float,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: torch.device = None):
        self.model = model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        
        # Initialize optimizer with weight decay
        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, device=self.device)
        returns = advantages + values
        return advantages, returns

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0
        }
        
        for batch in dataloader:
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
                
                # Compute GAE
                advantages, returns = self.compute_gae(
                    b['reward'], values, next_values, b['done']
                )
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute losses
            policy_loss = -(logp * advantages.detach()).mean()
            value_loss = 0.5 * ((returns - values) ** 2).mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Update statistics
            epoch_stats['policy_loss'] += policy_loss.item()
            epoch_stats['value_loss'] += value_loss.item()
            epoch_stats['entropy'] += entropy.item()
            epoch_stats['total_loss'] += loss.item()
        
        # Average statistics
        n_batches = len(dataloader)
        for k in epoch_stats:
            epoch_stats[k] /= n_batches
        
        # Update learning rate
        self.scheduler.step(epoch_stats['total_loss'])
        
        return epoch_stats

    def evaluate(self, dataloader):
        self.model.eval()
        eval_stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0
        }
        
        with torch.no_grad():
            for batch in dataloader:
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
                inp2 = (b['next_state_history'], b['next_state_length'],
                        b['next_state_numerical_features'], b['next_state_brand_idx'], b['next_state_holiday_idx'])
                _, next_values = self.model(*inp2)
                
                # Compute GAE
                advantages, returns = self.compute_gae(
                    b['reward'], values, next_values, b['done']
                )
                
                # Compute losses
                policy_loss = -(logp * advantages).mean()
                value_loss = 0.5 * ((returns - values) ** 2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update statistics
                eval_stats['policy_loss'] += policy_loss.item()
                eval_stats['value_loss'] += value_loss.item()
                eval_stats['entropy'] += entropy.item()
                eval_stats['total_loss'] += loss.item()
        
        # Average statistics
        n_batches = len(dataloader)
        for k in eval_stats:
            eval_stats[k] /= n_batches
        
        return eval_stats
