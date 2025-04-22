import torch
from torch import nn
from torch.distributions import Categorical

class ActorCriticTrainer:
    """
    A2C-подобный тренер:
      - policy_loss = −E[ log π(a|s) · A ]
      - value_loss  = A²
      - entropy bonus
    """
    def __init__(self,
                 model: nn.Module,
                 optimizer_cls,
                 lr: float,
                 gamma: float = 0.99,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 device: torch.device = None):
        self.model         = model
        self.gamma         = gamma
        self.value_coef    = value_coef
        self.entropy_coef  = entropy_coef
        self.opt           = optimizer_cls(model.parameters(), lr=lr)
        self.device        = device or torch.device('cpu')
        self.model.to(self.device)

    def train_epoch(self, dataloader):
        self.model.train()
        pl, vl, ent = 0.0, 0.0, 0.0
        for batch in dataloader:
            b = {k: v.to(self.device) for k, v in batch.items()}
            # forward
            inp = (b['state_history'], b['state_length'],
                   b['state_numerical_features'], b['state_brand_idx'], b['state_holiday_idx'])
            logits, values = self.model(*inp)
            dist   = Categorical(logits=logits)
            logp   = dist.log_prob(b['action'])
            entropy= dist.entropy().mean()
            # next‐state values
            with torch.no_grad():
                inp2 = (b['next_state_history'], b['next_state_length'],
                        b['next_state_numerical_features'], b['next_state_brand_idx'], b['next_state_holiday_idx'])
                _, next_v = self.model(*inp2)
                td_target = b['reward'] + self.gamma * next_v * (~b['done'])
            adv = td_target - values
            # losses
            policy_loss = -(logp * adv.detach()).mean()
            value_loss  = adv.pow(2).mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            # step
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            pl += policy_loss.item()
            vl += value_loss .item()
            ent+= entropy.item()
        N = len(dataloader)
        return {'policy_loss': pl/N, 'value_loss': vl/N, 'entropy': ent/N}

    def evaluate(self, dataloader):
        self.model.eval()
        pl, vl = 0.0, 0.0
        with torch.no_grad():
            for batch in dataloader:
                b = {k: v.to(self.device) for k, v in batch.items()}
                inp = (b['state_history'], b['state_length'],
                       b['state_numerical_features'], b['state_brand_idx'], b['state_holiday_idx'])
                logits, values = self.model(*inp)
                dist   = Categorical(logits=logits)
                logp   = dist.log_prob(b['action'])
                inp2   = (b['next_state_history'], b['next_state_length'],
                          b['next_state_numerical_features'], b['next_state_brand_idx'], b['next_state_holiday_idx'])
                _, next_v = self.model(*inp2)
                td_target = b['reward'] + self.gamma * next_v * (~b['done'])
                adv = td_target - values
                pl += (-(logp * adv.detach()).mean()).item()
                vl += adv.pow(2).mean().item()
        N = len(dataloader)
        return {'policy_loss': pl/N, 'value_loss': vl/N}
