import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.configs import settings
from tqdm import tqdm


class DQLTrainer:
    def __init__(self, 
                 policy_net, 
                 target_net, 
                 criterion, 
                 optimizer, 
                 lr, 
                 gamma, 
                 target_update_freq):
        self.policy_net = policy_net
        self.target_net = target_net
        self.criterion = criterion
        self.optimizer = optimizer(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.step_count = 0
        self.target_update_freq = target_update_freq

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def train_epoch(self, dataloader):
        self.policy_net.train()
        epoch_loss = 0.0
    
        for batch in dataloader:
            batch = {k: v.to(settings.DEVICE, non_blocking=True) for k, v in batch.items()}

            # Общие входные параметры для обеих сетей
            network_inputs = (
                batch['state_history'],
                batch['state_length'],
                batch['state_numerical_features'],
                batch['state_brand_idx'],
                batch['state_holiday_idx']
            )

            # 1. Получаем Q(s, a) от policy сети
            q_values_all = self.policy_net(*network_inputs)
            q_values_selected = q_values_all.gather(1, batch['action'].unsqueeze(1)).squeeze(1)

            # 2. Получаем TD Target с использованием Double DQN
            with torch.no_grad():
                # Получаем действия от policy сети
                next_network_inputs = (
                    batch['next_state_history'],
                    batch['next_state_length'],
                    batch['next_state_numerical_features'],
                    batch['next_state_brand_idx'],
                    batch['next_state_holiday_idx']
                )
                next_q_values_policy = self.policy_net(*next_network_inputs)
                best_next_actions = next_q_values_policy.argmax(1, keepdim=True)
                
                # Получаем Q-значения от target сети
                next_q_values_target = self.target_net(*next_network_inputs)
                q_values_next_state = next_q_values_target.gather(1, best_next_actions).squeeze(1)
                q_values_next_state.masked_fill_(batch['done'], False)

            # 3. Вычисляем целевые значения TD
            target_q_values = batch['reward'] + self.gamma * q_values_next_state

            # 4-5. Считаем ошибку и оптимизируем
            loss = self.criterion(q_values_selected, target_q_values)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            self.step_count += 1

            # 6. Обновление целевой сети
            if self.step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        return epoch_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.policy_net.eval()
        total_loss = 0.0
    
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(settings.DEVICE, non_blocking=True) for k, v in batch.items()}

                # Общие входные параметры для обеих сетей
                network_inputs = (
                    batch['state_history'],
                    batch['state_length'],
                    batch['state_numerical_features'],
                    batch['state_brand_idx'],
                    batch['state_holiday_idx']
                )

                # 1. Получаем Q(s, a) от policy сети
                q_values_all = self.policy_net(*network_inputs)
                q_values_selected = q_values_all.gather(1, batch['action'].unsqueeze(1)).squeeze(1)

                # 2. Получаем TD Target с использованием Double DQN
                next_network_inputs = (
                    batch['next_state_history'],
                    batch['next_state_length'],
                    batch['next_state_numerical_features'],
                    batch['next_state_brand_idx'],
                    batch['next_state_holiday_idx']
                )
                next_q_values_policy = self.policy_net(*next_network_inputs)
                best_next_actions = next_q_values_policy.argmax(1, keepdim=True)
                
                # Получаем Q-значения от target сети
                next_q_values_target = self.target_net(*next_network_inputs)
                q_values_next_state = next_q_values_target.gather(1, best_next_actions).squeeze(1)
                q_values_next_state.masked_fill_(batch['done'], False)

                target_q_values = batch['reward'] + self.gamma * q_values_next_state

                # Считаем ошибку
                loss = self.criterion(q_values_selected, target_q_values)
                total_loss += loss.item()
        return total_loss / len(dataloader)
    