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
                 target_update_freq,
                 product_to_category_map,
                 device):
        self.policy_net = policy_net
        self.target_net = target_net
        self.criterion = criterion
        self.optimizer = optimizer(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.step_count = 0
        self.target_update_freq = target_update_freq
        self.category_match_reward = settings.CATEGORY_MATCH_REWARD
        self.device = device
        # Индекс - product_id_idx, Значение - category_id -1 для неизвестных.
        print(max(product_to_category_map.keys()))
        self.category_lookup = torch.full((max(product_to_category_map.keys()) + 1,), -1, 
                                          dtype=torch.long, device=self.device)
        
        self.category_lookup[
            torch.tensor(list(product_to_category_map.keys()), 
                         dtype=torch.long, device=self.device)] = \
                    torch.tensor(list(product_to_category_map.values()), 
                                 dtype=torch.long, device=self.device)

        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


    def _get_categories(self, product_indices: torch.Tensor) -> torch.Tensor:
        """ Получает категории для тензора продуктов с помощью lookup тензора. """

        # Определяем границы lookup тензора
        lookup_size = len(self.category_lookup)

        # Ограничиваем индексы, чтобы избежать ошибки при индексации
        clamped_indices = torch.clamp(product_indices, 0, lookup_size - 1)

        # Получаем категории с помощью ограниченных индексов
        categories = self.category_lookup[clamped_indices]

        # Устанавливаем -1 для тех индексов, которые изначально были вне диапазона
        categories[(product_indices < 0) | (product_indices >= lookup_size)] = -1

        return categories

    def train_epoch(self, dataloader):
        self.policy_net.train()
        epoch_loss = 0.0
    
        for batch in tqdm(dataloader):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

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

            # Получаем предсказанные действия из s_t для сравнения категорий
            predicted_actions_from_st = q_values_all.argmax(1).detach()

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

            actual_actions = batch['action']
            # Получаем категории для предсказанных и реальных действий
            predicted_categories = self._get_categories(predicted_actions_from_st)
            actual_categories = self._get_categories(actual_actions)

            # Создаем маску: действие предсказано неверно, НО категория верна И известна
            product_mismatch_mask = (predicted_actions_from_st != actual_actions)
            category_match_mask = (predicted_categories == actual_categories) & (actual_categories != -1)
            reward_boost_mask = product_mismatch_mask & category_match_mask

            # Применяем бонус
            batch['reward'][reward_boost_mask] += self.category_match_reward

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
            for batch in tqdm(dataloader):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

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
    