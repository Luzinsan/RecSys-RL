from datetime import datetime
import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
from collections import Counter
from tqdm import tqdm
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import plotly.graph_objects as go
import plotly.express as px


logger = logging.getLogger(__name__)



def calculate_conversion_rate(df: pd.DataFrame, 
                                   session_col='user_session', 
                                   event_type_col='event_type') -> float:
    """ Рассчитывает Conversion Rate (доля сессий с покупкой) из DataFrame. """
    logger.info("Calculating Conversion Rate from DataFrame...")
    if df.empty or session_col not in df.columns or event_type_col not in df.columns:
        logger.warning("DataFrame is empty or missing required columns for Conversion Rate calculation.")
        return 0.0
    try:
        sessions_with_purchase = df[df[event_type_col] == 'purchase'][session_col].nunique()
        total_sessions = df[session_col].nunique()
        conversion_rate = sessions_with_purchase / total_sessions if total_sessions > 0 else 0.0
        logger.info(f"Conversion Rate calculated: {conversion_rate:.6f}")
        return conversion_rate
    except Exception as e:
        logger.error(f"Error calculating Conversion Rate from DataFrame: {e}", exc_info=True)
        return float('nan')

def calculate_arpu(df: pd.DataFrame, session_col='user_session', event_type_col='event_type', price_col='price') -> float:
    """ Рассчитывает Average Revenue Per User (ARPU - здесь Per Session) из DataFrame. """
    logger.info("Calculating ARPU (per session) from DataFrame...")
    if df.empty or session_col not in df.columns or event_type_col not in df.columns or price_col not in df.columns:
        logger.warning("DataFrame is empty or missing required columns for ARPU calculation.")
        return 0.0
    try:
        purchase_df = df[df[event_type_col] == 'purchase']
        total_revenue = purchase_df[price_col].sum()
        total_sessions = df[session_col].nunique()
        arpu = total_revenue / total_sessions if total_sessions > 0 else 0.0
        logger.info(f"ARPU (per session) calculated: {arpu:.6f}")
        return arpu
    except Exception as e:
        logger.error(f"Error calculating ARPU from DataFrame: {e}", exc_info=True)
        return float('nan')

def calculate_ctr_at_k(all_top_k_indices: torch.Tensor, all_actual_actions: torch.Tensor, k: int) -> float:
    """ CTR@K - доля "кликов" (реальное действие в топ-K) по "показам" (все слоты топ-K). """
    logger.info(f"Calculating CTR@{k}...")
    num_samples = all_actual_actions.numel()
    if num_samples == 0: return 0.0

    actual_next_expanded = all_actual_actions.unsqueeze(1)
    hits_tensor = (all_top_k_indices == actual_next_expanded).any(dim=1)
    total_clicks = hits_tensor.sum().item()
    total_impressions = all_top_k_indices.numel() # num_samples * k

    ctr = total_clicks / total_impressions if total_impressions > 0 else 0.0
    logger.info(f"CTR@{k} calculated: {ctr:.6f}")
    return ctr

def calculate_avg_session_length(df: pd.DataFrame, session_col='user_session') -> float:
    """ Рассчитывает среднюю длину сессии (количество событий) из DataFrame. """
    logger.info("Calculating Average Session Length from DataFrame...")
    if df.empty or session_col not in df.columns:
        logger.warning("DataFrame is empty or missing session column for Average Session Length calculation.")
        return 0.0
    try:
        session_lengths = df.groupby(session_col).size()
        avg_length = session_lengths.mean()
        logger.info(f"Average Session Length calculated: {avg_length:.6f}")
        return avg_length
    except Exception as e:
        logger.error(f"Error calculating Average Session Length from DataFrame: {e}", exc_info=True)
        return float('nan')


def analyze_reward_distribution(dataloader, return_raw=False) -> Union[Dict[str, float], Tuple[Dict[str, float], torch.Tensor]]:
    """ Analyzes the distribution of rewards in the dataloader. """
    logger.info("Analyzing Reward Distribution from transitions...")
    all_rewards = []
    for batch in tqdm(dataloader, desc="Analyzing Rewards", leave=False, ncols=100):
        try:
            rewards = batch['reward']
            all_rewards.append(rewards)
        except KeyError:
            logger.warning("Key 'reward' not found in batch. Skipping.")
            continue
        except Exception as e:
             logger.error(f"Error processing batch for reward analysis: {e}", exc_info=True)
             continue
    stats = {'reward_mean': 0.0, 'reward_std': 0.0, 'reward_min': 0.0, 'reward_max': 0.0, 'reward_median': 0.0}
    all_rewards_tensor = torch.tensor([]) 

    if all_rewards:
        all_rewards_tensor_np = torch.cat(all_rewards).numpy()
        stats = {
            'reward_mean': np.mean(all_rewards_tensor_np),
            'reward_std': np.std(all_rewards_tensor_np),
            'reward_min': np.min(all_rewards_tensor_np),
            'reward_max': np.max(all_rewards_tensor_np),
            'reward_median': np.median(all_rewards_tensor_np)
        }
        all_rewards_tensor = torch.from_numpy(all_rewards_tensor_np)
    else:
         logger.warning("No rewards found to analyze.")

    logger.info("Reward Distribution analyzed.")
    if return_raw:
        return stats, all_rewards_tensor
    else:
        return stats


def calculate_auc(all_q_values: torch.Tensor, all_actual_actions: torch.Tensor) -> float:
    logger.info("Calculating AUC Score...")
    num_samples, num_actions = all_q_values.shape
    if num_samples == 0: return 0.0

    y_true_flat = []
    y_score_flat = []
    processed_samples = 0
    for i in range(num_samples):
        q_scores = all_q_values[i].numpy()
        actual_action = all_actual_actions[i].item()
        labels = np.zeros(num_actions, dtype=int)
        if 0 <= actual_action < num_actions:
            labels[actual_action] = 1
            y_true_flat.extend(labels)
            y_score_flat.extend(q_scores)
            processed_samples += 1
        else:
            logger.warning(f"Action {actual_action} out of bounds for sample {i}. Skipping for AUC.")

    if processed_samples == 0 or len(set(y_true_flat)) < 2:
        logger.warning("Not enough valid data or only one class for AUC calculation.")
        return 0.5
    try:
        auc = roc_auc_score(y_true_flat, y_score_flat)
        logger.info(f"AUC Score calculated: {auc:.6f}")
        return auc
    except ValueError as e:
        logger.error(f"Error calculating AUC: {e}. Returning 0.5.")
        return 0.5


def calculate_average_precision(all_q_values: torch.Tensor, all_actual_actions: torch.Tensor) -> float:
    """ Average Precision (AP) - Q-значения как скоры. """
    logger.info("Calculating Average Precision Score...")
    num_samples, num_actions = all_q_values.shape
    if num_samples == 0: return 0.0

    y_true_flat = []
    y_score_flat = []
    processed_samples = 0
    positive_samples = 0
    for i in range(num_samples):
        q_scores = all_q_values[i].numpy()
        actual_action = all_actual_actions[i].item()
        labels = np.zeros(num_actions, dtype=int)
        if 0 <= actual_action < num_actions:
            labels[actual_action] = 1
            y_true_flat.extend(labels)
            y_score_flat.extend(q_scores)
            processed_samples += 1
            if labels[actual_action] == 1:
                positive_samples += 1
        else:
            logger.warning(f"Action {actual_action} out of bounds for sample {i}. Skipping for AP.")

    if processed_samples == 0 or positive_samples == 0:
        logger.warning("Not enough valid data or no positive samples for Average Precision calculation.")
        return 0.0
    try:
        ap = average_precision_score(y_true_flat, y_score_flat)
        logger.info(f"Average Precision Score calculated: {ap:.6f}")
        return ap
    except ValueError as e:
        logger.error(f"Error calculating Average Precision: {e}. Returning 0.0.")
        return 0.0


def calculate_recommendation_diversity(all_top_k_indices: torch.Tensor, k: int) -> float:
    """ Diversity@K - разнообразие топ-K рекомендаций (нормализованная энтропия). """
    logger.info(f"Calculating Recommendation Diversity@{k}...")
    if all_top_k_indices.numel() == 0: return 0.0

    recommended_items = all_top_k_indices.flatten().tolist()
    if not recommended_items: return 0.0

    item_counts = Counter(recommended_items)
    total_recommendations = len(recommended_items)
    num_unique_items = len(item_counts)

    entropy = -sum(
        (count / total_recommendations) * np.log2(count / total_recommendations)
        for count in item_counts.values() if count > 0
    )
    max_entropy = np.log2(num_unique_items) if num_unique_items > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    logger.info(f"Recommendation Diversity@{k} calculated: {normalized_entropy:.6f}")
    return normalized_entropy


def calculate_accuracy_at_k(all_top_k_indices: torch.Tensor, all_actual_actions: torch.Tensor, k: int) -> float:
    """ Accuracy@K - доля случаев, когда реальное действие в топ-K предсказаний. """
    logger.info(f"Calculating Accuracy@{k}...")
    num_samples = all_actual_actions.numel()
    if num_samples == 0: return 0.0

    actual_next_expanded = all_actual_actions.unsqueeze(1)
    hits = (all_top_k_indices == actual_next_expanded).any(dim=1).sum().item()
    accuracy = hits / num_samples
    logger.info(f"Accuracy@{k} calculated: {accuracy:.6f}")
    return accuracy

def calculate_category_accuracy_at_k(
        all_top_k_indices: torch.Tensor,
        all_actual_actions: torch.Tensor,
        product_to_category_map: Dict[int, int],
        k: int
    ) -> float:
    """
    Category Accuracy@K - доля случаев, когда категория реального действия присутствует
    среди категорий top-K рекомендаций.
    """
    logger.info(f"Calculating Category Accuracy@{k}...")
    num_samples = all_actual_actions.numel()
    if num_samples == 0:
        return 0.0

    hits = 0
    
    for recs, actual in zip(all_top_k_indices.tolist(), all_actual_actions.tolist()):
        
        actual_cat = product_to_category_map.get(actual)
        if actual_cat is None:
            continue
        
        rec_cats = [product_to_category_map.get(prod) for prod in recs]
        
        if actual_cat in rec_cats:
            hits += 1

    category_accuracy = hits / num_samples
    logger.info(f"Category Accuracy@{k} calculated: {category_accuracy:.6f}")
    return category_accuracy

def calculate_mrr_at_k(all_top_k_indices: torch.Tensor, all_actual_actions: torch.Tensor, k: int) -> float:
    """ Mean Reciprocal Rank @ K - средний обратный ранг реального действия в топ-K. """
    logger.info(f"Calculating MRR@{k}...")
    num_samples = all_actual_actions.numel()
    if num_samples == 0: return 0.0

    actual_next_expanded = all_actual_actions.unsqueeze(1)
    match_mask = (all_top_k_indices == actual_next_expanded)
    ranks = torch.where(match_mask)[1] + 1

    rr_tensor = torch.zeros_like(all_actual_actions, dtype=torch.float32)
    hit_indices = torch.where(match_mask.any(dim=1))[0]
    if len(hit_indices) > 0:
        rr_tensor[hit_indices] = 1.0 / ranks.float()

    mrr = rr_tensor.mean().item()
    logger.info(f"MRR@{k} calculated: {mrr:.6f}")
    return mrr



def get_recommendations_and_q_values(policy_net, dataloader, k: int, device: torch.device) -> tuple:
    """
    Проходит по даталоадеру (переходов), возвращает топ-k рекомендаций,
    реальные действия и Q-значения для каждого перехода.

    Args:
        policy_net: Обученная модель (policy network).
        dataloader: DataLoader с данными переходов для оценки.
        k: Количество рекомендаций (топ-k).
        device: Устройство ('cuda' или 'cpu').

    Returns:
        Кортеж из:
        - all_top_k_indices (torch.Tensor): Тензор с индексами топ-k продуктов [N, k].
        - all_actual_actions (torch.Tensor): Тензор с реальными действиями [N].
        - all_q_values (torch.Tensor): Тензор всех Q-значений [N, num_products].
    """
    policy_net.eval()
    all_top_k_indices = []
    all_actual_actions = []
    all_q_values = []
    logger.info(f"Generating Top-{k} recommendations and Q-values from transitions...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Generating Recs & QVals", leave=False, ncols=100):
            try:
                batch_gpu = {key: val.to(device, non_blocking=True) for key, val in batch.items()}
                actual_next_items = batch_gpu['action']

                network_inputs = (
                    batch_gpu['state_history'], batch_gpu['state_length'], batch_gpu['state_numerical_features'],
                    batch_gpu['state_brand_idx'], batch_gpu['state_holiday_idx']
                )
                q_values_all = policy_net(*network_inputs) # [batch_size, num_products]
                _, top_k_indices = torch.topk(q_values_all, k, dim=1) # [batch_size, k]

                all_top_k_indices.append(top_k_indices.cpu())
                all_actual_actions.append(actual_next_items.cpu())
                all_q_values.append(q_values_all.cpu())
            except KeyError as e:
                logger.error(f"Missing key in batch: {e}. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Error processing batch: {e}", exc_info=True)
                continue

    if not all_top_k_indices:
        logger.warning("No recommendations were generated.")
        try:
            num_products = policy_net.out_layer.out_features
        except AttributeError:
             logger.warning("Cannot determine num_products from policy_net. Returning empty tensors with placeholder dim.")
             num_products = 1 
        return (torch.empty((0, k), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
                torch.empty((0, num_products), dtype=torch.float32))

    all_top_k_indices = torch.cat(all_top_k_indices, dim=0)
    all_actual_actions = torch.cat(all_actual_actions, dim=0)
    all_q_values = torch.cat(all_q_values, dim=0)
    logger.info(f"Generated recommendations and Q-values for {all_actual_actions.shape[0]} transitions.")
    return all_top_k_indices, all_actual_actions, all_q_values


def calculate_loss(trainer, dataloader) -> float:
    """Рассчитывает средний лосс DQN."""
    logger.info("Calculating Loss...")
    try:
        loss = trainer.evaluate(dataloader)
        logger.info(f"Loss calculated: {loss:.6f}")
        return loss
    except Exception as e:
        logger.error(f"Error calculating loss: {e}", exc_info=True)
        return float('nan')


def calculate_ndcg_at_k(all_top_k_indices: torch.Tensor, all_actual_actions: torch.Tensor, k: int) -> float:
    """ Normalized Discounted Cumulative Gain @ K. """
    logger.info(f"Calculating NDCG@{k}...")
    num_samples = all_actual_actions.numel()
    if num_samples == 0: return 0.0

    actual_next_expanded = all_actual_actions.unsqueeze(1)
    match_mask = (all_top_k_indices == actual_next_expanded)
    ranks = torch.where(match_mask)[1] + 1

    dcg_tensor = torch.zeros_like(all_actual_actions, dtype=torch.float32)
    hit_indices = torch.where(match_mask.any(dim=1))[0]
    if len(hit_indices) > 0:
        dcg_tensor[hit_indices] = 1.0 / torch.log2(ranks.float() + 1.0)

    total_dcg = dcg_tensor.sum().item()
    total_idcg = float(num_samples)

    ndcg = total_dcg / total_idcg if total_idcg > 0 else 0.0
    logger.info(f"NDCG@{k} calculated: {ndcg:.6f}")
    return ndcg




def calculate_all_metrics(policy_net,
                          trainer,
                          test_dataloader,
                          test_df: pd.DataFrame,
                          k: int,
                          settings,
                          product_to_category_map: Optional[Dict[int, int]] = None) -> Tuple[Dict[str, float], Dict[str, Optional[torch.Tensor]]]:
    """
    Объединенная функция для расчета всех метрик и сбора промежуточных данных.

    Args:
        policy_net: Обученная модель (policy network).
        trainer: Экземпляр DQLTrainer.
        test_dataloader: DataLoader с тестовыми данными (переходы).
        test_df: DataFrame с тестовыми данными (события/сессии).
        k: Параметр K для метрик.
        settings: Объект настроек проекта.
        product_to_category_map: Mapping from product IDs to category IDs (optional).

    Returns:
        Кортеж из двух словарей:
        - final_metrics: Словарь с рассчитанными итоговыми метриками.
        - intermediate_data: Словарь с промежуточными тензорами для построения графиков
                             ('top_k', 'actual', 'q_values', 'rewards').
    """
    logger.info(f"Starting calculation of all metrics and intermediate data (K={k})...")
    metrics = {}
    intermediate_data = {}
    device = settings.DEVICE

    
    all_top_k_indices, all_actual_actions, all_q_values = get_recommendations_and_q_values(
        policy_net, test_dataloader, k, device
    )
    intermediate_data['top_k'] = all_top_k_indices
    intermediate_data['actual'] = all_actual_actions
    intermediate_data['q_values'] = all_q_values

    
    metrics['loss'] = calculate_loss(trainer, test_dataloader)
    metrics[f'accuracy@{k}'] = calculate_accuracy_at_k(all_top_k_indices, all_actual_actions, k)
    metrics[f'mrr@{k}'] = calculate_mrr_at_k(all_top_k_indices, all_actual_actions, k)
    metrics[f'ndcg@{k}'] = calculate_ndcg_at_k(all_top_k_indices, all_actual_actions, k)
    metrics['auc'] = calculate_auc(all_q_values, all_actual_actions)
    metrics['average_precision'] = calculate_average_precision(all_q_values, all_actual_actions)
    metrics[f'diversity@{k}'] = calculate_recommendation_diversity(all_top_k_indices, k)
    metrics[f'ctr@{k}'] = calculate_ctr_at_k(all_top_k_indices, all_actual_actions, k)
    # 2.1 Category-level accuracy
    if product_to_category_map is not None:
        metrics[f'category_accuracy@{k}'] = calculate_category_accuracy_at_k(
            all_top_k_indices, all_actual_actions, product_to_category_map, k)

    
    reward_stats, all_rewards_tensor = analyze_reward_distribution(test_dataloader, return_raw=True)
    metrics.update(reward_stats)
    intermediate_data['rewards'] = all_rewards_tensor

    
    metrics['conversion_rate'] = calculate_conversion_rate(test_df)
    metrics['arpu'] = calculate_arpu(test_df)
    metrics['avg_session_length'] = calculate_avg_session_length(test_df)


    logger.info("Finished calculation of all metrics and intermediate data.")
    metrics_rounded = {key: round(val, 6) if isinstance(val, (float, np.floating)) and not np.isnan(val) else val for key, val in metrics.items()}

    final_metrics = {
        'loss': metrics_rounded.get('loss'),
        f'accuracy_at_{k}': metrics_rounded.get(f'accuracy@{k}'),
        f'mean_reciprocal_rank': metrics_rounded.get(f'mrr@{k}'),
        f'ndcg_at_{k}': metrics_rounded.get(f'ndcg@{k}'),
        'auc_score': metrics_rounded.get('auc'),
        'average_precision': metrics_rounded.get('average_precision'),
        'diversity_score': metrics_rounded.get(f'diversity@{k}'),
        'ctr': metrics_rounded.get(f'ctr@{k}'),
        'category_accuracy': metrics_rounded.get(f'category_accuracy@{k}'),
        'conversion_rate': metrics_rounded.get('conversion_rate'),
        'arpu': metrics_rounded.get('arpu'),
        'avg_session_length': metrics_rounded.get('avg_session_length'),
        'reward_mean': metrics_rounded.get('reward_mean'),
        'reward_std': metrics_rounded.get('reward_std'),
        'reward_min': metrics_rounded.get('reward_min'),
        'reward_max': metrics_rounded.get('reward_max'),
        'reward_median': metrics_rounded.get('reward_median'),
    }
    final_metrics = {k: v for k, v in final_metrics.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
    return final_metrics, intermediate_data



def plot_reward_distribution_plotly(
        rewards_tensor: torch.Tensor, 
        title: str = "Reward Distribution") -> go.Figure:
    """Creates a Plotly histogram of the reward distribution."""
    if rewards_tensor is None or rewards_tensor.numel() == 0:
        logger.warning(f"No data provided for {title}.")
        return go.Figure(layout_title_text=f"{title} (No Data)")
    fig = px.histogram(rewards_tensor.numpy(), nbins=50, title=title, labels={'value': 'Reward'})
    fig.update_layout(bargap=0.1, xaxis_title="Reward Value", yaxis_title="Frequency")
    return fig

def plot_recommendation_distribution_plotly(
        top_k_indices: torch.Tensor, 
        k: int, 
        top_n_items: int = 20, 
        title: str = "Top Recommended Items") -> go.Figure:
    """Creates a Plotly bar chart of the most frequent recommendations."""
    if top_k_indices is None or top_k_indices.numel() == 0:
        logger.warning(f"No data provided for {title}.")
        return go.Figure(layout_title_text=f"{title} (No Data)")
    all_recs = top_k_indices.flatten().tolist()
    if not all_recs:
         return go.Figure(layout_title_text=f"{title} (No Recommendations)")
    counts = Counter(all_recs).most_common(top_n_items)
    if not counts:
        return go.Figure(layout_title_text=f"{title} (No Recommendations Found)")
    item_ids, frequencies = zip(*counts)
    fig = px.bar(x=[str(i) for i in item_ids], y=list(frequencies), title=f"{title} (Top {top_n_items})",
                 labels={'x': 'Product ID', 'y': 'Frequency'})
    fig.update_layout(xaxis_title="Product ID", yaxis_title="Recommendation Frequency")
    return fig

def plot_q_value_distribution_plotly(
        q_values: torch.Tensor, 
        actual_actions: torch.Tensor, 
        title: str = "Q-Value Distribution") -> go.Figure:
    """Creates a Plotly histogram comparing Q-values for actual vs. all actions."""
    if q_values is None or q_values.numel() == 0 or actual_actions is None or actual_actions.numel() == 0:
        logger.warning(f"No data provided for {title}.")
        return go.Figure(layout_title_text=f"{title} (No Data)")

    try:
        q_values_np = q_values.numpy().flatten()
        valid_mask = (actual_actions >= 0) & (actual_actions < q_values.shape[1])
        valid_actions = actual_actions[valid_mask]
        valid_q_values = q_values[valid_mask]
        if valid_actions.numel() == 0:
             logger.warning("No valid actual actions found for Q-value plot.")
             q_for_actual_action_np = np.array([])
        else:
            q_for_actual_action_np = valid_q_values[torch.arange(len(valid_actions)), valid_actions].numpy()


        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=q_values_np, 
            name='All Q-values', 
            nbinsx=100, 
            opacity=0.6, 
            histnorm='probability density'))
        if q_for_actual_action_np.size > 0:
            fig.add_trace(go.Histogram(
                x=q_for_actual_action_np, 
                name='Q-values for Actual Action', 
                nbinsx=100, 
                opacity=0.6, 
                histnorm='probability density'))
        fig.update_layout(
            title_text=title,
            xaxis_title_text='Q-value',
            yaxis_title_text='Density',
            barmode='overlay',
            legend_title_text='Q-Value Type'
        )
        return fig
    except Exception as e:
        logger.error(f"Error plotting Q-value distribution: {e}", exc_info=True)
        return go.Figure(layout_title_text=f"{title} (Error during plotting)")


def plot_metrics_comparison_plotly(
        metrics_main: Dict, 
        metrics_baseline: Dict, 
        title: str = "Model Comparison") -> go.Figure:
    """Creates a Plotly grouped bar chart comparing metrics between two models."""
    all_keys = sorted(list(set(metrics_main.keys()) | set(metrics_baseline.keys())))
    main_vals = [metrics_main.get(k, np.nan) for k in all_keys]
    baseline_vals = [metrics_baseline.get(k, np.nan) for k in all_keys]

    df_plot = pd.DataFrame({'Metric': all_keys, 'Main': main_vals, 'Baseline': baseline_vals})
    df_melted = df_plot.melt(id_vars='Metric', var_name='Model', value_name='Value')
    df_melted = df_melted.dropna(subset=['Value'])
    df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')
    df_melted = df_melted.dropna(subset=['Value'])

    if df_melted.empty:
        logger.warning("No valid metric values available for comparison plot.")
        return go.Figure(layout_title_text=f"{title} (No Data)")

    fig = px.bar(df_melted, x='Metric', y='Value', color='Model', barmode='group', title=title,
                 labels={'Value': 'Metric Value'})
    fig.update_layout(xaxis_tickangle=-45, yaxis_title="Metric Value", xaxis_title=None, legend_title="Model")
    return fig



def generate_html_report(
        metrics_main: Dict,
        intermediate_data_main: Dict,
        metrics_baseline: Optional[Dict] = None,
        intermediate_data_baseline: Optional[Dict] = None,
        k: int = 10,
        save_path: str = "evaluation_report.html",
        product_to_category_map: Optional[Dict[int, int]] = None
    ):
    """
    Generates an HTML report with metrics and Plotly graphs.

    Args:
        metrics_main: Dictionary of metrics for the main model.
        intermediate_data_main: Dictionary of intermediate data for the main model.
        metrics_baseline: Dictionary of metrics for the baseline model (optional).
        intermediate_data_baseline: Dictionary of intermediate data for baseline (optional).
        k: K value used for metrics.
        save_path: Path to save the HTML file.
        product_to_category_map: Mapping from product IDs to category IDs (optional).
    """
    logger.info(f"Generating HTML report at {save_path}...")

    plots_html = {}

    logger.info("Generating plots...")
    plots_html['reward_main'] = plot_reward_distribution_plotly(
        intermediate_data_main.get('rewards'), 
        title="Main Model: Reward Distribution"
        ).to_html(full_html=False, 
                  include_plotlyjs=True)

    plots_html['recs_main'] = plot_recommendation_distribution_plotly(
        intermediate_data_main.get('top_k'), 
        k=k, 
        title="Main Model: Top Recommended Items"
        ).to_html(full_html=False, 
                  include_plotlyjs=True)

    plots_html['q_main'] = plot_q_value_distribution_plotly(
        intermediate_data_main.get('q_values'), 
        intermediate_data_main.get('actual'), 
        title="Main Model: Q-Value Distribution"
        ).to_html(full_html=False, 
                  include_plotlyjs=True)

    # Category distribution for main model
    if product_to_category_map is not None:
        plots_html['cats_main'] = plot_category_distribution_plotly(
            intermediate_data_main.get('top_k'),
            product_to_category_map,
            k,
            title="Main Model: Top Recommended Categories"
        ).to_html(full_html=False, include_plotlyjs=True)

    if metrics_baseline and intermediate_data_baseline:
        plots_html['reward_base'] = plot_reward_distribution_plotly(
            intermediate_data_baseline.get('rewards'), 
            title="Baseline Model: Reward Distribution"
            ).to_html(full_html=False, 
                      include_plotlyjs=True)

        plots_html['recs_base'] = plot_recommendation_distribution_plotly(
            intermediate_data_baseline.get('top_k'), 
            k=k, 
            title="Baseline Model: Top Recommended Items"
            ).to_html(full_html=False, 
                      include_plotlyjs=True)

        plots_html['q_base'] = plot_q_value_distribution_plotly(
            intermediate_data_baseline.get('q_values'), 
            intermediate_data_baseline.get('actual'), 
            title="Baseline Model: Q-Value Distribution"
            ).to_html(full_html=False, 
                      include_plotlyjs=True)

        # Category distribution for baseline model
        if product_to_category_map is not None:
            plots_html['cats_base'] = plot_category_distribution_plotly(
                intermediate_data_baseline.get('top_k'),
                product_to_category_map,
                k,
                title="Baseline Model: Top Recommended Categories"
            ).to_html(full_html=False, include_plotlyjs=True)

        plots_html['compare_metrics'] = plot_metrics_comparison_plotly(
            metrics_main, metrics_baseline
            ).to_html(full_html=False, 
                      include_plotlyjs=True)

    logger.info("Assembling HTML content...")
    style = """
<style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f9f9f9; color: #333;}
    h1, h2, h3 { color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px;}
    h1 { text-align: center; }
    table { border-collapse: collapse; width: auto; margin: 20px auto; box-shadow: 0 2px 3px rgba(0,0,0,0.1); background-color: white; }
    th, td { border: 1px solid #ddd; padding: 10px 15px; text-align: left; }
    th { background-color: #3498db; color: white; font-weight: bold; }
    tr:nth-child(even) { background-color: #f2f9fc; }
    .plot-container { margin: 30px auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: white; box-shadow: 0 2px 3px rgba(0,0,0,0.1); width: 80%; max-width: 1000px;}
    .plot-container h3 { margin-top: 0; color: #3498db; }
    .metrics-table-container { text-align: center; } /* Center align table container */
</style>
"""
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    {style}
</head>
<body>
    <h1>DQN Model Evaluation Report</h1>
    <p style="text-align: center;">Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p style="text-align: center;">K for Ranking Metrics: {k}</p>
"""
    # --- Metrics Table ---
    html_content += "<div class='metrics-table-container'><h2>Metrics Summary</h2>"
    if metrics_baseline:
        all_keys = sorted(list(set(metrics_main.keys()) | set(metrics_baseline.keys())))
        comparison_data = {
            'Metric': all_keys,
            'Main Model': [f"{metrics_main.get(key, 'N/A'):.6f}" if isinstance(metrics_main.get(key), float) else metrics_main.get(key, 'N/A') for key in all_keys],
            'Baseline Model': [f"{metrics_baseline.get(key, 'N/A'):.6f}" if isinstance(metrics_baseline.get(key), float) else metrics_baseline.get(key, 'N/A') for key in all_keys]
        }
        df_compare = pd.DataFrame(comparison_data)
        html_content += df_compare.to_html(index=False, na_rep='N/A', classes='metrics-table', justify='center') # Added class
    else:
        df_main = pd.DataFrame(list(metrics_main.items()), columns=['Metric', 'Value'])
        df_main['Value'] = df_main['Value'].apply(lambda x: f"{x:.6f}" if isinstance(x, float) else x)
        html_content += "<h3>Main Model Metrics</h3>"
        html_content += df_main.to_html(index=False, na_rep='N/A', classes='metrics-table', justify='center') # Added class
    html_content += "</div>"
    # --- Comparison Plot ---
    if 'compare_metrics' in plots_html:
        html_content += f"""
        <div class="plot-container">
            <h3>Metrics Comparison</h3>
            {plots_html['compare_metrics']}
        </div>
        """
    # --- Baseline Model Plots ---
    html_content += "<h2>Baseline Model Analysis</h2>"
    html_content += f'<div class="plot-container">{plots_html.get("reward_base", "Plot unavailable.")}</div>'
    html_content += f'<div class="plot-container">{plots_html.get("recs_base", "Plot unavailable.")}</div>'
    html_content += f'<div class="plot-container">{plots_html.get("q_base", "Plot unavailable.")}</div>'
    # Show category distribution for baseline
    if 'cats_base' in plots_html:
        html_content += f'<div class="plot-container">{plots_html.get("cats_base")}</div>'
    # --- Main Model Plots ---
    html_content += "<h2>Main Model Analysis</h2>"
    html_content += f'<div class="plot-container">{plots_html.get("reward_main", "Plot unavailable.")}</div>'
    html_content += f'<div class="plot-container">{plots_html.get("recs_main", "Plot unavailable.")}</div>'
    html_content += f'<div class="plot-container">{plots_html.get("q_main", "Plot unavailable.")}</div>'
    # Show category distribution for main
    if 'cats_main' in plots_html:
        html_content += f'<div class="plot-container">{plots_html.get("cats_main")}</div>'
    # --- HTML Footer ---
    html_content += """
</body>
</html>
"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    logger.info(f"HTML report saved successfully to {save_path}")




def calculate_all_metrics_and_report(
        policy_net,
        trainer,
        test_dataloader,
        test_df: pd.DataFrame,
        k: int,
        settings,
        product_to_category_map: Optional[Dict[int, int]] = None,
        policy_net_baseline=None,
        trainer_baseline=None,
        generate_report: bool = True,
        report_save_path: str = "evaluation_report.html"
    ) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    """
    Calculates metrics for main model and optionally baseline, then generates an HTML report.

    Args:
        policy_net: Trained main policy network.
        trainer: Trainer for the main model.
        test_dataloader: DataLoader for test transitions.
        test_df: DataFrame for test session events.
        k: K value for ranking metrics.
        settings: Project settings object.
        product_to_category_map: Mapping from product IDs to category IDs (optional).
        policy_net_baseline: Trained baseline policy network (optional).
        trainer_baseline: Trainer for the baseline model (optional).
        generate_report: Whether to generate the HTML report.
        report_save_path: Path to save the HTML report.

    Returns:
        A tuple containing:
        - Dictionary of metrics for the main model.
        - Dictionary of metrics for the baseline model (or None).
    """
    # --- Evaluate Baseline Model  ---
    logger.info("="*20 + " Evaluating Baseline Model " + "="*20)
    metrics_baseline, intermediate_data_baseline = calculate_all_metrics(
        policy_net=policy_net_baseline,
        trainer=trainer_baseline,
        test_dataloader=test_dataloader,
        test_df=test_df.copy(),
        k=k,
        settings=settings,
        product_to_category_map=product_to_category_map
    )
    logger.info("Baseline Model Metrics:")
    for name in sorted(metrics_baseline.keys()):
        logger.info(f"\t{name}: {metrics_baseline[name]:.6f}")
   
    # --- Evaluate Main Model ---
    logger.info("="*20 + " Evaluating Main Model " + "="*20)
    metrics_main, intermediate_data_main = calculate_all_metrics(
        policy_net=policy_net,
        trainer=trainer,
        test_dataloader=test_dataloader,
        test_df=test_df,
        k=k,
        settings=settings,
        product_to_category_map=product_to_category_map
    )
    logger.info("Main Model Metrics:")
    for name in sorted(metrics_main.keys()):
        logger.info(f"\t{name}: {metrics_main[name]:.6f}")

    
    # --- Generate Report  ---
    if generate_report:
        generate_html_report(
            metrics_main=metrics_main,
            intermediate_data_main=intermediate_data_main,
            metrics_baseline=metrics_baseline,
            intermediate_data_baseline=intermediate_data_baseline,
            k=k,
            save_path=report_save_path,
            product_to_category_map=product_to_category_map
        )

    return metrics_main, metrics_baseline

def plot_category_distribution_plotly(
        top_k_indices: torch.Tensor,
        product_to_category_map: Dict[int, int],
        k: int,
        top_n_categories: int = 20,
        title: str = "Top Recommended Categories") -> go.Figure:
    """Creates a Plotly bar chart of the most frequent recommended categories."""
    logger.info(f"Generating category distribution plot: {title}")
    if top_k_indices is None or top_k_indices.numel() == 0:
        logger.warning(f"No recommendations provided for {title}.")
        return go.Figure(layout_title_text=f"{title} (No Data)")
    # Flatten item recommendations and map to categories
    rec_items = top_k_indices.flatten().tolist()
    rec_cats = [product_to_category_map.get(item) for item in rec_items if product_to_category_map.get(item) is not None]
    if not rec_cats:
        return go.Figure(layout_title_text=f"{title} (No Categories)")
    counts = Counter(rec_cats).most_common(top_n_categories)
    cats, freqs = zip(*counts)
    fig = px.bar(x=[str(c) for c in cats], y=list(freqs), title=title,
                 labels={'x': 'Category ID', 'y': 'Frequency'})
    fig.update_layout(xaxis_title="Category ID", yaxis_title="Recommendation Frequency")
    return fig
