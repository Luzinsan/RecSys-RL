import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from dataclasses import dataclass
from datetime import datetime
import torch
from dqn import DQNAgent, DEVICE, MAX_HISTORY_LENGTH

@dataclass
class TestMetrics:
    conversion_rate: float
    arpu: float
    ctr: float
    avg_session_length: float
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    reward_median: float
    auc_score: Optional[float] = None
    average_precision: Optional[float] = None
    diversity_score: Optional[float] = None
    novelty_score: Optional[float] = None
    accuracy_at_k: Optional[float] = None
    mean_reciprocal_rank: Optional[float] = None

class DQNEvaluator:
    def __init__(self, model_path: str, reward_weights: Dict[str, float] = None):
        """
        Initialize the DQN evaluator
        
        Args:
            model_path: Path to the trained DQN model
            reward_weights: Dictionary mapping event types to their reward values
        """
        self.reward_weights = reward_weights or {
            'view': 0.1,
            'cart': 1.0,
            'purchase': 5.0
        }
        
        # Load test data to get product mappings
        test_df = pd.read_csv('datasets/synthetic_test_data.csv')
        
        # Initialize agent with correct dimensions
        state_size = 10 + MAX_HISTORY_LENGTH * 3
        action_size = len(test_df['product_id'].unique())
        self.agent = DQNAgent(state_size, action_size)
        
        # Load model
        self.agent.policy_net.load_state_dict(torch.load(model_path))
        self.agent.policy_net.eval()
        
        # Initialize action mapping
        self.agent.label_encoders['product_id'].fit(test_df['product_id'])
        unique_products = test_df['product_id'].unique()
        self.agent.action_to_product = {i: product for i, product in enumerate(unique_products)}
        self.agent.product_to_action = {product: i for i, product in enumerate(unique_products)}
        
        self.metrics_history = defaultdict(list)
    
    def calculate_conversion_rate(self, sessions: List[Dict]) -> float:
        """Calculate the conversion rate (purchases / total sessions)"""
        total_sessions = len(sessions)
        purchase_sessions = sum(1 for session in sessions if any(event['event_type'] == 'purchase' for event in session['events']))
        return purchase_sessions / total_sessions if total_sessions > 0 else 0.0
    
    def calculate_arpu(self, sessions: List[Dict]) -> float:
        """Calculate Average Revenue Per User"""
        total_revenue = sum(
            sum(event['price'] for event in session['events'] if event['event_type'] == 'purchase')
            for session in sessions
        )
        return total_revenue / len(sessions) if sessions else 0.0
    
    def calculate_ctr(self, sessions: List[Dict], recommendations: List[List[int]]) -> float:
        """Calculate Click-Through Rate"""
        total_recommendations = sum(len(recs) for recs in recommendations)
        clicks = sum(
            sum(1 for event in session['events'] if event['event_type'] == 'view' and event['product_id'] in recs)
            for session, recs in zip(sessions, recommendations)
        )
        return clicks / total_recommendations if total_recommendations > 0 else 0.0
    
    def calculate_average_session_length(self, sessions: List[Dict]) -> float:
        """Calculate average number of events per session"""
        return np.mean([len(session['events']) for session in sessions]) if sessions else 0.0
    
    def analyze_reward_distribution(self, sessions: List[Dict]) -> Dict[str, float]:
        """Analyze the distribution of rewards across different event types"""
        rewards = []
        for session in sessions:
            session_reward = sum(
                self.reward_weights[event['event_type']]
                for event in session['events']
            )
            rewards.append(session_reward)
        
        return {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'median': np.median(rewards)
        }
    
    def calculate_auc_score(self, sessions: List[Dict], recommendations: List[List[int]]) -> float:
        """Calculate AUC score for purchase prediction"""
        y_true = []
        y_score = []
        
        for session, recs in zip(sessions, recommendations):
            purchased_items = {event['product_id'] for event in session['events'] 
                             if event['event_type'] == 'purchase'}
            
            for item in recs:
                y_true.append(1 if item in purchased_items else 0)
                y_score.append(1.0 / (recs.index(item) + 1))
        
        if len(set(y_true)) < 2:
            return 0.5
        
        return roc_auc_score(y_true, y_score)
    
    def calculate_average_precision(self, sessions: List[Dict], recommendations: List[List[int]]) -> float:
        """Calculate Average Precision for purchase prediction"""
        y_true = []
        y_score = []
        
        for session, recs in zip(sessions, recommendations):
            purchased_items = {event['product_id'] for event in session['events'] 
                             if event['event_type'] == 'purchase'}
            
            for item in recs:
                y_true.append(1 if item in purchased_items else 0)
                y_score.append(1.0 / (recs.index(item) + 1))
        
        if sum(y_true) == 0:
            return 0.0
        
        return average_precision_score(y_true, y_score)
    
    def calculate_diversity_score(self, recommendations: List[List[int]]) -> float:
        """Calculate the diversity of recommendations using entropy"""
        all_items = [item for recs in recommendations for item in recs]
        unique_items = set(all_items)
        item_counts = {item: all_items.count(item) for item in unique_items}
        total_recommendations = len(all_items)
        
        if total_recommendations == 0:
            return 0.0
            
        # Calculate entropy
        entropy = -sum(
            (count / total_recommendations) * np.log2(count / total_recommendations)
            for count in item_counts.values()
        )
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(unique_items)) if unique_items else 0
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def calculate_accuracy_at_k(self, recommendations: List[List[int]], 
                              ground_truth: List[int], k: int = 5) -> float:
        """
        Calculate accuracy@k for recommendations
        
        Args:
            recommendations: List of recommended product IDs
            ground_truth: List of actual purchased product IDs
            k: Number of recommendations to consider
            
        Returns:
            float: Accuracy@k score
        """
        correct = 0
        total = len(ground_truth)
        
        for recs, truth in zip(recommendations, ground_truth):
            if truth in recs[:k]:
                correct += 1
                
        return correct / total if total > 0 else 0.0
    
    def calculate_mrr(self, recommendations: List[List[int]], 
                     ground_truth: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank
        
        Args:
            recommendations: List of recommended product IDs
            ground_truth: List of actual purchased product IDs
            
        Returns:
            float: MRR score
        """
        reciprocal_ranks = []
        
        for recs, truth in zip(recommendations, ground_truth):
            if truth in recs:
                rank = recs.index(truth) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
                
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def evaluate_episode(self, sessions: List[Dict], 
                        recommendations: List[List[int]]) -> TestMetrics:
        """
        Evaluate a single episode of the DQN agent
        
        Args:
            sessions: List of session dictionaries
            recommendations: List of recommended product IDs
            
        Returns:
            TestMetrics: Comprehensive test metrics
        """
        # Calculate basic metrics
        conversion_rate = self.calculate_conversion_rate(sessions)
        arpu = self.calculate_arpu(sessions)
        ctr = self.calculate_ctr(sessions, recommendations)
        avg_session_length = self.calculate_average_session_length(sessions)
        
        # Calculate reward distribution
        reward_stats = self.analyze_reward_distribution(sessions)
        
        # Calculate advanced metrics
        auc_score = self.calculate_auc_score(sessions, recommendations)
        avg_precision = self.calculate_average_precision(sessions, recommendations)
        diversity_score = self.calculate_diversity_score(recommendations)
        
        # Calculate ranking metrics
        ground_truth = [session['events'][-1]['product_id'] 
                       for session in sessions 
                       if session['events'][-1]['event_type'] == 'purchase']
        accuracy_at_k = self.calculate_accuracy_at_k(recommendations, ground_truth)
        mrr = self.calculate_mrr(recommendations, ground_truth)
        
        # Store metrics in history
        metrics = {
            'conversion_rate': conversion_rate,
            'arpu': arpu,
            'ctr': ctr,
            'avg_session_length': avg_session_length,
            'reward_mean': reward_stats['mean'],
            'reward_std': reward_stats['std'],
            'reward_min': reward_stats['min'],
            'reward_max': reward_stats['max'],
            'reward_median': reward_stats['median'],
            'auc_score': auc_score,
            'average_precision': avg_precision,
            'diversity_score': diversity_score,
            'accuracy_at_k': accuracy_at_k,
            'mean_reciprocal_rank': mrr
        }
        
        for metric, value in metrics.items():
            if value is not None:  # Only store non-None values
                self.metrics_history[metric].append(value)
        
        return TestMetrics(**metrics)
    
    def plot_learning_curve(self, metrics_history: Dict[str, List[float]], 
                          window_size: int = 5) -> None:
        """
        Plot learning curve of metrics over episodes
        
        Args:
            metrics_history: Dictionary of metric histories
            window_size: Size of the moving average window
        """
        plt.figure(figsize=(15, 10))
        
        for metric, values in metrics_history.items():
            # Calculate moving average
            moving_avg = pd.Series(values).rolling(window=window_size).mean()
            
            plt.plot(moving_avg, label=metric)
        
        plt.xlabel('Episodes')
        plt.ylabel('Metric Value')
        plt.title('DQN Agent Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def compare_with_baseline(self, baseline_metrics: TestMetrics, 
                            dqn_metrics: TestMetrics) -> None:
        """
        Compare DQN performance with baseline model
        
        Args:
            baseline_metrics: Test metrics from baseline model
            dqn_metrics: Test metrics from DQN model
        """
        metrics = ['conversion_rate', 'arpu', 'ctr', 'avg_session_length', 
                  'reward_mean', 'auc_score', 'average_precision', 
                  'diversity_score', 'accuracy_at_k', 'mean_reciprocal_rank']
        
        baseline_values = [getattr(baseline_metrics, metric) for metric in metrics]
        dqn_values = [getattr(dqn_metrics, metric) for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(15, 8))
        rects1 = ax.bar(x - width/2, baseline_values, width, label='Baseline')
        rects2 = ax.bar(x + width/2, dqn_values, width, label='DQN')
        
        ax.set_ylabel('Metric Value')
        ax.set_title('Comparison of DQN with Baseline')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        
        # Add value labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        plt.show()

def main():
    # Example usage
    evaluator = DQNEvaluator('dqn_model.pth')
    
    # Load test data
    test_df = pd.read_csv('datasets/synthetic_test_data.csv')
    
    # Group by sessions
    sessions = []
    for _, group in test_df.groupby('user_session'):
        sessions.append({
            'events': group.to_dict('records')
        })
    
    # Generate recommendations
    recommendations = []
    for session in sessions:
        # Get state for the session
        state = evaluator.agent._get_state(session['events'], -1)
        # Get recommendations
        recs = evaluator.agent.get_recommendations(state)
        recommendations.append(recs)
    
    # Evaluate
    metrics = evaluator.evaluate_episode(sessions, recommendations)
    print("DQN Metrics:", metrics)
    
    # Plot learning curve
    evaluator.plot_learning_curve(evaluator.metrics_history)

if __name__ == "__main__":
    main()

