import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from dataclasses import dataclass
from datetime import datetime

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

class RLAgentEvaluator:
    def __init__(self, reward_weights: Dict[str, float] = None):
        """
        Initialize the evaluator with optional reward weights.
        
        Args:
            reward_weights: Dictionary mapping event types to their reward values
        """
        self.reward_weights = reward_weights or {
            'view': 1.0,
            'cart': 2.0,
            'purchase': 5.0
        }
        self.metrics_history = defaultdict(list)
        
    def calculate_conversion_rate(self, sessions: List[Dict]) -> float:
        """
        Calculate the conversion rate (purchases / total sessions)
        
        Args:
            sessions: List of session dictionaries containing events
            
        Returns:
            float: Conversion rate
        """
        total_sessions = len(sessions)
        purchase_sessions = sum(1 for session in sessions if any(event['type'] == 'purchase' for event in session['events']))
        return purchase_sessions / total_sessions if total_sessions > 0 else 0.0
    
    def calculate_arpu(self, sessions: List[Dict]) -> float:
        """
        Calculate Average Revenue Per User
        
        Args:
            sessions: List of session dictionaries containing events and prices
            
        Returns:
            float: Average revenue per user
        """
        total_revenue = sum(
            sum(event['price'] for event in session['events'] if event['type'] == 'purchase')
            for session in sessions
        )
        return total_revenue / len(sessions) if sessions else 0.0
    
    def calculate_ctr(self, sessions: List[Dict], recommendations: List[List[int]]) -> float:
        """
        Calculate Click-Through Rate
        
        Args:
            sessions: List of session dictionaries
            recommendations: List of recommended product IDs for each session
            
        Returns:
            float: Click-through rate
        """
        total_recommendations = sum(len(recs) for recs in recommendations)
        clicks = sum(
            sum(1 for event in session['events'] if event['type'] == 'view' and event['product_id'] in recs)
            for session, recs in zip(sessions, recommendations)
        )
        return clicks / total_recommendations if total_recommendations > 0 else 0.0
    
    def calculate_average_session_length(self, sessions: List[Dict]) -> float:
        """
        Calculate average number of events per session
        
        Args:
            sessions: List of session dictionaries
            
        Returns:
            float: Average session length
        """
        return np.mean([len(session['events']) for session in sessions]) if sessions else 0.0
    
    def analyze_reward_distribution(self, sessions: List[Dict]) -> Dict[str, float]:
        """
        Analyze the distribution of rewards across different event types
        
        Args:
            sessions: List of session dictionaries
            
        Returns:
            Dict[str, float]: Reward distribution statistics
        """
        rewards = []
        for session in sessions:
            session_reward = sum(
                self.reward_weights[event['type']]
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
    
    def evaluate_episode(self, sessions: List[Dict], recommendations: List[List[int]]) -> Dict[str, float]:
        """
        Evaluate a single episode of the RL agent
        
        Args:
            sessions: List of session dictionaries
            recommendations: List of recommended product IDs for each session
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        metrics = {
            'conversion_rate': self.calculate_conversion_rate(sessions),
            'arpu': self.calculate_arpu(sessions),
            'ctr': self.calculate_ctr(sessions, recommendations),
            'avg_session_length': self.calculate_average_session_length(sessions),
            **self.analyze_reward_distribution(sessions)
        }
        
        # Store metrics in history
        for metric, value in metrics.items():
            self.metrics_history[metric].append(value)
            
        return metrics
    
    def plot_metrics_history(self, window_size: int = 10) -> None:
        """
        Plot the history of metrics over episodes
        
        Args:
            window_size: Size of the moving average window
        """
        plt.figure(figsize=(15, 10))
        
        for metric, values in self.metrics_history.items():
            # Calculate moving average
            moving_avg = pd.Series(values).rolling(window=window_size).mean()
            
            plt.plot(moving_avg, label=metric)
        
        plt.xlabel('Episodes')
        plt.ylabel('Metric Value')
        plt.title('RL Agent Performance Metrics Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def compare_with_baseline(self, baseline_sessions: List[Dict], 
                            rl_sessions: List[Dict],
                            baseline_recommendations: List[List[int]],
                            rl_recommendations: List[List[int]]) -> Dict[str, Dict[str, float]]:
        """
        Compare RL agent performance with a baseline model
        
        Args:
            baseline_sessions: Sessions from baseline model
            rl_sessions: Sessions from RL agent
            baseline_recommendations: Recommendations from baseline
            rl_recommendations: Recommendations from RL agent
            
        Returns:
            Dict[str, Dict[str, float]]: Comparison metrics
        """
        baseline_metrics = self.evaluate_episode(baseline_sessions, baseline_recommendations)
        rl_metrics = self.evaluate_episode(rl_sessions, rl_recommendations)
        
        comparison = {}
        for metric in baseline_metrics:
            comparison[metric] = {
                'baseline': baseline_metrics[metric],
                'rl': rl_metrics[metric],
                'improvement': (rl_metrics[metric] - baseline_metrics[metric]) / baseline_metrics[metric] * 100
            }
        
        return comparison

class TestEvaluator:
    def __init__(self, reward_weights: Dict[str, float] = None):
        """
        Initialize the test evaluator with optional reward weights.
        
        Args:
            reward_weights: Dictionary mapping event types to their reward values
        """
        self.reward_weights = reward_weights or {
            'view': 1.0,
            'cart': 2.0,
            'purchase': 5.0
        }
        self.metrics_history = defaultdict(list)
        
    def calculate_diversity_score(self, recommendations: List[List[int]]) -> float:
        """
        Calculate the diversity of recommendations using entropy
        
        Args:
            recommendations: List of recommended product IDs for each session
            
        Returns:
            float: Diversity score (higher is better)
        """
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
    
    def calculate_novelty_score(self, recommendations: List[List[int]], 
                              historical_items: set) -> float:
        """
        Calculate the novelty of recommendations
        
        Args:
            recommendations: List of recommended product IDs
            historical_items: Set of items that have been recommended before
            
        Returns:
            float: Novelty score (higher is better)
        """
        if not recommendations:
            return 0.0
            
        total_items = sum(len(recs) for recs in recommendations)
        if total_items == 0:
            return 0.0
            
        novel_items = sum(
            sum(1 for item in recs if item not in historical_items)
            for recs in recommendations
        )
        
        return novel_items / total_items
    
    def calculate_auc_score(self, sessions: List[Dict], recommendations: List[List[int]]) -> float:
        """
        Calculate AUC score for purchase prediction
        
        Args:
            sessions: List of session dictionaries
            recommendations: List of recommended product IDs
            
        Returns:
            float: AUC score
        """
        y_true = []
        y_score = []
        
        for session, recs in zip(sessions, recommendations):
            purchased_items = {event['product_id'] for event in session['events'] 
                             if event['type'] == 'purchase'}
            
            for item in recs:
                y_true.append(1 if item in purchased_items else 0)
                # Simple scoring: higher score for items recommended earlier
                y_score.append(1.0 / (recs.index(item) + 1))
        
        if len(set(y_true)) < 2:
            return 0.5  # Return 0.5 if all predictions are the same
            
        return roc_auc_score(y_true, y_score)
    
    def calculate_average_precision(self, sessions: List[Dict], 
                                  recommendations: List[List[int]]) -> float:
        """
        Calculate Average Precision for purchase prediction
        
        Args:
            sessions: List of session dictionaries
            recommendations: List of recommended product IDs
            
        Returns:
            float: Average Precision score
        """
        y_true = []
        y_score = []
        
        for session, recs in zip(sessions, recommendations):
            purchased_items = {event['product_id'] for event in session['events'] 
                             if event['type'] == 'purchase'}
            
            for item in recs:
                y_true.append(1 if item in purchased_items else 0)
                y_score.append(1.0 / (recs.index(item) + 1))
        
        if sum(y_true) == 0:
            return 0.0
            
        return average_precision_score(y_true, y_score)
    
    def evaluate_test_set(self, test_sessions: List[Dict], 
                         test_recommendations: List[List[int]],
                         historical_items: Optional[set] = None) -> TestMetrics:
        """
        Evaluate model performance on test dataset
        
        Args:
            test_sessions: List of session dictionaries from test set
            test_recommendations: List of recommended product IDs
            historical_items: Set of items that have been recommended before
            
        Returns:
            TestMetrics: Comprehensive test metrics
        """
        # Calculate basic metrics
        conversion_rate = self.calculate_conversion_rate(test_sessions)
        arpu = self.calculate_arpu(test_sessions)
        ctr = self.calculate_ctr(test_sessions, test_recommendations)
        avg_session_length = self.calculate_average_session_length(test_sessions)
        
        # Calculate reward distribution
        reward_stats = self.analyze_reward_distribution(test_sessions)
        
        # Calculate advanced metrics
        auc_score = self.calculate_auc_score(test_sessions, test_recommendations)
        avg_precision = self.calculate_average_precision(test_sessions, test_recommendations)
        diversity_score = self.calculate_diversity_score(test_recommendations)
        
        # Calculate novelty if historical items are provided
        novelty_score = None
        if historical_items is not None:
            novelty_score = self.calculate_novelty_score(test_recommendations, historical_items)
        
        return TestMetrics(
            conversion_rate=conversion_rate,
            arpu=arpu,
            ctr=ctr,
            avg_session_length=avg_session_length,
            reward_mean=reward_stats['mean'],
            reward_std=reward_stats['std'],
            reward_min=reward_stats['min'],
            reward_max=reward_stats['max'],
            reward_median=reward_stats['median'],
            auc_score=auc_score,
            average_precision=avg_precision,
            diversity_score=diversity_score,
            novelty_score=novelty_score
        )
    
    def plot_test_metrics_comparison(self, baseline_metrics: TestMetrics, 
                                   rl_metrics: TestMetrics) -> None:
        """
        Plot comparison of test metrics between baseline and RL model
        
        Args:
            baseline_metrics: Test metrics from baseline model
            rl_metrics: Test metrics from RL model
        """
        metrics = ['conversion_rate', 'arpu', 'ctr', 'avg_session_length', 
                  'reward_mean', 'auc_score', 'average_precision', 
                  'diversity_score']
        
        if rl_metrics.novelty_score is not None:
            metrics.append('novelty_score')
        
        baseline_values = [getattr(baseline_metrics, metric) for metric in metrics]
        rl_values = [getattr(rl_metrics, metric) for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(15, 8))
        rects1 = ax.bar(x - width/2, baseline_values, width, label='Baseline')
        rects2 = ax.bar(x + width/2, rl_values, width, label='RL Model')
        
        ax.set_ylabel('Metric Value')
        ax.set_title('Comparison of Test Metrics')
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
    evaluator = RLAgentEvaluator(reward_weights={
        'view': 1.0,
        'cart': 2.0,
        'purchase': 5.0
    })
    
    # Generate multiple episodes of synthetic data
    num_episodes = 50
    num_sessions_per_episode = 100
    
    for episode in range(num_episodes):
        sessions = []
        recommendations = []
        
        for _ in range(num_sessions_per_episode):
            # Generate random number of events (1-5)
            num_events = np.random.randint(1, 6)
            events = []
            
            # Generate events with increasing probability of purchase
            for i in range(num_events):
                if i == num_events - 1 and np.random.random() < 0.3:  # 30% chance of purchase at the end
                    event_type = 'purchase'
                elif np.random.random() < 0.6:  # 60% chance of view
                    event_type = 'view'
                else:
                    event_type = 'cart'
                
                events.append({
                    'type': event_type,
                    'product_id': np.random.randint(1, 100),
                    'price': np.random.uniform(10, 1000)
                })
            
            sessions.append({'events': events})
            
            # Generate random recommendations (3-5 items)
            num_recs = np.random.randint(3, 6)
            recs = np.random.randint(1, 100, size=num_recs).tolist()
            recommendations.append(recs)
        
        # Evaluate the episode
        metrics = evaluator.evaluate_episode(sessions, recommendations)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Current metrics: {metrics}")
            print("-" * 50)
    
    # Plot metrics history with a window size of 5
    evaluator.plot_metrics_history(window_size=5)

