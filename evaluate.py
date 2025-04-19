import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

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

if __name__ == "__main__":
    main()
