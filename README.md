# RecSys-RL: Reinforcement Learning for E-commerce Recommendations

## Problem Formulation

This project implements two Reinforcement Learning (RL) based recommender systems for e-commerce platforms:
1. Deep Q-Network (DQN)
2. Actor-Critic

Both systems aim to optimize product recommendations by learning from user interaction patterns and maximizing long-term engagement and conversion.

### Key Challenges
1. Sequential Decision Making: Recommendations must consider the temporal nature of user sessions
2. Delayed Rewards: The impact of recommendations may not be immediate
3. Exploration vs Exploitation: Balancing between showing popular items and discovering new ones
4. Cold Start: Handling new users and products
5. Multi-objective Optimization: Balancing various metrics (CTR, conversion, revenue)

## Model Architectures

### 1. Deep Q-Network (DQN)

The DQN model learns to make optimal recommendation decisions through Q-learning. The model consists of:

1. **State Representation**:
   - User features (views, carts, purchases history)
   - Product features (price, category, popularity)
   - Session features (current session length, time)
   - Historical interaction features (last N interactions)

2. **Action Space**:
   - Set of possible product recommendations
   - Size: Number of unique products in the catalog

3. **Reward Function**:
   ```python
   REWARD_MAP = {
       'view': 0.1,    # Basic engagement
       'cart': 1.0,    # Stronger interest
       'purchase': 5.0 # Conversion
   }
   ```

4. **Network Architecture**:
   - Feature extraction layers
   - Q-value prediction layers
   - Target network for stable learning

### 2. Actor-Critic

The Actor-Critic model combines policy-based and value-based approaches:

1. **Actor Network**:
   - Policy network for action selection
   - Outputs action probabilities
   - Optimized for direct policy improvement

2. **Critic Network**:
   - Value network for state evaluation
   - Estimates state-value function
   - Provides baseline for policy updates

3. **Advantages**:
   - Better sample efficiency
   - Reduced variance in updates
   - More stable training

## Project Structure

```
RecSys-RL/
├── src/
│   ├── models/
│   │   ├── dqn/
│   │   │   ├── dqn_model.py         # DQN model implementation
│   │   │   ├── dqn_trainer.py       # DQN training logic
│   │   │   └── dqn_tuning.py        # Hyperparameter tuning
│   │   ├── actor_critic/
│   │   │   ├── actor_critic_model.py # Actor-Critic implementation
│   │   │   ├── actor_critic_trainer.py # Training logic
│   │   │   └── actor_critic_tuning.py  # Hyperparameter tuning
│   │   └── evaluate.py              # Evaluation framework
│   ├── scripts/
│   │   ├── preprocess.py            # Data preprocessing
│   │   └── train.py                 # Training orchestration
│   └── utils/
│       ├── data_utils.py            # Data handling utilities
│       └── metrics.py               # Metric calculations
├── datasets/
│   └── synthetic_test_data.csv      # Generated test data
└── requirements.txt                 # Dependencies
```

### Key Components

1. **DQN Implementation** (`src/models/dqn/`):
   - `dqn_model.py`: Core DQN architecture
   - `dqn_trainer.py`: Training pipeline
   - `dqn_tuning.py`: Hyperparameter optimization

2. **Actor-Critic Implementation** (`src/models/actor_critic/`):
   - `actor_critic_model.py`: Model architecture
   - `actor_critic_trainer.py`: Training logic
   - `actor_critic_tuning.py`: Parameter tuning

3. **Evaluation** (`src/models/evaluate.py`):
   - Metric calculations
   - Performance visualization
   - Model comparison

4. **Scripts** (`src/scripts/`):
   - `preprocess.py`: Data preprocessing
   - `train.py`: Training orchestration

5. **Utilities** (`src/utils/`):
   - `data_utils.py`: Data handling
   - `metrics.py`: Metric calculations

## Usage

1. **Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preprocessing**:
   ```bash
   python src/scripts/preprocess.py
   ```

3. **Model Training**:
   ```bash
   # Train DQN
   python src/scripts/train.py --model dqn
   
   # Train Actor-Critic
   python src/scripts/train.py --model actor_critic
   ```

4. **Evaluation**:
   ```bash
   python src/models/evaluate.py
   ```

## Model Customization

### DQN Architecture
- Modify network layers
- Adjust feature extraction
- Change state representation
- Implement new reward functions

### Actor-Critic Architecture
- Modify actor network
- Adjust critic network
- Change policy gradient method
- Implement new baseline

### Training Process
- Adjust hyperparameters
- Modify exploration strategy
- Change batch processing
- Implement new training techniques

### Evaluation
- Add custom metrics
- Modify visualization
- Implement new baselines
- Add A/B testing

## Future Improvements

1. **Model Enhancements**:
   - Implement Dueling DQN
   - Add Prioritized Experience Replay
   - Incorporate attention mechanisms
   - Add multi-task learning

2. **Feature Engineering**:
   - Add temporal features
   - Implement product embeddings
   - Add user preference modeling
   - Include contextual information

3. **Evaluation**:
   - Add offline/online evaluation
   - Implement counterfactual evaluation
   - Add fairness metrics
   - Include business metrics

## Contributing

Feel free to submit issues and enhancement requests!




