# RecSys-RL: Reinforcement Learning for E-commerce Recommendations

## Problem Formulation

This project implements a Reinforcement Learning (RL) based recommender system for e-commerce platforms. The system aims to optimize product recommendations by learning from user interaction patterns and maximizing long-term engagement and conversion.

### Key Challenges
1. Sequential Decision Making: Recommendations must consider the temporal nature of user sessions
2. Delayed Rewards: The impact of recommendations may not be immediate
3. Exploration vs Exploitation: Balancing between showing popular items and discovering new ones
4. Cold Start: Handling new users and products
5. Multi-objective Optimization: Balancing various metrics (CTR, conversion, revenue)

## Model Architecture

### Deep Q-Network (DQN)

The core of our system is a Deep Q-Network that learns to make optimal recommendation decisions. The model consists of:

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

### Training Process

1. **Data Preprocessing** (`src/scripts/preprocess.py`):
   - Session filtering and normalization
   - Feature engineering
   - Action space mapping

2. **Training Loop** (`src/models/dqn.py`):
   - Experience replay for sample efficiency
   - Target network updates
   - Epsilon-greedy exploration

3. **Hyperparameters**:
   ```python
   LEARNING_RATE = 0.00001
   GAMMA = 0.99        # Discount factor
   BATCH_SIZE = 32
   EPOCHS = 10
   MEMORY_SIZE = 10000
   ```

## Evaluation Framework

### Metrics

1. **Basic Metrics** (`src/models/evaluate.py`):
   - Conversion Rate (CR): Ratio of sessions with purchases
   - Average Revenue Per User (ARPU): Total revenue / number of users
   - Click-Through Rate (CTR): Ratio of recommended items that were viewed
   - Average Session Length: Mean number of events per session

2. **Advanced Metrics**:
   - AUC Score: Area under ROC curve for purchase prediction
   - Average Precision: Precision-recall trade-off
   - Diversity Score: Entropy-based measure of recommendation variety
   - Accuracy@k: Precision of top-k recommendations
   - Mean Reciprocal Rank (MRR): Ranking quality measure

3. **Reward Analysis**:
   - Mean, standard deviation
   - Min/max values
   - Distribution statistics

### Visualization

1. **Learning Curves**:
   - Metric trends over time
   - Moving averages for smoothing
   - Baseline comparisons

2. **Performance Analysis**:
   - Reward distribution plots
   - Metric correlation analysis
   - Session-level insights

## Project Structure

```
RecSys-RL/
├── src/
│   ├── models/
│   │   ├── dqn.py                    # DQN implementation
│   │   └── evaluate.py               # Evaluation framework
│   └── scripts/
│       ├── preprocess.py             # Data preprocessing
│       └── train.py                  # Training script
├── datasets/
│   └── synthetic_test_data.csv       # Generated test data
└── requirements.txt                  # Dependencies
```

### Key Files

1. **src/models/dqn.py**:
   - DQN model implementation
   - Training pipeline
   - State/action processing
   - Model saving/loading

2. **src/models/evaluate.py**:
   - Metric calculations
   - Performance visualization
   - Baseline comparison
   - Results analysis

3. **src/scripts/preprocess.py**:
   - Data preprocessing
   - Feature engineering
   - Data validation

4. **src/scripts/train.py**:
   - Training orchestration
   - Hyperparameter management
   - Model checkpointing

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
   python src/scripts/train.py
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




