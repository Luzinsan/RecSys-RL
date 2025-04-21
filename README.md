# RecSys-RL

A Reinforcement Learning-based Recommender System for E-commerce using Deep Q-Networks (DQN).

## Project Structure

```
RecSys-RL/
├── datasets/
│   ├── synthetic_test_data.csv    # Generated synthetic test data
├── dqn.py                         # DQN model implementation
├── evaluate.py                    # Evaluation framework
├── generate_test_data.py          # Synthetic data generation
└── README.md                      # Project documentation
```

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Workflow

### 1. Generate Synthetic Test Data

The project includes a script to generate synthetic e-commerce data for testing:

```bash
python generate_test_data.py
```

This will create `datasets/synthetic_test_data.csv` with the following features:
- User sessions and events (view, cart, purchase)
- Product information (price, category, brand)
- User and product statistics
- Session-level features

### 2. Train DQN Model

Train the DQN model on the synthetic data:

```bash
python dqn.py
```

The training process:
1. Loads and preprocesses the synthetic data
2. Initializes the DQN agent with:
   - State size: 10 + MAX_HISTORY_LENGTH * 3
   - Action size: Number of unique products
   - Hidden size: 128
3. Trains for 10 epochs with:
   - Learning rate: 0.00001
   - Gamma: 0.99
   - Batch size: 32
4. Saves the trained model to `dqn_model.pth`

### 3. Evaluate Model Performance

Evaluate the trained model:

```bash
python evaluate.py
```

The evaluation includes:

#### Basic Metrics
- Conversion Rate (CR)
- Average Revenue Per User (ARPU)
- Click-Through Rate (CTR)
- Average Session Length

#### Advanced Metrics
- Reward Distribution Analysis
- AUC Score
- Average Precision
- Diversity Score
- Accuracy@k
- Mean Reciprocal Rank (MRR)

#### Visualization
- Learning curve plot showing metric trends
- Comparison with baseline (if available)

## Configuration

Key parameters in `dqn.py`:
```python
RANDOM_SEED = 42
N_RECOMMENDATIONS = 5
MIN_SESSION_SIZE = 3
MAX_HISTORY_LENGTH = 10
LEARNING_RATE = 0.00001
GAMMA = 0.99
BATCH_SIZE = 32
EPOCHS = 10
MEMORY_SIZE = 10000
```

Reward mapping:
```python
REWARD_MAP = {
    'view': 0.1,
    'cart': 1.0,
    'purchase': 5.0
}
```

## Customization

### Data Generation
Modify `generate_test_data.py` to adjust:
- Number of users, products, categories
- Event type probabilities
- Price ranges
- Session characteristics

### Model Training
Adjust in `dqn.py`:
- Network architecture
- Hyperparameters
- Reward function
- State representation

### Evaluation
Customize in `evaluate.py`:
- Metric calculations
- Visualization settings
- Baseline comparison

## Example Usage

1. Generate test data:
```python
python generate_test_data.py
```

2. Train the model:
```python
python dqn.py
```

3. Evaluate results:
```python
python evaluate.py
```

The evaluation will output metrics and generate plots showing the model's performance over time (there is a problem with plot - I fix it later).




