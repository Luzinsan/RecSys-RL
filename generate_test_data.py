import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict

def generate_synthetic_data(num_rows: int = 100) -> pd.DataFrame:
    """
    Generate synthetic e-commerce data based on example.csv structure
    
    Args:
        num_rows: Number of rows to generate
        
    Returns:
        pd.DataFrame: Generated synthetic data
    """
    # Initialize lists for each column
    data = {
        'event_time': [],
        'event_type': [],
        'product_id': [],
        'category_id': [],
        'category_code': [],
        'brand': [],
        'price': [],
        'user_id': [],
        'user_session': [],
        'session_event_num': [],
        'user_global_event_num': [],
        'user_views_before': [],
        'user_carts_before': [],
        'user_purchases_before': [],
        'product_views_before': [],
        'product_purchases_before': [],
        'product_avg_price': [],
        'category_views_before': [],
        'category_avg_price': []
    }
    
    # Generate some base data
    num_users = 20
    num_products = 50
    num_categories = 10
    num_brands = 15
    
    # Generate category codes
    category_codes = [
        'electronics.smartphone',
        'electronics.tablet',
        'electronics.laptop',
        'electronics.smartwatch',
        'electronics.headphones',
        'electronics.speaker',
        'electronics.camera',
        'electronics.tv',
        'electronics.gaming',
        'electronics.accessories'
    ]
    
    # Generate brand names
    brands = [
        'apple', 'samsung', 'sony', 'lg', 'xiaomi',
        'huawei', 'oneplus', 'google', 'microsoft', 'dell',
        'hp', 'lenovo', 'asus', 'acer', 'razer'
    ]
    
    # Generate base timestamps
    base_time = datetime(2023, 1, 1)
    
    # Generate user sessions
    user_sessions = {}
    for user_id in range(1, num_users + 1):
        num_sessions = random.randint(1, 5)
        for _ in range(num_sessions):
            session_id = f"{user_id}_{random.randint(1000, 9999)}"
            user_sessions[session_id] = {
                'user_id': user_id,
                'start_time': base_time + timedelta(days=random.randint(0, 30))
            }
    
    # Generate product data
    products = {}
    for product_id in range(1, num_products + 1):
        category_id = random.randint(1, num_categories)
        products[product_id] = {
            'category_id': category_id,
            'category_code': category_codes[category_id - 1],
            'brand': random.choice(brands),
            'price': round(random.uniform(50, 2000), 2),
            'views': 0,
            'carts': 0,
            'purchases': 0
        }
    
    # Generate category data
    categories = {}
    for category_id in range(1, num_categories + 1):
        categories[category_id] = {
            'views': 0,
            'avg_price': round(random.uniform(100, 1500), 2)
        }
    
    # Generate user data
    users = {}
    for user_id in range(1, num_users + 1):
        users[user_id] = {
            'views': 0,
            'carts': 0,
            'purchases': 0,
            'global_events': 0
        }
    
    # Generate events
    for session_id, session_info in user_sessions.items():
        user_id = session_info['user_id']
        current_time = session_info['start_time']
        session_events = random.randint(1, 10)
        
        for event_num in range(1, session_events + 1):
            # Select random product
            product_id = random.randint(1, num_products)
            product = products[product_id]
            category_id = product['category_id']
            
            # Determine event type with weighted probabilities
            event_type = random.choices(
                ['view', 'cart', 'purchase'],
                weights=[0.7, 0.2, 0.1]
            )[0]
            
            # Update counters
            users[user_id][event_type + 's'] += 1
            users[user_id]['global_events'] += 1
            products[product_id][event_type + 's'] += 1
            categories[category_id]['views'] += 1
            
            # Add row to data
            data['event_time'].append(current_time)
            data['event_type'].append(event_type)
            data['product_id'].append(product_id)
            data['category_id'].append(category_id)
            data['category_code'].append(product['category_code'])
            data['brand'].append(product['brand'])
            data['price'].append(product['price'])
            data['user_id'].append(user_id)
            data['user_session'].append(session_id)
            data['session_event_num'].append(event_num)
            data['user_global_event_num'].append(users[user_id]['global_events'])
            data['user_views_before'].append(users[user_id]['views'])
            data['user_carts_before'].append(users[user_id]['carts'])
            data['user_purchases_before'].append(users[user_id]['purchases'])
            data['product_views_before'].append(products[product_id]['views'])
            data['product_purchases_before'].append(products[product_id]['purchases'])
            data['product_avg_price'].append(product['price'])
            data['category_views_before'].append(categories[category_id]['views'])
            data['category_avg_price'].append(categories[category_id]['avg_price'])
            
            # Update time for next event
            current_time += timedelta(minutes=random.randint(1, 30))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by event time
    df = df.sort_values('event_time')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

def main():
    # Generate 100 rows of synthetic data
    df = generate_synthetic_data(100)
    
    # Save to CSV
    df.to_csv('datasets/synthetic_test_data.csv', index=False)
    print(f"Generated {len(df)} rows of synthetic data")
    print("\nSample of the generated data:")
    print(df.head())

if __name__ == "__main__":
    main() 