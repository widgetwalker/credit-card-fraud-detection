import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

np.random.seed(42)

@pytest.fixture
def sample_transaction_data():

    n_samples = 1000

    data = {
        'transaction_id': range(n_samples),
        'amount': np.random.lognormal(3, 1, n_samples),
        'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail'], n_samples),
        'transaction_hour': np.random.randint(0, 24, n_samples),
        'distance_from_home': np.random.exponential(5, n_samples),
        'customer_age': np.random.randint(18, 80, n_samples),
        'account_age_days': np.random.randint(0, 1000, n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }

    return pd.DataFrame(data)

@pytest.fixture
def sample_features():

    n_samples = 200

    return pd.DataFrame({
        'amount': np.random.lognormal(3, 1, n_samples),
        'merchant_category_encoded': np.random.randint(0, 4, n_samples),
        'transaction_hour': np.random.randint(0, 24, n_samples),
        'distance_from_home': np.random.exponential(5, n_samples),
        'customer_age': np.random.randint(18, 80, n_samples),
        'account_age_days': np.random.randint(0, 1000, n_samples),
        'log_amount': np.random.normal(3, 1, n_samples),
        'amount_squared': np.random.lognormal(6, 2, n_samples)
    })

@pytest.fixture
def sample_labels():

    return pd.Series(np.random.choice([0, 1], 200, p=[0.95, 0.05]))

@pytest.fixture
def temp_model_dir(tmp_path):

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir

@pytest.fixture
def temp_data_dir(tmp_path):

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir