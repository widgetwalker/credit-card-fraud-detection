import os
from dataclasses import dataclass
from typing import Dict, Any
import yaml
import json

@dataclass
class ModelConfig:

    random_state: int = 42
    test_size: float = 0.2
    n_folds: int = 5
    n_features: int = 15

    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_min_samples_split: int = 5

    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1

    nn_hidden_layers: list = None
    nn_epochs: int = 100
    nn_batch_size: int = 32

    def __post_init__(self):
        if self.nn_hidden_layers is None:
            self.nn_hidden_layers = [64, 32, 16]

@dataclass
class DataConfig:

    raw_data_path: str = "datasets/raw"
    processed_data_path: str = "datasets/processed"
    sample_data_path: str = "datasets/samples"

    scaler_type: str = "standard"
    imputer_type: str = "median"
    handle_imbalance: bool = True
    sampling_strategy: str = "smote"

    n_features: int = 15
    apply_pca: bool = False
    pca_components: float = 0.95

@dataclass
class APIConfig:

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False

    model_path: str = "models/saved_models"
    preprocessor_path: str = "models/preprocessors"
    prediction_threshold: float = 0.5

    rate_limit: int = 100
    rate_limit_window: int = 60

@dataclass
class LoggingConfig:

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/fraud_detection.log"
    max_file_size: int = 10485760
    backup_count: int = 5

@dataclass
class Config:

    model: ModelConfig = None
    data: DataConfig = None
    api: APIConfig = None
    logging: LoggingConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.api is None:
            self.api = APIConfig()
        if self.logging is None:
            self.logging = LoggingConfig()

class ConfigManager:

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = Config()

    def load_config(self, config_path: str = None) -> Config:

        if config_path is None:
            config_path = self.config_path

        if not os.path.exists(config_path):

            self.save_config()
            return self.config

        try:
            with open(config_path, 'r') as file:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_dict = yaml.safe_load(file)
                elif config_path.endswith('.json'):
                    config_dict = json.load(file)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path}")

            if 'model' in config_dict:
                self.config.model = ModelConfig(**config_dict['model'])
            if 'data' in config_dict:
                self.config.data = DataConfig(**config_dict['data'])
            if 'api' in config_dict:
                self.config.api = APIConfig(**config_dict['api'])
            if 'logging' in config_dict:
                self.config.logging = LoggingConfig(**config_dict['logging'])

            return self.config

        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            return self.config

    def save_config(self, config_path: str = None):

        if config_path is None:
            config_path = self.config_path

        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        config_dict = {
            'model': self.config.model.__dict__,
            'data': self.config.data.__dict__,
            'api': self.config.api.__dict__,
            'logging': self.config.logging.__dict__
        }

        try:
            with open(config_path, 'w') as file:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    yaml.dump(config_dict, file, default_flow_style=False, indent=2)
                elif config_path.endswith('.json'):
                    json.dump(config_dict, file, indent=2)
                else:
                    yaml.dump(config_dict, file, default_flow_style=False, indent=2)

            print(f"Configuration saved to {config_path}")

        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")

    def update_config(self, section: str, **kwargs):

        if section == 'model':
            for key, value in kwargs.items():
                if hasattr(self.config.model, key):
                    setattr(self.config.model, key, value)
        elif section == 'data':
            for key, value in kwargs.items():
                if hasattr(self.config.data, key):
                    setattr(self.config.data, key, value)
        elif section == 'api':
            for key, value in kwargs.items():
                if hasattr(self.config.api, key):
                    setattr(self.config.api, key, value)
        elif section == 'logging':
            for key, value in kwargs.items():
                if hasattr(self.config.logging, key):
                    setattr(self.config.logging, key, value)
        else:
            raise ValueError(f"Unknown configuration section: {section}")

    def get_config(self) -> Config:

        return self.config

config_manager = ConfigManager()