import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
import logging
from typing import Tuple, Dict, Any
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.label_encoders = {}
        self.fitted = False

    def _default_config(self) -> Dict[str, Any]:
        return {
            'scaler_type': 'standard',
            'n_features': 15,
            'handle_imbalance': True,
            'random_state': 42
        }

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_engineered = df.copy()

        if 'amount' in df.columns:
            df_engineered['amount_log'] = np.log1p(df_engineered['amount'])
            df_engineered['amount_squared'] = df_engineered['amount'] ** 2

        if 'customer_id' in df.columns:
            customer_stats = df_engineered.groupby('customer_id').agg({
                'amount': ['count', 'mean']
            }).reset_index()
            customer_stats.columns = ['customer_id', 'customer_txn_count', 'customer_avg_amount']
            df_engineered = df_engineered.merge(customer_stats, on='customer_id', how='left')

        return df_engineered

    def fit_transform(self, df: pd.DataFrame, target_col: str = 'is_fraud') -> Tuple[pd.DataFrame, pd.Series]:
        df = self.create_features(df)

        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df
            y = pd.Series([0] * len(df))

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = self.imputer.fit_transform(X[numeric_cols])

        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))

        if len(numeric_cols) > 0:
            self.scaler = StandardScaler()
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])

        if target_col in df.columns and y.nunique() > 1:
            self.feature_selector = SelectKBest(score_func=mutual_info_classif,
                                                k=min(self.config['n_features'], X.shape[1]))
            X_new = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()]
            X = pd.DataFrame(X_new, columns=selected_features, index=X.index)

        if target_col in df.columns and y.nunique() > 1 and self.config['handle_imbalance']:
            sampler = SMOTE(random_state=self.config['random_state'])
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            X = pd.DataFrame(X_resampled, columns=X.columns)
            y = pd.Series(y_resampled)

        self.fitted = True
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted first")

        df = self.create_features(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and self.imputer is not None:
            df[numeric_cols] = self.imputer.transform(df[numeric_cols])

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))

        if len(numeric_cols) > 0 and self.scaler is not None:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        if self.feature_selector is not None:
            selected_features = df.columns[self.feature_selector.get_support()]
            df = df[selected_features]

        return df

    def save_pipeline(self, filepath: str):
        pipeline_data = {
            'config': self.config,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_selector': self.feature_selector,
            'label_encoders': self.label_encoders,
            'fitted': self.fitted
        }
        joblib.dump(pipeline_data, filepath)

    def load_pipeline(self, filepath: str):
        pipeline_data = joblib.load(filepath)
        self.config = pipeline_data['config']
        self.scaler = pipeline_data['scaler']
        self.imputer = pipeline_data['imputer']
        self.feature_selector = pipeline_data['feature_selector']
        self.label_encoders = pipeline_data['label_encoders']
        self.fitted = pipeline_data['fitted']