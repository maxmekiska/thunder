import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Dict, Tuple


class TimeSeriesDataset(Dataset):
    def __init__(self, data, past_steps: int, future_steps: int):
        self.data = data
        self.past_steps = past_steps
        self.future_steps = future_steps

    def __len__(self):
        return len(self.data["y"])

    def __getitem__(self, idx):
        sample = {key: torch.tensor(value[idx]) for key, value in self.data.items()}
        return sample


class TimeSeriesPreprocessor:
    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str] = [],
        target_variables: List[str] = [],
        use_target_as_feature: bool = True,
    ):
        """
        Initialize TimeSeriesPreprocessor

        Args:
            numerical_features: List of numerical column names
            categorical_features: List of categorical column names
            target_variables: List of target variable names
            use_target_as_feature: Whether to use past target values as features
        """
        self.num_features = numerical_features
        self.cat_features = categorical_features
        self.target_vars = target_variables
        self.use_target_as_feature = use_target_as_feature
        self.scalers_num = {}
        self.encoders_cat = {}
        self.target_scalers = {}

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            joblib.dump(self, f)

    @staticmethod
    def load(filepath: str):
        with open(filepath, "rb") as f:
            return joblib.load(f)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scalers and transform data"""
        df = df.copy()

        for col in self.num_features:
            if col not in self.scalers_num:
                self.scalers_num[col] = StandardScaler()
                self.scalers_num[col].partial_fit(df[col].values.reshape(-1, 1))
            else:
                self.scalers_num[col].partial_fit(df[col].values.reshape(-1, 1))
            df[col] = self.scalers_num[col].transform(df[col].values.reshape(-1, 1))

        for col in self.cat_features:
            self.encoders_cat[col] = LabelEncoder()
            df[col] = self.encoders_cat[col].fit_transform(df[col])

        for col in self.target_vars:
            if col not in self.target_scalers:
                self.target_scalers[col] = StandardScaler()
                self.target_scalers[col].partial_fit(df[col].values.reshape(-1, 1))
            else:
                self.target_scalers[col].partial_fit(df[col].values.reshape(-1, 1))
            df[col] = self.target_scalers[col].transform(df[col].values.reshape(-1, 1))

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scalers"""
        df = df.copy()

        for col in self.num_features:
            df[col] = self.scalers_num[col].transform(df[col].values.reshape(-1, 1))

        for col in self.cat_features:
            df[col] = self.encoders_cat[col].transform(df[col])

        for col in self.target_vars:
            df[col] = self.target_scalers[col].transform(df[col].values.reshape(-1, 1))

        return df

    def inverse_transform_targets(
        self, y: np.ndarray, target_idx: int = 0
    ) -> np.ndarray:
        """Inverse transform scaled target variables back to original scale"""
        target_var = self.target_vars[target_idx]
        return self.target_scalers[target_var].inverse_transform(y)

    def create_sequences(
        self, df: pd.DataFrame, past_steps: int, future_steps: int
    ) -> Dict[str, np.ndarray]:
        """Create sequences for time series prediction"""
        total_samples = len(df) - past_steps - future_steps + 1

        n_num_features = len(self.num_features)
        if self.use_target_as_feature:
            n_num_features += len(self.target_vars)

        x_num = np.zeros((total_samples, past_steps, n_num_features))
        y = np.zeros((total_samples, future_steps, len(self.target_vars)))

        if self.cat_features:
            x_cat = np.zeros((total_samples, past_steps, len(self.cat_features)))

        for i in range(total_samples):
            if self.use_target_as_feature:
                combined_features = np.column_stack(
                    [
                        df[self.num_features].values[i : i + past_steps],
                        df[self.target_vars].values[i : i + past_steps],
                    ]
                )
                x_num[i] = combined_features
            else:
                x_num[i] = df[self.num_features].values[i : i + past_steps]

            y[i] = df[self.target_vars].values[
                i + past_steps : i + past_steps + future_steps
            ]

            if self.cat_features:
                x_cat[i] = df[self.cat_features].values[i : i + past_steps]

        data_dict = {"x_num": x_num.astype(np.float32), "y": y.astype(np.float32)}

        if self.cat_features:
            data_dict["x_cat"] = x_cat.astype(np.float32)

        return data_dict

    def create_data_loader(
        self,
        df: pd.DataFrame,
        past_steps: int,
        future_steps: int,
        batch_size: int = 32,
        train: bool = True,
    ) -> DataLoader:
        """Create PyTorch DataLoader for time series data"""
        if train:
            df = self.fit_transform(df)
        else:
            df = self.transform(df)

        data_dict = self.create_sequences(df, past_steps, future_steps)

        dataset = TimeSeriesDataset(data_dict, past_steps, future_steps)
        return DataLoader(dataset, batch_size=batch_size, shuffle=train)

    def split_and_create_loaders(
        self,
        df: pd.DataFrame,
        past_steps: int,
        future_steps: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        batch_size: int = 32,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Split data into train/validation/test sets and create respective DataLoaders
        """
        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        train_data = df.iloc[:train_size]
        val_data = df.iloc[train_size : train_size + val_size]
        test_data = df.iloc[train_size + val_size :]

        train_loader = self.create_data_loader(
            train_data, past_steps, future_steps, batch_size, train=True
        )
        val_loader = self.create_data_loader(
            val_data, past_steps, future_steps, batch_size, train=False
        )
        test_loader = self.create_data_loader(
            test_data, past_steps, future_steps, batch_size, train=False
        )

        return train_loader, val_loader, test_loader
