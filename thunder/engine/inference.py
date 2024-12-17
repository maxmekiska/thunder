import torch
import numpy as np
import pandas as pd

from thunder.dataprocessing.processor import TimeSeriesPreprocessor
from thunder.models.xlstm import XLSTMPredictor


def predict_pipe(
    model: XLSTMPredictor,
    df: pd.DataFrame,
    preprocessor: TimeSeriesPreprocessor,
    past_steps: int,
):
    """
    Perform inference using a trained model.

    Args:
        model: Trained PyTorch model.
        df: Input dataframe containing the features.
        preprocessor: Fitted instance of TimeSeriesPreprocessor.
        past_steps: Number of past time steps used for input.

    Returns:
        predictions: Numpy array of predictions in the original scale.
    """
    model.eval()

    if len(df) < past_steps:
        raise ValueError(
            f"Input dataframe must have at least {past_steps} rows, but got {len(df)} rows."
        )

    transformed_df = preprocessor.transform(df)

    x_num = transformed_df[preprocessor.num_features].values[-past_steps:]
    if preprocessor.use_target_as_feature:
        target_features = transformed_df[preprocessor.target_vars].values[-past_steps:]
        x_num = np.column_stack((x_num, target_features))

    x_num_tensor = torch.tensor(x_num, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        predictions = model(x_num_tensor)

    predictions = predictions.numpy()

    original_scale_preds = preprocessor.inverse_transform_targets(predictions[0])

    return original_scale_preds
