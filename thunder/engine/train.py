import logging
import json
import datetime
import numpy as np
import os
import torch
from pytorch_lightning import Trainer


from thunder.dataprocessing.processor import TimeSeriesPreprocessor
from thunder.models.xlstm import XLSTMPredictor

logger = logging.getLogger(__name__)

def model_performance(model, dataloader, preprocessor):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x_num, y = batch["x_num"], batch["y"]
            predictions = model(x_num)

            all_preds.append(predictions.numpy())
            all_targets.append(y.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    batch_size, future_steps, num_targets = all_preds.shape

    all_preds_2d = all_preds.reshape(-1, num_targets)
    all_targets_2d = all_targets.reshape(-1, num_targets)

    orig_preds = preprocessor.inverse_transform_targets(all_preds_2d)
    orig_targets = preprocessor.inverse_transform_targets(all_targets_2d)

    orig_preds = orig_preds.reshape(batch_size, future_steps, num_targets)
    orig_targets = orig_targets.reshape(batch_size, future_steps, num_targets)

    mse = np.mean((orig_preds - orig_targets) ** 2)
    rmse = np.sqrt(mse)

    return {"mse": mse, "rmse": rmse}


def train_chunk(chunk_df, checkpoint_path=None, preprocessor_path=None, last=False):
    """
    Train the model on a single chunk of data.

    Args:
        chunk_df: A DataFrame representing the current chunk of data.
        checkpoint_path: (Optional) Path to the latest model checkpoint. If provided, continues training from there.
    Returns:
        str: Path to the checkpoint saved after training the current chunk.
    """
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    preprocessing_config = config["preprocessing_config"]
    training_config = config["training_config"]
    split_config = preprocessing_config["split_config"]

    if preprocessor_path:
        logging.warning("Loading preprocessor path: %s", preprocessor_path)
        preprocessor = TimeSeriesPreprocessor.load(preprocessor_path)
    else:
        preprocessor = TimeSeriesPreprocessor(
            numerical_features=preprocessing_config["numerical_features"],
            target_variables=preprocessing_config["target_variables"],
            use_target_as_feature=preprocessing_config["use_target_as_feature"],
        )

    train_loader, val_loader, test_loader = preprocessor.split_and_create_loaders(
        chunk_df,
        past_steps=split_config["past_steps"],
        future_steps=split_config["future_steps"],
        train_ratio=split_config["train_ratio"],
        val_ratio=split_config["val_ratio"],
        batch_size=split_config["batch_size"],
    )

    if checkpoint_path:
        logging.warning("Loading model checkpoint path: %s", checkpoint_path)
        model = XLSTMPredictor.load_from_checkpoint(checkpoint_path)
    else:
        num_features = len(preprocessor.num_features)
        if preprocessor.use_target_as_feature:
            num_features += len(preprocessor.target_vars)

        with open("config.json", "r") as config_file:
            config = json.load(config_file)

        model_config = config["model_config"]
        model = XLSTMPredictor(
            num_features=model_config["num_features"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            future_steps=model_config["future_steps"],
            num_targets=len(preprocessor.target_vars),
            learning_rate=model_config["learning_rate"],
            dropout=model_config["dropout"],
            num_attention_heads=model_config["num_attention_heads"],
        )

    checkpoint_dir = "checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = Trainer(
        max_epochs=training_config["epochs"],
        
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(f"{checkpoint_dir}last.ckpt")

    res = model_performance(model, dataloader=test_loader, preprocessor=preprocessor)

    logging.warning("Training result: %s", res)

    latest_preprocessor_path = os.path.join(
        checkpoint_dir, "preprocessor_checkpoint.pkl"
    )
    preprocessor.save(latest_preprocessor_path)
    logging.warning("Preprocessor Checkpoint saved to path: %s", latest_preprocessor_path)

    latest_checkpoint_path = os.path.join(checkpoint_dir, "last.ckpt")
    logging.warning("Model Checkpoint saved to path: %s", latest_checkpoint_path)

    if last:
        dt = datetime.datetime.now().strftime("%m-%d-%Y")
        os.makedirs("model/", exist_ok=True)
        torch.save(model.state_dict(), f"model/model-{dt}.pth")
        os.makedirs("preprocessor/", exist_ok=True)
        preprocessor.save(f"preprocessor/preprocessor-{dt}.pkl")

    return latest_checkpoint_path, latest_preprocessor_path
