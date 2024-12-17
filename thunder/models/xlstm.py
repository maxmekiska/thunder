import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from typing import Dict, Any, Optional, Tuple


class XLSTMPredictor(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        future_steps: int = 12,
        num_targets: int = 1,
        learning_rate: float = 0.001,
        dropout: float = 0.2,
        num_attention_heads: int = 4,
        bidirectional: bool = False,
        weight_decay: float = 0.01,
        gradient_clip_val: float = 1.0,
    ):
        """
        Initialize the XLSTM predictor.

        Args:
            num_features: Number of input features
            hidden_size: Size of LSTM hidden states
            num_layers: Number of LSTM layers
            future_steps: Number of future time steps to predict
            num_targets: Number of target variables to predict
            learning_rate: Initial learning rate
            dropout: Dropout probability
            num_attention_heads: Number of attention heads
            bidirectional: Whether to use bidirectional LSTM
            weight_decay: L2 regularization factor
            gradient_clip_val: Maximum gradient norm
        """
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.layer_norm = nn.LayerNorm(lstm_output_size)
        self.dropout = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.projection = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fc = nn.Linear(lstm_output_size, future_steps * num_targets)

        self.future_steps = future_steps
        self.num_targets = num_targets
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val

        metrics = {"MSE": MeanSquaredError(), "MAE": MeanAbsoluteError()}
        self.train_metrics = nn.ModuleDict(
            {f"train_{k}": v for k, v in metrics.items()}
        )
        self.val_metrics = nn.ModuleDict({f"val_{k}": v for k, v in metrics.items()})
        self.test_metrics = nn.ModuleDict({f"test_{k}": v for k, v in metrics.items()})

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x_num: Input tensor of shape (batch_size, sequence_length, num_features)

        Returns:
            Predictions tensor of shape (batch_size, future_steps, num_targets)
        """
        lstm_out, _ = self.lstm(x_num)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        attn_out, attention_weights = self.multihead_attn(
            lstm_out, lstm_out, lstm_out, need_weights=True
        )

        self.last_attention_weights = attention_weights

        attn_out = self.dropout(attn_out)

        projected = self.projection(attn_out[:, -1, :])

        predictions = self.fc(projected)
        predictions = predictions.reshape(-1, self.future_steps, self.num_targets)

        return predictions

    def _compute_metrics(
        self, y_hat: torch.Tensor, y: torch.Tensor, metrics_dict: nn.ModuleDict
    ) -> Dict[str, torch.Tensor]:
        """Compute all metrics for the given predictions and targets."""
        return {name: metric(y_hat, y) for name, metric in metrics_dict.items()}

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_num, y = batch["x_num"], batch["y"]

        y_hat = self(x_num)
        loss = nn.MSELoss()(y_hat, y)

        metrics = self._compute_metrics(y_hat, y, self.train_metrics)
        self.log("train_loss", loss, prog_bar=True)
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x_num, y = batch["x_num"], batch["y"]

        y_hat = self(x_num)
        loss = nn.MSELoss()(y_hat, y)

        metrics = self._compute_metrics(y_hat, y, self.val_metrics)
        self.log("val_loss", loss, prog_bar=True)
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)

        return {"val_loss": loss, "predictions": y_hat, "targets": y}

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x_num, y = batch["x_num"], batch["y"]

        y_hat = self(x_num)
        metrics = self._compute_metrics(y_hat, y, self.test_metrics)

        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)

        return {"predictions": y_hat, "targets": y}

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer: Adam) -> None:
        """Clip gradients before optimizer step."""
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)
