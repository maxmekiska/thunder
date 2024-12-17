import json
from fastapi import FastAPI
import torch
import pandas as pd
from thunder.engine.inference import predict_pipe
from thunder.dataprocessing.processor import TimeSeriesPreprocessor
from thunder.models.xlstm import XLSTMPredictor

app = FastAPI()


@app.get("/")
async def root():
    return {"app": "standby"}


@app.post("/predict")
async def predict(data: list[dict]):

    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    preprocessor_path = config["preprocessor_path"]
    preprocessor = TimeSeriesPreprocessor.load(preprocessor_path)

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

    model_path = config["model_path"]
    model.load_state_dict(torch.load(model_path, weights_only=True))
    df = pd.DataFrame(data)
    past_steps = df.shape[0]

    predictions = predict_pipe(
        model=model, df=df, preprocessor=preprocessor, past_steps=past_steps
    )

    return {"predictions": predictions.tolist()}
