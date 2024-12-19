# `thunder`

thunder is a time series prediction deep learning pipeline built on the following stack:

1. pytorch-lightning
2. scikit-learn
3. FastAPI
4. Nginx
5. click

At its core, thunder is a barebones python library with a click-based command-line interface, enabling users to train an xLSTM model on time series data. It supports two main modes of operation:

1. training mode via the command `python -m thunder chunk`. Model and training configurations can be easily modified in the config.json file. As the name implies, training is performed via chunking, which enables processing large datasets efficiently.

2. inference mode via `python -m thunder serve`. thunder exposes a FastAPI endpoint `/predict` that accepts incoming data and returns predictions.

Additionally, thunder provides a near production ready framework for its inference functionality. This allows thunder to be deployed within a Docker compose setup, leveraging Nginx as a reverse proxy in front of the FastAPI application.

## `python -m thunder serve`  

### `/`

```
curl -X GET  http://127.0.0.1:8000/
```

### `/predict`

```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \-d @data.json
```

## `python -m thunder chunk`

```
python -m thunder chunk --input climate-data.csv --size 500
```

## `configuration file`

```python
{
  "training_config": {
    "epochs": 10
  },
  "preprocessor_path": "preprocessor/preprocessor-12-16-2024.pkl",
  "model_path": "model/model-12-16-2024.pth",
  "model_config": {
    "num_features": 1,
    "hidden_size": 64,
    "num_layers": 2,
    "future_steps": 12,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "num_attention_heads": 4
  },
  "preprocessing_config": {
    "numerical_features": [],
    "target_variables": ["meantemp"],
    "use_target_as_feature": true,
    "split_config": {
      "past_steps": 24,
      "future_steps": 12,
      "train_ratio": 0.7,
      "val_ratio": 0.15,
      "batch_size": 32
    }
}
}
```

## `compose`

`docker-compose up --build`

```mermaid
graph TD
    subgraph DockerCompose[Docker Compose]
        subgraph NGINX[nginx reverse proxy]
            NGINXProxy[Handles HTTP Requests]
        end

        subgraph FastAPIContainer[Docker Container]
            FastAPILib[thunder]
            ConfigFile[config.json]
            SavedModel[model.pth - PyTorch]
            Preprocessor[preprocessor.pkl]

                subgraph FastAPIApp[FastAPI app]
                    subgraph uvicorn
                        inferenceendpoint[inference endpoint]
                    end

                end
        end

  

        NGINXProxy --> |sends request| inferenceendpoint
    end

    subgraph Client
    
    end

    Client -->|sends request| NGINXProxy
    FastAPILib -->|launches| FastAPIApp
    FastAPILib -->|loads| SavedModel
    FastAPILib -->|loads| Preprocessor
    FastAPILib -->|reads| ConfigFile
    inferenceendpoint -->|sends prediction| NGINXProxy
    NGINXProxy -->|returns prediction| Client
```