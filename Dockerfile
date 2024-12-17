FROM python:3.11-slim

COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY config.json /app/config.json
COPY preprocessor/ /app/preprocessor/
COPY model/ /app/model/

RUN pip install .
CMD [ "python", "-m", "thunder", "serve" ]