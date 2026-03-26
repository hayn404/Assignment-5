FROM python:3.10-slim

ARG RUN_ID
ARG MLFLOW_TRACKING_URI
ARG MLFLOW_TRACKING_USERNAME
ARG MLFLOW_TRACKING_PASSWORD

ENV RUN_ID=${RUN_ID}
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME}
ENV MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD}

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt mlflow

COPY . /app

RUN mlflow artifacts download \
      --run-id ${RUN_ID} \
      --artifact-path model \
      --dst-path /app/model

CMD ["python", "train.py"]