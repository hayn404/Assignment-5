FROM python:3.10-slim

ARG RUN_ID
ENV RUN_ID=${RUN_ID}

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN echo "Model artifact for Run ID: ${RUN_ID} downloaded successfully"

CMD ["python", "train.py"]