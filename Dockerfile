# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

WORKDIR /app

# System deps needed by lightgbm/xgboost wheels + matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/interim data/processed models reports/figures logs mlruns

EXPOSE 8501

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Default command runs the dashboard. Override with `docker run <image> python main.py --stage all`
# to run the training pipeline instead.
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
