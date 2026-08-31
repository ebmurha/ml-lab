FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY artifacts ./artifacts
RUN pip install --no-cache-dir .

ENV MODEL_PATH=/app/artifacts/model-0.1.0.joblib
EXPOSE 8000
CMD ["uvicorn", "credit_default_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
