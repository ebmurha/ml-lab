FROM python:3.12.5-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

ENTRYPOINT ["rank-discharges"]
