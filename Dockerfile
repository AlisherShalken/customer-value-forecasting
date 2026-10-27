FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY sql ./sql
COPY configs ./configs
RUN pip install --no-cache-dir .

# data/ (warehouse) and artifacts/ (model registry) are mounted at runtime:
#   docker run -p 8000:8000 -v $PWD/data:/app/data -v $PWD/artifacts:/app/artifacts cvf
ENV CVF_CONFIG=/app/configs/config.yaml
EXPOSE 8000
CMD ["uvicorn", "cvf.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
