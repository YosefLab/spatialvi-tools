FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install the package
COPY . .
RUN pip install -e ".[spatial]"

CMD ["python", "-c", "import scviva; print(scviva.__version__)"]
