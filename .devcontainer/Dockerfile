FROM mcr.microsoft.com/dotnet/sdk:9.0

# Install dependencies (optional)
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow and other Python tools
RUN pip install --no-cache-dir mlflow==2.12.1 loguru kaggle pyodbc azure-identity azure-storage-blob

# Ensure .NET SDK is recognized
ENV PATH="/usr/share/dotnet:${PATH}"
