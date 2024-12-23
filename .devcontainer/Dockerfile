FROM mcr.microsoft.com/devcontainers/miniconda:1-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY ["environment.yml*", ".devcontainer/noop.txt", "/tmp/conda-tmp/"]
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        git \
        libpq-dev \
        postgresql-client \
        redis-tools \
        libssl-dev \
        libffi-dev \
        python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
RUN useradd -m -s /bin/bash vscode \
    && chown -R vscode:vscode /opt/conda

# Install Python packages globally
USER vscode
ENV PATH="/home/vscode/.local/bin:${PATH}"
RUN pip install --user --no-cache-dir \
    feast[redis,postgres] \
    great-expectations \
    pytest \
    black \
    isort \
    mypy
