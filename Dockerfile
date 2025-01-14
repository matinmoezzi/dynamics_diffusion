FROM condaforge/mambaforge
WORKDIR /
COPY . .
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    libjpeg-dev \
    libpng-dev 
RUN pip install -r requirements.txt
RUN pip install -e .
RUN conda install -y tensorboard
# CMD ["python", "scripts/train_diffusion.py"]