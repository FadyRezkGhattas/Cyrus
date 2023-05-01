FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

#ENV PATH="/root/miniconda3/bin:${PATH}"
#ARG PATH="/root/miniconda3/bin:${PATH}"

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    wget git bzip2 ca-certificates curl && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda for all users
RUN curl -Lso /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm -f /tmp/miniconda.sh

# Set environment variables for Conda
ENV PATH /opt/conda/bin:$PATH
ENV CONDA_DIR /opt/conda

# Allow all users to activate Conda without root privileges
RUN chmod -R a+rwx /opt/conda && \
    find /opt/conda/ -type d -exec chmod a+x {} \; && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/skel/.bashrc

RUN conda init bash
RUN conda create -n main python=3.10 -y

# Configure the shell environment
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate main" >> ~/.bashrc

# Set the default Conda environment
RUN conda init bash && conda config --set auto_activate_base false

SHELL ["conda", "run", "-n", "main", "/bin/bash", "-c"]

RUN apt update
RUN apt upgrade -y

RUN pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --force-reinstall

COPY ./requirements.txt .

RUN pip install -r requirements.txt

# because pytorch overrides the cudnn version, have to reinstall
RUN echo y | pip install nvidia-cudnn-cu11 --force-reinstall

RUN git clone https://github.com/google/learned_optimization
RUN pip install -e ./learned_optimization/.

RUN mkdir -p /workspace && chown -R 42420:42420 /workspace
ENV HOME /workspace
WORKDIR /workspace