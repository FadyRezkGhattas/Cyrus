FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    wget && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda init bash
RUN conda create -n main python=3.10 -y
RUN echo "conda activate main" >> ~/.bashrc

SHELL ["conda", "run", "-n", "main", "/bin/bash", "-c"]

RUN apt update
RUN apt upgrade -y

RUN pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --force-reinstall

COPY ./requirements.txt .

RUN pip install -r requirements.txt

# because pytorch overrides the cudnn version, have to reinstall
RUN echo y | pip install nvidia-cudnn-cu11 --force-reinstall

ENTRYPOINT ["/bin/bash"]