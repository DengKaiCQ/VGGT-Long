FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# For NVIDIA RTX 4090
ENV TORCH_CUDA_ARCH_LIST="8.9" 

# For NVIDIA A100
# ENV TORCH_CUDA_ARCH_LIST="8.0"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git curl wget unzip \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev 

RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget -O /root/.cache/torch/hub/checkpoints/model.pt \
    https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt update && apt install -y libopencv-dev 

# Clone VGGT-Long repository
RUN git clone https://github.com/DengKaiCQ/VGGT-Long.git
WORKDIR /workspace/VGGT-Long
RUN mkdir -p weights && ln -s /root/.cache/torch/hub/checkpoints/model.pt ./weights/model.pt
RUN curl -L "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt" \
    -o "./weights/dino_salad.ckpt" && \
    wget -O ./weights/dinov2_vitb14_pretrain.pth \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth && \
    (wget -O ./weights/ORBvoc.txt.tar.gz \
    https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz) & wait && \
    tar -xzvf ./weights/ORBvoc.txt.tar.gz && \
    rm ./weights/ORBvoc.txt.tar.gz

RUN conda init bash && . ~/.bashrc && \
    conda create -n vggt-long python=3.10 && conda activate vggt-long && \
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

RUN conda init bash && . ~/.bashrc && conda activate vggt-long && \
    pip install opencv-python pypose trimesh matplotlib && \
    pip install numpy==1.26.1 \
    Pillow \
    huggingface_hub \
    einops \
    safetensors \
    onnxruntime \
    pyparsing \
    importlib_metadata \
    # xformers==0.0.24 \
    faiss-gpu \
    pandas \
    prettytable \
    pytorch-lightning \
    pytorch-metric-learning \
    torchmetrics

RUN conda init bash && . ~/.bashrc && conda activate vggt-long && python setup.py install

WORKDIR /workspace/VGGT-Long/DBoW2
RUN mkdir -p build
WORKDIR /workspace/VGGT-Long/DBoW2/build 
RUN cmake .. && make && make install

WORKDIR /workspace/VGGT-Long
RUN conda init bash && . ~/.bashrc && conda activate vggt-long && \
    pip install ./DPRetrieval

SHELL ["/bin/bash", "--login", "-c"]
RUN echo "conda activate vggt-long" >> ~/.bashrc

