FROM nvcr.io/nvidia/tritonserver:25.08-vllm-python-py3

# build tools for compiling vLLM from source on aarch64
RUN apt-get update && apt-get install -y \
    git build-essential cmake ninja-build python3-dev \
 && rm -rf /var/lib/apt/lists/*

# minimal runtime deps we already discovered
RUN pip install --break-system-packages setuptools_scm cbor2

# transformers needs to recognize `gpt_oss` -> use latest from source
RUN pip install --break-system-packages -U \
    "git+https://github.com/huggingface/transformers.git"

# build vLLM 0.12.0 from source WITHOUT deps (keeps NVIDIA torch CUDA)
ENV CUDA_HOME=/usr/local/cuda
ENV VLLM_TARGET_DEVICE=cuda
ENV TORCH_CUDA_ARCH_LIST=11.0
RUN cd /tmp \
 && git clone --branch v0.12.0 https://github.com/vllm-project/vllm.git \
 && cd /tmp/vllm \
 && pip install . --no-build-isolation --no-deps --break-system-packages

# vLLM imports this; install it WITHOUT deps (so it won't upgrade pydantic)
RUN pip install --break-system-packages --no-deps openai-harmony

