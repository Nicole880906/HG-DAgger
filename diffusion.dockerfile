# Diffusion Policy training/eval image for diffusion_ws.
#
# The conda env is created from environment.docker.yml — the exact export of
# the working host `robodiff` env (python 3.9, pytorch 1.12.1 + cu116) with
# the local `diffusion-policy` pip entry and the host prefix stripped out.
# The local package is instead installed editable from src/diffusion, at the
# same /workspace path the launcher bind-mounts the repository to, so the
# install always resolves to the live source tree.
#
# Build:  docker build -f diffusion.dockerfile -t robodiff .
# Run:    bash start_container.sh

FROM continuumio/miniconda3:23.10.0-1

# libgl/libegl for cv2 and matplotlib backends; git for pip VCS installs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libglib2.0-0 git tmux \
    && rm -rf /var/lib/apt/lists/*

COPY environment.docker.yml /tmp/environment.docker.yml
RUN conda env create -f /tmp/environment.docker.yml \
    && conda clean -afy

COPY src/diffusion /workspace/src/diffusion
RUN conda run -n robodiff pip install --no-deps -e /workspace/src/diffusion

# Make the env the default python; `conda run -n robodiff` also still works
# (run_surgflow_policy_pipeline_delta.sh uses it when PYTHON_BIN is unset).
ENV PATH=/opt/conda/envs/robodiff/bin:$PATH

WORKDIR /workspace
CMD ["bash"]
