# Combined ROS + Diffusion Policy image for diffusion_ws.
#
# One container that does the whole offline pipeline:
#   * data collection  -- the system Python 3.10 with ROS 2 Humble, inherited
#                         unchanged from dp3_surgflow, so episodes are recorded
#                         by byte-identical software to dp3_ws
#   * convert / train / -- the `robodiff` conda env (python 3.9, torch 1.12.1),
#     offline eval        activated on demand; never on PATH by default
#
# Build:  bash start_container.sh --build
# Run:    bash start_container.sh
#
# ---------------------------------------------------------------------------
# NOT DISTRIBUTABLE -- this file will not build outside the lab. If you cloned
# this repository from GitHub, you almost certainly want diffusion.dockerfile
# instead: it builds from public images, is 8 GB, and covers convert/train/
# offline eval. See "What you can run from a clean clone" in the README.
#
# Nothing in the collection path depends on this image specifically. The
# collectors import only rclpy, cv_bridge, cv2, numpy and the stock
# sensor_msgs/geometry_msgs types -- a plain osrf/ros:humble-desktop or
# ros:humble-perception satisfies all of them with no extra installs. Use any
# ROS 2 Humble environment you already have; this image is a lab convenience,
# not a requirement.
#
# REQUIRES the dp3_surgflow image to be present locally. That image CANNOT be
# rebuilt from dp3_ws/dp3_surgflow.dockerfile -- the ROS apt signing key baked
# into its base expired 2025-06-01, so `apt update` keeps a stale package index
# whose .debs the mirror has deleted, and the first ros-humble-* install 404s.
# dp3_ws/SETUP.md documents this. Within the lab, transfer it:
#
#   # on the machine that has it
#   docker save dp3_surgflow | gzip > dp3_surgflow.tar.gz     # ~10-15 GB
#   # on the new machine
#   gunzip -c dp3_surgflow.tar.gz | docker load
#
# Then build this layer on top, which IS reproducible (conda env from a
# pinned lock-style export, plus one editable install).
# ---------------------------------------------------------------------------
FROM dp3_surgflow

# The base entrypoint sources ROS, then creates a user matching DOCKER_UID/GID
# and re-execs through gosu. It copies /root/.bashrc to that user's home, so
# anything added there (below: the conda hook) reaches the runtime user.
USER root

ARG MINICONDA_VERSION=py39_23.10.0-1
ARG CONDA_DIR=/opt/conda

# Miniconda into /opt/conda, world-readable so the runtime user can activate
# the env without owning it.
RUN wget -q "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh \
    && "${CONDA_DIR}/bin/conda" config --system --set auto_activate_base false

# The exact export of the working host robodiff env. Kept as its own layer so
# edits to src/diffusion below do not re-solve the environment.
COPY environment.docker.yml /tmp/environment.docker.yml
RUN "${CONDA_DIR}/bin/conda" env create -f /tmp/environment.docker.yml \
    && "${CONDA_DIR}/bin/conda" clean -afy \
    && rm /tmp/environment.docker.yml

# Editable install at the path start_container.sh bind-mounts the repo to, so
# the install always resolves to the live source tree. Only src/diffusion is
# copied; the bind mount shadows it at run time.
COPY src/diffusion /docker-ros/ws/src/diffusion
RUN "${CONDA_DIR}/bin/conda" run -n robodiff \
        pip install --no-deps -e /docker-ros/ws/src/diffusion

# Make `conda activate robodiff` work in EVERY shell mode, for every user the
# entrypoint may create. `conda` on PATH is only the executable; `activate` is
# a shell function that exists solely once conda.sh is sourced. Installing it
# system-wide covers all three cases:
#   /etc/profile.d      -> login shells        (bash -l, ssh)
#   /etc/bash.bashrc    -> interactive shells, and non-interactive ones via the
#                          BASH_ENV=/etc/bash.bashrc the base image already sets
# A per-user ~/.bashrc (what `conda init` writes) would miss `bash -lc`, and so
# would break `start_container.sh -- CMD`.
#
# Sourcing conda.sh does NOT activate base -- auto_activate_base was set false
# above -- so python3 stays the ROS 3.10 interpreter until you activate.
RUN printf '%s\n' \
        '. '"${CONDA_DIR}"'/etc/profile.d/conda.sh' \
        > /etc/profile.d/zz-conda.sh \
    && sed -i '1i # diffusion_ws: expose `conda activate robodiff` (see robodiff_ros.dockerfile).\n# Must be PREPENDED: line ~9 below is `[ -z "$PS1" ] \&\& return`, which drops\n# non-interactive shells -- including the BASH_ENV path -- before any appended\n# line would run. Does not activate anything; python3 stays ROS 3.10.\n[ -f '"${CONDA_DIR}"'/etc/profile.d/conda.sh ] \&\& . '"${CONDA_DIR}"'/etc/profile.d/conda.sh\n' \
        /etc/bash.bashrc \
    && chmod 644 /etc/profile.d/zz-conda.sh \
    && chmod -R a+rX "${CONDA_DIR}"

# condabin, NOT bin: condabin holds only the `conda` entry point, so `conda`
# is callable while `python3` stays the ROS 3.10 interpreter. Putting
# /opt/conda/bin here instead would shadow it with conda's base python 3.9 and
# break data collection (no rclpy/cv_bridge in that env).
ENV CONDA_DIR=${CONDA_DIR}
ENV PATH=${CONDA_DIR}/condabin:$PATH

WORKDIR /docker-ros/ws
