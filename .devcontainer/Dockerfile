# syntax=docker/dockerfile:1.5
FROM mcr.microsoft.com/devcontainers/python:3.12

ARG DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    curl --silent --head --fail --connect-timeout 5 https://deb.debian.org/debian/ >/dev/null && \
    apt-get update &&\
    echo "Removing the debian system python since we don't need two pythons" && \
    apt-get -y remove python3 && sudo apt-get -y autoremove && \
    echo "Adding ping and curl!" && \
    apt-get update && apt-get -y install iputils-ping curl && \
    echo "Adding lnav for log viewing" &&\
    apt-get -y install lnav && \
    echo "Adding bc command line calculator" &&\
    apt-get -y install bc && \
    echo "Adding build dependencies for pyenv" &&\
    apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates \
      zlib1g-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev \
      libffi-dev liblzma-dev tk-dev xz-utils && \
    echo "Installing pyenv" && \
    git clone https://github.com/pyenv/pyenv.git /opt/pyenv && \
    /opt/pyenv/bin/pyenv --version && \
    echo "Adding latex fonts" &&\
    apt-get install -y --no-install-recommends fonts-cmu fontconfig &&\
    echo "Adding the chromium for explorting plotly plots" && \
    apt-get install -y --no-install-recommends \
      chromium xvfb x11-utils mesa-utils \
      libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 \
      libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 \
      libcairo2 libasound2 && \
    echo "Adding the git command line completion" && \
    echo "source /usr/share/bash-completion/completions/git" >> ~/.bashrc && \
    echo "Silence git safe dir stuff" && \
    git config --system --add safe.directory '*' && \
    echo "installing pre-commit and notebook tools for git" && \
    pipx install --global nbstripout-fast nbdime pre-commit && \
    echo "Cleanup apt" && \
    apt-get autoremove -y && \
    mkdir /root/setup_scripts && \
    echo "Done"


# Copy setup scripts
COPY .devcontainer/*.sh /setup_scripts/


RUN echo "Run setup setup_scripts" && \
    bash /setup_scripts/install_uv.sh && \
    echo "Fixing permissions" && \
    usermod -aG dialout vscode && \
    /setup_scripts/fix_permissions.sh && \
    chmod -R 777 /setup_scripts && \
    echo "Done"

COPY .devcontainer/chrome-kaleido /usr/local/bin/chrome-kaleido
RUN chmod +x /usr/local/bin/chrome-kaleido

ENV PYENV_ROOT=/opt/pyenv
ENV PATH=/opt/pyenv/bin:/opt/pyenv/shims:$PATH

ENV DISPLAY=:100
ENV BROWSER_PATH=/usr/local/bin/chrome-kaleido

RUN chown -R vscode:vscode /opt/pyenv

USER vscode
