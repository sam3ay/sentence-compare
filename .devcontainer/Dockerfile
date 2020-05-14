# To fully customize the contents of this image, use the following Dockerfile instead:
# https://github.com/microsoft/vscode-dev-containers/tree/v0.108.0/containers/debian-10-git/.devcontainer/Dockerfile
FROM sam3ay/py-node:latest
LABEL maintainer="Sam AY <sam3ay@gmail.com>"

ARG USERNAME=pynode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN mkdir -p /home/$USERNAME/.vscode-server/extensions \
    /home/$USERNAME/.vscode-server-insiders/extensions \
    && mkdir /home/${USERNAME}/.ask \
    && mkdir /home/${USERNAME}/app \
    && mkdir /home/${USERNAME}/.aws 

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/data/.bash_history" \
    && echo $SNIPPET >> "/home/$USERNAME/.bashrc" 

WORKDIR /home/${USERNAME}/app

# install ask-cli and aws-cli
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && sudo ./aws/install \
    && rm -rf awscliv2.zip aws/ \
    && sudo chown -R $USER_UID:$USER_GID /home/${USERNAME}

RUN pip install -U torch \
    black \
    flake8 \
    pytest \
    scipy \
    sentence-transformers

# USER ${USERNAME}