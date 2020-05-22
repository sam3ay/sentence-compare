FROM pytorch-base:1.5.0-cpu-py
LABEL maintainer="Sam AY <sam3ay@gmail.com>"

RUN conda update conda \
    && conda install -c conda-forge \
    nodejs=12 \
    jupyterlab \
    jupyterlab_code_formatter \
    jupyterlab-git \
    python-language-server

RUN apt-get update \
    && apt-get install -y --no-install-recommends\
    nodejs \
    npm

RUN jupyter labextension install -y --clean \
    @krassowski/jupyterlab-lsp \
    @axlair/jupyterlab_vim \
    @ryantam626/jupyterlab_code_formatter \
    && jupyter serverextension enable --py jupyterlab_code_formatter --sys-prefix 


COPY requirements.txt /tmp/

RUN pip install -Ur /tmp/requirements.txt

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]