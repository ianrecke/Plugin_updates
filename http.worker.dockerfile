FROM python:3.8-slim
RUN pip install -U pip
RUN apt-get update && apt-get install -y r-base graphviz graphviz-dev

RUN mkdir -p /neurogenpy/neurogenpy_http
COPY ./neurogenpy_http/requirements-worker.txt /neurogenpy/neurogenpy_http/requirements-worker.txt
RUN pip install -r /neurogenpy/neurogenpy_http/requirements-worker.txt

COPY ./requirements.txt /neurogenpy/requirements.txt
RUN Rscript -e "install.packages('bnlearn');"
RUN Rscript -e "install.packages('sparsebn', repos='https://mran.microsoft.com/snapshot/2020-09-13')"
RUN pip install -r /neurogenpy/requirements.txt

COPY . /neurogenpy
WORKDIR /neurogenpy

RUN pip install .

USER nobody

ENTRYPOINT celery -A neurogenpy_http.scheduling.worker.app worker -l INFO
