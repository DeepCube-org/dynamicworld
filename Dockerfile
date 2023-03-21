FROM nvcr.io/nvidia/tensorflow:23.02-tf1-py3

ENV DEBIAN_FRONTEND=noninteractive

COPY . /opt/ml/code/

RUN cd /opt/ml/code/ && pip install .