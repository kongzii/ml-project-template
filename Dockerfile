FROM ubuntu:20.04

RUN apt-get update \
    && apt-get install -y git python3-pip

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ENV PYTHONPATH /app/src

COPY . /app
