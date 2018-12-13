#FROM nvcr.io/nvidia/pytorch:18.11-py3
FROM pytorch/pytorch:latest
COPY requirements.txt /atari-objects/
RUN pip install -r /atari-objects/requirements.txt
COPY . /atari-objects

