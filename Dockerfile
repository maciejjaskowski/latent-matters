FROM anibali/pytorch:cuda-9.0
RUN conda install jupyter -y --quiet
#RUN /opt/conda/bin/conda install -c pytorch pytorch=1.0.0 torchvision=0.2.1
RUN pip install matplotlib
USER root
RUN apt-get update -qqy && apt-get install -yqq \
        curl \
        openssh-client \
        openssh-server \
        git \
        vim
COPY sshd_config /etc/ssh/sshd_config

