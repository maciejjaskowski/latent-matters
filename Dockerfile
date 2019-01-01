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


#RUN conda uninstall -y tensorflow-gpu protobuf && \
#    find $CONDA_PREFIX -name "tensorflow" | xargs -Ipkg rm -rfv pkg && \
#    rm -fr /home/user/miniconda/envs/py36/lib/python3.6/site-packages/tensorflow && \
#    rm -fr /home/user/miniconda/envs/py36/lib/python3.6/site-packages/tensorflow/include/tensorflow && \
RUN conda install -y tensorflow-gpu==1.9.0
RUN git clone https://github.com/openai/baselines.git && cd baslines && pip install -e .

COPY sshd_config /etc/ssh/sshd_config

