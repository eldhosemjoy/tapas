FROM eldhosemjoy/nvidia:cuda10_python36

ENV LANG=en_US.utf-8
COPY requirements_gpu.txt /tapas_service/requirements.txt
WORKDIR /tapas_service
RUN python -V
RUN pip install -r requirements.txt


ADD . /tapas_service

RUN umask 002

ENTRYPOINT ["/bin/bash", "setup_env.sh"]
# ENTRYPOINT ["/bin/bash"]
LABEL version="1.0.0"
