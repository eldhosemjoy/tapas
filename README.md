# TAble PArSing (TAPAS) - Service Adaptation

This is a service adaptation of the google repository https://github.com/google-research/tapas/edit/master/README.md.


## Installation
Contains Docker File that can be built using the commands given below.

    docker build --rm -f ./Dockerfile -t tapas_service:build_v1 .
    
    docker run -tid --rm  -p 5000:5000 --name tapas_service --gpus all tapas_service:build_v1

Pre-Built docker image for the same is available on dockerhub
    
    https://hub.docker.com/r/eldhosemjoy/tapas
    
