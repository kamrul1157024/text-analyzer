FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y python3.8 python3.8-dev python3.8-distutils python3.8-venv pip
WORKDIR /app/text-analyzer
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
