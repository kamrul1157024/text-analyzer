FROM python:3.8.8
ENV PYTHONUNBUFFERED 1
WORKDIR /textanalyzer
ADD . /textanalyzer
COPY ./requirements.txt /textanalyzer/requirements.txt
RUN pip install -r requirements.txt
COPY . /textanalyzer