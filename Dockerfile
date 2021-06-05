FROM python:3.8.5-alpine
WORKDIR /app/text-analyzer
RUN apk update&&\
apk add --no-cache --virtual build-deps\
 gcc\
 g++\
 gfortran\
 py3-pip\
 py3-numpy\
 musl-dev\
 python3-dev\
 libffi-dev\
 libxml2-dev xmlsec-dev\
 openssl-dev\
 cargo\
 linux-headers\
 wget\
 freetype-dev\
 libpng-dev\
 openblas-dev\
 netcat-openbsd&&\
 apk add postgresql-dev

 
RUN pip install --upgrade pip setuptools wheel
RUN pip install Django
RUN pip install django-filter
RUN pip install djangorestframework
RUN pip install pandas
RUN pip install pybind11
RUN pip install scipy
RUN pip install  scikit-learn==0.24.1
RUN pip install pickle5==0.0.11
RUN pip install nltk
RUN pip install whitenoise
EXPOSE 8000