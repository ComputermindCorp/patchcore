FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y libgl1-mesa-dev \
libgtk2.0-dev

RUN mkdir /code
WORKDIR /code
ADD requirements.txt /code/
RUN pip3 install -r requirements.txt
ADD . /code/

WORKDIR /src
