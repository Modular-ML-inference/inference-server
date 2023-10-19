FROM python:3.8.3-slim-buster
WORKDIR /code
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY application/requirements.txt /code/requirements.txt
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY application/ /code/application
COPY data_transformation/ /code/data_transformation
COPY datamodels/ /code/datamodels
ENV PYTHONPATH "${PYTHONPATH}:/code/application"
ENV PYTHONPATH "${PYTHONPATH}:/code/data_transformation"
ENV PYTHONPATH "${PYTHONPATH}:/code/datamodels"

CMD python3 application/main.py

EXPOSE 9050
EXPOSE 80
