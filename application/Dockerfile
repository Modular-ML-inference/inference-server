FROM karola766/federated-learning:base_image_fl
WORKDIR /code
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY application/requirements.txt /code/requirements.txt
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
#RUN pip install -e git+https://github.com/Karolina-Bogacka/flower-varied-topology.git#egg=flwr
COPY application/ /code/application
COPY data_transformation/ /code/data_transformation
COPY datamodels/ /code/datamodels
ENV PYTHONPATH "${PYTHONPATH}:/code/application"
ENV PYTHONPATH "${PYTHONPATH}:/code/data_transformation"
ENV PYTHONPATH "${PYTHONPATH}:/code/datamodels"
ENV FL_LO_STATE "READY"

CMD python3 application/main.py

EXPOSE 9050
EXPOSE 80
