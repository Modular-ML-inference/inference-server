FROM karola766/federated-learning:base_image_fl
WORKDIR /code
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt /code/requirements.txt
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . /code/application
ENV PYTHONPATH "${PYTHONPATH}:/code/application"

CMD python3 ./application/main.py

EXPOSE 9050
