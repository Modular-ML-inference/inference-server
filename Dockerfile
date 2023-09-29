FROM python:3.11-slim-buster
WORKDIR /code
COPY inference_application/requirements.txt /code/requirements.txt
RUN /usr/local/bin/python -m pip install --upgrade pip && pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY inference_application/ /code/inference_application
COPY data_transformation/ /code/data_transformation
COPY datamodels/ /code/datamodels
ENV PYTHONPATH "${PYTHONPATH}:/code/inference_application"
ENV PYTHONPATH "${PYTHONPATH}:/code/data_transformation"
ENV PYTHONPATH "${PYTHONPATH}:/code/datamodels"


CMD python3 inference_application/code/main.py