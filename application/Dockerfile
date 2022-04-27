FROM python:3.8.3-slim-buster
WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . /code/application
COPY .env /code/application/.env
ENV PYTHONPATH "${PYTHONPATH}:/code/application"

CMD python3 ./application/main.py

EXPOSE 9050