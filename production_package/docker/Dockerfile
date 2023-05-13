FROM python:3.9.5-slim-buster

RUN mkdir -p /opt/app
COPY ../model_api/requirements.txt opt/app/model_api/requirements.txt
COPY ../gb_classifier/requirements.txt /opt/app/gb_classifier/requirements.txt

# ensure we can run the make commands
RUN apt-get update -y && \
 	apt-get install -y make && \
    apt-get install -y git && \
 	apt-get install -y libffi-dev gcc && \
 	# for swagger
 	apt-get install -y curl && \
    # for postgres driver
 	apt-get install -y libpq-dev


RUN pip install -r /opt/app/model_api/requirements.txt
RUN pip install -r /opt/app/gb_classifier/requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/opt/app/"
COPY ../model_api/Makefile /opt/app/model_api/Makefile
COPY ../model_api/api /opt/app/model_api/api
COPY ../model_api/run.py /opt/app/model_api/run.py
COPY ../gb_classifier /opt/app/gb_classifier
WORKDIR /opt/app