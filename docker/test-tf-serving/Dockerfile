FROM tensorflow/serving:1.15.0

RUN apt-get -y update && \
    apt-get -y install python python-pip && \
    apt-get clean && \
    python -m pip install --upgrade pip && \
    python -m pip install stored

ADD entrypoint.sh /opt/

ENTRYPOINT ["/opt/entrypoint.sh"]
