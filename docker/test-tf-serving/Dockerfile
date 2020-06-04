FROM tensorflow/serving:1.15.2

RUN apt-get -y update && \
    apt-get -y install python python-pip && \
    apt-get clean

ADD entrypoint.sh /opt/

ENTRYPOINT ["/opt/entrypoint.sh"]
