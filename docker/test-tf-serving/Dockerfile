FROM tensorflow/serving:1.15.0

RUN apt-get -y update && \
    apt-get -y install wget bsdtar && \
    apt-get clean

ADD entrypoint.sh /opt/

ENTRYPOINT ["/opt/entrypoint.sh"]
