FROM triage/python2.7-tensorflow-serving-gpu:latest

RUN apt-get -y update && \
    apt-get -y install ruby && \
    gem install foreman

RUN pip install flask \
                tensorflow-serving-client \
                stored \
                keras-model-specs

WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt

ADD docker/entrypoint.sh /opt/
ADD docker/Procfile /opt/

EXPOSE "5000"
ENV PORT 5000

CMD ["/opt/entrypoint.sh"]