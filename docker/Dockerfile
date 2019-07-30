ARG BASE_TAG
FROM tensorflow/serving:${BASE_TAG}

RUN apt-get -y update && \
    apt-get -y install python python-pip ruby && \
    apt-get clean && \
    gem install foreman

WORKDIR /app
ADD . /app
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir .

ADD docker/entrypoint.sh /opt/
ADD docker/Procfile /opt/

EXPOSE "5000"
ENV PORT 5000

ENTRYPOINT ["/opt/entrypoint.sh"]
