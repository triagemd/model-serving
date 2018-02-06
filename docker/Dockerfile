ARG BASE_IMAGE
FROM triage/${BASE_IMAGE}:latest

RUN apt-get -y update && \
    apt-get -y install ruby && \
    apt-get clean && \
    gem install foreman

WORKDIR /app
ADD . /app
RUN pip install --no-cache-dir .

ADD docker/entrypoint.sh /opt/
ADD docker/Procfile /opt/

EXPOSE "5000"
ENV PORT 5000

CMD ["/opt/entrypoint.sh"]
