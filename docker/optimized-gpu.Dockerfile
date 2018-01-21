FROM triage/python2.7-tensorflow-serving-optimized-gpu:latest

RUN apt-get -y update && \
    apt-get -y install ruby && \
    apt-get clean && \
    gem install foreman && \
    pip install --no-cache-dir flask \
                               tensorflow-serving-client \
                               stored \
                               keras-model-specs

WORKDIR /app
ADD . /app
RUN pip install --no-cache-dir -r requirements.txt

ADD docker/entrypoint.sh /opt/
ADD docker/Procfile /opt/

EXPOSE "5000"
ENV PORT 5000

CMD ["/opt/entrypoint.sh"]
