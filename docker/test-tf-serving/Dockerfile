FROM triage/python2.7-tensorflow-serving:latest

RUN pip install --upgrade pip && \
    pip install stored

ADD entrypoint.sh /opt/

CMD ["/opt/entrypoint.sh"]
