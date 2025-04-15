FROM python:3

ADD . /spam-class-python

WORKDIR /spam-class-python

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python","./app.py"]