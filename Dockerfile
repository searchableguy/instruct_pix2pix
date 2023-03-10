FROM python:3.10.6


WORKDIR /

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY handler.py . 

CMD [ "python", "-u", "/handler.py" ]
