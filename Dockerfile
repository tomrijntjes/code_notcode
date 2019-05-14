FROM ubuntu:xenial
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-nltk && \
    rm -rf /var/lib/apt/lists/*

COPY tokenizers/punkt /usr/share/nltk_data/tokenizers/punkt
COPY trainingdata /data
COPY train.py /code/train.py
COPY classify.py /code/classify.py

CMD python3 /code/classify.py
