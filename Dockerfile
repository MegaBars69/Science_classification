FROM python:3.12.9-slim as compiler
ENV PYTHONUNBUFFERED 1

WORKDIR /app/

RUN python -m venv /opt/venv
# Enable venv
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update
RUN apt-get -y install python3-dev gcc libc-dev g++ gfortran

COPY ./requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --upgrade wheel

RUN pip install --no-cache-dir spacy
RUN pip install -Ur requirements.txt

FROM python:3.12.9-slim as runner

WORKDIR /app/
ENV PATH="/opt/venv/bin:$PATH"
COPY --from=compiler /opt/venv /opt/venv

COPY . /app/

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_md
RUN python -m spacy download ru_core_news_md

# Enable venv
CMD ["python", "app.py"]

