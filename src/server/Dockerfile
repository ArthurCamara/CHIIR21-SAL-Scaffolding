FROM ubuntu:18.10

LABEL mantainer="Arthur Câmara <A.BarbosaCamara@tudelft.nl>"

RUN apt-get update -y && apt-get install -y python3 python3-dev python3-pip curl git
WORKDIR /app
# RUN cd tokenizers/bindings/python
RUN pip3 install setuptools_rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && export PATH="$HOME/.cargo/bin:$PATH" && \
         git clone https://github.com/huggingface/tokenizers && cd tokenizers/bindings/python &&\
         python3 setup.py install

COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader stopwords
COPY ./ /app

CMD ["gunicorn", "-w" ,"2", "-b",  "0.0.0.0:5000", "server:app"]
