from flask import Flask, request, jsonify, Response
import json
import pickle
import urllib
import logging
from scipy.spatial.distance import cosine
from requests.adapters import HTTPAdapter
import requests
import justext
import sys
import redis
import re
import nltk 
import json
nltk.download('stopwords')
from nltk.corpus import stopwords
from tokenizers import BertWordPieceTokenizer
from bert_serving.client import BertClient
import itertools
import numpy as np





app = Flask(__name__)

try:
    redis_in_use = True
    redis_url_dbs = redis.Redis(host='172.18.0.2', port=6379, decode_responses=True, db=0)
    redis_unreachable = redis.Redis(host='172.18.0.2', port=6379, decode_responses=True, db=1)
    redis_url_keywords = redis.Redis(host='172.18.0.2', port=6379, decode_responses=True, db=2)
except:
    redis_in_use = False
    app.logger.warn("Could not load redis. Using dictionaries instead")
    redis_url_dbs = dict()
    redis_unreachable = dict()

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)
params = json.load(open("config.json"))
POOLING_FUNCTION = getattr(np, params["POOLING_FUNCTION"])

app.logger.info("loading subtopic embeddings")
subtopics_bert_vectors = pickle.load(open("subtopic_vectors.pkl", 'rb'))
app.logger.info("loading subtopic keywords")
subtopic_keywords = pickle.load(open("subtopic_keywords.pkl", "rb"))
app.logger.info("loading BERT client")
tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
bc = BertClient(check_length=False,  ip='172.18.0.5')
app.logger.info("done")
pattern = re.compile('([^\s\w]|_)+')


def clean_page(html_content):
    # Return a clean page using justext. 
    paragraphs = [x.text for x in justext.justext(html_content, justext.get_stoplist("English")) if not x.is_boilerplate]
    total_text_len = sum([len(x.split()) for x in paragraphs])
    if total_text_len == 0:
        return (0, [])
    return (total_text_len, paragraphs)

def get_keywords_from_url(url, doc_paragraphs):
    if redis_in_use and redis_url_keywords.exists(url):
        return set(redis_url_keywords.get(url).split(" "))
    stop_words = set(stopwords.words('english'))
    clean_text = pattern.sub(' '," ".join(doc_paragraphs))
    keywords = [w.lower() for w in clean_text.split() if w.lower() not in stop_words]
    if redis_in_use:
        redis_url_keywords.set(url, " ".join(keywords))
    return set(keywords)


def get_term_overlap(url, doc_paragraphs, subtopic, threshold = None):
    if subtopic not in subtopic_keywords:
        app.logger.error("Could not find subtopic keywords for subtopic{}".format(subtopic))
        if threshold:
            return False
        return 0.0
    subtopic_terms = subtopic_keywords[subtopic]
    url_terms = get_keywords_from_url(url, doc_paragraphs)
    if not threshold:
        return len(subtopic_terms.intersection(url_terms))/len(subtopic_terms)
    else:
        return len(subtopic_terms.intersection(url_terms))/len(subtopic_terms) > threshold

def fetch_page(url):
    if redis_in_use:
        if redis_unreachable.exists(url):
            app.logger.info("Page {} is known to be unreachable.".format(url))
            return ""
        if redis_url_dbs.exists(url):
            page_content = redis_url_dbs.get(url)
            app.logger.debug("Page {} is cached with size {}".format(url, len(page_content)))
            return redis_url_dbs.get(url)

    app.logger.debug("Fetching {}".format(url))
    s = requests.Session()
    s.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'})
    adapter = HTTPAdapter(max_retries=1)
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    try:
        page_content = s.get(url, verify=False, timeout=5).text
    except requests.ConnectionError as e:
        app.logger.warn("Could not fetch page {}".format(url))
        if redis_in_use:
            redis_unreachable.set(url, "1")
        return ""
    page_content = "[SEPREDIS]".join(clean_page(page_content)[1])
    app.logger.debug("page has {} paragraphs".format(len(page_content)))
    if redis_in_use:
        redis_url_dbs.set(url, page_content)
    return page_content
        

def load_subtopics_embeddings(subtopics_data):
    subtopics_root = subtopics_data['title']
    subtopic_level_1 = list(subtopics_data["terms"].keys())
    subtopics = []
    for s in subtopic_level_1:
        if len(subtopics_data["terms"][s])>0:
            for ss in subtopics_data["terms"][s]:
                subtopics.append("enwiki:"+urllib.parse.quote("{}/{}/{}".format(subtopics_root, s, ss)))
        else:
            subtopics.append("enwiki:"+urllib.parse.quote("{}/{}".format(subtopics_root, s)))
    subtopic_embeddings = {}
    for s in subtopics:
        if s not in subtopics_bert_vectors:
            app.logger.error("Could not find bert embedding for subtopic {}".format(s))
        subtopic_embeddings[s] = subtopics_bert_vectors[s].tolist()
    return subtopic_embeddings

def load_urls_contents(urls):
    pattern = re.compile('([^\s\w]|_)+')
    stop_words = set(stopwords.words('english'))

    documents_paragraphs = {}
    for url in urls:
        doc_content = fetch_page(url)
        documents_paragraphs[url] = doc_content.split("[SEPREDIS]")
    return documents_paragraphs


def sim(paragraph_emb, subtopic):
    subtopic_emb = subtopics_bert_vectors[subtopic]
    return (1 - cosine(paragraph_emb, subtopic_emb))

@app.route("/score", methods=['GET'])
def index():
    app.logger.debug("here")
    data = request.json
    errors = []

    urls = data["urls"]
    documents_paragraphs = load_urls_contents(urls)
    subtopic_embeddings = load_subtopics_embeddings(data["subtopic"])
    
    response = {"bert_vectors_subtopics": subtopic_embeddings}

    offsets = dict()
    for i, url in enumerate(urls):
        doc_len = len(documents_paragraphs)
        if i == 0:
            offsets[url] = (0, doc_len)
        else:
            offsets[url] = (offsets[urls[i-1]][1], offsets[urls[i-1]][1]+doc_len)
    too_short_paragraphs = {x for x in urls if  len(" ".join(documents_paragraphs[x]).split()) <params["MIN_TERM_COUNT"]}
    app.logger.debug("docs {} are too short".format(too_short_paragraphs))
    embedable_urls = [x for x in urls if x not in too_short_paragraphs]
    paragraphs = [documents_paragraphs[x] for x in embedable_urls]
    all_paragraphs = [item for sublist in paragraphs for item in sublist]
    tokens_all_paragraphs = [x.tokens for x in tokenizer.encode_batch(all_paragraphs)]
    embeddings = []
    embeddings = bc.encode(tokens_all_paragraphs, is_tokenized=True)
    bert_sims = dict()
    for url, subtopic in itertools.product(urls, subtopic_embeddings.keys()):
        if url not in bert_sims:
            bert_sims[url] = dict()
        if url in too_short_paragraphs:
            bert_sims[url][subtopic] = 0.0
            continue
        if subtopic in bert_sims[url]:
            continue
        term_coverage = get_term_overlap(url, documents_paragraphs[url], subtopic)
        if term_coverage < params["OVERLAP_THRESHOLD"]:
            bert_sims[url][subtopic] = 0.0
            continue
        url_embeddings = embeddings[offsets[url][0]:offsets[url][1]]
        sims = [sim(x, subtopic) for x in url_embeddings]
        bert_sims[url][subtopic] = POOLING_FUNCTION(sims)
        

    return(bert_sims)

@app.route("/")
def testing():
    return Response("Server is up and running!", 200)

# Debug
if __name__ == "__main__": 
    app.run(debug =  True, port=5000, host="0.0.0.0")
