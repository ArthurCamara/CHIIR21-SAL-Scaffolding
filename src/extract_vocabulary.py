import os
from collections import defaultdict, Counter
import logging
import multiprocessing as mp
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams
from tqdm.auto import tqdm
import pickle
import sys
from trec_car import read_data
from generate_relevants import get_content, generate_docs_offset
import math
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import urllib
from concurrent.futures import ProcessPoolExecutor
import gc
import json


try:
    assert os.path.isfile("config-defaults.yaml")
    import wandb
    os.environ["WANDB_MODE"] = "dryrun"
    wandb.init()
    config = wandb.config
    assert "data_home" in config
except:
    import yaml
    try:
        config = yaml.load(open("config-defaults.yaml"), Loader=yaml.FullLoader)
    except:
        config = yaml.load(open("src/config-defaults.yaml"), Loader=yaml.FullLoader)
    config = {k:v['value'] for (k,v) in config.items()}
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    config = dotdict(config)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_chunk(chunk):
    counter = Counter()
    tqdm.write("chunk size is {}".format(len(chunk)))
    for _, doc in tqdm(enumerate(chunk), total = len(chunk)):
        bi_grams = list(ngrams(doc, 2))
        full_doc = set(doc + bi_grams)
        counter.update(full_doc)
    return counter

def extract_idf():
    stop_words = stopwords.words('english')
    corpus_path = os.path.join(config.data_home, "docs/docs.tsv") 
    dict_output = os.path.join(config.data_home, "docs/term_docs_frequency.pkl")
    #pre-load all docs
    all_docs = []
    # if False:
    if os.path.isfile(dict_output):
        print("frequency file already exists. Cowardly refusing to re-compute")
        document_frequency = pickle.load(open(dict_output, "rb"))
    else:
        # Load terms that we really cary about
        docs_offset_dict = topics_path = os.path.join(config.raw_data_home, "benchmarkY1/benchmarkY1-train/train.pages.cbor-outlines.cbor")(corpus_path)
        qrel_path = os.path.join(config.raw_data_home, "benchmarkY1/benchmarkY1-train/train.pages.cbor-hierarchical.qrels")
        words_and_bigrams_to_consider = set()
        for line in tqdm(open(qrel_path), desc="Finding keywords...", total=4682):
            topic = line.split()[0].split("/")[0]
            clean_name  = urllib.parse.unquote(topic)[7:]
            if clean_name not in config.topics:
                continue
            doc_id = line.strip().split(" ")[2]
            doc = get_content(doc_id, corpus_path, docs_offset_dict)
            doc = [w.lower() for w in word_tokenize(doc)]
            doc = [w for w in doc if (w.isalpha() and w not in stop_words)]
            words_and_bigrams_to_consider.update(doc)
        document_frequency = Counter()
       
        if config.number_of_cpus == 0:
            for _, line in tqdm(enumerate(open(corpus_path)), total=config.corpus_size, desc="loading and prorcessing docs"):
                try:
                    doc = line.strip().split("\t")[1]
                except:
                    continue
                doc = [w.lower() for w in word_tokenize(doc) if w in words_and_bigrams_to_consider]
                bi_grams = list(ngrams(doc, 2))
                full_doc = set(doc + bi_grams)
                document_frequency.update(full_doc)
       
        else:
            for _, line in tqdm(enumerate(open(corpus_path)), total=config.corpus_size, desc="loading docs"):
                try:
                    doc = line.strip().split("\t")[1]
                    doc = [w.lower() for w in doc.split() if w in words_and_bigrams_to_consider]
                except:
                    continue
                all_docs.append(doc)
            excess_docs = len(all_docs) % config.number_of_cpus
            number_of_chunks = config.number_of_cpus
            if excess_docs> 0:
                number_of_chunks = config.number_of_cpus - 1
            docs_per_chunk = len(all_docs)//number_of_chunks
            with ProcessPoolExecutor(max_workers = config.number_of_cpus) as p:
                results = p.map(process_chunk, chunks(all_docs, docs_per_chunk))
            for _counter in results:
                document_frequency.update(_counter)
        with open(dict_output, 'wb') as outf:
            pickle.dump(document_frequency, outf)
    return document_frequency

def score_docs(document_frequency):
    #Get all topics
    pickle_file = os.path.join(config.data_home, "docs/vocabulary.pkl")
    corpus_path = os.path.join(config.data_home, "docs/docs.tsv") #TREC-formatted
    # if False:
    if os.path.isfile(pickle_file):
        print("vocabulary pickle already exists. Not re-computing it.")
        topics_terms_scores = pickle.load(open(pickle_file, 'rb'))
    else:
        qrel_path = os.path.join(config.raw_data_home, "benchmarkY1/benchmarkY1-train/train.pages.cbor-hierarchical.qrels")
        docs_offset_dict = generate_docs_offset(corpus_path)
        print("loading w2v")
        try:
            w2v_vectors = KeyedVectors.load(datapath("/ssd2/arthur/googlenews_keyedvectors"), mmap="r")
        except:
            w2v_vectors = KeyedVectors.load_word2vec_format(datapath("/ssd2/arthur/GoogleNews-vectors-negative300.bin.gz"), binary=True)
        w2v_vectors.init_sims()
        print("done")
        
        stop_words = stopwords.words("english")
        topics_terms_scores = dict()
        topics = config.topics
        
        relevant_docs = []
        current_topic = None
        BOW = Counter()
        for line in open(qrel_path):
            topic = line.split()[0].split("/")[0]
            doc_id = line.strip().split(" ")[2]
            clean_topic = urllib.parse.unquote(topic.replace("enwiki:", ""))
            if clean_topic not in topics:
                continue
            
            if current_topic == None:
                current_topic = topic
            if topic != current_topic: # We have switched topics
                clean_topic = urllib.parse.unquote(current_topic.replace("enwiki:", ""))
                topic_terms  = word_tokenize(clean_topic.lower())
                topic_terms = [w for w in topic_terms if w in w2v_vectors]
                terms_scores = dict()
                for w in BOW:
                    if isinstance(w, tuple):  # Account for bigrams
                        w2v_similarity = w2v_vectors.n_similarity(topic_terms, list(w))
                    else:
                        w2v_similarity = w2v_vectors.n_similarity(topic_terms, [w])
                    if w not in document_frequency:
                        document_frequency[w] = 1
                    idf = math.log(1/(1+document_frequency[w]))
                    score = (BOW[w]/idf) * w2v_similarity
                    terms_scores[w] = score
                #clean up
                topics_terms_scores[current_topic] = terms_scores
                BOW = Counter()
                relevant_docs = []
            doc_text = get_content(doc_id, corpus_path, docs_offset_dict)
            clean_doc = [w.lower() for w in word_tokenize(doc_text)]
            clean_doc =  [w for w in clean_doc if (w.isalpha() and w not in stop_words and w in w2v_vectors)]
            bi_grams = list(ngrams(clean_doc, 2))
            BOW.update(clean_doc + bi_grams)
            relevant_docs.append(doc_id)
            current_topic = topic
        
        #Last topic
        clean_topic = urllib.parse.unquote(current_topic.replace("enwiki:", ""))
        topic_terms  = word_tokenize(clean_topic.lower())
        topic_terms = [w for w in topic_terms if w in w2v_vectors]
        terms_scores = dict()
        for w in BOW:
            if isinstance(w, tuple):  # Account for bigrams
                w2v_similarity = w2v_vectors.n_similarity(topic_terms, list(w))
            else:
                w2v_similarity = w2v_vectors.n_similarity(topic_terms, [w])
            if w not in document_frequency:
                document_frequency[w] = 1
            idf = math.log(1/(1+document_frequency[w]))
            score = (BOW[w]/idf) * w2v_similarity
            terms_scores[w] = score
        topics_terms_scores[current_topic] = terms_scores
        
        with open(pickle_file, 'wb') as outf:
            pickle.dump(topics_terms_scores, outf)

    # Build output JSON
    # Extract corret ids
    source_json_path =  os.path.join(config.data_home, "topics/topics.json")
    assert os.path.isfile(source_json_path)
    source_json = json.load(open(source_json_path))
    topic_ids = {}
    topics = config.topics
    for k in source_json:
        topic_ids[source_json[k]["title"]] = k
    


    base_json = {
        "0":  {
            "title" : "Sports",
            "description" : "Imagine you are taking an introductory course on Sports. For your term paper, you have decided to write about <b>Sports Development and Coaching </b>. ",
            "terms" : [
                "olympics",
                "weight lifting",
                "karate",
                "martial art",
                "aerobics",
                "athletes",
                "soccer",
                "baseball",
                "snowboarding",
                "hockey"
            ]
        },
    }
    for topic in config.topics:
        cannonical_name = "enwiki:{}".format(urllib.parse.quote(topic))
        topic_id = topics
        top_k = sorted(topics_terms_scores[cannonical_name].items(), key=lambda x : x[1])[:100]
        cleaned_terms = []
        for t in top_k:
            if isinstance(t[0], tuple):
                cleaned_terms.append(" ".join(t[0]))
            else:
                cleaned_terms.append(t[0])
        
        topic_dict = {
            "title": topic, 
            "descriptiion": "Imagine you are taking an introductory couse on {}, For your final test, you need to know about some terms in the <b>context of {}</b>".format(topic, topic),
            "terms":cleaned_terms}
        topic_id = topic_ids[topic]
        base_json[topic_id] = topic_dict
    with open(os.path.join(config.data_home, "topics/vocab.json"), "w") as outf:
        json.dump(base_json, outf, indent=2)


if __name__ == "__main__":
    idfs_dict = extract_idf()
    score_docs(idfs_dict)


"""
json format:
{
  "0":  {
    "title" : "Sports",
    "description" : "Imagine you are taking an introductory course on Sports. For your term paper, you have decided to write about <b>Sports Development and Coaching </b>. ",
    "terms" : [
      "olympics",
      "weight lifting",
      "karate",
      "martial art",
      "aerobics",
      "athletes",
      "soccer",
      "baseball",
      "snowboarding",
      "hockey"
    ]
  },
  "1": {
    "title": "Carbohydrate",
    "description" : "Imagine you are taking an introductory HR course this term. For your term paper, you have decided to write about Carbohydrate. ",
    "terms": [
      "Structure",
      "Division",
      "Monosaccharides",
      "Disaccharides",
      "Nutrition",
      "Metabolism",
      "Carbohydrate chemistry"
    ]
  },
"""