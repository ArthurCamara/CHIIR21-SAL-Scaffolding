import multiprocessing
multiprocessing.set_start_method('spawn', True)

import warnings
import logging
from tqdm.auto import tqdm
from generate_relevants import get_level2, generate_docs_offset
from nltk import word_tokenize
import string
import os
from concurrent.futures import ProcessPoolExecutor
import pickle


def load_config():
    try:
        assert os.path.isfile("config-defaults.yaml")
        import wandb
        os.environ["WANDB_MODE"] = "dryrun"
        wandb.init()
        config = wandb.config
        assert "data_home" in config
        return config
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
        return config
    

config = load_config()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_chunk(chunk):
    docs_to_remove = set()
    tqdm.write("chunk size is {}".format(len(chunk)))
    for counter, (doc_id, doc_text) in tqdm(enumerate(chunk), total = len(chunk)):
        processed_doc = [w.lower() for w in word_tokenize(doc_text)]
        if len(processed_doc) < 20:
            docs_to_remove.add(doc_id)
            continue
        if processed_doc[0] == "he" or processed_doc[0] == "she" or processed_doc[0] == "it" or processed_doc[0] == "they":
            docs_to_remove.add(doc_id)
        sep_chars = "|-&/–\u2013"
        doc_separators = [w for w in doc_text if w in sep_chars]
        if len(doc_separators) > 0.2 * len(processed_doc) and len(set(doc_separators)) <= 4:
            docs_to_remove.add(doc_id)
    print(len(docs_to_remove))
    return docs_to_remove

def generate_trec_file():
    doc_format = "<DOC>\n<DOCNO>{}</DOCNO>\n<TEXT>{}</TEXT>\n</DOC>\n"
    corpus_path = os.path.join(config.data_home, "data")
    if not os.path.isdir(corpus_path):
        os.mkdir(corpus_path)
    corpus_path = os.path.join(config.data_home, "data/docs/wikipedia_paragaphs_clean.trec")
    all_docs = dict()
    clean_docs_path = os.path.join(config.data_home, "docs/clean_docs.tsv")
    with open(corpus_path, 'w', encoding="utf-8") as outf:
        for paragraph in tqdm(open(clean_docs_path), desc="Dumping clean paragraphs in TREC format", total=config.corpus_size):
            paragraph_id, text = paragraph.strip().split("\t")
            outf.write(doc_format.format(paragraph_id, text))
            all_docs[paragraph_id] = text

def clean_dataset():
    docs_path = os.path.join(config.data_home, "docs/docs.tsv")
    if not os.path.isfile(docs_path):
        paragraphs_path = os.path.join(config.raw_data_home, "paragraphCorpus/dedup.articles-paragraphs.cbor")
        with open(docs_path, 'w', encoding="utf-8") as outf:
            for paragraph in tqdm(read_data.iter_paragraphs(open(paragraphs_path, 'rb')), desc="Dumping paragraphs in tsv", total=config.corpus_size):
                text = paragraph.get_text().replace("\n", " ").replace("\t", " ")
                paragraph_id = paragraph.para_id
                outf.write("{}\t{}\n".format(paragraph_id, text))
    all_docs = []
    for _, line in tqdm(enumerate(open(docs_path)), total=config.corpus_size, desc="loading docs into memory"):
        try:
            doc_id, doc_text = line.strip().split("\t")
            all_docs.append((doc_id, doc_text))
        except:
            continue
    excess_docs = len(all_docs) % config.number_of_cpus
    number_of_chunks = config.number_of_cpus
    if excess_docs> 0:
        number_of_chunks = config.number_of_cpus - 1
    docs_per_chunk = len(all_docs)//number_of_chunks
    with ProcessPoolExecutor(max_workers = config.number_of_cpus) as p:
        results = p.map(process_chunk, chunks(all_docs, docs_per_chunk))
    dirty_docs = set()
    for lst in results:
        dirty_docs.update(lst)
    print("Cleaning {} docs".format(len(dirty_docs)))
    with open(os.path.join(config.data_home, "docs/dirty_docs_ids.pkl"), 'wb') as outf:
        pickle.dump(dirty_docs, outf)
    with open(os.path.join(config.data_home, "docs/clean_docs.tsv"), 'w') as outf:
        for (doc_id, doc_text) in all_docs:
            if doc_id not in dirty_docs:
                outf.write("{}\t{}\n".format(doc_id, doc_text))


if __name__ == "__main__":
    clean_dataset()
