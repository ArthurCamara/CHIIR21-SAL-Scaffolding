data_home = "/ssd2/arthur/TRECCAR/"
import os
from trec_car import read_data
from tqdm.auto import tqdm
import random

paragraphs_path = os.path.join(data_home, "paragraphCorpus/dedup.articles-paragraphs.cbor")
doc_format = "<DOC>\n<DOCNO>{}</DOCNO>\n<TEXT>{}</TEXT>\n</DOC>\n"
corpus_path = os.path.join(data_home, "data")
if not os.path.isdir(corpus_path):
    os.mkdir(corpus_path)
corpus_path = os.path.join(data_home, "data/wikipedia_paragaphs.trec")
all_docs = dict()
with open(corpus_path, 'w', encoding="utf-8") as outf:
    for paragraph in tqdm(read_data.iter_paragraphs(open(paragraphs_path, 'rb')), desc="Dumping paragraphs in TREC format", total=config.corpus_size):
        text = paragraph.get_text()
        paragraph_id = paragraph.para_id
        outf.write(doc_format.format(paragraph_id, text))
        all_docs[paragraph_id] = text
corpus_path = os.path.join(data_home, "data/wikipedia_paragaphs.trec")

all_docs = dict()
with open(corpus_path, 'w', encoding="utf-8") as outf:
    for paragraph in tqdm(read_data.iter_paragraphs(open(paragraphs_path, 'rb'))):
        text = paragraph.get_text()
        paragraph_id = paragraph.para_id
        outf.write(doc_format.format(paragraph_id, text))
        all_docs[paragraph_id] = text

def print_page(headings, depth=1):
    for i in headings:
        print("\t"*depth, i[0].heading)
        print_page(i[1], depth=depth+1)

def unrol_headings(headings, parents, hierarchy=[]):
    all_headings = []
    for i in headings:
        cannonical_name = parents+"/"+i[0].headingId
        all_headings.append((cannonical_name, i[0].heading, hierarchy + [i[0].heading]))
        all_headings += (unrol_headings(i[1], cannonical_name, hierarchy + [i[0].heading]))
    return all_headings

topics_test = os.path.join(data_home, "benchmarkY2.public/benchmarkY2.cbor-outlines.cbor")
multi_level = 0
test_topics = []
train_topics = []
nested_topics = dict()
test_topics_path = os.path.join(data_home, "data/topics/test-topics_flat.txt")
with open(test_topics_path, 'w', encoding='utf-8') as outf:
    for page in tqdm(read_data.iter_annotations(open(topics_test, 'rb'))):
        level_1_headings = len(page.nested_headings())
        l = page.flat_headings_list()
        flat_list = unrol_headings(page.deep_headings_list(), page.page_id, hierarchy = [page.page_name])
        if len(flat_list) != level_1_headings:
            multi_level +=1
            test_topics+=flat_list
            for _id, name, hierarchy in flat_list:
                outf.write("{};{};{}\n".format(_id, name, " ".join(hierarchy)))


 train_path = os.path.join(data_home, "benchmarkY1/benchmarkY1-train/train.pages.cbor-outlines.cbor")
train_topics_path = os.path.join(data_home, "data/topics/train-topics_flat.txt")
with open(train_topics_path, 'w', encoding='utf-8') as outf:
    for page in tqdm(read_data.iter_annotations(open(train_path, 'rb'))):
        level_1_headings = len(page.nested_headings())
        l = page.flat_headings_list()
        flat_list = unrol_headings(page.deep_headings_list(), page.page_id, hierarchy = [page.page_name])
        train_topics+=flat_list
        for _id, name, hierarchy in flat_list:
            outf.write("{};{};{}\n".format(_id, name, " ".join(hierarchy)))
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
from collections import defaultdict
relevants = defaultdict(lambda: set())
for line in tqdm(open(os.path.join(data_home, 'benchmarkY1/benchmarkY1-train/train.pages.cbor-hierarchical.qrels'))):
    topic, _, doc, label = line.strip().split()
    if label == "1":
        relevants[topic].add(doc)

def truncate_seq_pair(tokens_a, tokens_b, max_length=509):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) < len(tokens_b):
            tokens_b.pop()
        else:
            tokens_a.pop()

with open(os.path.join(data_home, "data/samples/triples_tokenized.bert"), 'w', encoding='utf-8') as outf:
    for counter, (canonical_id, topic, hierarchy) in tqdm(enumerate(train_topics), total = len(train_topics)):
        query = " ".join(hierarchy)
        tokenized_query = tokenizer.tokenize(query)
        relevant_docs = relevants[canonical_id]
        for doc in relevants[canonical_id]:
            triple_id = "{}-{}".format(canonical_id, doc)
            doc_text = all_docs[doc]
            tokenized_doc = tokenizer.tokenize(doc_text)
            truncate_seq_pair(tokenized_query, tokenized_doc)
            assert len(final_version) <= 512
            final_version = ['[CLS]']+tokenized_query+['[SEP]']+tokenized_doc+['[SEP]']
            outf.write("{}\t{}\t1\n".format(triple_id, str(final_version)))
            #negative sampling
            neg_doc = random.choice(all_docs_ids)
            doc_text = all_docs[neg_doc]
            tokenized_doc = tokenizer.tokenize(doc_text)
            truncate_seq_pair(tokenized_query, tokenized_doc)
            final_version = ['[CLS]']+tokenized_query+['[SEP]']+tokenized_doc+['[SEP]']
            assert len(final_version) <= 512
            outf.write("{}\t{}\t0\n".format(triple_id, str(final_version)))
