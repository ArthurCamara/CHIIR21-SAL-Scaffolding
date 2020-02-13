import os
from clean_dataset import clean_dataset, generate_trec_file
from indri import generate_index, generate_custom_index
from generate_relevants import get_all_relevant_docs, generate_docs_offset, get_content
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm.auto import tqdm
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from trec_car import read_data
import urllib

def load_config():
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
    return config

config = load_config()


def get_level2(headings, parents, hierarchy=[], depth=1):
    """extract all leaves or level2 headings from an hierarchy"""
    all_headings = []
    for i in headings:
        cannonical_name = parents+"/"+i[0].headingId
        if len(i[1]) < 1 or depth == 2:
            all_headings.append((cannonical_name, i[0].heading, hierarchy + [i[0].heading]))
        all_headings += (get_level2(i[1], cannonical_name, hierarchy + [i[0].heading], depth=depth+1))
    return all_headings

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def tf_idf_all_docs():
    all_docs = []
    ids = []
    clean_docs_path = os.path.join(config.data_home, "docs/clean_docs.tsv")
    for line in tqdm(open(clean_docs_path), desc="loading files"):
        doc_id, doc = line.strip().split("\t")
        all_docs.append(doc)
        ids.append(doc_id)
    vectorizer = TfidfVectorizer(stop_words="english")
    print("computing tf-idfs")
    tf_idfs = vectorizer.fit_transform(all_docs)
    print("done")
    pickle.dump(tf_idfs, open(os.path.join(config.data_home, "tf-idfs.pkl"), 'wb'), protocol=4)


def get_all_relevant_docs_as_queries():
    indri_param_format = """<parameters>
    <threads>{}</threads>
    <trecFormat>true</trecFormat>
    <baseline>tfidf,k1:1.0,b:0.3</baseline>
    <index>{}</index>
    <count>{}</count>
    <runID>{}</runID>
    {}
</parameters>"""
    query_param_format = "  <query>\n    <number>{}</number>\n    <text>{}</text>\n  </query>"
    index_path = os.path.join(os.path.join(config.data_home, "indexes/wikipedia_paragraphs_clean"))
    param_path = os.path.join(config.data_home, "indri_params", "QL_for_TFIDF.indriparam")
    runID = "QL_indri"
    queries_lines = []
    corpus_path = os.path.join(config.data_home, "docs/clean_docs.tsv")
    docs_offset_dict = generate_docs_offset(corpus_path)
    qrel_path = os.path.join(config.raw_data_home, "benchmarkY1/benchmarkY1-train/train.pages.cbor-hierarchical.qrels")
    pattern = re.compile(r'([^\s\w]|_)+')  # noqa W605`
    with open(param_path,  'w', encoding="utf-8", errors="ignore") as outf:
        for _, line in tqdm(enumerate(open(qrel_path)), desc="loading all docs as queries"):
            doc_id = line.strip().split(" ")[2]
            try:
                doc_text = get_content(doc_id, corpus_path, docs_offset_dict)
            except:
                print("{} not found on clean docs".format(doc_id))
                continue
            query = pattern.sub(' ',doc_text)
            queries_lines.append(query_param_format.format(doc_id, query))
        all_queries_lines = "\n".join(queries_lines)
        indri_param_format = indri_param_format.format(config.number_of_cpus, index_path, config.indri_top_k, runID, all_queries_lines)
        outf.write(indri_param_format)
    print("Saved params file at {}".format(param_path))
    if not os.path.isdir(os.path.join(config.data_home, "runs")):
        os.mkdir(os.path.join(config.data_home, "runs"))
        print("Creating runs folder at %s" % os.path.join(config.data_home, "runs"))
    run_path = os.path.join(config.data_home, "runs/QL_TFIDF.run")
 
    indri_path = os.path.join(config.indri_bin_path, "IndriRunQuery")
    assert os.path.isfile(indri_path), ("could not find indri binaries at %s"% indri_path)
    print("Running Indri process with command %s %s with %d queries" % (indri_path, param_path, len(all_queries_lines)))
    output = subprocess.check_output([indri_path, param_path])
    with open(run_path, 'w', encoding="utf-8", errors="ignore") as outf:
        outf.write(output.decode("utf-8"))


def process_chunk(data):
    i, chunk = data
    output_dict = dict()
    for original_doc, original_doc_vec, target_doc, target_doc_vec in tqdm(chunk, position=i):
        sim = cosine_similarity(original_doc_vec, target_doc_vec)
        if sim >= 0.3:
            if original_doc not in output_dict:
                output_dict[original_doc] = []
            output_dict[original_doc].append(target_doc)
    with open("/ssd2/arthur/TRECCAR/data/tmp/{}.pkl".format(i), 'wb') as outf:
        pickle.dump(output_dict, outf, protocol=4)



def get_similar_docs():
    run_file = os.path.join(config.data_home, "runs/QL_TFIDF.run")
    # tf_idfs = pickle.load( open(os.path.join(config.data_home, "tf-idfs.pkl"), 'rb'))
    # get doc ids
    outf = os.path.join(config.data_home, "docs/clean_docs_indexes.pkl")
    if os.path.isfile(outf):
        doc_indexes = pickle.load(open(outf, 'rb'))
    else:
        doc_indexes = dict()
        clean_docs_path = os.path.join(config.data_home, "docs/clean_docs.tsv")
        for counter, line in tqdm(enumerate(open(clean_docs_path, encoding="utf-8", errors="surrogateescape")), desc="loading doc ids"):
            doc_id, _ = line.strip().split("\t")
            doc_indexes[doc_id] = counter
        pickle.dump(doc_indexes,open(outf, 'wb'))
    
    # all_lines = []
    # load every line
    # for line in tqdm(open(run_file), desc="loading all lines from indri run"):
    #     original_doc, _, target_doc, _, _, _ = line.split()
    #     original_doc_vec = tf_idfs[doc_indexes[original_doc]]
    #     target_doc_vec = tf_idfs[doc_indexes[target_doc]]
    #     all_lines.append((original_doc, original_doc_vec, target_doc, target_doc_vec))

    # excess_docs = len(all_lines) % config.number_of_cpus
    # number_of_chunks = config.number_of_cpus
    # if excess_docs> 0:
    #     number_of_chunks = config.number_of_cpus - 1
    # docs_per_chunk = len(all_lines)//number_of_chunks
    # with ProcessPoolExecutor(max_workers = config.number_of_cpus) as p:
    #     p.map(process_chunk, enumerate(chunks(all_lines, docs_per_chunk)))


    # TODO: Join dicts, create new .qrels, use previous script to generate .trec and index with relevants
    
    all_relevants = dict()
    for i in range(config.number_of_cpus):
        local_dict_path = os.path.join(config.data_home, "tmp/{}.pkl".format(i))
        local_dict = pickle.load(open(local_dict_path, 'rb'))
        for doc in local_dict:
            if doc not in all_relevants:
                all_relevants[doc] = []
            all_relevants[doc] += local_dict[doc]
    qrel_path = os.path.join(config.raw_data_home, "benchmarkY1/benchmarkY1-train/train.pages.cbor-hierarchical.qrels")
    new_qrel = os.path.join(config.data_home, "topics/expanded_relevants.qrel")
    qrel_format = "{} 0 {} 1\n"
    with open(new_qrel, 'w') as outf:
        for line in tqdm(open(qrel_path), desc="writing new qrels file"):
            topic, _, doc, _ = line.split()
            if doc in all_relevants:
                extra_relevants = list(set(all_relevants[doc]))
            else:
                extra_relevants = []
            outf.write(line)
            for extra_doc in extra_relevants:
                outf.write(qrel_format.format(topic, extra_doc))



def generate_expanded_index():
    docs_path = os.path.join(config.data_home, "docs/clean_docs.tsv")
    topics_path = os.path.join(config.raw_data_home, "benchmarkY1/benchmarkY1-train/train.pages.cbor-outlines.cbor")
    topics_to_use = []

    non_tokenized_queries = dict()
    for page in tqdm(read_data.iter_annotations(open(topics_path, 'rb')), total=65, desc="Reading and tokening topics"):
        # If the topic does not have hierarchy, we don't care about it
        if len(page.nested_headings()) == len(page.flat_headings_list()):
            continue
        topics_to_consider = get_level2(page.deep_headings_list(), page.page_id, hierarchy=[page.page_name])
        topics_to_use += topics_to_consider
        for topic_id, _, hierarchy in topics_to_consider:
            query = " ".join(hierarchy)
            non_tokenized_queries[topic_id] = query
    
    # Load relevants based on qrels

    relevant_per_doc = defaultdict(lambda:set())
    qrel_path = os.path.join(config.data_home, "topics/expanded_relevants.qrel")
    for line in open(qrel_path):
        topic, _, doc, _ = line.split(" ")
        hierarchy = urllib.parse.unquote(topic).split("/")
        if len(hierarchy) >= 2:
            cannonical_id = "/".join(topic.split("/")[:3])
            relevant_per_doc[doc].add(cannonical_id)
    doc_format = "<DOC>\n<DOCNO>{}</DOCNO>\n<text>{}</text>\n{}</DOC>\n"
    full_docs_offset = generate_docs_offset(docs_path)
    final_docs_path = os.path.join(config.data_home, "docs/docs_with_expanded_relevance.trec")
    with open(final_docs_path, 'w') as outf:
        for _, doc in tqdm(enumerate(full_docs_offset), desc="dumping trec docs with expanded relevants", total = len(full_docs_offset)):
            # doc_scores = []
            relevant_docs_format ="<relevants>{}</relevants>"
            doc_text = get_content(doc,docs_path, full_docs_offset)
            relevant_topics = list(relevant_per_doc[doc])
            if len(relevant_topics) > 0:
                relevants = relevant_docs_format.format(",".join(relevant_topics))
            else:
                relevants  = relevant_docs_format.format(" ")
            doc_full = doc_format.format(doc, doc_text, relevants)
            outf.write(doc_full)

                # break
    generate_custom_index(final_docs_path, "docs_with_expanded_relevants")
    




if __name__=="__main__":
    # clean_dataset()
    # generate_trec_file()
    # generate_index()
    
    # tf_idf_all_docs()
    # get_all_relevant_docs_as_queries()
    get_similar_docs()
    generate_expanded_index()


