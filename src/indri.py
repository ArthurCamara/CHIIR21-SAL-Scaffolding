import os
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

import os
import subprocess
import logging
import re
from collections import defaultdict

def generate_index():
    param_file_format = """
    <parameter>
        <corpus>
            <path>{}</path>
            <class>trectext</class>
        </corpus>
        <index>{}</index>
        <memory>32G</memory>
        <threads>{}</threads>
    </parameter>"""
    if not os.path.isdir(os.path.join(config.data_home, "indexes")):
        os.mkdir(os.path.join(config.data_home, "indexes"))
    if not os.path.isdir(os.path.join(config.data_home, "indri_params")):
        os.mkdir(os.path.join(config.data_home, "indri_params"))
    data_dir = os.path.join(os.path.join(config.data_home, "docs/wikipedia_paragaphs_clean.trec"))
    index_path = os.path.join(os.path.join(config.data_home, "indexes/wikipedia_paragraphs_clean"))
    param_file_format = param_file_format.format(data_dir, index_path, config.number_of_cpus)
    param_file = os.path.join(config.data_home, "indri_params/indexing.param")

    if os.path.isdir(index_path) and "index" not in config.force_steps:
        # wandb.save(os.path.join(index_path, "index/0/manifest"))
        logging.info("Index  already exists at %s. Skipping it.", index_path)
        # wandb.save(param_file)
        return
    with open(param_file, 'w') as outf:
        outf.write(param_file_format)
    # wandb.save(param_file)
    # Run indri processing
    cmd = "{} {}".format(os.path.join(config.indri_bin_path, "IndriBuildIndex"), param_file)
    logging.info(cmd)
    subprocess.run(cmd.split())
    # save manifest on wandb
    # wandb.save(os.path.join(index_path, "index/0/manifest"))

def generate_custom_index(data_dir, docs_name, metadata=None):
    param_file_format = """
    <parameter>
        <corpus>
            <path>{}</path>
            <class>trectext</class>
        </corpus>
        <index>{}</index>
        <memory>32G</memory>
        <threads>{}</threads>
        <field>
            <name>text</name>
        </field>
    </parameter>"""

    if metadata is None:
        metadata = ["text"]
    else:
        metadata.append("text")
    extra_fields_format = "<field>{}</field>"
    metadata_format = "<metadata>{}</metadata>"
    extra_fields = "\n".join([extra_fields_format.format(x) for x in metadata])
    
    metadata = metadata_format.format(extra_fields)
    param_file_format = param_file_format.replace("</parameter>",  "{}\n</parameter>".format(metadata))
    if not os.path.isdir(os.path.join(config.data_home, "indexes")):
        os.mkdir(os.path.join(config.data_home, "indexes"))
    if not os.path.isdir(os.path.join(config.data_home, "indri_params")):
        os.mkdir(os.path.join(config.data_home, "indri_params"))
    index_path = os.path.join(os.path.join(config.data_home, "indexes/{}".format(docs_name)))
    param_file_format = param_file_format.format(data_dir, index_path, config.number_of_cpus)
    param_file = os.path.join(config.data_home, "indri_params/indexing_{}.param".format(docs_name))

    if os.path.isdir(index_path) and "index" not in config.force_steps:
        logging.info("Index  already exists at %s. Skipping it.", index_path)
        return
    with open(param_file, 'w') as outf:
        outf.write(param_file_format)
    cmd = "{} {}".format(os.path.join(config.indri_bin_path, "IndriBuildIndex"), param_file)
    logging.info(cmd)
    subprocess.run(cmd.split())
 

def run_queries(queries):
    """Create params file and run it, based on list of queries"""
    if not os.path.isdir(os.path.join(config.data_home, "runs")):
        os.mkdir(os.path.join(config.data_home, "runs"))
    indri_param_format = """<parameters>
    <threads>{}</threads>
    <trecFormat>true</trecFormat>
    <index>{}</index>
    <count>{}</count>
    <runID>{}</runID>
    {}
</parameters>"""
    query_param_format = "  <query>\n    <number>{}</number>\n    <text>#combine({})</text>\n  </query>"
    index_path = os.path.join(os.path.join(config.data_home, "indexes/wikipedia_paragraphs_clean"))
    param_path = os.path.join(config.data_home, "indri_params", "QL.indriparam")
    runID = "QL_indri"
    queries_lines = []
    if not os.path.isfile(param_path) or "query" in config.force_steps:
        pattern = re.compile('([^\s\w]|_)+')  # noqa W605
        for query_id, query in queries:
            query = pattern.sub(' ',query)
            queries_lines.append(query_param_format.format(query_id, query))
        all_queries_lines = "\n".join(queries_lines)
        indri_param_format = indri_param_format.format(config.number_of_cpus, index_path, config.indri_top_k, runID, all_queries_lines)
        with open(param_path, 'w') as outf:
            outf.write(indri_param_format)
        logging.info("Saved params file at %s", param_path)
    else:
        logging.info("Already found file %s. Not recreating it", param_path)

    #actually run indri
    if not os.path.isdir(os.path.join(config.data_home, "runs")):
        os.mkdir(os.path.join(config.data_home, "runs"))
        logging.info("Creating runs folder at %s", os.path.join(config.data_home, "runs"))
    run_path = os.path.join(config.data_home, "runs/QL.run")
    if not os.path.isfile(run_path) or "query" in config.force_steps:
        indri_path = os.path.join(config.indri_bin_path, "IndriRunQuery")
        logging.info("Running Indri process with command %s %s", indri_path, param_path)
        output = subprocess.check_output([indri_path, param_path])
        with open(run_path, 'w') as outf:
            outf.write(output.decode("utf-8"))
    scores = dict()
    retrieved = defaultdict(lambda:[])
    for line in  open(run_path):
        query_id, _, doc_id, _, score, _ = line.split()
        scores["{}-{}".format(query_id, doc_id)] = float(score)
        retrieved[query_id].append(doc_id)
    return retrieved, scores






