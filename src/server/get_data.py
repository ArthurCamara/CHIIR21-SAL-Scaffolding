import pickle

def load_embeddings():
    subtopics_bert_vectors = pickle.load("subtopic_vectors.pkl", 'rb')