from data import tokenize_content
from gensim.models import Word2Vec

def get_word2vec(essay_contents, remove_stopwords, dim): 
    sentences = []
    for content in essay_contents:
        sentences += tokenize_content(content, remove_stopwords, level = "sentence")
    print("training word2vec model")
    word2vec_model = Word2Vec(sentences, workers = 10, size = dim)
    return word2vec_model.wv