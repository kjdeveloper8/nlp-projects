import re
from nltk.corpus import stopwords
from helper.util import Estimate

class TextEncoding:
    def __init__(self, text) -> None:
        self.text = text
        self.word = text.split()
        self.vocabulary = self.vocab()
        self.word_to_index = self.word2id()

    def vocab(self):
        vocabulary = []
        for w in self.word:
            if w not in vocabulary:
                vocabulary.append(w)
        return vocabulary
    
    def word2id(self):
        return {word: idx for idx, word in enumerate(self.vocabulary)}
    
    @Estimate.timer
    def one_hot_encoding(self):
        """ One hot encoding."""
        encoded_text = []
        for each in self.word:
            one_hot_encoded = [1 if word == each else 0 for word in self.word]
            encoded_text.append(one_hot_encoded)
        return encoded_text

    @Estimate.timer 
    def index_based_encoding(self):
        """ Index based encoding."""
        encoded_corpus = []
        encoded_sentence = [self.word_to_index[w]+1 for w in self.word]
        encoded_corpus.append(encoded_sentence)
        return encoded_corpus       

    @Estimate.timer
    def bow_encoding(self):
        """ Bag of words."""
        # Initialize a vector of zeros with the same length as the vocabulary
        bow_vector = [0] * len(self.vocabulary)
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in self.word if w.lower() not in stop_words]

        for word in filtered_words:        
            if word in self.word_to_index:
                bow_vector[self.word_to_index[word]] += 1
        return bow_vector

    @Estimate.timer
    def tfidf_encoding(self):
        """ Term Frequency Inverse Document Frequency. """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from pandas import DataFrame

        tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
        tfidf_matrix = tfidf_vectorizer.fit_transform([self.text])
        # tfidf_matrix : (doc_is, term_index) score/value -- generates csr_matrix
        feature_names = tfidf_vectorizer.get_feature_names_out()
        # print(f"{feature_names=}")

        tfidf_df = DataFrame(data=tfidf_matrix.toarray(), columns=feature_names)
        return tfidf_df, tfidf_matrix

    @Estimate.timer
    def word2vec_encoding(self, term1='', term2='', model="en_core_web_sm"):
        """ Word2vec. 
            term1: term to match
            term2: term to match
            model: model to use
        """
        from spacy import load, info
        # print(info()['pipelines'].keys()) # list out installed model
        spacy_model = load('en_core_web_sm')
        doc = spacy_model(self.text)
        match_term = spacy_model(term2)

        # print("Token pos, dep --> ",[(token.text, token.pos_, token.dep_) for token in doc])
        # vector
        # print("Vector norm --> ",[(token.text, token.has_vector, token.vector_norm, token.is_oov) for token in doc])
        
        # similarity
        # print("Similarity --> ",[(token.text, token2.text, token.similarity(token2)) for token in doc for token2 in doc])
        similarity = [token.similarity(term2) for token in doc for term2 in match_term if token.text==term1]

        return similarity


if __name__ == "__main__":
    sentence = "I know how to bake cake, that to know it very well"
    sentence2 ="Odd number is a good number and numbers are good"
    e = TextEncoding(sentence2)
    result = e.one_hot_encoding()
    # result = e.bow_encoding()
    # result = e.index_based_encoding()
    # result, csr_matrix = e.tfidf_encoding()
    # result = e.word2vec_encoding('bake', 'cake')
    print(result)
