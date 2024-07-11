# import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NLTKSimilarity:
    def text_similarity(text1, text2):
        # Tokenize and lemmatize the texts
        tokens1 = word_tokenize(text1)
        tokens2 = word_tokenize(text2)
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

        # Remove stopwords
        stop_words = stopwords.words('english')
        tokens1 = [token for token in tokens1 if token not in stop_words]
        tokens2 = [token for token in tokens2 if token not in stop_words]

        # Create the TF-IDF vectors
        vectorizer = TfidfVectorizer()
        vector1 = vectorizer.fit_transform(tokens1)
        vector2 = vectorizer.transform(tokens2)

        # Calculate the cosine similarity
        similarity = cosine_similarity(vector1, vector2)

        return similarity