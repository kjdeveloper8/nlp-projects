from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SklearnSimilarity:
    def text_similarity(text1, text2):
        # Convert the texts into TF-IDF vectors
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text1, text2])

        # Calculate the cosine similarity
        similarity = cosine_similarity(vectors)
        return similarity