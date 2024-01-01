from .models import Documentdata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def vector_model_search(query):
    # Tokenize the query
    query_terms = [query.lower()]

    # Vectorize the documents and the query
    vectorizer = CountVectorizer()
    corpus = [doc.content for doc in Documentdata.objects.all()]
    X = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform(query_terms)

    # Calculate cosine similarity between query and documents
    similarities = cosine_similarity(query_vector, X).flatten()

    # Sort documents by similarity and return the results
    results = [doc for doc, similarity in zip(Documentdata.objects.all(), similarities)]
    results.sort(key=lambda x: -similarities[x.id - 1])  # Sorting based on similarity score
    return results
