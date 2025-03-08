import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import time
# Download the necessary NLTK data (if not already downloaded)
nltk.download('punkt')

def nltk_tokenize(text: str):
    """Tokenizes text into lowercase alphanumeric tokens."""
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum()]
    return tokens

def load_index(file_path: str):
    """Loads the inverted index from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        index = json.load(f)
    # Convert postings (lists) to tuples for easier handling.
    for token in index:
        index[token] = [(doc_id, freq) for doc_id, freq in index[token]]
    return index

def boolean_search(index, query: str):
    """
    Processes the query by tokenizing and stemming, then performs an AND search.
    It returns a set of matching document IDs and a score dictionary where the score
    is the sum of term frequencies for the query tokens.
    """
    stemmer = PorterStemmer()
    tokens = nltk_tokenize(query)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Use the postings lists from the index to compute an intersection of documents.
    result_docs = None
    scores = {}  # For ranking: accumulate term frequencies
    
    for token in stemmed_tokens:
        postings = index.get(token, [])
        docs_with_token = {doc_id: freq for doc_id, freq in postings}
        if result_docs is None:
            result_docs = set(docs_with_token.keys())
            scores = docs_with_token.copy()
        else:
            # Keep only documents that contain the current token (AND query)
            result_docs = result_docs.intersection(docs_with_token.keys())
            # Update scores only for the intersection documents
            new_scores = {}
            for doc_id in result_docs:
                new_scores[doc_id] = scores.get(doc_id, 0) + docs_with_token[doc_id]
            scores = new_scores
    
    return result_docs, scores

def main():
    # This should match the filename you used when saving the index.
    index_file = "inverted_index_nltk.json"
    index = load_index(index_file)
    print("Inverted index loaded successfully.")
    
    while True:
        query = input("\nEnter query (or type 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        matching_docs, scores = boolean_search(index, query)
       
        if matching_docs:
            # Rank the documents by their accumulated score (higher is better)
            ranked_docs = sorted(matching_docs, key=lambda doc: scores[doc], reverse=True)
            print(f"\nFound {len(ranked_docs)} matching document(s). Top results:")
            for doc in ranked_docs[:5]:  # show top 5 results
                print(f"Document: {doc}, Score: {scores[doc]}")
        else:
            print("No documents found for the given query.")

if __name__ == "__main__":
    main()
