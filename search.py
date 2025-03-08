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

def build_auxiliary_dictionary(index_file_path: str) -> dict:
    """
    Builds an auxiliary term dictionary from a final inverted index file.
    Each line in the index file is assumed to be in the format:
        term|posting1,posting2,...
    The dictionary maps each term to the byte offset at which its posting list starts.
    """
    auxiliary_dict = {}
    with open(index_file_path, 'r', encoding='utf-8') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split('|', 1)
            if len(parts) < 2:
                continue
            term = parts[0].strip()
            auxiliary_dict[term] = offset
    return auxiliary_dict

def get_postings_for_term(term: str, aux_dict: dict, index_file_path: str):
    """
    Retrieves the posting list for the given term by using the auxiliary dictionary
    to seek to the correct position in the final index file.
    """
    if term not in aux_dict:
        return []
    offset = aux_dict[term]
    with open(index_file_path, 'r', encoding='utf-8') as f:
        f.seek(offset)
        line = f.readline().strip()
        parts = line.split('|', 1)
        if len(parts) < 2:
            return []
        postings_str = parts[1]
        postings = []
        for posting in postings_str.split(','):
            if ':' not in posting:
                continue
            # Split on the last occurrence of ':' to handle doc_ids that might contain ':'
            doc_id, freq_str = posting.rsplit(':', 1)
            # Decode the URL-encoded colon if present
            doc_id = doc_id.replace('%3A', ':')
            try:
                freq = int(freq_str)
                postings.append((doc_id, freq))
            except ValueError:
                continue
    return postings

def on_demand_boolean_search(query: str, aux_dict: dict, index_file_path: str):
    """
    Processes the query by tokenizing and stemming, then performing an AND search.
    Retrieves posting lists on-demand from disk using the auxiliary dictionary.
    Returns a set of matching document IDs and a dictionary of scores (sum of term frequencies).
    """
    stemmer = PorterStemmer()
    tokens = nltk_tokenize(query)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    result_docs = None
    scores = {}
    
    for token in stemmed_tokens:
        postings = get_postings_for_term(token, aux_dict, index_file_path)
        docs_with_token = {doc_id: freq for doc_id, freq in postings}
        if result_docs is None:
            result_docs = set(docs_with_token.keys())
            scores = docs_with_token.copy()
        else:
            result_docs = result_docs.intersection(docs_with_token.keys())
            new_scores = {}
            for doc_id in result_docs:
                new_scores[doc_id] = scores.get(doc_id, 0) + docs_with_token[doc_id]
            scores = new_scores
    
    return result_docs, scores

def main():
    # Path to the final merged index file
    index_file = "final_index.txt"
    
    # Build the auxiliary dictionary
    aux_dict = build_auxiliary_dictionary(index_file)
    print("Auxiliary dictionary built successfully.")
    
    while True:
        query = input("\nEnter query (or type 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        query_start = time.time()
        matching_docs, scores = on_demand_boolean_search(query, aux_dict, index_file)
        query_end = time.time()
        elapsed = query_end - query_start
        if matching_docs:
            ranked_docs = sorted(matching_docs, key=lambda d: scores[d], reverse=True)
            print(f"\nFound {len(ranked_docs)} matching document(s) in {elapsed:.3f} seconds: Top results:")
            for doc in ranked_docs[:5]:
                print(f"Document: {doc}, Score: {scores[doc]}")
        else:
            print(f"No documents found for the query: {query}")

if __name__ == "__main__":
    main()
