from typing import Dict, List, Tuple
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

def load_index(file_path: str) -> Dict:
    """
    Load index from text file format: token|doc1:freq1,doc2:freq2,...
    """
    index = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue
                
                try:
                    # Split line into token and postings
                    parts = line.strip().split('|')
                    if len(parts) != 2:
                        continue
                    
                    token, postings_str = parts
                    
                    # Process postings
                    postings = []
                    for posting in postings_str.split(','):
                        if not posting:
                            continue
                        try:
                            doc_id, freq = posting.rsplit(':', 1)  # Use rsplit to handle URLs with colons
                            postings.append((doc_id, int(freq)))
                        except (ValueError, IndexError):
                            continue
                    
                    if postings:
                        index[token] = postings
                        
                except Exception as e:
                    print(f"Error processing line: {line[:100]}...")
                    continue
                    
        print(f"Loaded {len(index)} tokens from index")
        return index
    
    except Exception as e:
        print(f"Error loading index: {e}")
        return {}


def compute_final_score(doc_id: str, basic_score: float, query_terms: List[str], 
                       pagerank_scores: Dict, hub_scores: Dict, 
                       auth_scores: Dict, anchor_texts: Dict) -> float:
    """
    Combine all ranking signals into a final score
    """
    # Get link-based scores (default to 0 if not found)
    pagerank = pagerank_scores.get(doc_id, 0)
    hub = hub_scores.get(doc_id, 0)
    auth = auth_scores.get(doc_id, 0)
    
    # Calculate anchor text score
    anchor_score = 0
    doc_anchors = anchor_texts.get(doc_id, [])
    for anchor in doc_anchors:
        matching_terms = sum(1 for term in query_terms if term in anchor.lower())
        anchor_score += matching_terms
    
    # Combine scores (weights can be adjusted)
    final_score = (
        0.4 * basic_score +     # Basic tf-idf
        0.3 * pagerank +        # PageRank
        0.1 * hub +            # HITS hub score
        0.1 * auth +           # HITS authority score
        0.1 * anchor_score     # Anchor text relevance
    )
    
    return final_score

def boolean_search(index: Dict, query: str, 
                  pagerank_scores: Dict={}, hub_scores: Dict={}, 
                  auth_scores: Dict={}, anchor_texts: Dict={}) -> Tuple[set, Dict]:
    """
    Enhanced boolean search with all ranking signals
    """
    stemmer = PorterStemmer()
    query_terms = [stemmer.stem(term.lower()) for term in query.split()]
    
    result_docs = None
    basic_scores = {}
    final_scores = {}
    
    for term in query_terms:
        postings = index.get(term, [])
        docs_with_term = {doc_id: freq for doc_id, freq in postings}
        
        if result_docs is None:
            result_docs = set(docs_with_term.keys())
            basic_scores = docs_with_term.copy()
        else:
            result_docs &= set(docs_with_term.keys())
            new_scores = {}
            for doc_id in result_docs:
                if doc_id in docs_with_term:
                    new_scores[doc_id] = basic_scores[doc_id] + docs_with_term[doc_id]
            basic_scores = new_scores
            
        if not result_docs:
            return set(), {}
    
    # Compute final scores using all ranking signals
    for doc_id in result_docs:
        final_scores[doc_id] = compute_final_score(
            doc_id,
            basic_scores[doc_id],
            query_terms,
            pagerank_scores,
            hub_scores,
            auth_scores,
            anchor_texts
        )
            
    return result_docs, final_scores


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
'''
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
        
        matching_docs, scores = boolean_search(index, query)
       
        if matching_docs:
            # Rank the documents by their accumulated score (higher is better)
            ranked_docs = sorted(matching_docs, key=lambda doc: scores[doc], reverse=True)
            print(f"\nFound {len(ranked_docs)} matching document(s). Top results:")
            for doc in ranked_docs[:5]:  # show top 5 results
                print(f"Document: {doc}, Score: {scores[doc]}")
        else:
            print("No documents found for the given query.")


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
'''