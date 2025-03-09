from collections import defaultdict
import json
import math
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

def compute_tf_idf_score(term_freq: int, doc_freq: int, total_docs: int) -> float:
    """
    Compute tf-idf score for a term
    """
    if doc_freq == 0:
        return 0
    tf = 1 + math.log10(term_freq)  # logarithmic tf
    idf = math.log10(total_docs / doc_freq)  # inverse document frequency
    return tf * idf

def load_scores(partial_index_num: str) -> Dict:
    """
    Load ranking scores for a partial index
    """
    scores_file = f"partial_index_{partial_index_num}_scores.json"
    try:
        with open(scores_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No scores file found for partial index {partial_index_num}")
        return {}

def compute_final_score(doc_id: str, query_terms: List[str], 
                       tf_idf_score: float, scores_data: Dict) -> Tuple[float, Dict]:
    """
    Compute final score combining all ranking signals.
    
    Args:
        doc_id: Document identifier
        query_terms: List of processed query terms
        tf_idf_score: Base tf-idf score for the document
        scores_data: Dictionary containing all ranking data
        
    Returns:
        Tuple of (final_score, score_components)
    """
    # Initialize score components
    score_components = {
        'tf_idf': tf_idf_score,
        'pagerank': 0.0,
        'hub': 0.0,
        'authority': 0.0,
        'anchor': 0.0
    }
    
    # Get ranking data
    pagerank = scores_data.get('pagerank', {})
    hub_scores = scores_data.get('hub', {})
    auth_scores = scores_data.get('authority', {})
    anchor_texts = scores_data.get('anchor_texts', {})
    
    # Add link-based scores
    score_components['pagerank'] = pagerank.get(doc_id, 0)
    score_components['hub'] = hub_scores.get(doc_id, 0)
    score_components['authority'] = auth_scores.get(doc_id, 0)
    
    # Compute anchor text score
    anchor_score = 0
    doc_anchors = anchor_texts.get(doc_id, [])
    for anchor in doc_anchors:
        matching_terms = sum(1 for term in query_terms if term in anchor.lower())
        anchor_score += matching_terms
    score_components['anchor'] = anchor_score
    
    # Combine all scores with weights
    final_score = (
        0.4 * score_components['tf_idf'] +
        0.3 * score_components['pagerank'] +
        0.1 * score_components['hub'] +
        0.1 * score_components['authority'] +
        0.1 * score_components['anchor']
    )
    
    return final_score, score_components

def boolean_search(index: Dict, query: str, scores_data: Dict) -> Tuple[set, Dict]:
    """
    Enhanced boolean search with multiple ranking signals
    """
    stemmer = PorterStemmer()
    query_terms = [stemmer.stem(term.lower()) for term in query.split()]
    
    result_docs = None
    tf_idf_scores = defaultdict(float)
    final_scores = {}
    score_details = {}
    
    # Get document frequency data
    doc_freqs = scores_data.get('doc_frequencies', {})
    total_docs = scores_data.get('total_docs', 1)
    
    # Process each query term
    for term in query_terms:
        postings = index.get(term, [])
        docs_with_term = {doc_id: freq for doc_id, freq in postings}
        
        if result_docs is None:
            result_docs = set(docs_with_term.keys())
        else:
            result_docs &= set(docs_with_term.keys())
            
        # Early termination if no matches
        if not result_docs:
            return set(), {}
            
        # Update tf-idf scores
        for doc_id, freq in docs_with_term.items():
            if doc_id in result_docs:
                tf_idf = compute_tf_idf_score(freq, doc_freqs.get(term, 0), total_docs)
                tf_idf_scores[doc_id] += tf_idf
    
    # Compute final scores for matching documents
    for doc_id in result_docs:
        final_score, components = compute_final_score(
            doc_id,
            query_terms,
            tf_idf_scores[doc_id],
            scores_data
        )
        final_scores[doc_id] = final_score
        score_details[doc_id] = components
    
    return result_docs, score_details







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
    """
    Enhanced main function that combines:
    1. Auxiliary dictionary for efficient disk access
    2. On-demand boolean search
    3. Enhanced ranking with multiple signals
    """
    # Path to the final merged index file
    index_file = "final_index.txt"
    
    # Load ranking scores
    scores_data = load_scores("final")  # Load final merged scores
    
    # Build the auxiliary dictionary
    print("Building auxiliary dictionary...")
    aux_dict = build_auxiliary_dictionary(index_file)
    print("Auxiliary dictionary built successfully.")
    
    while True:
        query = input("\nEnter query (or type 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        # Time the query processing
        query_start = time.time()
        
        # Get initial matches using on-demand boolean search
        matching_docs, basic_scores = on_demand_boolean_search(query, aux_dict, index_file)
        
        if matching_docs:
            # Enhance scores with additional ranking signals
            enhanced_scores = {}
            score_details = {}
            
            # Process query terms for anchor text matching
            stemmer = PorterStemmer()
            query_terms = [stemmer.stem(term.lower()) for term in query.split()]
            
            # Compute enhanced scores for matching documents
            for doc_id in matching_docs:
                # Use basic tf score as tf-idf score (simplified)
                tf_idf_score = basic_scores[doc_id]
                
                # Compute final score with all ranking signals
                final_score, components = compute_final_score(
                    doc_id,
                    query_terms,
                    tf_idf_score,
                    scores_data
                )
                enhanced_scores[doc_id] = final_score
                score_details[doc_id] = components
            
            # Rank documents by enhanced scores
            ranked_docs = sorted(matching_docs, 
                               key=lambda d: enhanced_scores[d], 
                               reverse=True)
            
            query_end = time.time()
            elapsed = query_end - query_start
            
            # Display results with detailed scoring
            print(f"\nFound {len(ranked_docs)} matching document(s) in {elapsed:.3f} seconds:")
            for i, doc in enumerate(ranked_docs[:5], 1):
                print(f"\n{i}. Document: {doc}")
                print(f"   Final Score: {enhanced_scores[doc]:.4f}")
                print("   Score Components:")
                for component, value in score_details[doc].items():
                    print(f"      {component}: {value:.4f}")
        else:
            query_end = time.time()
            elapsed = query_end - query_start
            print(f"No documents found for the query: {query} ({elapsed:.3f} seconds)")

if __name__ == "__main__":
    main()
